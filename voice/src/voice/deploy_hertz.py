"""hertz-dev on Modal: its image, the stock server start-up, a CPU dry run and the context probe.

    uv run voice deploy --backend hertz --system stock               # voice.deploy: app voice-serve, web function "hertz"
    uv run python -m voice.deploy_hertz --dry-run                     # CPU stand-in behind the real app; no Modal
    uv run python -m voice.deploy_hertz --probe-context 1024 2048     # what the stock model does when its cache fills
    uv run python -m voice.deploy_hertz --profile [--group exp-...]   # one cold start, timed per stage

`voice.deploy` holds the web function (app `voice-serve`, proxy auth, one
container, scale to zero); it runs `start` in a container built from `image`.
`start` loads the model, sets the stock prompt and serves upstream's protocol
on the port (`HertzBackend.serve`); clients connect to `<url>/audio`.

The stock prompt is #22's 30 s author segment, `voice-voice:/prompt-30s.wav`
(24 kHz, -24 LUFS; resampled to 16 kHz, not re-normalised), on the agent
channel with the user channel silent. Prompts are named by a bare file name
on `voice-voice`, checked by `safe_name` in the serving process; clients
never name one.

`image` is the workspace versions (root uv.lock: torch 2.14.1, numpy 2.3.5,
einops 0.9.0.dev0, hf-hub 1.32.0); #18 reuses it for training. hertz-dev pins
torch 2.5.1 and runs here on 2.14 with three patches from `voice.patches`,
applied in `HertzBackend.load` (upstream's tree is unchanged): `hertz_sdpa`
(flash/efficient attention; 2.14's default kernel is ~2.5x slower, #21),
`hertz_md5_stamp` and `hertz_fast_build` (#27, below). Outputs match 2.5.1 at
the bf16 floor (`voice/scripts/hertz_parity.py`). Upstream's code is cloned at
`UPSTREAM_COMMIT` into /opt/hertz-dev; its four checkpoints (20.6 GB) download
from the HF repo si-pbc/hertz-dev into `voice-data:hf-cache` on first start
and load from there after.

Cold start (#27, H200, v1 Volume reads 1.2-1.7 GB/s cold): about 31 s from
container start to listening, 91 s on the first start after a checkpoint
changes, down from ~352 s. Upstream MD5s every checkpoint on every start
(38 s) and random-initialises the fp32 model on the CPU before loading it
(293 s). `hertz_md5_stamp` hashes once and leaves `<file>.md5stamp` next to
the checkpoint, trusted while size and mtime_ns match; `loaded_backend`
commits the Volume so stamps persist. `hertz_fast_build` builds with
parameters on meta and assigns the checkpoint tensors. `--profile` times
each stage of one cold start.
"""

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import modal

from voice.backends.hertz import CHECKPOINT, CONTEXT_FRAMES, UPSTREAM, UPSTREAM_COMMIT

T_IMPORT = time.time()  # wall clock once this module's imports are done; in a container, ~Python start + imports
logger = logging.getLogger(__name__)

GPU = "H200"
DATA, VOICE = "/data", "/voice"
DEFAULT_PROMPT = "prompt-30s.wav"  # #22's 30 s author segment on voice-voice

# the workspace versions (root uv.lock), as voice.train.image; #18 reuses it
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "libsndfile1")
    .uv_pip_install(
        "torch==2.14.1", "numpy==2.3.5", "einops==0.9.0.dev0", "huggingface-hub==1.32.0", "tqdm==4.70.1",
        "requests==2.34.2", "scipy==1.18.1", "soundfile==0.14.0", "fastapi==0.141.1", "uvicorn==0.53.0",
        "aiohttp==3.14.3", "pydantic==2.14.0b2", "safetensors==0.9.0rc0",
        "websockets==16.1.1",  # uvicorn answers a WebSocket upgrade with 404 without it
        pre=True,
    )
    .run_commands(f"git clone {UPSTREAM} /opt/hertz-dev && git -C /opt/hertz-dev checkout {UPSTREAM_COMMIT}")
    .env({"HERTZ_DEV_DIR": "/opt/hertz-dev", "HF_HOME": f"{DATA}/hf-cache"})
    .add_local_python_source("voice", "shared")  # voice.deploy imports voice.train, which imports shared
)

data_volume = modal.Volume.from_name("voice-data", create_if_missing=True)
voice_volume = modal.Volume.from_name("voice-voice", create_if_missing=True)
volumes = {DATA: data_volume, VOICE: voice_volume}


def safe_name(name: str) -> str:
    """A prompt is named by a bare file name: no separators, no '..', not absolute."""
    if not name or "/" in name or "\\" in name or ".." in name or Path(name).is_absolute() or Path(name).name != name:
        raise ValueError(f"prompt must be a bare file name on voice-voice, got {name!r}")
    return name


def prompt_path(name: str) -> str:
    """Container path of a prompt on voice-voice."""
    return f"{VOICE}/{safe_name(name)}"


def loaded_backend(prompt: str = DEFAULT_PROMPT):
    """In the container: the hertz backend, loaded, with its stock prompt set."""
    from voice import backends

    b = backends.get("hertz", {"checkpoint": CHECKPOINT})
    t0 = time.monotonic()
    b.load("cuda")
    b.load_s = time.monotonic() - t0
    data_volume.commit()  # keep checkpoints fetched on a first start
    b.set_prompt(prompt_path(prompt))
    return b


def start(port: int):
    """In the container: load, set the stock prompt and serve on `port` (voice.deploy's web function)."""
    t0 = time.monotonic()
    b = loaded_backend()
    server = b.serve(None, port)
    server.timings = {"load_s": round(b.load_s, 1), "ready_s": round(time.monotonic() - t0, 1)}
    logger.info("hertz serving %s (prompt %s, %.1f s): %s", server.url, DEFAULT_PROMPT,
                b.prompt.shape[1] / 16000, server.timings)
    return server


probe_app = modal.App("voice-hertz-probe", image=image)


@probe_app.function(gpu=GPU, volumes=volumes, secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=3600)
def probe_context(prompt: str, cache_lens: list[int]) -> list[dict]:
    """Run upstream's live path (no guard) on silence until it fails; one row per cache length."""
    import torch

    from voice.backends.hertz import CHUNK, TEMPS

    b = loaded_backend(prompt)
    m, rows = b.model, []
    silence = torch.zeros(1, 1, CHUNK, device="cuda")
    for length in cache_lens:
        prompt_frames = b.prompt.shape[1] // CHUNK
        frames, error = prompt_frames, None
        with b._ctx():
            m.init_cache(bsize=1, device="cuda", dtype=torch.bfloat16, length=length)
            nxt = m.next_audio_from_audio(b._tensor(b.prompt)[None], temps=TEMPS)
            t0 = time.perf_counter()
            try:
                while frames < 4 * CONTEXT_FRAMES:
                    nxt = m.next_audio_from_audio(torch.cat([silence.to(nxt.dtype), nxt], 1), temps=TEMPS)
                    frames += 1
            except Exception as e:  # what the stock model does past its cache
                error = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            step_ms = 1000 * (time.perf_counter() - t0) / max(frames - prompt_frames, 1)  # live steps only
            m.deinit_cache()
        rows.append({"torch": str(torch.__version__), "cache_len": length, "prompt_frames": prompt_frames,
                     "frames_at_failure": frames, "error": error, "ms_per_step": round(step_ms, 1),
                     "load_s": round(b.load_s, 1), "max_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        print(rows[-1], flush=True)
    return rows


@probe_app.function(gpu=GPU, volumes=volumes, secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=3600)
def profile_load(t_call: float, prompt: str) -> list[dict]:
    """One cold start in the server's order, one row per stage (seconds). The
    first reader of each checkpoint reads it cold: the MD5 when a file has no
    valid stamp, else `torch.load`. A warm re-read of the same files follows."""
    import numpy as np
    import torch

    from voice.backends.hertz import CHUNK
    from voice.patches import hertz_fast_build, hertz_md5_stamp

    rows = [{"stage": "container start to Python + module imports", "s": round(T_IMPORT - t_call, 1)}]
    t0 = time.perf_counter()
    sys.path.insert(0, os.environ.get("HERTZ_DEV_DIR", "/opt/hertz-dev"))
    import model  # noqa: F401  upstream; cached for load()
    rows.append({"stage": "imports (upstream)", "s": round(time.perf_counter() - t0, 1)})

    calls = {"md5": [], "torch.load": [], "build": []}

    def timed(fn, key):
        def wrapped(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            path = a[0] if key != "build" else None
            calls[key].append((path, os.path.getsize(path) if path else 0, time.perf_counter() - t))
            return out
        return wrapped

    md5, load, build = hertz_md5_stamp.md5, torch.load, hertz_fast_build.build
    hertz_md5_stamp.md5, torch.load, hertz_fast_build.build = timed(md5, "md5"), timed(load, "torch.load"), timed(
        build, "build")
    try:
        b = loaded_backend(prompt)
    finally:
        hertz_md5_stamp.md5, torch.load, hertz_fast_build.build = md5, load, build

    def stage(name, key):
        n, s = sum(c[1] for c in calls[key]), sum(c[2] for c in calls[key])
        return {"stage": name, "s": round(s, 1), "GB": round(n / 1e9, 2), "MB_s": round(n / 1e6 / s) if s else None,
                "files": [Path(c[0]).name for c in calls[key]]}

    rows += [stage("MD5 (files without a valid stamp)", "md5"), stage("torch.load", "torch.load")]
    build_s = calls["build"][0][2]
    rows.append({"stage": "model build + load_state_dict (fast build)",
                 "s": round(build_s - rows[-1]["s"] - rows[-2]["s"], 1),
                 "missing_keys": sum(len(r["missing"]) for r in b.loads)})
    rows.append({"stage": ".bfloat16().to(cuda)", "s": round(b.load_s - build_s, 1),
                 "non_contiguous_params": sum(not p.is_contiguous() for p in b.model.parameters())})

    t0 = time.perf_counter()
    s = b.session()
    for _ in range(8):
        s.step(np.zeros(CHUNK, np.float32))
    torch.cuda.synchronize()
    warm_s, t1 = time.perf_counter() - t0, time.perf_counter()
    for _ in range(100):
        s.step(np.zeros(CHUNK, np.float32))  # returns audio on the CPU, so each step is synced
    s.close()
    rows.append({"stage": "warm-up (prompt prefill + 8 live steps)", "s": round(warm_s, 1),
                 "live_step_ms": round(10 * (time.perf_counter() - t1), 1)})
    t0 = time.perf_counter()
    server = b.serve(None, 8998)
    rows.append({"stage": "server listening", "s": round(time.perf_counter() - t0, 1)})
    server.server.should_exit = True
    rows.append({"stage": "total (container start to listening)", "s": round(sum(r["s"] for r in rows), 1)})

    n, t0 = 0, time.perf_counter()
    for path, *_ in calls["torch.load"]:
        with open(path, "rb") as f:
            while block := f.read(64 << 20):
                n += len(block)
    s = time.perf_counter() - t0
    rows.append({"stage": "raw read, warm (after the total)", "s": round(s, 1), "GB": round(n / 1e9, 2),
                 "MB_s": round(n / 1e6 / s)})
    rows.append({"stage": "peak VRAM GiB", "s": None, "GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    return rows


def log_profile(rows: list[dict], run) -> None:
    """The stage table and `profile/<stage>_s` summaries on a W&B run."""
    import wandb

    run.log({"profile/stages": wandb.Table(columns=["stage", "s", "detail"], data=[
        [r["stage"], r["s"], json.dumps({k: v for k, v in r.items() if k not in ("stage", "s")})] for r in rows])})
    run.summary.update({f"profile/{r['stage'].split(' (')[0].replace(' ', '_')}_s": r["s"] for r in rows})


def dry_run(prompt: str | None = None, cache_len: int = CONTEXT_FRAMES, steps: int = 16) -> dict:
    """CPU, no Modal: the backend with `CpuStandIn` behind the real app, driven
    through the wire protocol for `steps` chunks, then a rollout. `prompt` is a
    local wav (default: 2 s of noise)."""
    import numpy as np
    from fastapi.testclient import TestClient

    from voice import backends
    from voice.backends.hertz import CHUNK, CpuStandIn, make_app
    from voice.hertz_transport import pack, unpack

    b = backends.get("hertz", {"checkpoint": CHECKPOINT, "device": "cpu"})
    b.model = CpuStandIn()
    rng = np.random.default_rng(0)
    b.set_prompt(prompt if prompt and Path(prompt).is_file() else rng.uniform(-.1, .1, (1, 16 * CHUNK)).astype(np.float32))
    with TestClient(make_app(b, cache_len)).websocket_connect("/audio") as ws:
        for _ in range(steps):
            ws.send_text(pack(rng.uniform(-.1, .1, CHUNK).astype(np.float32)))
            assert unpack(ws.receive_text()).shape == (CHUNK,)
    d = b.rollout(None, rng.uniform(-.1, .1, steps * CHUNK).astype(np.float32), max_s=steps / 8, cache_len=cache_len)
    out = {"prompt_s": b.prompt.shape[1] / 16000, "steps": steps, "cache_len": cache_len, "rollout": d.agent_wav}
    print(f"dry run ok: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="CPU stand-in model behind the real app; no Modal")
    p.add_argument("--prompt", default=None, help="--dry-run: a local wav; --probe-context: a bare name on voice-voice")
    p.add_argument("--probe-context", type=int, nargs="+", metavar="CACHE_LEN",
                   help="run the stock model past these cache lengths on Modal")
    p.add_argument("--profile", action="store_true", help="time each stage of one cold start on Modal")
    p.add_argument("--group", default="", help="--profile: W&B group exp-YYYYMMDD-<slug>; empty: no W&B run")
    args = p.parse_args(argv)
    if args.dry_run:
        dry_run(args.prompt)
    elif args.probe_context:
        with modal.enable_output(), probe_app.run():
            rows = probe_context.remote(safe_name(args.prompt or DEFAULT_PROMPT), args.probe_context)
        print(json.dumps(rows, indent=2))
    elif args.profile:
        prompt = safe_name(args.prompt or DEFAULT_PROMPT)
        with contextlib.ExitStack() as stack:
            run = None
            if args.group:
                from shared import tracking

                run = stack.enter_context(tracking.start_run(
                    "voice", args.group, vars(args), name="hertz/stock/serve-smoke", job_type="eval", gpu=GPU,
                    tags=["check", "hertz", "profile"]))
            with modal.enable_output(), probe_app.run():
                rows = profile_load.remote(time.time(), prompt)
            if run is not None:
                log_profile(rows, run)
        print(json.dumps(rows, indent=2))
    else:
        p.error("deploy with `voice deploy --backend hertz`; here: --dry-run, --probe-context or --profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
