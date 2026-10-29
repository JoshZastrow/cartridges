"""hertz-dev on Modal: its image, the stock server start-up, a CPU dry run and the context probe.

    uv run voice deploy --backend hertz --system stock               # voice.deploy: app voice-serve, web function "hertz"
    uv run python -m voice.deploy_hertz --dry-run                     # CPU stand-in behind the real app; no Modal
    uv run python -m voice.deploy_hertz --probe-context 1024 2048     # what the stock model does when its cache fills

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
torch 2.5.1 but runs unpatched on 2.14 (bf16-level differences, measured on
#21). Upstream's code is cloned at `UPSTREAM_COMMIT` into /opt/hertz-dev; its
four checkpoints download from the HF repo si-pbc/hertz-dev into
`voice-data:hf-cache` on first start (about 17 GB) and load from there after.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import modal

from voice.backends.hertz import CHECKPOINT, CONTEXT_FRAMES, UPSTREAM, UPSTREAM_COMMIT

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
        frames, error = b.prompt.shape[1] // CHUNK, None
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
            step_ms = 1000 * (time.perf_counter() - t0) / max(frames, 1)
            m.deinit_cache()
        rows.append({"torch": str(torch.__version__), "cache_len": length, "prompt_frames": b.prompt.shape[1] // CHUNK,
                     "frames_at_failure": frames, "error": error, "ms_per_step": round(step_ms, 1),
                     "load_s": round(b.load_s, 1), "max_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        print(rows[-1], flush=True)
    return rows


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
    args = p.parse_args(argv)
    if args.dry_run:
        dry_run(args.prompt)
    elif args.probe_context:
        with modal.enable_output(), probe_app.run():
            rows = probe_context.remote(safe_name(args.prompt or DEFAULT_PROMPT), args.probe_context)
        print(json.dumps(rows, indent=2))
    else:
        p.error("deploy with `voice deploy --backend hertz`; here: --dry-run or --probe-context")
    return 0


if __name__ == "__main__":
    sys.exit(main())
