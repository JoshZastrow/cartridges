"""Modal app "voice-serve-hertz": stock hertz-dev behind upstream's WebSocket protocol.

    uv run python -m voice.deploy_hertz --system stock --prompt <wav> [--cache-len 2048]
    uv run python -m voice.deploy_hertz --probe-context 1024 2048     # what happens when the cache fills
    uv run python -m voice.deploy_hertz --dry-run [--prompt <local wav>]  # CPU, no Modal

The endpoint requires Modal proxy auth (Modal-Key / Modal-Secret headers;
`HertzTransport` sends them from MODAL_PROXY_TOKEN_ID / _SECRET). The prompt
is named by a bare file name, validated in the container (`safe_name`).
The default prompt is #22's 30 s author segment, `voice-voice:/prompt-30s.wav`
(24 kHz, -24 LUFS; resampled to 16 kHz, not re-normalised), placed on the
agent channel with the user channel silent. `deploy` uploads a local
`--prompt` wav to the Volume `voice-voice` under `prompts/`, writes the
serving config to `voice-data:serve/hertz.json`, deploys the app and prints
the WebSocket URL (`wss://.../audio`). The
container loads the model, reads that config and runs `HertzBackend.serve`
on port 8000 (`modal.web_server`). One GPU container, one session at a time.

`image` is the workspace versions (root uv.lock: torch 2.14.1, numpy 2.3.5,
einops 0.9.0.dev0, hf-hub 1.32.0), the image #18 reuses for training. hertz-dev
pins torch 2.5.1, but runs unpatched on 2.14 (bf16-level differences, measured
on #21). Upstream's code is cloned at
`UPSTREAM_COMMIT` into /opt/hertz-dev; its four checkpoints download from the
HF repo si-pbc/hertz-dev into `voice-data:hf-cache` on first start (about
17 GB) and load from there after.

`voice deploy --backend hertz --system stock` (cli.py, #20) calls `deploy`.
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import modal

from voice.backends.hertz import CHECKPOINT, CONTEXT_FRAMES, UPSTREAM, UPSTREAM_COMMIT

logger = logging.getLogger(__name__)

APP_NAME = "voice-serve-hertz"
PORT = 8000
GPU = "H200"
DATA, VOICE = "/data", "/voice"
CONFIG = "serve/hertz.json"  # on voice-data
DEFAULT_PROMPT = "prompt-30s.wav"  # #22's 30 s author segment on voice-voice


# the workspace versions (root uv.lock), as voice.train.image; #18 reuses it
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "libsndfile1")
    .uv_pip_install(
        "torch==2.14.1", "numpy==2.3.5", "einops==0.9.0.dev0", "huggingface-hub==1.32.0", "tqdm==4.70.1",
        "requests==2.34.2", "scipy==1.18.1", "soundfile==0.14.0", "fastapi==0.141.1", "uvicorn==0.53.0",
        "aiohttp==3.14.3", "pydantic==2.14.0b2", "safetensors==0.9.0rc0",
        pre=True,
    )
    .run_commands(f"git clone {UPSTREAM} /opt/hertz-dev && git -C /opt/hertz-dev checkout {UPSTREAM_COMMIT}")
    .env({"HERTZ_DEV_DIR": "/opt/hertz-dev", "HF_HOME": f"{DATA}/hf-cache"})
    .add_local_python_source("voice")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name("voice-data", create_if_missing=True)
voice_volume = modal.Volume.from_name("voice-voice", create_if_missing=True)
volumes = {DATA: data_volume, VOICE: voice_volume}


def _loaded_backend():
    from voice import backends

    b = backends.get("hertz", {"checkpoint": CHECKPOINT})
    t0 = time.perf_counter()
    b.load("cuda")
    b.load_s = time.perf_counter() - t0
    data_volume.commit()  # keep freshly downloaded checkpoints
    return b


@app.function(gpu=GPU, volumes=volumes, secrets=[modal.Secret.from_name("HF_TOKEN")],
              timeout=3600, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def serve():
    cfg = json.loads((Path(DATA) / CONFIG).read_text())
    b = _loaded_backend()
    b.set_prompt(prompt_path(cfg["prompt"], cfg["uploaded"]))
    print(f"hertz loaded in {b.load_s:.1f}s; prompt {cfg['prompt']} "
          f"({b.prompt.shape[1] / 16000:.1f}s); cache_len {cfg['cache_len']}", flush=True)
    b.serve(None, PORT, cache_len=cfg["cache_len"])


def _probe_context(prompt: str, cache_lens: list[int]) -> list[dict]:
    """Run upstream's live path (no guard) on silence until it fails; one row per cache length."""
    import torch

    from voice.backends.hertz import CHUNK, TEMPS

    b = _loaded_backend()
    b.set_prompt(prompt)
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
        rows.append({"torch": torch.__version__, "cache_len": length, "prompt_frames": b.prompt.shape[1] // CHUNK,
                     "frames_at_failure": frames, "error": error, "ms_per_step": round(step_ms, 1),
                     "load_s": round(b.load_s, 1), "max_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        print(rows[-1], flush=True)
    return rows


gpu_fn = dict(gpu=GPU, volumes=volumes, secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=3600)
probe_context = app.function(**gpu_fn)(_probe_context)


def safe_name(name: str) -> str:
    """A prompt is named by a bare file name: no separators, no '..', not absolute."""
    if not name or "/" in name or "\\" in name or ".." in name or Path(name).is_absolute() or Path(name).name != name:
        raise ValueError(f"prompt must be a bare file name on voice-voice, got {name!r}")
    return name


def prompt_path(name: str, uploaded: bool) -> str:
    """Container path of a prompt: voice-voice:/<name>, or /prompts/<name> for an uploaded one."""
    return f"{VOICE}/prompts/{safe_name(name)}" if uploaded else f"{VOICE}/{safe_name(name)}"


def upload_prompt(prompt: str) -> tuple[str, bool]:
    """A local wav goes to voice-voice:/prompts/<name>; anything else must be a bare
    file name already on voice-voice. Returns (name, uploaded)."""
    if not Path(prompt).is_file():
        return safe_name(prompt), False
    name = safe_name(Path(prompt).name)
    with voice_volume.batch_upload(force=True) as up:
        up.put_file(prompt, f"/prompts/{name}")
    return name, True


def deploy(system: str = "stock", prompt: str = DEFAULT_PROMPT,
           cache_len: int = CONTEXT_FRAMES) -> str:
    """Deploy the stock server and return its WebSocket URL."""
    if system != "stock":
        raise NotImplementedError("serving a cartridge on hertz is #12/#18")
    if not 0 < cache_len <= CONTEXT_FRAMES:
        raise ValueError(f"cache_len must be in 1..{CONTEXT_FRAMES} (the LM's RoPE table ends there)")
    name, uploaded = upload_prompt(prompt)
    cfg = {"system": system, "prompt": name, "uploaded": uploaded, "cache_len": cache_len}
    with data_volume.batch_upload(force=True) as up:
        up.put_file(io.BytesIO(json.dumps(cfg).encode()), f"/{CONFIG}")
    with modal.enable_output():
        app.deploy()
    url = serve.get_web_url().replace("https://", "wss://") + "/audio"
    print(f"config {cfg}\n{url}")
    return url


def dry_run(prompt: str | None = None, cache_len: int = CONTEXT_FRAMES, steps: int = 16) -> dict:
    """CPU, no Modal: the backend with `CpuStandIn` behind the real app, driven
    through the wire protocol for `steps` chunks, then a rollout. `prompt` is a
    local wav (default: 2 s of noise)."""
    import numpy as np
    from fastapi.testclient import TestClient

    from voice import backends
    from voice.backends.hertz import CHUNK, CpuStandIn, make_app, pack, unpack

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
    p.add_argument("--system", default="stock")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="local wav (uploaded to voice-voice:/prompts/) or a bare file name on voice-voice")
    p.add_argument("--cache-len", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--probe-context", type=int, nargs="+", metavar="CACHE_LEN",
                   help="instead of deploying, run the stock model past these cache lengths")
    p.add_argument("--dry-run", action="store_true", help="CPU stand-in model behind the real app; no Modal")
    args = p.parse_args(argv)
    if args.dry_run:
        dry_run(args.prompt, args.cache_len)
        return 0
    if args.probe_context:
        prompt = prompt_path(*upload_prompt(args.prompt))
        with modal.enable_output(), app.run():
            print(json.dumps(probe_context.remote(prompt, args.probe_context), indent=2))
        return 0
    deploy(args.system, args.prompt, args.cache_len)
    return 0


if __name__ == "__main__":
    sys.exit(main())
