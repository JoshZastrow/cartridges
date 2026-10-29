"""Modal app "voice-serve-hertz": stock hertz-dev behind upstream's WebSocket protocol.

    uv run python -m voice.deploy_hertz --system stock --prompt <wav> [--cache-len 2048]
    uv run python -m voice.deploy_hertz --probe-context 1024 2048     # what happens when the cache fills
    uv run python -m voice.deploy_hertz --dry-run [--prompt <local wav>]  # CPU, no Modal

`deploy` uploads the prompt wav (a local file) to the Volume `voice-voice`
under `prompts/`, writes the serving config to `voice-data:serve/hertz.json`,
deploys the app and prints the WebSocket URL (`wss://.../audio`). The
container loads the model, reads that config and runs `HertzBackend.serve`
on port 8000 (`modal.web_server`). One GPU container, one session at a time.

The image is separate from `voice.train.image`: hertz-dev needs torch 2.5.1,
which the workspace lock cannot hold (the root override pins torch>=2.14.1).
Upstream's code is cloned at `UPSTREAM_COMMIT` into /opt/hertz-dev; its four
checkpoints download from the HF repo si-pbc/hertz-dev into `voice-data:hf-cache`
on first start (about 17 GB) and load from there after.

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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1")
    .uv_pip_install(
        "torch==2.5.1",  # hertz-dev's pin; cu124 wheels
        "einops==0.8.0",
        "tqdm==4.66.6",
        "requests==2.32.3",
        "huggingface-hub[hf_transfer]==0.26.2",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "soundfile==0.12.1",
        "fastapi==0.115.4",
        "uvicorn==0.32.0",
        "aiohttp==3.11.9",  # voice.hertz_transport, imported by the backend
        "pydantic==2.10.3",
        "safetensors==0.4.5",
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
@modal.web_server(PORT, startup_timeout=1200)
def serve():
    cfg = json.loads((Path(DATA) / CONFIG).read_text())
    b = _loaded_backend()
    b.set_prompt(cfg["prompt"])
    print(f"hertz loaded in {b.load_s:.1f}s; prompt {cfg['prompt']} "
          f"({b.prompt.shape[1] / 16000:.1f}s); cache_len {cfg['cache_len']}", flush=True)
    b.serve(None, PORT, cache_len=cfg["cache_len"])


@app.function(gpu=GPU, volumes=volumes, secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=3600)
def probe_context(prompt: str, cache_lens: list[int]) -> list[dict]:
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
        rows.append({"cache_len": length, "prompt_frames": b.prompt.shape[1] // CHUNK,
                     "frames_at_failure": frames, "error": error, "ms_per_step": round(step_ms, 1),
                     "max_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        print(rows[-1], flush=True)
    return rows


def upload_prompt(prompt: str) -> str:
    """A local wav goes to voice-voice:/prompts/<name>; anything else is a container path."""
    if not Path(prompt).is_file():
        return prompt
    with voice_volume.batch_upload(force=True) as up:
        up.put_file(prompt, f"/prompts/{Path(prompt).name}")
    return f"{VOICE}/prompts/{Path(prompt).name}"


def deploy(system: str = "stock", prompt: str = f"{VOICE}/prompts/hertz-stock.wav",
           cache_len: int = CONTEXT_FRAMES) -> str:
    """Deploy the stock server and return its WebSocket URL."""
    if system != "stock":
        raise NotImplementedError("serving a cartridge on hertz is #12/#18")
    if not 0 < cache_len <= CONTEXT_FRAMES:
        raise ValueError(f"cache_len must be in 1..{CONTEXT_FRAMES} (the LM's RoPE table ends there)")
    cfg = {"system": system, "prompt": upload_prompt(prompt), "cache_len": cache_len}
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
    p.add_argument("--prompt", default=f"{VOICE}/prompts/hertz-stock.wav",
                   help="local wav (uploaded to voice-voice) or a path in the container")
    p.add_argument("--cache-len", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--probe-context", type=int, nargs="+", metavar="CACHE_LEN",
                   help="instead of deploying, run the stock model past these cache lengths")
    p.add_argument("--dry-run", action="store_true", help="CPU stand-in model behind the real app; no Modal")
    args = p.parse_args(argv)
    if args.dry_run:
        dry_run(args.prompt, args.cache_len)
        return 0
    if args.probe_context:
        prompt = upload_prompt(args.prompt)
        with modal.enable_output(), app.run():
            print(json.dumps(probe_context.remote(prompt, args.probe_context), indent=2))
        return 0
    deploy(args.system, args.prompt, args.cache_len)
    return 0


if __name__ == "__main__":
    sys.exit(main())
