"""Modal app "voice-serve": a backend's live server behind a WebSocket URL.

    uv run voice deploy --backend personaplex --system stock     # deploys, prints wss://.../api/chat?...
    uv run voice serve-local --backend personaplex [--port 8998] # the same server on this machine's GPU

A backend plugs in through `serve(cartridge, port)` (brief 4b) and
`client_path` (the path and query a client connects to). Each backend has one
web function here, because each needs its own image; `FUNCTIONS` maps backend
names to them. #20 ships personaplex with the stock system; #21 adds hertz
(image, start-up and prompt in `voice.deploy_hertz`);
#12 adds `--system <cartridge artifact>`.

Cold start is the cost driver (a ~16 GB checkpoint onto the GPU), so weights
live in the HF cache on the `voice-data` Volume (`HF_HOME=/data/hf`), the app
scales to zero after `SCALEDOWN_S` idle, and at most one container runs. The
container logs its startup split (`resolve_s`: weights on disk, `ready_s`:
server listening); the client measures the whole cold start as the handshake
time of the first connection.

The endpoint requires Modal proxy auth: a connection without a workspace proxy
token is refused (401) before it reaches a container. `voice.client` sends
the token from MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET (voice/README.md).
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urlsplit, urlunsplit

import modal

from voice import backends, deploy_hertz
from voice.train import DATA_MOUNT, image

logger = logging.getLogger(__name__)

APP_NAME = "voice-serve"
PORT = 8998
HF_HOME = f"{DATA_MOUNT}/hf"
SCALEDOWN_S = 300
CHECKPOINTS = {"personaplex": "nvidia/personaplex-7b-v1", "hertz": "si-pbc/hertz-dev"}
FUNCTIONS = {"personaplex": "personaplex", "hertz": "hertz"}  # backend -> web function in this app

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("voice-data", create_if_missing=True)


def backend_for(name: str):
    if name not in FUNCTIONS:
        raise ValueError(f"no server for backend {name!r}; served: {sorted(FUNCTIONS)}")
    return backends.get(name, {"checkpoint": CHECKPOINTS[name]})


def start(name: str, port: int = PORT):
    """Start backend `name`'s stock server on `port` and return it (in the container)."""
    t0 = time.monotonic()
    server = backend_for(name).serve(None, port)
    data_volume.commit()  # keep weights fetched on a first start
    logger.info("voice-serve %s ready after %.1f s: %s", name, time.monotonic() - t0, server.timings)
    return server


@app.function(image=image, gpu="H200", volumes={DATA_MOUNT: data_volume}, secrets=[modal.Secret.from_name("HF_TOKEN")],
              env={"HF_HOME": HF_HOME}, timeout=3600, scaledown_window=SCALEDOWN_S, max_containers=1)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def personaplex():
    logging.basicConfig(level=logging.INFO)
    start("personaplex")


@app.function(image=deploy_hertz.image, gpu="H200", volumes=deploy_hertz.volumes,
              secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=3600, scaledown_window=SCALEDOWN_S,
              max_containers=1)
@modal.web_server(PORT, startup_timeout=1200, requires_proxy_auth=True)
def hertz():
    logging.basicConfig(level=logging.INFO)
    deploy_hertz.start(PORT)  # its own image (no moshi fork) and prompt; see voice.deploy_hertz


def ws_url(web_url: str, client_path: str) -> str:
    """`https://host` + `/api/chat?...` -> `wss://host/api/chat?...`."""
    web, path = urlsplit(web_url), urlsplit(client_path)
    scheme = {"https": "wss", "http": "ws"}[web.scheme]
    return urlunsplit((scheme, web.netloc, path.path, path.query, ""))


def deploy(name: str, system: str = "stock", dry_run: bool = False) -> str:
    """Deploy `voice-serve` and return the WebSocket URL of backend `name`
    serving `system`. `dry_run`: no Modal calls; the URL with a placeholder host."""
    if system != "stock":
        raise NotImplementedError("serving a cartridge ships with #12; only --system stock for now")
    backend = backend_for(name)
    if dry_run:
        return ws_url(f"https://<{APP_NAME}-{FUNCTIONS[name]}>.modal.run", backend.client_path)
    with modal.enable_output():
        app.deploy()
    web = modal.Function.from_name(APP_NAME, FUNCTIONS[name]).get_web_url()
    return ws_url(web, backend.client_path)


def serve_local(name: str, port: int = PORT) -> int:
    """Run the stock server on this machine until it exits."""
    server = backend_for(name).serve(None, port)
    print(server.url, flush=True)
    return server.wait()
