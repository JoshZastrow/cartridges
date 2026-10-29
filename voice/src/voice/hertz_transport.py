"""hertz-dev wire protocol (upstream inference_server.py) as a `voice.client` Transport.

WebSocket at `<url>/audio` (the URL `voice.deploy_hertz` prints already ends
in `/audio`). Every message both ways is a text frame

    data:audio/raw;base64,<int16 little-endian mono PCM at 16 kHz>

and the server answers each 2000-sample (125 ms) user chunk with exactly one
2000-sample agent chunk: strict request/response, no handshake. The server
closes with 1011 "context full ..." when its caches are full (see
`voice.backends.hertz.ContextFull`) and 1013 when another session is live.
The deployed endpoint requires Modal proxy auth: `connect` sends Modal-Key /
Modal-Secret from MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET when set.
"""

import base64
import os
from collections.abc import AsyncIterator

import aiohttp
import numpy as np

SAMPLE_RATE = 16_000
FRAME_SIZE = 2000  # 125 ms
PREFIX = "data:audio/raw;base64,"


def pack(pcm: np.ndarray) -> str:
    """float32 [-1, 1] or int16 mono -> one text frame."""
    if pcm.dtype != np.int16:
        pcm = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    return PREFIX + base64.b64encode(pcm.astype("<i2").tobytes()).decode()


def unpack(frame: str) -> np.ndarray:
    """One text frame -> float32 mono in [-1, 1]."""
    return np.frombuffer(base64.b64decode(frame.split(",", 1)[1]), dtype="<i2").astype(np.float32) / 32767.0


def proxy_auth_headers() -> dict[str, str]:
    key, secret = os.environ.get("MODAL_PROXY_TOKEN_ID"), os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    return {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}


class HertzTransport:
    sample_rate = SAMPLE_RATE
    frame_size = FRAME_SIZE
    codec_hold_s = 0.0  # raw PCM both ways: no decoder holds audio

    def __init__(self):
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.closed_with: tuple[int | None, str] | None = None  # e.g. (1011, "context full: ...")

    async def connect(self, session: aiohttp.ClientSession, url: str) -> None:
        self.ws = await session.ws_connect(url, max_msg_size=0, headers=proxy_auth_headers())

    async def send(self, pcm: np.ndarray) -> int:
        """Send one chunk; returns the bytes put on the wire (always sent, never held)."""
        assert pcm.shape == (FRAME_SIZE,), f"hertz takes {FRAME_SIZE}-sample chunks, got {pcm.shape}"
        frame = pack(pcm)
        await self.ws.send_str(frame)
        return len(frame)

    async def received(self) -> AsyncIterator[tuple[str, np.ndarray | str | int]]:
        """Per reply ("wire", n_bytes) on arrival, then ("audio", float32 [2000]),
        until the socket closes; the server's close code and reason are then in
        `closed_with`."""
        while True:
            msg = await self.ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                yield "wire", len(msg.data)
                yield "audio", unpack(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise msg.data
            else:  # CLOSE, CLOSING, CLOSED
                self.closed_with = (self.ws.close_code, msg.extra if isinstance(msg.extra, str) else "")
                return

    async def close(self) -> None:
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
