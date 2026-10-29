"""Offline client for a served duplex endpoint (#20): a user WAV in, in
real-time-sized chunks, the agent channel out with wall-clock timestamps.

    d = Client("wss://<app>.modal.run/api/chat?voice_prompt=NATF2.pt&text_prompt=...").converse("q.wav", max_s=30)
    d.agent_wav, d.response_text, d.handshake_s, d.chunks, d.text

The wire protocol is a `Transport`, picked by backend name (`TRANSPORTS`);
everything else is the same for every backend. `MoshiTransport` speaks the
PersonaPlex fork's `moshi.server`, unchanged: after the system prompt the
server sends b"\\x00"; both sides then send b"\\x01" + Ogg/Opus bytes of
24 kHz mono audio, and the server also sends b"\\x02" + UTF-8 text pieces of
the agent's text stream. The system (voice, persona) is in the URL query, so
the URL names what is being scored.

Time zero is when the first user frame is due, right after `connect`. User
frame n (`frame_size` samples) is sent at n * frame_size / sample_rate s;
after the WAV the client keeps sending silence until `max_s`, so the agent can
finish. The agent wav is laid on the same timeline as the user wav: each
received chunk starts at the sample of its arrival time, or right after the
previous chunk if that is later (what a playback buffer does). So agent onset
minus user offset in the two wavs is the latency a listener hears, transport
included. `Conversation.transport_s` is the transport's share, measured per
conversation as the sum of
    send_hold_s           median(page left - PCM due) of the user's Opus pages
    server_decode_hold_s  the transport's `codec_hold_s`: the server's Opus reader
                          holds that long before its first PCM (a constant of the
                          pinned sphn, asserted in tests/test_client.py)
    client_decode_hold_s  first decoded agent PCM - first raw agent audio message

    python -m voice.client smoke --url <ws-url> --prompts <dir of wavs> --out outputs/smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import numpy as np

from voice.backends.base import Dialogue

logger = logging.getLogger(__name__)


class Conversation(Dialogue):
    """A `Dialogue` plus what the client measured (seconds on the client clock)."""

    url: str = ""
    sample_rate: int = 0
    max_s: float = 0.0
    user_s: float = 0.0  # length of the user wav before the silence tail
    handshake_s: float = 0.0  # connect -> server ready (system prompt; cold start on a cold app)
    chunks: list[tuple[float, int]] = []  # (arrival_s, samples) per decoded agent chunk
    wire: list[tuple[float, int]] = []  # (arrival_s, bytes) per raw agent audio message, before decoding
    sent: list[tuple[float, float, int]] = []  # (due_s, left_s, bytes) per user chunk; 0 bytes if the encoder held it
    text: list[tuple[float, str]] = []  # (arrival_s, piece) of the agent text stream, if the backend has one
    transport: dict[str, float | None] = {}  # send_hold_s, server_decode_hold_s, client_decode_hold_s
    transport_s: float | None = None  # their sum: the transport's share of any latency measured on the wavs

    @property
    def first_audio_s(self) -> float | None:
        return self.chunks[0][0] if self.chunks else None


class Transport(Protocol):
    """One backend's wire protocol over one WebSocket session."""

    sample_rate: int
    frame_size: int  # samples per user chunk, sent every frame_size / sample_rate s
    codec_hold_s: float  # how long the server's decoder holds user audio before its first PCM

    async def connect(self, session, url: str) -> None: ...  # returns once the server takes audio
    async def send(self, pcm: np.ndarray) -> int: ...  # one float32 chunk of frame_size samples; bytes put on the wire
    # ("wire", n_bytes) for each raw agent audio message as it arrives, then ("audio", pcm) once decoded; ("text", str)
    def received(self) -> AsyncIterator[tuple[str, np.ndarray | str | int]]: ...
    async def close(self) -> None: ...


class MoshiTransport:
    """The PersonaPlex fork's `moshi.server` protocol (24 kHz, 80 ms frames, Ogg/Opus)."""

    sample_rate = 24000
    frame_size = 1920
    OPUS_HOLD_FRAMES = 3  # sphn 0.1.12's OpusStreamReader: first PCM after the 4th page (tests/test_client.py)
    codec_hold_s = OPUS_HOLD_FRAMES * frame_size / sample_rate  # 0.24 s
    HANDSHAKE, AUDIO, TEXT = 0, 1, 2

    def __init__(self, connect_timeout_s: float = 1500.0):
        import sphn

        self.connect_timeout_s = connect_timeout_s  # covers a cold start (container + weights)
        self.writer, self.reader = sphn.OpusStreamWriter(self.sample_rate), sphn.OpusStreamReader(self.sample_rate)

    async def connect(self, session, url: str) -> None:
        self.ws = await session.ws_connect(url, max_msg_size=0)
        first = await self.ws.receive_bytes(timeout=self.connect_timeout_s)
        if first[:1] != bytes([self.HANDSHAKE]):
            raise RuntimeError(f"expected the handshake byte, got {first[:8]!r}")

    async def send(self, pcm: np.ndarray) -> int:
        self.writer.append_pcm(pcm)
        if msg := self.writer.read_bytes():
            await self.ws.send_bytes(bytes([self.AUDIO]) + msg)
        return len(msg)

    async def received(self) -> AsyncIterator[tuple[str, np.ndarray | str | int]]:
        import aiohttp

        async for message in self.ws:
            if message.type != aiohttp.WSMsgType.BINARY or not message.data:
                continue
            kind, payload = message.data[0], message.data[1:]
            if kind == self.AUDIO:
                yield "wire", len(payload)
                self.reader.append_bytes(payload)
                pcm = self.reader.read_pcm()
                if pcm.shape[-1]:
                    yield "audio", np.asarray(pcm, dtype=np.float32).reshape(-1)
            elif kind == self.TEXT:
                yield "text", payload.decode("utf-8", errors="replace")

    async def close(self) -> None:
        await self.ws.close()


TRANSPORTS = {"personaplex": MoshiTransport}


def place(chunks: list[tuple[float, np.ndarray]], n_samples: int, sample_rate: int) -> np.ndarray:
    """Lay received chunks on the user timeline: each starts at its arrival
    sample or at the end of the previous chunk, whichever is later; cut at
    `n_samples`."""
    out = np.zeros(n_samples, dtype=np.float32)
    cursor = 0
    for t, pcm in chunks:
        start = max(cursor, round(t * sample_rate))
        if start >= n_samples:
            break
        n = min(len(pcm), n_samples - start)
        out[start:start + n] = pcm[:n]
        cursor = start + len(pcm)
    return out


def load_user(wav: str | Path | np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 at `sample_rate`; an array is taken as already at that rate."""
    if isinstance(wav, np.ndarray):
        return wav.astype(np.float32)
    import sphn

    pcm, sr = sphn.read(str(wav))
    return sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=sample_rate)[0].astype(np.float32)


class Client:
    def __init__(self, url: str, backend: str = "personaplex", out_dir: str | Path = "outputs/client",
                 drain_s: float = 1.0):
        self.url = url
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.drain_s = drain_s  # keep receiving this long after the last user frame

    def converse(self, user_wav: str | Path | np.ndarray, max_s: float, id: str | None = None) -> Conversation:
        return asyncio.run(self.aconverse(user_wav, max_s, id))

    async def aconverse(self, user_wav: str | Path | np.ndarray, max_s: float, id: str | None = None) -> Conversation:
        import aiohttp

        tr = TRANSPORTS[self.backend]()
        sr, size = tr.sample_rate, tr.frame_size
        user = load_user(user_wav, sr)
        n_frames = max(1, round(max_s * sr / size))
        padded = np.zeros(n_frames * size, dtype=np.float32)
        padded[: min(len(user), len(padded))] = user[: len(padded)]
        id = id or (Path(user_wav).stem if isinstance(user_wav, (str, Path)) else f"conv-{int(time.time())}")
        received: list[tuple[float, np.ndarray]] = []
        wire: list[tuple[float, int]] = []
        sent: list[tuple[float, float, int]] = []
        text: list[tuple[float, str]] = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            c0 = time.monotonic()
            await tr.connect(session, self.url)
            t0 = time.monotonic()
            handshake_s = t0 - c0

            async def send():
                for n in range(n_frames):
                    due = n * size / sr
                    await asyncio.sleep(max(0.0, t0 + due - time.monotonic()))
                    n_bytes = await tr.send(padded[n * size:(n + 1) * size])
                    sent.append((due, time.monotonic() - t0, n_bytes))
                await asyncio.sleep(self.drain_s)
                await tr.close()

            async def recv():
                log = {"audio": received, "wire": wire, "text": text}
                async for kind, item in tr.received():
                    log[kind].append((time.monotonic() - t0, item))

            await asyncio.gather(send(), recv())

        user_path, agent_path = self.out_dir / f"{id}.user.wav", self.out_dir / f"{id}.agent.wav"
        write_wav(user_path, padded, sr)
        write_wav(agent_path, place(received, len(padded), sr), sr)
        transport = transport_split(sent, wire, received, tr.codec_hold_s)
        return Conversation(
            id=id, user_wav=str(user_path), agent_wav=str(agent_path), response_text="".join(p for _, p in text).strip(),
            url=self.url, sample_rate=sr, max_s=max_s, user_s=len(user) / sr, handshake_s=handshake_s,
            chunks=[(t, len(p)) for t, p in received], wire=wire, sent=sent, text=text, transport=transport,
            transport_s=None if None in transport.values() else sum(transport.values()),
        )


def transport_split(sent, wire, received, codec_hold_s: float) -> dict[str, float | None]:
    """The transport's share of latency, in seconds, from one conversation's logs (module docstring)."""
    holds = [left - due for due, left, n in sent if n > 0]
    return {
        "send_hold_s": float(np.median(holds)) if holds else None,
        "server_decode_hold_s": codec_hold_s,
        "client_decode_hold_s": received[0][0] - wire[0][0] if received and wire else None,
    }


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), pcm, sample_rate)


# ---------------------------------------------------------------- smoke check


def transcribe(wav: str) -> dict:
    """Phonon-2 transcript of `wav`: {text, words[{w, start_s, end_s}]}. A stopgap
    until #9's `evals.transcribe`; reads `phonon-2 transcribe --json`."""
    import subprocess

    r = subprocess.run(["phonon-2", "transcribe", "--json", wav], capture_output=True, text=True, check=True)
    raw = json.loads(r.stdout)
    words = [{"w": w["text"], "start_s": w["start"], "end_s": w["end"]} for w in raw["words"]]
    return {"text": raw["text"], "words": words}


def response_latency(c: Conversation, words: list[dict]) -> float | None:
    """First agent word start (Phonon-2, agent wav timeline) minus the end of the user wav."""
    after = [w["start_s"] for w in words if w["start_s"] >= c.user_s]
    return min(after) - c.user_s if after else None


def smoke(url: str, backend: str, prompts: list[str], out_dir: str, max_s: float, tail_s: float = 12.0) -> list[dict]:
    """Converse each prompt wav (for `max(max_s, prompt + tail_s)` seconds),
    transcribe the agent with Phonon-2, and return one row per prompt."""
    import soundfile

    client = Client(url, backend=backend, out_dir=out_dir)
    rows = []
    for i, wav in enumerate(prompts):
        user_s = soundfile.info(wav).duration
        c = client.converse(wav, max_s=max(max_s, user_s + tail_s))
        tr = transcribe(c.agent_wav)
        latency = response_latency(c, tr["words"])
        rows.append({"id": c.id, "user_s": c.user_s, "handshake_s": c.handshake_s,
                     "first_audio_s": c.first_audio_s, "latency_s": latency, "transport_s": c.transport_s,
                     **c.transport,
                     "latency_minus_transport_s": None if latency is None or c.transport_s is None
                     else latency - c.transport_s,
                     "text_stream": c.response_text, "transcript": tr["text"], "agent_wav": c.agent_wav})
        logger.info("%d/%d %s", i + 1, len(prompts), json.dumps(rows[-1]))
    Path(out_dir, "smoke.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m voice.client")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="round-trip prompt wavs through an endpoint and transcribe the agent")
    s.add_argument("--url", required=True)
    s.add_argument("--backend", default="personaplex", choices=sorted(TRANSPORTS))
    s.add_argument("--prompts", required=True, help="directory of user wavs (sorted) or one wav")
    s.add_argument("--out", default="outputs/smoke")
    s.add_argument("--max-s", type=float, default=20.0)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--group", default="", help="W&B group exp-YYYYMMDD-<slug>; empty: no W&B run")
    s.add_argument("--name", default="personaplex/stock/eval-smoke", help="W&B run name <backend>/<system>/<stage>")
    args = p.parse_args(argv)
    src = Path(args.prompts)
    wavs = sorted(str(x) for x in src.glob("*.wav")) if src.is_dir() else [str(src)]
    rows = smoke(args.url, args.backend, wavs[: args.limit], args.out, args.max_s)
    print(json.dumps(rows, indent=1))
    if args.group:
        log_smoke(rows, args)
    return 0


def log_smoke(rows: list[dict], args) -> None:
    """One W&B run in project `voice`: the per-prompt table and the latency summary.
    Tagged `serve-check`, not `baseline`, and summarised under `smoke/`, so the
    report never mixes it with #9's section 8 metrics."""
    import wandb

    from shared import tracking

    with tracking.start_run("voice", args.group, vars(args), name=args.name, job_type="eval", gpu="CPU",
                            tags=["serve-check", args.backend]) as run:
        cols = list(rows[0])
        run.log({"smoke/prompts": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
        def median(key):
            xs = [r[key] for r in rows if r[key] is not None]
            return float(np.median(xs)) if xs else None

        run.summary.update({f"smoke/{k}": median(k) for k in (
            "latency_s", "transport_s", "send_hold_s", "server_decode_hold_s", "client_decode_hold_s",
            "latency_minus_transport_s")})
        run.summary.update({"smoke/answered": sum(r["latency_s"] is not None for r in rows), "smoke/n": len(rows),
                            "smoke/handshake_s_first": rows[0]["handshake_s"]})


if __name__ == "__main__":
    raise SystemExit(main())
