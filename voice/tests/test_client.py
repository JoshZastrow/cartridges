"""client.py against a fake server speaking the PersonaPlex fork's protocol; deploy.ws_url."""

import asyncio
import time

import numpy as np
import pytest
import soundfile
from voice.client import Client, Conversation, MoshiTransport, place
from voice.deploy import ws_url

SAMPLE_RATE, FRAME_SIZE = MoshiTransport.sample_rate, MoshiTransport.frame_size
aiohttp = pytest.importorskip("aiohttp")
sphn = pytest.importorskip("sphn")
from aiohttp import web  # noqa: E402


async def echo_handler(request):
    """The fork's protocol: handshake, then echo each user frame back as agent
    audio (Opus both ways) with one text piece per frame."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    assert request.query["voice_prompt"] == "NATF2.pt"
    await ws.send_bytes(b"\x00")
    reader, writer = sphn.OpusStreamReader(SAMPLE_RATE), sphn.OpusStreamWriter(SAMPLE_RATE)
    buf = np.zeros(0, dtype=np.float32)
    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.BINARY or msg.data[:1] != b"\x01":
            continue
        reader.append_bytes(msg.data[1:])
        buf = np.concatenate([buf, reader.read_pcm().reshape(-1)])
        while len(buf) >= FRAME_SIZE:
            writer.append_pcm(buf[:FRAME_SIZE])
            buf = buf[FRAME_SIZE:]
            if out := writer.read_bytes():
                await ws.send_bytes(b"\x01" + out)
            await ws.send_bytes(b"\x02" + " hi".encode())
    return ws


def run_with_server(coro_fn):
    async def main():
        app = web.Application()
        app.router.add_get("/api/chat", echo_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await coro_fn(f"ws://127.0.0.1:{port}/api/chat?voice_prompt=NATF2.pt&text_prompt=hello")
        finally:
            await runner.cleanup()

    return asyncio.run(main())


def tone(seconds: float, start_s: float, length_s: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    on = (t >= start_s) & (t < start_s + length_s)
    return (0.5 * np.sin(2 * np.pi * 440 * t) * on).astype(np.float32)


def onset_s(pcm: np.ndarray, thresh: float = 0.05) -> float:
    return int(np.argmax(np.abs(pcm) > thresh)) / SAMPLE_RATE


def test_converse_round_trips_in_real_time_on_one_timeline(tmp_path):
    wav = tmp_path / "q1.wav"
    soundfile.write(str(wav), tone(0.8, 0.3, 0.3), SAMPLE_RATE)
    max_s = 1.2

    async def go(url):
        t0 = time.monotonic()
        c = await Client(url, out_dir=tmp_path / "out", drain_s=0.3).aconverse(str(wav), max_s=max_s)
        return c, time.monotonic() - t0

    c, elapsed = run_with_server(go)
    assert isinstance(c, Conversation) and c.id == "q1"
    assert elapsed >= max_s  # paced in real time, not as fast as possible
    assert c.user_s == pytest.approx(0.8, abs=1e-3) and c.handshake_s >= 0
    user, sr = soundfile.read(c.user_wav)
    agent, _ = soundfile.read(c.agent_wav)
    assert sr == SAMPLE_RATE and len(user) == len(agent) == round(max_s / 0.08) * FRAME_SIZE
    # the echo server adds no model time, so the measured transport split explains the whole delay
    delay = onset_s(agent) - onset_s(user)
    assert c.transport["server_decode_hold_s"] == MoshiTransport.codec_hold_s
    assert 0 <= c.transport["send_hold_s"] < 0.05 and 0.15 < c.transport["client_decode_hold_s"] < 0.4
    assert c.transport_s == pytest.approx(delay, abs=0.12)
    arrivals = [t for t, _ in c.chunks]
    assert arrivals == sorted(arrivals) and c.first_audio_s == arrivals[0]
    assert c.wire and c.wire[0][0] <= c.first_audio_s and all(n > 0 for _, n in c.wire)
    assert len(c.sent) == round(max_s / 0.08) and all(left >= due for due, left, _ in c.sent)
    assert c.response_text.startswith("hi hi") and len(c.text) >= 10


def test_place_queues_chunks_that_arrive_in_a_burst():
    chunks = [(0.0, np.ones(10, np.float32)), (0.0, 2 * np.ones(10, np.float32)), (1.0, 3 * np.ones(10, np.float32))]
    out = place(chunks, 40, sample_rate=25)  # 1.0 s -> sample 25
    assert out[:10].tolist() == [1] * 10 and out[10:20].tolist() == [2] * 10
    assert out[20:25].tolist() == [0] * 5 and out[25:35].tolist() == [3] * 10 and out[35:].tolist() == [0] * 5


def test_ws_url_rewrites_scheme_and_keeps_the_system_query():
    url = ws_url("https://me--voice-serve-personaplex.modal.run", "/api/chat?voice_prompt=NATF2.pt&text_prompt=Hi+there")
    assert url == "wss://me--voice-serve-personaplex.modal.run/api/chat?voice_prompt=NATF2.pt&text_prompt=Hi+there"


def test_cli_deploy_dry_run_prints_the_stock_url_without_modal(capsys):
    from voice import cli

    assert cli.main(["deploy", "--backend", "personaplex", "--system", "stock", "--dry-run"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("wss://") and out.endswith("/api/chat?voice_prompt=NATF2.pt&text_prompt=You+enjoy+having+a+good+conversation.")
    assert cli.main(["deploy", "--backend", "personaplex", "--system", "voice-cartridge:v0", "--dry-run"]) == 2


def test_sphn_opus_holds_exactly_the_pinned_frames():
    """The server-side share of transport_s is a constant of the pinned sphn: its
    OpusStreamWriter emits a page on every 80 ms chunk (no encode hold) and its
    OpusStreamReader returns no PCM until OPUS_HOLD_FRAMES pages later."""
    writer, reader = sphn.OpusStreamWriter(SAMPLE_RATE), sphn.OpusStreamReader(SAMPLE_RATE)
    pcm = tone(2.0, 0.0, 2.0)
    out = 0
    for k in range(1, len(pcm) // FRAME_SIZE + 1):
        writer.append_pcm(pcm[(k - 1) * FRAME_SIZE:k * FRAME_SIZE])
        page = writer.read_bytes()
        assert page, f"writer held chunk {k}"
        reader.append_bytes(page)
        out += reader.read_pcm().shape[-1]
        assert out == max(0, k - MoshiTransport.OPUS_HOLD_FRAMES) * FRAME_SIZE, k
