"""backends/hertz.py inference half on CPU, with a stub in place of upstream's model.

`CpuStandIn` mirrors upstream's shapes and its fixed-length caches: a forward
that would write past `length` frames raises, as upstream's cache assignment does.
"""

import numpy as np
import pytest
import soundfile as sf
import torch
from voice.backends.base import Dialogue, Frames
from voice.backends.hertz import (
    CHUNK,
    ContextFull,
    CpuStandIn,
    HertzBackend,
    agent_prompt,
    make_app,
    read_wav,
)
from voice.backends.registry import HertzConfig
from voice.hertz_transport import HertzTransport, pack, unpack

from voice import backends


@pytest.fixture
def backend():
    b = backends.get("hertz", {"checkpoint": "si-pbc/hertz-dev", "device": "cpu"})
    b.model, b.device = CpuStandIn(), "cpu"
    b.set_prompt(np.random.default_rng(0).uniform(-0.5, 0.5, (1, 10 * CHUNK)).astype(np.float32))
    return b


def test_registered_with_hertz_spec():
    b = backends.get("hertz", HertzConfig(checkpoint="si-pbc/hertz-dev"))
    assert isinstance(b, HertzBackend)
    s = b.spec
    assert (s.sample_rate, s.frame_rate_hz, s.context_frames, s.has_text) == (16000, 8.0, 2048, False)
    assert s.cache_groups == ["lm", "resyn"]
    kinds = {(st.name, st.kind): (st.vocab, st.dim) for st in s.streams}
    assert kinds == {("user", "audio_token"): (32768, None), ("agent", "audio_token"): (32768, None),
                     ("user", "latent"): (None, 32), ("agent", "latent"): (None, 32)}


def test_load_requires_cuda():
    b = backends.get("hertz", {"checkpoint": "si-pbc/hertz-dev", "device": "cpu"})
    with pytest.raises(RuntimeError, match="CUDA only"):
        b.load("cpu")


@pytest.mark.parametrize("method,args", [
    ("build_prefix", (None, "", "")), ("cartridge_from_prefill", (None,)), ("forward", (None, None)),
    ("sft_loss", (None, None, None)), ("distill_loss", (None, None)),
])
def test_cartridge_half_is_18s(method, args):
    b = backends.get("hertz", {"checkpoint": "si-pbc/hertz-dev"})
    with pytest.raises(NotImplementedError, match="#18"):
        getattr(b, method)(*args)


def test_pack_unpack_roundtrip_is_upstream_format():
    x = np.linspace(-1, 1, CHUNK, dtype=np.float32)
    frame = pack(x)
    assert frame.startswith("data:audio/raw;base64,")
    y = unpack(frame)
    assert y.shape == (CHUNK,) and np.abs(y - x).max() < 1e-4


def test_read_wav_resamples_and_trims(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(24000 * 2 + 100, dtype=np.float32), 24000)
    x = read_wav(tmp_path / "a.wav")
    assert x.shape == (1, 32000)  # 2 s at 16 kHz, whole frames
    assert read_wav(tmp_path / "a.wav", channels=2).shape == (2, 32000)
    sf.write(tmp_path / "s.wav", np.zeros((16000, 2), dtype=np.float32), 16000)
    assert read_wav(tmp_path / "s.wav", channels=None).shape == (2, 16000)
    p = agent_prompt(x)
    assert p.shape == (2, 32000) and not p[0].any()  # user channel silent


def test_set_prompt_puts_mono_on_agent_channel(backend):
    assert backend.prompt.shape == (2, 10 * CHUNK) and not backend.prompt[0].any() and backend.prompt[1].any()


def test_encode_dialogue_shapes(backend, tmp_path):
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "u.wav", rng.uniform(-1, 1, 5 * CHUNK).astype(np.float32), 16000)
    sf.write(tmp_path / "a.wav", rng.uniform(-1, 1, 7 * CHUNK + 50).astype(np.float32), 16000)
    d = Dialogue(id="d1", split="train", user_wav=str(tmp_path / "u.wav"), agent_wav=str(tmp_path / "a.wav"),
                 words=[{"w": "hi", "start_s": 0.3, "end_s": 0.5}])
    f = backend.encode_dialogue(d)
    assert f.tokens["user"].shape == f.tokens["agent"].shape == (7,)
    assert f.tokens["agent"].dtype == torch.long
    assert f.latents["user"].shape == f.latents["agent"].shape == (7, 32)
    assert f.meta["words"][0]["frame"] == 2 and f.id == "d1"
    f2 = Frames.load(f.save(tmp_path / "f.safetensors"))
    assert torch.equal(f2.tokens["agent"], f.tokens["agent"])
    assert backend.decode_agent(f).shape == (7 * CHUNK,)


@pytest.mark.parametrize("user_frames,max_s,cache_len,expected", [
    (20, 60.0, 2048, 20),  # ends with the user wav
    (20, 1.0, 2048, 8),  # max_s at 8 Hz
    (20, 60.0, 16, 6),  # cache: 16 - 10 prompt frames
])
def test_rollout_stops_at_the_first_limit(backend, user_frames, max_s, cache_len, expected):
    user = np.random.default_rng(2).uniform(-1, 1, user_frames * CHUNK).astype(np.float32)
    d = backend.rollout(None, user, max_s=max_s, cache_len=cache_len)
    agent, sr = sf.read(d.agent_wav)
    assert sr == 16000 and agent.shape == (expected * CHUNK,)
    assert sf.read(d.user_wav)[0].shape == (expected * CHUNK,)
    assert backend.model.length is None  # caches released


def test_rollout_needs_prompt_and_no_cartridge(backend):
    with pytest.raises(NotImplementedError, match="#18"):
        backend.rollout(object(), np.zeros(CHUNK, np.float32), 1.0)
    backend.prompt = None
    with pytest.raises(RuntimeError, match="set_prompt"):
        backend.rollout(None, np.zeros(CHUNK, np.float32), 1.0)


def test_session_steps_until_context_full(backend):
    s = backend.session(cache_len=13)  # 10 prompt frames + 3 live
    out = [s.step(np.full(CHUNK, 0.1, np.float32)) for _ in range(3)]
    assert all(o.shape == (CHUNK,) for o in out) and s.remaining == 0
    with pytest.raises(ContextFull, match="13 frames"):
        s.step(np.zeros(CHUNK, np.float32))
    assert backend.model.used == 13  # the guard fires before the stand-in's own overflow
    s.close()


def test_session_rejects_prompt_longer_than_cache(backend):
    with pytest.raises(ContextFull):
        backend.session(cache_len=10)


def test_websocket_protocol_and_context_full_close(backend):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(make_app(backend, cache_len=12))
    assert client.get("/health").json()["cache_len"] == 12
    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect("/audio") as ws:
        for _ in range(2):
            ws.send_text(pack(np.full(CHUNK, 0.2, np.float32)))
            assert unpack(ws.receive_text()).shape == (CHUNK,)
        ws.send_text(pack(np.zeros(CHUNK, np.float32)))
        ws.receive_text()
    assert e.value.code == 1011 and "context full" in e.value.reason
    assert backend.model.length is None  # session closed, caches released
    with client.websocket_connect("/audio") as ws:  # the next session starts fresh
        ws.send_text(pack(np.zeros(CHUNK, np.float32)))
        assert unpack(ws.receive_text()).shape == (CHUNK,)


def test_deploy_dry_run_drives_the_real_app_on_cpu():
    from voice.deploy_hertz import dry_run

    out = dry_run(steps=4, cache_len=64)
    assert out["steps"] == 4 and sf.read(out["rollout"])[0].shape == (4 * CHUNK,)


def test_transport_against_served_app_until_context_full(backend):
    """`serve` on a real port (CPU stand-in), driven by `HertzTransport` over aiohttp."""
    import asyncio
    import socket

    import aiohttp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = backend.serve(None, port, cache_len=13)  # 10 prompt frames + 3 live

    async def talk():
        t = HertzTransport()
        async with aiohttp.ClientSession() as session:
            await t.connect(session, server.url)
            got = []
            stream = t.received()
            for _ in range(4):  # the fourth exceeds the cache
                sent = await t.send(np.full(CHUNK, 0.1, np.float32))
                async for kind, item in stream:
                    got.append((kind, item if kind == "wire" else item.shape))
                    if kind == "audio":
                        break
            async for _ in stream:  # drains to the close
                pass
            await t.close()
        return got, t.closed_with, sent

    try:
        got, closed, sent = asyncio.run(talk())
    finally:
        server.server.should_exit = True
        server.thread.join(5)
    assert got == [("wire", sent), ("audio", (CHUNK,))] * 3  # same-size frames both ways
    assert closed[0] == 1011 and closed[1].startswith("context full")


@pytest.mark.parametrize("bad", ["", "../x.wav", "a/b.wav", "/voice/x.wav", "..", "a\\b.wav", "x..wav"])
def test_prompt_names_must_be_bare(bad):
    from voice.deploy_hertz import safe_name

    with pytest.raises(ValueError, match="bare file name"):
        safe_name(bad)


def test_prompt_path():
    from voice.deploy_hertz import prompt_path

    assert prompt_path("prompt-30s.wav", False) == "/voice/prompt-30s.wav"
    assert prompt_path("mine.wav", True) == "/voice/prompts/mine.wav"


def test_server_takes_no_paths_from_clients(backend):
    """The prompt is fixed at start-up from the operator's config (validated by
    `prompt_path` in the serving process); the app's only routes take audio or nothing."""
    routes = {r.path for r in make_app(backend).routes if not r.path.startswith(("/docs", "/openapi", "/redoc"))}
    assert routes == {"/health", "/audio"}


def test_transport_has_no_codec_hold():
    """Raw PCM both ways: nothing is held before decode, so a reply's "wire"
    arrival is its "audio" arrival (the end-to-end test checks the pairing)."""
    assert HertzTransport.codec_hold_s == 0.0
    assert (HertzTransport.sample_rate, HertzTransport.frame_size) == (16000, CHUNK)


def test_transport_sends_no_auth_headers_of_its_own(monkeypatch):
    """Proxy auth rides only on the caller's session (voice.client scopes it to our
    workspace); with the token in the env, a bare session still sends no Modal-* header."""
    import asyncio

    import aiohttp
    from aiohttp import web

    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-secret")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-secret")
    seen = {}

    async def handler(request):
        seen.update(request.headers)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close()
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/audio", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                t = HertzTransport()
                await t.connect(session, f"ws://127.0.0.1:{port}/audio")
                async for _ in t.received():
                    pass
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen and not any(k.lower().startswith("modal-") for k in seen)
