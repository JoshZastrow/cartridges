"""patches/personaplex_voice_prompt.py: the check, the middleware, and that the
server entry point applies it before the fork's server sets up its app."""

import asyncio
import sys

import pytest
from voice.backends.personaplex import PersonaPlexBackend
from voice.backends.registry import PersonaPlexConfig
from voice.patches import personaplex_voice_prompt as patch

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402


@pytest.mark.parametrize("name", ["NATF2.pt", "author-30s.wav"])
def test_bare_pt_or_wav_names_pass(name):
    assert patch.valid_voice_prompt(name)


@pytest.mark.parametrize("name", ["../x.wav", "/etc/x.pt", "sub/NATF2.pt", "sub\\x.pt", "C:\\x.pt", "NATF2.mp3",
                                  "NATF2", ".hidden.pt", "..", "", None])
def test_paths_and_other_suffixes_are_rejected(name):
    assert not patch.valid_voice_prompt(name)


def test_middleware_rejects_before_the_handler_runs():
    calls = []

    async def handler(request):
        calls.append(request.query["voice_prompt"])
        return web.Response(text="ok")

    async def go():
        app = web.Application(middlewares=[patch.middleware()])
        app.router.add_get("/api/chat", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        statuses = {}
        try:
            async with aiohttp.ClientSession() as s:
                for vp in ("../x.wav", "/etc/passwd.pt", "sub/x.pt", "x.mp3", "NATF2.pt"):
                    async with s.get(f"http://127.0.0.1:{port}/api/chat", params={"voice_prompt": vp}) as r:
                        statuses[vp] = r.status
                async with s.get(f"http://127.0.0.1:{port}/api/chat") as r:
                    statuses[None] = r.status
        finally:
            await runner.cleanup()
        return statuses

    statuses = asyncio.run(go())
    assert statuses == {"../x.wav": 400, "/etc/passwd.pt": 400, "sub/x.pt": 400, "x.mp3": 400, "NATF2.pt": 200,
                        None: 400}
    assert calls == ["NATF2.pt"]


FAKE_SERVER = """
import sys
from aiohttp import web
SEEN_ARGV = list(sys.argv)
async def chat(request):
    return web.Response(text="ok")
app = web.Application()
app.router.add_get("/api/chat", chat)
web.run_app(app, port=0)   # what moshi.server does at the end of its import-time main()
"""


def test_entry_point_patches_before_the_server_module_runs(tmp_path, monkeypatch):
    (tmp_path / "fake_moshi_server.py").write_text(FAKE_SERVER)
    monkeypatch.syspath_prepend(str(tmp_path))
    started = []
    monkeypatch.setattr(web, "run_app", lambda app, **kw: started.append((app, kw)))  # the "original" run_app
    monkeypatch.setattr(sys, "argv", list(sys.argv))

    patch.main(["--port", "8998"], server_module="fake_moshi_server")

    (app, kw), = started
    assert getattr(web.run_app, "voice_prompt_patched", False)
    assert [m.__name__ for m in app.middlewares] == ["check_voice_prompt"]
    assert sys.argv == ["fake_moshi_server", "--port", "8998"]


def test_serve_launches_the_server_through_the_patch():
    b = PersonaPlexBackend(PersonaPlexConfig(checkpoint="nvidia/personaplex-7b-v1"))
    files = {"tokenizer": "t", "moshi": "m", "mimi": "mi", "voices": "v"}
    cmd = b.server_command(files, 8998)
    assert cmd[1:3] == ["-m", "voice.patches.personaplex_voice_prompt"]
    assert patch.SERVER_MODULE == "moshi.server" and "--voice-prompt-dir" in cmd
