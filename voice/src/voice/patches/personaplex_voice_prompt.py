"""Reject unsafe `voice_prompt` query values before the fork's `/api/chat` handler sees them.

The fork joins `voice_prompt` onto its voices directory and loads the file
(`torch.load` for .pt), so a value like "../x" or an absolute path would load
an arbitrary file. Only a bare file name ending in .pt or .wav is accepted;
anything else gets HTTP 400 before the WebSocket opens.

`moshi.server` runs its server at import, so the patch cannot touch the fork's
classes first. It wraps `aiohttp.web.run_app` (which the fork looks up at call
time) to add a middleware to the fork's app. It must run inside the server
process, so `serve` launches this module, which applies the patch and then
runs `moshi.server`:

    python -m voice.patches.personaplex_voice_prompt --port 8998 ...   # same args as moshi.server
"""

from __future__ import annotations

import runpy
import sys
from pathlib import PurePosixPath, PureWindowsPath

SERVER_MODULE = "moshi.server"
SUFFIXES = (".pt", ".wav")


def valid_voice_prompt(name: str | None) -> bool:
    """A bare file name with a .pt or .wav suffix: no directory part, not absolute, not hidden."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return False
    if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
        return False
    return name.endswith(SUFFIXES)


def middleware():
    from aiohttp import web

    @web.middleware
    async def check_voice_prompt(request, handler):
        if request.path == "/api/chat" and not valid_voice_prompt(request.query.get("voice_prompt")):
            raise web.HTTPBadRequest(text="voice_prompt must be a bare .pt or .wav file name")
        return await handler(request)

    return check_voice_prompt


def apply() -> None:
    """Make every `aiohttp.web.run_app` call serve its app behind the check. Idempotent."""
    from aiohttp import web

    if getattr(web.run_app, "voice_prompt_patched", False):
        return
    original = web.run_app

    def run_app(app, *args, **kwargs):
        app.middlewares.append(middleware())
        return original(app, *args, **kwargs)

    run_app.voice_prompt_patched = True
    web.run_app = run_app


def main(argv: list[str] | None = None, server_module: str = SERVER_MODULE) -> None:
    """Apply the patch, then run the fork's server with `argv` as its arguments."""
    apply()
    sys.argv = [server_module, *(sys.argv[1:] if argv is None else argv)]
    runpy.run_module(server_module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
