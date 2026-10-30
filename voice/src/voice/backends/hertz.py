"""hertz-dev backend, inference half (#21): load, encode, decode, rollout, serve.

hertz-dev (Standard Intelligence, si-pbc/hertz-dev) is a duplex audio LM over
2 channels at 16 kHz: channel 0 the user, channel 1 the agent. The codec turns
each 2000-sample chunk (125 ms) of one channel into one 32-d latent, so frames
run at 8 Hz; the LM reads both channels' latents and predicts each channel's
next FSQ token (8^5 = 32,768 levels); the resynthesizer turns those tokens
into next latents (a Gaussian mixture). Context is 2048 frames (256 s).

Shapes (T = frames of one dialogue at 8 Hz):

    Frames.tokens["user" | "agent"]    [T] long      FSQ index of that channel's latent
    Frames.latents["user" | "agent"]   [T, 32] float codec posterior mean (`mu`), deterministic
    rollout / serve audio              float32 mono at 16 kHz, 2000 samples per frame

The model code is upstream's (github.com/Standard-Intelligence/hertz-dev at
`UPSTREAM_COMMIT`), imported from `HERTZ_DEV_DIR`, and runs only on CUDA: its
autocast contexts are hard-coded to "cuda". It runs unpatched on the workspace
torch (2.14), not only on upstream's pinned 2.5.1; the difference was measured
at the bf16 floor (#21). The Modal image is `voice.deploy_hertz.image`.

Serving speaks upstream `inference_server.py`'s WebSocket protocol unchanged
(`voice.hertz_transport`), and `HertzSession` runs its live path, minus two things: the
3 s replay of the prompt tail it sends before going live, so every reply is
model output, and the recording it writes to disk. Context overflow: the
caches are fixed tensors of `cache_len` frames; past them the stock model
raises (see `ContextFull`). The session refuses the step instead.

The cartridge methods (build_prefix, cartridge_from_prefill, forward,
sft_loss, distill_loss) are #18's.
"""

import asyncio
import contextlib
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from voice.backends.base import BackendSpec, Cartridge, Dialogue, Frames, StreamSpec
from voice.backends.registry import HertzConfig, register
from voice.hertz_transport import pack, unpack

logger = logging.getLogger(__name__)

UPSTREAM = "https://github.com/Standard-Intelligence/hertz-dev"
UPSTREAM_COMMIT = "21670d74be1d85ca29212c100be578797a54ea19"
CHECKPOINT = "si-pbc/hertz-dev"  # upstream load_ckpt hard-codes this repo
SAMPLE_RATE = 16_000
CHUNK = 2000  # samples per frame (125 ms)
LATENT = 32
VOCAB = 8**5
CONTEXT_FRAMES = 2048  # LM seq_len; the RoPE table ends here
TEMPS = (0.8, (0.4, 0.1))  # upstream inference_server: (token, (categorical, gaussian))


class ContextFull(RuntimeError):
    """The session's KV caches hold `cache_len` frames and the next step needs
    one more. Stock hertz-dev raises at that step too (a shape mismatch writing
    into the cache, or the RoPE assert at 2048); #12 adds the sliding window."""


def read_wav(path: str | Path, channels: int | None = 1) -> np.ndarray:
    """A wav at 16 kHz as float32 [channels, N], N a whole number of frames.
    channels=1 downmixes, 2 repeats mono (as upstream does with its prompt),
    None keeps the file's channels."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.T
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g, axis=-1).astype(np.float32)
    if x.shape[0] == 1 and channels == 2:
        x = np.repeat(x, 2, axis=0)
    if channels == 1:
        x = x.mean(0, keepdims=True)
    return x[:, : x.shape[1] // CHUNK * CHUNK]


def agent_prompt(voice: np.ndarray) -> np.ndarray:
    """The stock prompt layout: the voice clip on the agent channel, the user
    channel silent, as #18's `build_prefix` lays out its prefix. [2, N]."""
    voice = voice.reshape(-1)
    return np.stack([np.zeros_like(voice), voice])


@register("hertz")
class HertzBackend:
    def __init__(self, config: HertzConfig):
        self.config = config
        self.spec = BackendSpec(
            name="hertz", sample_rate=SAMPLE_RATE, frame_rate_hz=SAMPLE_RATE / CHUNK,
            context_frames=CONTEXT_FRAMES, has_text=False, cache_groups=["lm", "resyn"],
            streams=[StreamSpec(name=n, role=n, kind=k, **{"vocab": VOCAB} if k == "audio_token" else {"dim": LATENT})
                     for k in ("audio_token", "latent") for n in ("user", "agent")],
        )
        self.model = None  # upstream HertzDevModel, set by load()
        self.device = config.device
        self.prompt: np.ndarray | None = None  # [2, N] at 16 kHz; the stock system's prompt

    # ------------------------------------------------------------ loading

    def load(self, device: str | None = None) -> None:
        """Build upstream's split (duplex) model in bf16; downloads the four
        checkpoints into the HF cache on first use."""
        device = device or self.config.device
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            raise RuntimeError(f"hertz-dev runs on CUDA only (upstream autocast is hard-coded to cuda); got {device!r}")
        if self.config.checkpoint != CHECKPOINT:
            raise ValueError(f"checkpoint must be {CHECKPOINT!r} (upstream load_ckpt hard-codes it)")
        sys.path.insert(0, os.environ.get("HERTZ_DEV_DIR", "/opt/hertz-dev"))
        from model import get_hertz_dev_config

        from voice.patches import hertz_sdpa

        hertz_sdpa.apply()  # flash/efficient attention: torch 2.14's default backend is ~2.5x slower here
        self.model = get_hertz_dev_config(is_split=True)().eval().bfloat16().to(device)
        self.device = device

    def set_prompt(self, wav: str | Path | np.ndarray) -> None:
        """The stock system's prompt: a path (mono goes on the agent channel) or [2, N] audio."""
        x = wav if isinstance(wav, np.ndarray) else read_wav(wav, channels=None)
        self.prompt = x if x.shape[0] == 2 else agent_prompt(x)

    def _ctx(self):
        stack = contextlib.ExitStack()
        if str(self.device).startswith("cuda"):
            stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        stack.enter_context(torch.inference_mode())
        return stack

    def _tensor(self, x: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

    # ------------------------------------------------------------ encoding

    def latents(self, wav: np.ndarray) -> Tensor:
        """Mono 16 kHz [N] -> codec posterior mean [N // 2000, 32]. Whole wav at once:
        the codec's streaming path is broken upstream."""
        x = self._tensor(wav.reshape(-1)[: wav.size // CHUNK * CHUNK])[None, None]
        with self._ctx():
            return self.model.audio_tokenizer.latent_from_data(x, get_parameters=True)["mu"][0].float()

    def fsq_tokens(self, latents: Tensor) -> Tensor:
        """[T, 32] -> [T] FSQ index, the LM's output vocabulary."""
        with self._ctx():
            _, tokens = self.model.resynthesizer.quantizer(latents[None].to(self.device), return_latent=True)
        return tokens.reshape(-1).long()

    def encode_dialogue(self, d: Dialogue) -> Frames:
        user, agent = read_wav(d.user_wav)[0], read_wav(d.agent_wav)[0]
        n = max(user.size, agent.size)
        user, agent = np.pad(user, (0, n - user.size)), np.pad(agent, (0, n - agent.size))
        lat = {"user": self.latents(user).cpu(), "agent": self.latents(agent).cpu()}
        tok = {k: self.fsq_tokens(v).cpu() for k, v in lat.items()}
        rate = self.spec.frame_rate_hz
        words = [{**w, "frame": int(w["start_s"] * rate)} for w in d.words]
        return Frames(tokens=tok, latents=lat, meta={"id": d.id, "split": d.split, "words": words})

    def decode(self, latents: Tensor) -> np.ndarray:
        """[T, 32] -> mono float32 [T * 2000] at 16 kHz."""
        with self._ctx():
            x = self.model.audio_tokenizer.data_from_latent(latents[None].to(self.device, torch.bfloat16))
        return x.float().reshape(-1).cpu().numpy()

    def decode_agent(self, frames: Frames) -> np.ndarray:
        return self.decode(frames.latents["agent"])

    # ------------------------------------------------------------ generation

    def rollout(self, cartridge: Cartridge | None, user_wav: np.ndarray, max_s: float,
                temps=TEMPS, cache_len: int = CONTEXT_FRAMES) -> Dialogue:
        """`completion`'s loop with the user channel teacher-forced: prefill the
        prompt, then each step feeds [user latent t, agent latent t] and keeps the
        agent half of the predicted frame t+1. Stops at the end of `user_wav`,
        at `max_s`, or when the caches are full. Wavs are written to a temp dir."""
        if cartridge is not None:
            raise NotImplementedError("rollout from a cartridge is #18")
        if self.prompt is None:
            raise RuntimeError("set_prompt() first: the stock system needs its prompt wav")
        m = self.model
        prompt = torch.cat([self.latents(self.prompt[0]), self.latents(self.prompt[1])], -1)[None]
        user = self.latents(np.asarray(user_wav, dtype=np.float32))
        n = min(user.shape[0], int(max_s * self.spec.frame_rate_hz), cache_len - prompt.shape[1])
        agent = []
        with self._ctx():
            m.init_cache(bsize=1, device=self.device, dtype=torch.bfloat16, length=cache_len)
            try:
                nxt = m.next_latent(prompt.to(torch.bfloat16), temps)
                for t in range(n):
                    a = nxt[..., LATENT:]
                    agent.append(a)
                    if t + 1 < n:
                        nxt = m.next_latent(torch.cat([user[None, t:t + 1].to(a.dtype), a], -1), temps)
            finally:
                m.deinit_cache()
        wav = self.decode(torch.cat(agent, 1)[0].float())
        out = Path(tempfile.mkdtemp(prefix="hertz-rollout-"))
        import soundfile as sf

        sf.write(out / "user.wav", np.asarray(user_wav, dtype=np.float32).reshape(-1)[: n * CHUNK], SAMPLE_RATE)
        sf.write(out / "agent.wav", wav, SAMPLE_RATE)
        return Dialogue(id=f"rollout-{uuid.uuid4().hex[:8]}", user_wav=str(out / "user.wav"),
                        agent_wav=str(out / "agent.wav"))

    def session(self, temps=TEMPS, cache_len: int = CONTEXT_FRAMES) -> "HertzSession":
        if self.prompt is None:
            raise RuntimeError("set_prompt() first: the stock system needs its prompt wav")
        return HertzSession(self, self.prompt, temps, cache_len)

    client_path = "/audio"  # where a client connects, under the server's URL

    def serve(self, cartridge: Cartridge | None, port: int, cache_len: int = CONTEXT_FRAMES) -> "HertzServer":
        """Start the WebSocket server (upstream protocol) on `port` in a daemon
        thread; returns once it accepts connections. Loads the model if needed;
        the stock prompt must be set (`set_prompt`)."""
        if cartridge is not None:
            raise NotImplementedError("serving a cartridge is #12/#18")
        if self.prompt is None:
            raise RuntimeError("set_prompt() first: the stock system needs its prompt wav")
        if self.model is None:
            self.load()
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(make_app(self, cache_len), host="0.0.0.0", port=port, log_level="info"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("hertz server failed to start")
            time.sleep(0.1)
        return HertzServer(url=f"ws://localhost:{port}{self.client_path}", server=server, thread=thread)

    # ------------------------------------------------------------ cartridge half (#18)

    def build_prefix(self, voice_wav, persona_text, digest_text) -> Frames:
        raise NotImplementedError("hertz build_prefix is #18")

    def cartridge_from_prefill(self, prefix):
        raise NotImplementedError("hertz cartridge_from_prefill is #18")

    def forward(self, frames, cartridge):
        raise NotImplementedError("hertz forward is #18")

    def sft_loss(self, out, frames, w):
        raise NotImplementedError("hertz sft_loss is #18")

    def distill_loss(self, out, targets):
        raise NotImplementedError("hertz distill_loss is #18")


class HertzSession:
    """One live conversation: upstream `AudioProcessor`'s live path. The prompt
    is prefilled through `next_audio_from_audio`; each `step` takes 125 ms of
    user audio and returns the agent's next 125 ms. The model holds the caches,
    so one session at a time per model."""

    def __init__(self, backend: HertzBackend, prompt: np.ndarray, temps, cache_len: int):
        self.backend, self.temps, self.cache_len = backend, temps, cache_len
        self.model = backend.model
        self.frames = prompt.shape[1] // CHUNK
        if self.frames >= cache_len:
            raise ContextFull(f"prompt is {self.frames} frames; cache holds {cache_len}")
        with backend._ctx():
            self.model.init_cache(bsize=1, device=backend.device, dtype=torch.bfloat16, length=cache_len)
            self.next_agent = self.model.next_audio_from_audio(backend._tensor(prompt)[None], temps=temps)

    @property
    def remaining(self) -> int:
        return self.cache_len - self.frames

    def step(self, user: np.ndarray) -> np.ndarray:
        """[2000] float32 user audio -> [2000] float32 agent audio (the frame after it)."""
        if self.frames >= self.cache_len:
            raise ContextFull(f"context full: {self.frames} frames (cache_len {self.cache_len})")
        x = self.backend._tensor(user.reshape(-1)).reshape(1, 1, -1)
        with self.backend._ctx():
            audio = torch.cat([x.to(self.next_agent.dtype), self.next_agent], dim=1)
            self.next_agent = self.model.next_audio_from_audio(audio, temps=self.temps)
        self.frames += 1
        return self.next_agent.float().reshape(-1).cpu().numpy()

    def close(self) -> None:
        self.model.deinit_cache()


class CpuStandIn:
    """Upstream `HertzDevModel`'s inference surface on CPU, for tests and
    `--dry-run`: same shapes (2000 samples per 32-d latent, an FSQ index per
    frame, [B, 2, N] duplex audio) and the same fixed-length caches, so a
    forward that would write past `length` frames raises."""

    def __init__(self):
        class Tokenizer:
            def latent_from_data(self, x, get_parameters=False):
                mu = x.reshape(x.shape[0], -1, CHUNK)[..., :LATENT]
                return {"mu": mu, "logvar": mu * 0, "z": mu} if get_parameters else mu

            def data_from_latent(self, lat):
                return lat.mean(-1, keepdim=True).expand(*lat.shape[:2], CHUNK).reshape(lat.shape[0], 1, -1)

        class Resynthesizer:
            @staticmethod
            def quantizer(x, return_latent=False):
                return x, (x.sum(-1).abs() * 100).long() % VOCAB

        self.audio_tokenizer, self.resynthesizer = Tokenizer(), Resynthesizer()
        self.length, self.used = None, 0

    def init_cache(self, bsize, device, dtype, length=None):
        self.length, self.used = length, 0

    def deinit_cache(self):
        self.length = None

    def _write(self, frames: int):
        if self.length is None:
            raise RuntimeError("no cache")
        if self.used + frames > self.length:
            raise RuntimeError(f"cache of {self.length} frames overflowed")
        self.used += frames

    def next_latent(self, x, temps):
        self._write(x.shape[1])
        return x[:, -1:] + 0.01

    def next_audio_from_audio(self, audio, temps):
        self._write(audio.shape[-1] // CHUNK)
        return audio[:, 1:2, -CHUNK:] * 0.5


@dataclass
class HertzServer:
    url: str
    server: object = None  # uvicorn.Server
    thread: threading.Thread | None = None
    timings: dict = field(default_factory=dict)  # seconds: load_s, ready_s (set by the deployer)

    def wait(self) -> int:
        """Block until the server stops; 0 on a clean exit."""
        self.thread.join()
        return 0


def make_app(backend: HertzBackend, cache_len: int = CONTEXT_FRAMES):
    """FastAPI app with upstream's `/audio` WebSocket: one text frame in (125 ms
    of user audio), one text frame out (the agent's next 125 ms). One session
    at a time; a second connection is closed with 1013. When the context is
    full the session is closed with 1011 and reason "context full ...".
    `/health` reports load time and peak VRAM."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app = FastAPI()
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        return {"loaded": backend.model is not None, "load_s": getattr(backend, "load_s", None),
                "max_vram_gib": round(mem, 2), "cache_len": cache_len}

    @app.websocket("/audio")
    async def audio(ws: WebSocket):
        await ws.accept()
        if lock.locked():
            await ws.close(code=1013, reason="busy: one session at a time")
            return
        async with lock:
            session = await asyncio.to_thread(backend.session, TEMPS, cache_len)
            steps, compute = 0, 0.0
            try:
                while True:
                    user = unpack(await ws.receive_text())
                    t0 = time.perf_counter()
                    try:
                        agent = await asyncio.to_thread(session.step, user)
                    except ContextFull as e:
                        logger.warning("%s", e)
                        await ws.close(code=1011, reason=str(e)[:120])
                        return
                    compute += time.perf_counter() - t0
                    steps += 1
                    await ws.send_text(pack(agent))
            except WebSocketDisconnect:
                pass
            finally:
                logger.info("session: %d steps, %.1f ms/step compute, %d frames used of %d",
                            steps, 1000 * compute / max(steps, 1), session.frames, cache_len)
                await asyncio.to_thread(session.close)

    return app
