"""PersonaPlex 7B backend (NVIDIA's moshi fork at `FORK_COMMIT`): the inference
half (#20). The cartridge half (`build_prefix`, `cartridge_from_prefill`,
`forward`, `sft_loss`, `distill_loss`) is #8 and raises NotImplementedError.

Streams (17, 12.5 Hz, 24 kHz audio), in the model's codebook order:

    agent_text                 [T]  SentencePiece ids (32k); PAD=3 between words, EPAD=0 the frame before a word
    agent_audio_0..7           [T]  Mimi codebooks of the agent channel (2048 each); 0 is semantic
    user_audio_0..7            [T]  Mimi codebooks of the user channel

`Frames` hold them undelayed: frame t of every stream is audio window
[t, t+1) / 12.5 s. The model's acoustic delay of one frame
(`DELAYS`) is applied inside the model, not in the frames.

The stock system is a shipped voice embedding (`STOCK_VOICE`, from the
checkpoint's voices.tgz) and a persona text prompt (`STOCK_PERSONA`). The
server receives both per connection as query parameters of `/api/chat`.

    backend = backends.get("personaplex", {"checkpoint": "nvidia/personaplex-7b-v1"})
    backend.load("cuda")
    frames = backend.encode_dialogue(dialogue)
    d = backend.rollout(None, user_wav, max_s=30.0, temps=None)
    server = backend.serve(None, port=8998)        # the fork's moshi.server, unchanged
"""

from __future__ import annotations

import logging
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import torch

from voice.backends.base import BackendSpec, Dialogue, Frames, StreamSpec
from voice.backends.registry import PersonaPlexConfig, register

logger = logging.getLogger(__name__)

FORK_COMMIT = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples
CONTEXT_FRAMES = 3000
TEXT_VOCAB, AUDIO_VOCAB, CODEBOOKS = 32000, 2048, 8
PAD, EPAD = 3, 0  # text padding and end-of-padding ids (the fork's `existing_text_padding_id` and EPAD)
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  # per stream, in STREAMS order

AGENT_AUDIO = [f"agent_audio_{i}" for i in range(CODEBOOKS)]
USER_AUDIO = [f"user_audio_{i}" for i in range(CODEBOOKS)]
STREAMS = ["agent_text", *AGENT_AUDIO, *USER_AUDIO]

STOCK_VOICE = "NATF2.pt"  # the fork's README example voice
STOCK_PERSONA = "You enjoy having a good conversation."
DEFAULT_TEMPS = {"audio": 0.8, "text": 0.7}  # the fork's defaults (offline.py)


def spec() -> BackendSpec:
    streams = [StreamSpec(name="agent_text", role="agent", kind="text", vocab=TEXT_VOCAB)]
    streams += [StreamSpec(name=s, role="agent", kind="audio_token", vocab=AUDIO_VOCAB) for s in AGENT_AUDIO]
    streams += [StreamSpec(name=s, role="user", kind="audio_token", vocab=AUDIO_VOCAB) for s in USER_AUDIO]
    return BackendSpec(name="personaplex", sample_rate=SAMPLE_RATE, frame_rate_hz=FRAME_RATE,
                       context_frames=CONTEXT_FRAMES, streams=streams, has_text=True, cache_groups=["temporal"])


def chat_path(voice: str = STOCK_VOICE, persona: str = STOCK_PERSONA) -> str:
    """The server's WebSocket path with a system (voice embedding + persona) as its query."""
    return "/api/chat?" + urlencode({"voice_prompt": voice, "text_prompt": persona})


def text_stream(words: list[tuple[list[int], float, float]], n_frames: int) -> torch.Tensor:
    """The agent text stream [n_frames] from (token ids, start_s, end_s) per word.

    moshi-finetune's `Interleaver.build_token_stream` with audio_delay 0 and
    keep_and_shift=True (a word's leftover tokens are emitted after the next
    word starts instead of dropped): a word's tokens go one per frame from its
    start frame, EPAD replaces the PAD in the frame before, and PAD fills the
    rest (in-word padding is PAD too)."""
    out = [PAD] * n_frames
    words = sorted((w for w in words if w[1] < w[2]), key=lambda w: w[1])
    pending: list[int] = []
    i, last_end = 0, -1
    for t in range(n_frames):
        while i < len(words) and words[i][1] * FRAME_RATE < t + 1:
            pending.extend(words[i][0])
            last_end = int(words[i][2] * FRAME_RATE)
            i += 1
        if pending:
            if t > 0 and out[t - 1] == PAD:
                out[t - 1] = EPAD
            out[t] = pending.pop(0)
        elif t <= last_end:
            out[t] = PAD
    return torch.tensor(out, dtype=torch.long)


def read_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 at `sample_rate` (first channel, as the fork reads prompts)."""
    import sphn

    pcm, sr = sphn.read(path)
    return sphn.resample(pcm, src_sample_rate=sr, dst_sample_rate=sample_rate)[0].astype(np.float32)


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    import soundfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.asarray(pcm, dtype=np.float32), sample_rate)
    return str(path)


def pad_to_frames(pcm: np.ndarray, n_frames: int) -> np.ndarray:
    out = np.zeros(n_frames * FRAME_SIZE, dtype=np.float32)
    n = min(len(pcm), len(out))
    out[:n] = pcm[:n]
    return out


def wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"moshi.server exited with code {proc.returncode} before listening")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"moshi.server did not listen on {port} within {timeout_s:.0f} s")


@dataclass
class PersonaPlexServer:
    """The fork's `moshi.server` in a subprocess. `url` carries the stock system."""

    url: str
    process: subprocess.Popen
    timings: dict = field(default_factory=dict)  # seconds: resolve (weights on disk), ready (listening)

    def wait(self) -> int:
        return self.process.wait()

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=30)


@register("personaplex")
class PersonaPlexBackend:
    def __init__(self, config: PersonaPlexConfig):
        self.config = config
        self.spec = spec()
        self.voice = STOCK_VOICE
        self.persona = STOCK_PERSONA

    @property
    def client_path(self) -> str:
        """Path and query a client connects to on `serve`'s host; deploy.py reads it."""
        return chat_path(self.voice, self.persona)

    # ------------------------------------------------------------ weights

    def files(self) -> dict[str, str]:
        """Local paths of the tokenizer, LM, Mimi and voice-prompt dir; downloads
        from the HF repo `config.checkpoint` unless it is a local directory."""
        from moshi.models import loaders

        names = {"tokenizer": loaders.TEXT_TOKENIZER_NAME, "moshi": loaders.MOSHI_NAME, "mimi": loaders.MIMI_NAME}
        root = Path(self.config.checkpoint)
        if root.is_dir():
            return {k: str(root / v) for k, v in names.items()} | {"voices": str(root / "voices")}
        from huggingface_hub import hf_hub_download

        repo = self.config.checkpoint
        out = {k: hf_hub_download(repo, v) for k, v in names.items()}
        voices = Path(hf_hub_download(repo, "voices.tgz")).parent / "voices"
        if not voices.exists():
            import tarfile

            with tarfile.open(voices.parent / "voices.tgz", "r:gz") as tar:
                tar.extractall(path=voices.parent)
        return out | {"voices": str(voices)}

    def load(self, device: str) -> None:
        """Mimi, the SentencePiece tokenizer and the frozen LM, as the fork's server loads them."""
        import sentencepiece
        from moshi.models import LMGen, loaders

        self.device = device
        f = self.files()
        self.paths = f
        self.mimi = loaders.get_mimi(f["mimi"], device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(f["tokenizer"])
        self.lm = loaders.get_moshi_lm(f["moshi"], device=device).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm_gen = LMGen(self.lm, device=device, sample_rate=SAMPLE_RATE, frame_rate=FRAME_RATE,
                            audio_silence_frame_cnt=int(0.5 * FRAME_RATE))

    # ------------------------------------------------------------ encoding

    @torch.no_grad()
    def _mimi_encode(self, pcm: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(pcm).to(self.device)[None, None]
        return self.mimi.encode(x)[0].cpu()  # [8, T]

    def encode_dialogue(self, d: Dialogue) -> Frames:
        """Mimi codes of both wavs (padded to the longer one) and the agent text
        stream from `d.words` (the agent wav's word timestamps)."""
        user, agent = read_wav(d.user_wav), read_wav(d.agent_wav)
        n = math.ceil(max(len(user), len(agent)) / FRAME_SIZE)
        user_codes, agent_codes = self._mimi_encode(pad_to_frames(user, n)), self._mimi_encode(pad_to_frames(agent, n))
        words = [(self.tokenizer.encode(w["w"].strip()), w["start_s"], w["end_s"]) for w in d.words]
        tokens = {"agent_text": text_stream(words, n)}
        tokens |= {s: agent_codes[i].long() for i, s in enumerate(AGENT_AUDIO)}
        tokens |= {s: user_codes[i].long() for i, s in enumerate(USER_AUDIO)}
        frame_words = [{**w, "frame": int(w["start_s"] * FRAME_RATE)} for w in d.words]
        return Frames(tokens=tokens, meta={"id": d.id, "split": d.split, "words": frame_words})

    @torch.no_grad()
    def decode_agent(self, frames: Frames) -> np.ndarray:
        codes = torch.stack([frames.tokens[s] for s in AGENT_AUDIO])[None].to(self.device)  # [1, 8, T]
        return self.mimi.decode(codes)[0, 0].float().cpu().numpy()

    # ------------------------------------------------------------ cartridge half (#8)

    def build_prefix(self, voice_wav, persona_text, digest_text):
        raise NotImplementedError("personaplex build_prefix ships with #8")

    def cartridge_from_prefill(self, prefix):
        raise NotImplementedError("personaplex cartridge_from_prefill ships with #8")

    def forward(self, frames, cartridge):
        raise NotImplementedError("personaplex forward ships with #8")

    def sft_loss(self, out, frames, w):
        raise NotImplementedError("personaplex sft_loss ships with #8")

    def distill_loss(self, out, targets):
        raise NotImplementedError("personaplex distill_loss ships with #8")

    # ------------------------------------------------------------ generation and serving

    @torch.no_grad()
    def rollout(self, cartridge, user_wav: np.ndarray, max_s: float, temps: dict | None = None,
                out_dir: str | None = None) -> Dialogue:
        """Stock system prompt (`self.voice`, `self.persona`) then `user_wav`
        (24 kHz mono, padded with silence or cut to `max_s`) frame by frame, as
        the fork's `offline.py` does. Returns a Dialogue whose agent wav is
        frame-aligned with the user wav and whose `response_text` is the
        sampled text stream."""
        if cartridge is not None:
            raise NotImplementedError("serving a cartridge needs the pinned prefix (#12)")
        t ={**DEFAULT_TEMPS, **(temps or {})}
        gen = self.lm_gen
        gen.temp, gen.temp_text = t["audio"], t["text"]
        voice = os.path.join(self.paths["voices"], self.voice)
        user = pad_to_frames(np.asarray(user_wav, dtype=np.float32), round(max_s * FRAME_RATE))
        agent, pieces = [], []
        with self.mimi.streaming(1), gen.streaming(1):
            gen.load_voice_prompt_embeddings(voice) if voice.endswith(".pt") else gen.load_voice_prompt(voice)
            # moshi.server.wrap_with_system_tags (that module runs its server on import)
            gen.text_prompt_tokens = self.tokenizer.encode(f"<system> {self.persona.strip()} <system>")
            gen.step_system_prompts(self.mimi)
            self.mimi.reset_streaming()
            for start in range(0, len(user), FRAME_SIZE):
                chunk = torch.from_numpy(user[start:start + FRAME_SIZE]).to(self.device)[None, None]
                codes = self.mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = gen.step(codes[:, :, c:c + 1])
                    if tokens is None:
                        continue
                    agent.append(self.mimi.decode(tokens[:, 1:9])[0, 0].float().cpu().numpy())
                    text = tokens[0, 0, 0].item()
                    if text not in (EPAD, PAD):
                        pieces.append(self.tokenizer.id_to_piece(text).replace("▁", " "))
        out = Path(out_dir or tempfile.mkdtemp(prefix="personaplex-rollout-"))
        rid = uuid.uuid4().hex[:8]
        agent_pcm = pad_to_frames(np.concatenate(agent) if agent else np.zeros(0, np.float32), len(user) // FRAME_SIZE)
        return Dialogue(id=rid, user_wav=write_wav(out / f"{rid}.user.wav", user),
                        agent_wav=write_wav(out / f"{rid}.agent.wav", agent_pcm),
                        response_text="".join(pieces).strip())

    def serve(self, cartridge, port: int, startup_timeout_s: float = 1200.0) -> PersonaPlexServer:
        """Start the fork's `moshi.server` unchanged (no static UI) on `port`
        and return once it listens. Weights are resolved first (HF cache) so the
        server never downloads; `timings` splits the cold start."""
        if cartridge is not None:
            raise NotImplementedError("serving a cartridge needs the pinned prefix (#12)")
        t0 = time.monotonic()
        f = self.files()
        resolved = time.monotonic() - t0
        cmd = [sys.executable, "-m", "moshi.server", "--host", "0.0.0.0", "--port", str(port), "--static", "none",
               "--device", self.config.device, "--tokenizer", f["tokenizer"], "--moshi-weight", f["moshi"],
               "--mimi-weight", f["mimi"], "--voice-prompt-dir", f["voices"]]
        if not Path(self.config.checkpoint).is_dir():
            cmd += ["--hf-repo", self.config.checkpoint]
        logger.info("starting %s", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        wait_for_port(port, proc, startup_timeout_s)
        timings = {"resolve_s": resolved, "ready_s": time.monotonic() - t0}
        logger.info("moshi.server listening on %d: %s", port, timings)
        return PersonaPlexServer(url=f"ws://localhost:{port}{self.client_path}",
                                 process=proc, timings=timings)
