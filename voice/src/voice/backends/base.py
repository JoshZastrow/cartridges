"""Backend-agnostic types and the `DuplexBackend` protocol (brief section 4b).

The data pipeline, trainer, evals and report code against these types only and
get a backend from `voice.backends.get(name, config)`.

Shapes (T = frames of one dialogue at `spec.frame_rate_hz`, p = prefix frames):

    Frames.tokens[stream]    [T] int      discrete streams (text, audio codebook, FSQ index)
    Frames.latents[stream]   [T, D] float continuous streams (hertz only)
    Outputs.logits[stream]   [B, T, V]    agent discrete streams; position t predicts frame t
    Outputs.gmm[stream]      GMMParams    agent latent streams (hertz only)
    Targets.topk_*[stream]   [T, k]       stored teacher top-k for frame t
    KVGroup.keys[l]          [1, H, p, D] per layer; the prefix axis is dim 2

A `Cartridge` is the trainable prefix: one `KVGroup` per name in
`spec.cache_groups` (PersonaPlex: "temporal"; hertz: "lm", "resyn"). The first
`num_frozen` prefix positions of every group are buffers (the attention sink);
the rest are fp32 parameters. On disk it is a directory with

    cartridge.safetensors  <group>.keys.<layer>, <group>.values.<layer>  (full [1, H, p, D])
    cartridge.json         {p, num_frozen_tokens, backend, spec_hash, base_model, fork_commit,
                            voice_sha, persona, digest_sha, stage, step}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

WEIGHTS = "cartridge.safetensors"
CONFIG = "cartridge.json"


class StreamSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    role: Literal["user", "agent"]
    kind: Literal["text", "audio_token", "latent"]
    vocab: int | None = None  # discrete streams
    dim: int | None = None  # latent streams


class BackendSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    sample_rate: int
    frame_rate_hz: float
    context_frames: int
    streams: list[StreamSpec]
    has_text: bool
    cache_groups: list[str]

    @property
    def hash(self) -> str:
        """12 hex chars identifying the spec; a cartridge records it."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class Dialogue(BaseModel):
    """One dialogue as the data pipeline (#10) writes it and `rollout` returns it.
    Wavs are paths; `words` are the agent wav's Phonon-2 word timestamps."""

    id: str
    family: str = ""
    split: str = ""
    question: str = ""
    response_text: str = ""
    passage_ids: list[str] = []
    coverage: float = 0.0
    user_wav: str = ""
    agent_wav: str = ""
    words: list[dict] = []  # {w, start_s, end_s}


class Frames(BaseModel):
    """One dialogue encoded for one backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokens: dict[str, Tensor] = {}
    latents: dict[str, Tensor] = {}
    meta: dict = {}  # id, split, words with frame indices

    @property
    def id(self) -> str:
        return self.meta["id"]

    def save(self, path: str | Path) -> Path:
        """One safetensors file: `tokens.<stream>`, `latents.<stream>`, meta as JSON metadata."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"tokens.{k}": v.contiguous().cpu() for k, v in self.tokens.items()}
        tensors |= {f"latents.{k}": v.contiguous().cpu() for k, v in self.latents.items()}
        save_file(tensors, str(path), metadata={"meta": json.dumps(self.meta)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> Frames:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["meta"])
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        split = lambda prefix: {k.removeprefix(prefix): v for k, v in tensors.items() if k.startswith(prefix)}
        return cls(tokens=split("tokens."), latents=split("latents."), meta=meta)


class Targets(BaseModel):
    """A teacher's stored top-k for one dialogue, per discrete agent stream."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topk_ids: dict[str, Tensor]  # [T, k] long
    topk_logprobs: dict[str, Tensor]  # [T, k] float

    @classmethod
    def from_rows(cls, rows) -> Targets:
        """From the `shared.targets.TargetRow`s of one dialogue (one per stream)."""
        return cls(topk_ids={r.stream: torch.tensor(r.topk_ids, dtype=torch.long) for r in rows},
                   topk_logprobs={r.stream: torch.tensor(r.topk_logprobs, dtype=torch.float32) for r in rows})


class GMMParams(NamedTuple):
    locs: Tensor  # [B, T, K, D]
    scales: Tensor  # [B, T, K, D]
    weights: Tensor  # [B, T, K]


class Outputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: dict[str, Tensor] = {}
    gmm: dict[str, GMMParams] = {}


class LossWeights(BaseModel):
    """Per-stream weights for `sft_loss`; a stream not listed gets `default`.
    Backends may read `extra` for their own knobs (e.g. PersonaPlex's text pad weight)."""

    streams: dict[str, float] = {}
    default: float = 1.0
    extra: dict[str, float] = Field(default_factory=dict)

    def of(self, stream: str) -> float:
        return self.streams.get(stream, self.default)


class KVGroup(nn.Module):
    """One cache group of a cartridge: per-layer K/V [1, H, p, D], the first
    `num_frozen` prefix positions frozen."""

    def __init__(self, keys: list[Tensor], values: list[Tensor], num_frozen: int = 1):
        super().__init__()
        assert len(keys) == len(values) and keys, "a group needs at least one layer"
        self.num_frozen = num_frozen
        n = num_frozen
        for name, kv in (("keys", keys), ("values", values)):
            frozen = [t[:, :, :n].detach().float().contiguous().clone() for t in kv]
            for i, t in enumerate(frozen):
                self.register_buffer(f"frozen_{name}_{i}", t)
            setattr(self, f"trainable_{name}", nn.ParameterList(
                [nn.Parameter(t[:, :, n:].detach().float().contiguous().clone()) for t in kv]))

    def __len__(self) -> int:
        return len(self.trainable_keys)

    def _full(self, name: str) -> list[Tensor]:
        return [torch.cat([getattr(self, f"frozen_{name}_{i}"), t], dim=2)
                for i, t in enumerate(getattr(self, f"trainable_{name}"))]

    @property
    def keys(self) -> list[Tensor]:
        return self._full("keys")

    @property
    def values(self) -> list[Tensor]:
        return self._full("values")


class Cartridge(nn.Module):
    """The trainable prefix: named `KVGroup`s, all with the same `p`."""

    def __init__(self, groups: dict[str, KVGroup], meta: dict | None = None):
        super().__init__()
        self.groups = nn.ModuleDict(groups)
        ps = {g.keys[0].shape[2] for g in self.groups.values()}
        frozen = {g.num_frozen for g in self.groups.values()}
        assert len(ps) == 1 and len(frozen) == 1, f"groups disagree on p {ps} or num_frozen {frozen}"
        self.p, self.num_frozen_tokens = ps.pop(), frozen.pop()
        self.meta = dict(meta or {})

    def config_dict(self) -> dict:
        return {"p": self.p, "num_frozen_tokens": self.num_frozen_tokens, **self.meta}

    def save(self, path: str | Path) -> Path:
        """Write `cartridge.safetensors` and `cartridge.json` into directory `path`."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {f"{name}.{kind}.{i}": t.detach().contiguous().cpu()
                   for name, g in self.groups.items() for kind in ("keys", "values")
                   for i, t in enumerate(getattr(g, kind))}
        save_file(tensors, str(path / WEIGHTS))
        (path / CONFIG).write_text(json.dumps(self.config_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> Cartridge:
        from safetensors.torch import load_file

        path = Path(path)
        meta = json.loads((path / CONFIG).read_text())
        meta.pop("p")
        num_frozen = meta.pop("num_frozen_tokens")
        sd = load_file(str(path / WEIGHTS), device=str(device or "cpu"))
        names = sorted({k.split(".")[0] for k in sd})

        def layers(group: str, kind: str) -> list[Tensor]:
            n = sum(k.startswith(f"{group}.{kind}.") for k in sd)
            return [sd[f"{group}.{kind}.{i}"] for i in range(n)]

        groups = {g: KVGroup(layers(g, "keys"), layers(g, "values"), num_frozen) for g in names}
        return cls(groups, meta)


class Server(Protocol):
    url: str


@runtime_checkable
class DuplexBackend(Protocol):
    spec: BackendSpec

    def load(self, device: str) -> None: ...
    def encode_dialogue(self, d: Dialogue) -> Frames: ...
    def decode_agent(self, frames: Frames) -> np.ndarray: ...
    def build_prefix(self, voice_wav: np.ndarray, persona_text: str, digest_text: str) -> Frames: ...
    def cartridge_from_prefill(self, prefix: Frames) -> Cartridge: ...
    def forward(self, frames: Frames, cartridge: Cartridge | None) -> Outputs: ...
    def sft_loss(self, out: Outputs, frames: Frames, w: LossWeights) -> dict[str, Tensor]: ...
    def distill_loss(self, out: Outputs, targets: Targets) -> Tensor: ...
    def rollout(self, cartridge: Cartridge, user_wav: np.ndarray, max_s: float, temps: Any) -> Dialogue: ...
    def serve(self, cartridge: Cartridge, port: int) -> Server: ...
