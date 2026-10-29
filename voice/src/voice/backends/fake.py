"""FakeBackend: a CPU test double with two tiny discrete streams.

Streams `user` and `agent` (vocab `config.vocab`), one cache group "temporal"
with one layer, K/V [1, 1, p, dim]. The frozen "model" is a seeded embedding,
one causal attention layer over [prefix; frames] and an output head, so
`forward` is differentiable w.r.t. the cartridge and training it lowers the
loss. Alignment as in base.py: the input is shifted right by one frame, so
logits position t sees frames < t and predicts agent frame t.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from voice.backends.base import (
    BackendSpec,
    Cartridge,
    Dialogue,
    Frames,
    KVGroup,
    LossWeights,
    Outputs,
    StreamSpec,
    Targets,
)
from voice.backends.registry import FakeConfig, register


def kl_to_topk(logits: Tensor, topk_ids: Tensor, topk_logprobs: Tensor) -> Tensor:
    """KL(teacher || student) over the teacher's top-k entries, mean over
    positions, as `cartridges.cartridge.train_distill.distill_loss` computes it.
    logits [T, V], topk_* [T, k]; row t of each is frame t (base.py alignment)."""
    student = logits.float().log_softmax(-1).gather(-1, topk_ids)
    return (topk_logprobs.exp() * (topk_logprobs - student)).sum(-1).mean()


@register("fake")
class FakeBackend:
    def __init__(self, config: FakeConfig):
        self.config = config
        v, d = config.vocab, config.dim
        self.spec = BackendSpec(
            name="fake", sample_rate=8000, frame_rate_hz=10.0, context_frames=256, has_text=False,
            streams=[StreamSpec(name="user", role="user", kind="audio_token", vocab=v),
                     StreamSpec(name="agent", role="agent", kind="audio_token", vocab=v)],
            cache_groups=["temporal"],
        )
        g = torch.Generator().manual_seed(config.seed)
        self.embed = {s: torch.randn(v, d, generator=g) for s in ("user", "agent")}
        self.wq, self.wk, self.wv = (torch.randn(d, d, generator=g) / d**0.5 for _ in range(3))
        self.head = torch.randn(d, v, generator=g) / d**0.5

    def load(self, device: str) -> None:
        self.device = device

    def _x(self, frames: Frames) -> Tensor:
        return self.embed["user"][frames.tokens["user"]] + self.embed["agent"][frames.tokens["agent"]]

    def encode_dialogue(self, d: Dialogue) -> Frames:
        g = torch.Generator().manual_seed(int.from_bytes(d.id.encode()[:7].ljust(7, b"\0"), "little"))
        t = max(2, len(d.response_text.split()))
        tokens = {s: torch.randint(self.config.vocab, (t,), generator=g) for s in ("user", "agent")}
        return Frames(tokens=tokens, meta={"id": d.id, "split": d.split})

    def decode_agent(self, frames: Frames) -> np.ndarray:
        n = int(len(frames.tokens["agent"]) * self.spec.sample_rate / self.spec.frame_rate_hz)
        return np.zeros(n, dtype=np.float32)

    def build_prefix(self, voice_wav, persona_text: str, digest_text: str) -> Frames:
        g = torch.Generator().manual_seed(self.config.seed + 1)
        p = self.config.p
        return Frames(tokens={s: torch.randint(self.config.vocab, (p,), generator=g) for s in ("user", "agent")},
                      meta={"id": "prefix"})

    @torch.no_grad()
    def cartridge_from_prefill(self, prefix: Frames) -> Cartridge:
        x = self._x(prefix)
        k, v = (x @ self.wk)[None, None], (x @ self.wv)[None, None]
        return Cartridge({"temporal": KVGroup([k], [v], num_frozen=1)},
                         meta={"backend": "fake", "spec_hash": self.spec.hash})

    def forward(self, frames: Frames, cartridge: Cartridge | None) -> Outputs:
        x = self._x(frames)  # [T, d]
        x = torch.cat([torch.zeros_like(x[:1]), x[:-1]])  # position t sees frames < t
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        t, p = len(x), 0
        if cartridge is not None:
            group = cartridge.groups["temporal"]
            pk, pv = group.keys[0][0, 0], group.values[0][0, 0]
            k, v, p = torch.cat([pk, k]), torch.cat([pv, v]), len(pk)
        mask = torch.ones(t, p + t, dtype=torch.bool).tril(diagonal=p)
        h = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask[None])[0]
        return Outputs(logits={"agent": ((x + h) @ self.head)[None]})

    def sft_loss(self, out: Outputs, frames: Frames, w: LossWeights) -> dict[str, Tensor]:
        return {"agent": w.of("agent") * F.cross_entropy(out.logits["agent"][0], frames.tokens["agent"])}

    def distill_loss(self, out: Outputs, targets: Targets) -> Tensor:
        return kl_to_topk(out.logits["agent"][0], targets.topk_ids["agent"], targets.topk_logprobs["agent"])

    def rollout(self, cartridge, user_wav, max_s, temps) -> Dialogue:
        raise NotImplementedError("the fake backend has no audio")

    def serve(self, cartridge, port):
        raise NotImplementedError("the fake backend has no server")


def fixture(dir, n: int = 6, frames: int = 12, config: FakeConfig | None = None) -> list[str]:
    """Write `n` fake Frames files with a learnable pattern (agent[t+1] = user[t])
    under `dir`; returns their paths."""
    config = config or FakeConfig()
    g = torch.Generator().manual_seed(config.seed + 2)
    paths = []
    for i in range(n):
        user = torch.randint(config.vocab, (frames,), generator=g)
        agent = torch.cat([torch.zeros(1, dtype=torch.long), user[:-1]])
        f = Frames(tokens={"user": user, "agent": agent}, meta={"id": f"fake-{i}", "split": "train"})
        paths.append(str(f.save(f"{dir}/fake-{i}.safetensors")))
    return paths
