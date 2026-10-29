"""backends/base.py: Cartridge save/load, KVGroup freezing, Frames IO, the protocol."""

import json

import pytest
import torch
from voice.backends.base import BackendSpec, Cartridge, DuplexBackend, Frames, KVGroup, LossWeights, StreamSpec, Targets
from voice.backends.fake import FakeBackend
from voice.backends.registry import FakeConfig


def kv(layers: int, p: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 2, p, 4, generator=g) for _ in range(layers)]


def cartridge() -> Cartridge:
    groups = {"lm": KVGroup(kv(3, 5, 0), kv(3, 5, 1), num_frozen=2), "resyn": KVGroup(kv(2, 5, 2), kv(2, 5, 3), 2)}
    return Cartridge(groups, meta={"backend": "hertz", "stage": "sft", "step": 7})


def test_cartridge_round_trip(tmp_path):
    c = cartridge()
    with torch.no_grad():
        c.groups["lm"].trainable_keys[1].add_(1.0)
    c.save(tmp_path)
    cfg = json.loads((tmp_path / "cartridge.json").read_text())
    assert cfg == {"p": 5, "num_frozen_tokens": 2, "backend": "hertz", "stage": "sft", "step": 7}
    from safetensors.torch import load_file
    assert sorted(load_file(tmp_path / "cartridge.safetensors")) == sorted(
        [f"lm.{k}.{i}" for k in ("keys", "values") for i in range(3)]
        + [f"resyn.{k}.{i}" for k in ("keys", "values") for i in range(2)])
    d = Cartridge.load(tmp_path)
    assert (d.p, d.num_frozen_tokens, d.meta) == (5, 2, c.meta)
    for name in ("lm", "resyn"):
        for a, b in zip(c.groups[name].keys + c.groups[name].values, d.groups[name].keys + d.groups[name].values):
            assert torch.equal(a, b)


def test_frozen_prefix_positions_get_no_parameters():
    c = cartridge()
    params = list(c.parameters())
    assert len(params) == 2 * (3 + 2) and all(p.shape[2] == 3 for p in params)
    loss = sum(k.sum() for g in c.groups.values() for k in g.keys)
    loss.backward()
    assert all(p.grad is not None for p in c.groups["lm"].trainable_keys)


def test_groups_must_agree_on_p():
    with pytest.raises(AssertionError):
        Cartridge({"a": KVGroup(kv(1, 4, 0), kv(1, 4, 0)), "b": KVGroup(kv(1, 5, 0), kv(1, 5, 0))})


def test_frames_round_trip(tmp_path):
    f = Frames(tokens={"user": torch.arange(5), "agent": torch.arange(5) * 2},
               latents={"agent": torch.randn(5, 3)}, meta={"id": "d1", "words": [{"w": "hi", "frame": 2}]})
    g = Frames.load(f.save(tmp_path / "x/d1.safetensors"))
    assert g.meta == f.meta and g.id == "d1"
    assert all(torch.equal(f.tokens[k], g.tokens[k]) for k in f.tokens) and torch.equal(f.latents["agent"], g.latents["agent"])


def test_targets_from_rows_stacks_frames_per_stream():
    from shared.targets import TargetRow

    rows = [TargetRow(id="d", frame=f, stream=s, topk_ids=[f, f + 1], topk_logprobs=[-0.5, -1.0])
            for s in ("agent", "agent_text") for f in (2, 0, 1)]  # out of order
    t = Targets.from_rows(rows)
    assert t.topk_ids["agent"].tolist() == [[0, 1], [1, 2], [2, 3]] and t.topk_logprobs["agent_text"].shape == (3, 2)
    with pytest.raises(ValueError, match="frames are not"):
        Targets.from_rows([rows[0], rows[1]])  # frames 2, 0: gap at 1


def test_loss_weights():
    w = LossWeights(streams={"agent_text": 0.3}, default=0.02)
    assert (w.of("agent_text"), w.of("agent_audio_3")) == (0.3, 0.02)


def test_spec_hash_and_protocol():
    s = BackendSpec(name="x", sample_rate=24000, frame_rate_hz=12.5, context_frames=3000, has_text=True,
                    streams=[StreamSpec(name="agent_text", role="agent", kind="text", vocab=32000)],
                    cache_groups=["temporal"])
    assert len(s.hash) == 12 and s.hash != s.model_copy(update={"context_frames": 1}).hash
    assert isinstance(FakeBackend(FakeConfig()), DuplexBackend)
