"""train.py: the generic loop end to end against the FakeBackend on CPU."""

import pytest
import torch
from shared.targets import TargetRow, write_shard
from voice.backends.base import Cartridge, Frames, LossWeights, Targets
from voice.backends.fake import FakeBackend, fixture, kl_to_topk
from voice.backends.registry import FakeConfig
from voice.train import TrainConfig, load_targets, train, validate

from voice import cli


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def data(tmp_path):
    paths = fixture(tmp_path / "frames", n=6)
    return paths[:4], paths[4:]


def config(tmp_path, data, **kw) -> TrainConfig:
    train_paths, val_paths = data
    defaults = {"out_dir": str(tmp_path / "out"), "gpu": "CPU", "max_steps": 40, "val_every": 10, "patience": 3}
    return TrainConfig(train_frames=train_paths, val_frames=val_paths, **(defaults | kw))


def test_training_lowers_held_out_loss_and_saves_best(tmp_path, data):
    cfg = config(tmp_path, data)
    backend = FakeBackend(FakeConfig())
    start = validate(backend, backend.cartridge_from_prefill(backend.build_prefix(None, "", "")), data[1], cfg, {})
    result = train(cfg)
    assert result["best_val"] < start["total"] - 0.1
    c = Cartridge.load(result["best_dir"])
    assert c.meta["backend"] == "fake" and c.meta["stage"] == "sft" and c.meta["step"] == result["best_step"]
    assert c.meta["spec_hash"] == backend.spec.hash and c.p == 4


def test_only_cartridge_parameters_change(tmp_path, data):
    backend = FakeBackend(FakeConfig())
    head = backend.head.clone()
    train(config(tmp_path, data, max_steps=5))
    assert torch.equal(FakeBackend(FakeConfig()).head, head)  # the frozen model is seeded, never a parameter


def test_early_stopping_ends_a_flat_run(tmp_path, data):
    result = train(config(tmp_path, data, lr=0.0, patience=2, val_every=1))
    assert (result["steps"], result["best_step"]) == (3, 1)


def test_init_from_saved_cartridge(tmp_path, data):
    first = train(config(tmp_path, data, max_steps=10))
    again = train(config(tmp_path, data, max_steps=1, val_every=1, lr=0.0, init=first["best_dir"],
                         out_dir=str(tmp_path / "out2")))
    assert again["best_val"] == pytest.approx(first["best_val"], abs=1e-5)


def teacher_rows(f: Frames, shift: int = 1) -> list[TargetRow]:
    """A teacher sure of agent[t + shift] at position t (shift=1 is next-frame)."""
    agent = f.tokens["agent"].tolist()
    n = len(agent) - 1
    return [TargetRow(id=f.id, frame=t, stream="agent", topk_ids=[agent[(t + shift) % len(agent)]], topk_logprobs=[0.0])
            for t in range(n)]


def test_distillation_term_uses_stored_targets(tmp_path, data):
    rows = [r for p in data[0] + data[1] for r in teacher_rows(Frames.load(p))]
    shard = write_shard(rows, tmp_path / "targets/run/part-00000.parquet")
    cfg = config(tmp_path, data, targets=[shard], distill_weight=0.5, max_steps=5, val_every=5)
    result = train(cfg)
    backend = FakeBackend(FakeConfig())
    val = validate(backend, Cartridge.load(result["best_dir"]), data[1], cfg, load_targets([shard], None))
    assert val["distill"] > 0 and val["total"] == pytest.approx(val["agent"] + 0.5 * val["distill"], rel=1e-5)


def test_sft_and_distill_losses_share_next_frame_alignment(tmp_path):
    """Position t predicts frame t+1 in both losses: a one-hot teacher on frame t+1
    at row t makes the distill loss equal the SFT loss; off by one either way, it does not."""
    backend = FakeBackend(FakeConfig())
    frames = Frames.load(fixture(tmp_path, n=1)[0])
    with torch.no_grad():
        out = backend.forward(frames, backend.cartridge_from_prefill(backend.build_prefix(None, "", "")))
    sft = float(backend.sft_loss(out, frames, LossWeights())["agent"])
    distill = {shift: float(backend.distill_loss(out, Targets.from_rows(teacher_rows(frames, shift))))
               for shift in (0, 1, 2)}
    assert distill[1] == pytest.approx(sft, rel=1e-6)
    assert distill[0] != pytest.approx(sft, rel=1e-3) and distill[2] != pytest.approx(sft, rel=1e-3)


def test_kl_to_topk_is_zero_when_student_matches():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    ids = torch.tensor([[0, 1]])
    assert float(kl_to_topk(logits, ids, logits.log_softmax(-1).gather(-1, ids))) == pytest.approx(0.0, abs=1e-6)


def test_cli_dry_run_exits_zero(tmp_path):
    (tmp_path / "c.toml").write_text('lr = 0.05\n\n[backend]\nname = "personaplex"\ncheckpoint = "nvidia/personaplex-7b-v1"\n')
    assert cli.main(["train", "--config", str(tmp_path / "c.toml"), "--dry-run"]) == 0
    assert cli.main(["train", "--dry-run"]) == 0


@pytest.mark.parametrize("command", ["encode"])  # eval and report ship with #9 (tests/test_evals.py)
def test_cli_stubs_take_backend(command, capsys):
    assert cli.main([command, "--backend", "hertz"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


@pytest.mark.parametrize("backend,path", [("hertz", "/audio"), ("personaplex", "/api/chat?")])
def test_cli_deploy_takes_backend_dry_run(backend, path, capsys):
    """Every backend is served now; --dry-run exercises the CLI path without reaching Modal."""
    assert cli.main(["deploy", "--backend", backend, "--system", "stock", "--dry-run"]) == 0
    assert path in capsys.readouterr().out
