"""Generic cartridge trainer (brief section 4b); Modal app "voice-train".

Backend-agnostic: build (or load) the `Cartridge`, Adam over its parameters
only (`shared.optim`), loop over `Frames` shards, and minimise

    loss = sum(backend.sft_loss(out, frames, w).values()) + distill_weight * backend.distill_loss(out, targets)

where the distillation term applies to dialogues that have stored teacher
targets (`shared.targets` shards, row id = dialogue id). Every `val_every`
steps: held-out loss over `val_frames`, an optional smoke eval, early stopping
on the held-out loss (the best cartridge is saved to <out_dir>/best). At the
end the best cartridge is logged as the W&B artifact `voice-cartridge`
(aliases `best`, `step-N`). Stage 2 (on-policy, #15) reuses this loop.

    uv run voice train --config sft.toml             # on Modal, gpu="H200"
    uv run voice train --config sft.toml --dry-run   # CPU, FakeBackend, tiny fixture, W&B disabled
"""

from __future__ import annotations

import itertools
import logging
import os
import random
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import modal
import numpy as np
import torch
from pydantic import BaseModel, Field

from shared import optim, tracking
from voice import backends
from voice.backends.base import Cartridge, Frames, LossWeights, Targets
from voice.backends.registry import BackendConfig, FakeConfig

logger = logging.getLogger(__name__)

APP_NAME = "voice-train"
FORK_COMMIT = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
FORK = f"moshi-personaplex @ git+https://github.com/NVIDIA/personaplex@{FORK_COMMIT}#subdirectory=moshi"
DATA_MOUNT = "/data"


class TrainConfig(BaseModel):
    backend: BackendConfig = Field(default_factory=FakeConfig)
    stage: Literal["sft", "onpolicy", "ablation"] = "sft"
    system: str = "voice-cartridge"  # W&B run name <backend>/<system>/<stage>
    group: str = ""  # exp-YYYYMMDD-<slug>; default group_name(stage)

    train_frames: list[str] = []  # Frames files, or directories of *.safetensors
    val_frames: list[str] = []
    targets: list[str] = []  # shared.targets shard uris (row id = dialogue id)
    targets_root: str | None = None  # where Volume-relative target uris resolve
    distill_weight: float = 0.0  # lambda
    loss_weights: LossWeights = LossWeights()

    init: str | None = None  # cartridge dir to start from; else build_prefix + cartridge_from_prefill
    voice_wav: str | None = None
    persona: str = ""
    digest_path: str | None = None

    lr: float = 2e-2
    relative_lr: bool = True
    warmup_steps: int = 10
    max_steps: int = 1000
    val_every: int = 100
    patience: int = 3
    seed: int = 0
    out_dir: str = "outputs/voice-train"  # on Modal, a relative out_dir lands under /data
    gpu: str | None = None  # Modal gpu= for cost accounting; detected when None


# ---------------------------------------------------------------- data


def frame_paths(entries: list[str]) -> list[str]:
    """Files as given; directories expanded to their *.safetensors, sorted."""
    out = []
    for e in entries:
        p = Path(e)
        out += sorted(str(x) for x in p.glob("*.safetensors")) if p.is_dir() else [e]
    return out


def shuffled_forever(paths: list[str], seed: int) -> Iterator[Frames]:
    """Frames from `paths`, reshuffled every epoch."""
    if not paths:
        raise ValueError("no training frames")
    rng = random.Random(seed)
    for epoch in itertools.count():
        order = list(paths)
        rng.shuffle(order)
        for path in order:
            yield Frames.load(path)


def load_targets(uris: list[str], root: str | None) -> dict[str, Targets]:
    from shared.targets import read_shards

    by_id: dict[str, list] = {}
    for row in read_shards(uris, root=root):
        by_id.setdefault(row.id, []).append(row)
    return {k: Targets.from_rows(rows) for k, rows in by_id.items()}


def initial_cartridge(backend, cfg: TrainConfig) -> Cartridge:
    if cfg.init:
        return Cartridge.load(cfg.init)
    wav = np.zeros(0, dtype=np.float32)
    if cfg.voice_wav:
        import soundfile

        wav = soundfile.read(cfg.voice_wav, dtype="float32")[0]
    digest = Path(cfg.digest_path).read_text() if cfg.digest_path else ""
    return backend.cartridge_from_prefill(backend.build_prefix(wav, cfg.persona, digest))


# ---------------------------------------------------------------- loop


def step_loss(backend, cartridge, frames: Frames, cfg: TrainConfig, targets: dict[str, Targets]) -> dict[str, torch.Tensor]:
    """Per-stream SFT losses, `distill` when the dialogue has targets, and `total`."""
    out = backend.forward(frames, cartridge)
    losses = dict(backend.sft_loss(out, frames, cfg.loss_weights))
    total = sum(losses.values())
    if cfg.distill_weight and frames.id in targets:
        losses["distill"] = backend.distill_loss(out, targets[frames.id])
        total = total + cfg.distill_weight * losses["distill"]
    losses["total"] = total
    return losses


@torch.no_grad()
def validate(backend, cartridge, paths: list[str], cfg: TrainConfig, targets: dict[str, Targets]) -> dict[str, float]:
    """Mean of every loss over the held-out frames."""
    sums: dict[str, float] = {}
    for path in paths:
        for k, v in step_loss(backend, cartridge, Frames.load(path), cfg, targets).items():
            sums[k] = sums.get(k, 0.0) + float(v)
    return {k: v / len(paths) for k, v in sums.items()}


def train(cfg: TrainConfig, smoke_eval: Callable[[object, Cartridge], dict] | None = None) -> dict:
    """Run the loop; returns {best_val, best_step, steps, best_dir, artifact}.
    `smoke_eval(backend, cartridge) -> metrics` runs at every validation step
    (the shared evals plug in here; #9)."""
    torch.manual_seed(cfg.seed)
    backend = backends.get(cfg.backend.name, cfg.backend)
    backend.load(cfg.backend.device)
    cartridge = initial_cartridge(backend, cfg).to(cfg.backend.device)
    cartridge.meta |= {"backend": cfg.backend.name, "spec_hash": backend.spec.hash, "stage": cfg.stage}
    if cfg.backend.name == "personaplex":
        cartridge.meta.setdefault("fork_commit", FORK_COMMIT)
    _, optimizer, warmup = optim.make_optimizer(cartridge.parameters(), cfg.lr, cfg.relative_lr, cfg.warmup_steps)

    train_paths, val_paths = frame_paths(cfg.train_frames), frame_paths(cfg.val_frames)
    targets = load_targets(cfg.targets, cfg.targets_root) if cfg.targets else {}
    out_dir = Path(cfg.out_dir)
    best_dir = out_dir / "best"
    state = {"step": 0}

    def save_best():
        cartridge.meta["step"] = state["step"]
        cartridge.save(best_dir)

    stopper = optim.EarlyStopping(cfg.patience, save=save_best)
    run = tracking.start_run(
        "voice", cfg.group or tracking.group_name(cfg.stage), cfg.model_dump(mode="json"),
        name=f"{cfg.backend.name}/{cfg.system}/{cfg.stage}", job_type="train", gpu=cfg.gpu,
        tags=["candidate", cfg.backend.name],
    )
    with run:
        frames_iter = shuffled_forever(train_paths, cfg.seed)
        for step in range(1, cfg.max_steps + 1):
            state["step"] = step
            losses = step_loss(backend, cartridge, next(frames_iter), cfg, targets)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            warmup.step()
            run.log({f"train/loss/{k}" if k != "total" else "train/loss": v.item() for k, v in losses.items()}
                    | {"train/lr": warmup.get_last_lr()[0]}, step=step)
            if val_paths and (step % cfg.val_every == 0 or step == cfg.max_steps):
                val = validate(backend, cartridge, val_paths, cfg, targets)
                metrics = {f"val/loss/{k}" if k != "total" else "val/loss": v for k, v in val.items()}
                if smoke_eval is not None:
                    metrics |= smoke_eval(backend, cartridge)
                run.log(metrics, step=step)
                logger.info("step %d val %.4f", step, val["total"])
                if stopper.update(val["total"], step):
                    break
        if not stopper.saved:  # no validation or patience 0: the last cartridge is the result
            save_best()
        artifact = tracking.log_artifact(run, best_dir, type="voice-cartridge",
                                         metadata=cartridge.config_dict(),
                                         aliases=["best", f"step-{stopper.step or state['step']}"])
        run.summary.update({"best/val_loss": stopper.best, "best/step": stopper.step})
    return {"best_val": stopper.best, "best_step": stopper.step, "steps": state["step"],
            "best_dir": str(best_dir), "artifact": artifact}


# ---------------------------------------------------------------- dry run


def dry_run(cfg: TrainConfig, steps: int = 20) -> dict:
    """The same loop on CPU: FakeBackend, a tiny fixture, W&B disabled. The
    config's optimiser and loop settings are kept."""
    from voice.backends.fake import fixture

    os.environ["WANDB_MODE"] = "disabled"
    with tempfile.TemporaryDirectory() as tmp:
        paths = fixture(tmp, n=6)
        cfg = cfg.model_copy(update={
            "backend": FakeConfig(), "train_frames": paths[:4], "val_frames": paths[4:], "targets": [],
            "init": None, "voice_wav": None, "digest_path": None, "max_steps": steps,
            "val_every": max(1, steps // 4), "out_dir": f"{tmp}/out", "gpu": "CPU",
        })
        result = train(cfg)
    logger.info("dry run ok: %s", result)
    return result


# ---------------------------------------------------------------- Modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "libsndfile1", "libopus0")
    .uv_pip_install(
        "torch==2.14.1",
        "numpy==2.3.5",
        "safetensors==0.9.0rc0",
        "pydantic==2.14.0b2",
        "wandb==0.30.0",
        "pyarrow==25.0.1",
        "soundfile==0.14.0",
        "sentencepiece==0.2.2",
        # the fork's own deps at the versions of the root lock; its pins (torch<2.5,
        # numpy<2.2, ...) are overridden there too ([tool.uv] override-dependencies)
        "aiohttp==3.14.3",
        "einops==0.9.0.dev0",
        "huggingface-hub==1.32.0",
        "sounddevice==0.5.0",
        "sphn==0.1.12",
        pre=True,
    )
    .uv_pip_install(FORK, extra_options="--no-deps")
    .add_local_python_source("voice", "shared")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name("voice-data", create_if_missing=True)


@app.function(
    gpu="H200", timeout=6 * 3600, volumes={DATA_MOUNT: data_volume},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("HF_TOKEN")],
)
def train_remote(config: dict, git: dict) -> dict:
    os.environ.update(git)
    cfg = TrainConfig.model_validate(config)
    if not cfg.out_dir.startswith("/"):  # keep the best dir on the Volume, not the container disk
        cfg = cfg.model_copy(update={"out_dir": f"{DATA_MOUNT}/{cfg.out_dir}"})
    result = train(cfg)
    data_volume.commit()
    return result


def launch(cfg: TrainConfig) -> dict:
    """Run `train` on Modal (gpu="H200"); a relative out_dir is placed under /data."""
    with modal.enable_output(), app.run():
        return train_remote.remote(cfg.model_dump(mode="json"), tracking.modal_env())
