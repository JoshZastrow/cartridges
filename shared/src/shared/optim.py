"""Optimizer and early stopping for trainable prefixes.

Copied from `cartridges.cartridge.train_distill` (`make_optimizer`,
`EarlyStopping`) at commit a9a7588. Changed: `make_optimizer` takes the
parameters instead of a cartridge, so any module's trainable tensors can use
it; `EarlyStopping` tracks any lower-is-better metric, not only val KL.

    params, optimizer, warmup = make_optimizer(cartridge.parameters(), lr=2e-2, relative_lr=True, warmup_steps=10)
    stopper = EarlyStopping(patience=3, save=lambda: cartridge.save(out))
    ...
    if stopper.update(val_loss, step):
        break
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


def make_optimizer(params: Iterable, lr: float, relative_lr: bool = False, warmup_steps: int = 0):
    """(params, Adam, warmup LambdaLR) over the tensors in `params` that require
    grad. warmup_steps: linear lr warmup from lr/warmup_steps, then constant.

    relative_lr: each tensor's lr is `lr` x its initial RMS. Adam moves every
    entry by about lr per step, which suits a KV prefix with RMS ~1 but is large
    for tensors with a much smaller scale."""
    import torch

    params = [x for x in params if x.requires_grad]
    if relative_lr:
        groups = [{"params": [x], "lr": lr * x.detach().pow(2).mean().sqrt().clamp(min=1e-3).item()} for x in params]
        optimizer = torch.optim.Adam(groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(warmup_steps, 1)))
    return params, optimizer, warmup


class EarlyStopping:
    """Tracks the best value of a lower-is-better validation metric. update()
    calls `save` on a new best (when patience > 0, so there is a checkpoint to
    end with) and returns True once `patience` evals in a row bring no
    improvement."""

    def __init__(self, patience: int, save: Callable[[], object], best: float | None = None, step: int = 0):
        self.patience, self.save = patience, save
        self.best, self.step, self.bad = float("inf") if best is None else best, step, 0
        self.saved = False

    def update(self, value: float, step: int) -> bool:
        if value < self.best:
            self.best, self.step, self.bad = value, step, 0
            if self.patience:
                self.save()
                self.saved = True
        else:
            self.bad += 1
        return bool(self.patience) and self.bad >= self.patience
