"""shared.optim: relative-LR Adam with warmup, EarlyStopping."""

import pytest
import torch
from shared.optim import EarlyStopping, make_optimizer


def test_make_optimizer_skips_frozen_and_scales_lr_by_rms():
    big, small, frozen = torch.full((4,), 2.0, requires_grad=True), torch.full((4,), 0.01, requires_grad=True), torch.ones(4)
    params, opt, warmup = make_optimizer([big, small, frozen], lr=0.1, relative_lr=True, warmup_steps=4)
    assert params == [big, small]
    assert [g["initial_lr"] for g in opt.param_groups] == pytest.approx([0.2, 0.001])
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2 / 4)  # warmup step 0
    for _ in range(5):
        opt.step()
        warmup.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)


def test_make_optimizer_plain_lr():
    x = torch.zeros(3, requires_grad=True)
    _, opt, _ = make_optimizer([x], lr=0.05)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


def test_early_stopping_saves_on_best_and_stops_after_patience():
    saves = []
    stop = EarlyStopping(patience=2, save=lambda: saves.append(1))
    assert [stop.update(v, s) for s, v in enumerate([3.0, 2.0, 2.5, 1.5, 1.6, 1.7])] == [
        False, False, False, False, False, True]
    assert (stop.best, stop.step, len(saves), stop.saved) == (1.5, 3, 3, True)


def test_early_stopping_patience_zero_never_stops_or_saves():
    stop = EarlyStopping(patience=0, save=lambda: pytest.fail("saved"))
    assert not any(stop.update(v, s) for s, v in enumerate([1.0, 2.0, 3.0]))
