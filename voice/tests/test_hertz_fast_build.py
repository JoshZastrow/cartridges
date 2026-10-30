"""patches/hertz_fast_build.py on toy modules shaped like upstream's: weights
loaded inside `__init__` (strict and strict=False), a child that loads its own
checkpoint after the parent's load, weight norm, buffers built at init. The fast
build must equal the ordinary one, and a weight no checkpoint has must raise."""

import pytest
import torch
from torch import nn
from voice.patches import hertz_fast_build as patch


class Child(nn.Module):
    """Like upstream's tokenizer: weight norm, a buffer, its own strict=False load."""

    def __init__(self, ckpt):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(2, 3, 3))
        self.register_buffer("levels", torch.tensor([8, 8, 5]), persistent=False)
        self.load_state_dict(ckpt, strict=False)


class Parent(nn.Module):
    """Like upstream's HertzDevModel: loads its own weights strict=False, then builds a child."""

    def __init__(self, ckpts):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.freqs = 1.0 / (10_000 ** (torch.arange(0, 4, 2).float() / 4))  # plain tensor, like RoPE
        self.load_state_dict(ckpts["parent"], strict=False)
        self.child = Child(ckpts["child"])

    def forward(self, x):
        return self.norm(self.lin(x)).sum() + self.child.conv(x[None, :2]).sum() + self.freqs.sum()


@pytest.fixture
def ckpts():
    torch.manual_seed(1)
    ref = Parent({"parent": {}, "child": {}})
    sd = {k: v.clone() for k, v in ref.state_dict().items()}
    child = {k.removeprefix("child."): v for k, v in sd.items() if k.startswith("child.")}
    parent = {k: v for k, v in sd.items() if not k.startswith("child.")}
    return {"parent": parent, "child": child}


def fresh(ckpts):
    return {k: {n: t.clone() for n, t in v.items()} for k, v in ckpts.items()}


def test_fast_build_equals_the_ordinary_build(ckpts):
    torch.manual_seed(2)
    slow = Parent(fresh(ckpts))
    fast, loads = patch.build(lambda: Parent(fresh(ckpts)))
    assert slow.state_dict().keys() == fast.state_dict().keys()
    for k, v in slow.state_dict().items():
        assert torch.equal(v, fast.state_dict()[k]), k
    assert torch.equal(fast.child.levels, torch.tensor([8, 8, 5]))
    x = torch.randn(3, 4)
    assert torch.equal(slow(x), fast(x))
    assert [r["module"] for r in loads] == ["Parent", "Child"]
    assert loads[0]["missing"] == [] and loads[1]["missing"] == []


def test_parameters_are_the_checkpoint_tensors_not_copies(ckpts):
    c = fresh(ckpts)
    fast, _ = patch.build(lambda: Parent(c))
    assert fast.lin.weight.data_ptr() == c["parent"]["lin.weight"].data_ptr()
    assert isinstance(fast.lin.weight, nn.Parameter)


def test_a_weight_in_no_checkpoint_raises(ckpts):
    c = fresh(ckpts)
    del c["parent"]["lin.bias"]
    with pytest.raises(patch.MissingWeights, match="lin.bias"):
        patch.build(lambda: Parent(c))


def test_loads_report_missing_and_unexpected_keys(ckpts):
    c = fresh(ckpts)
    c["parent"]["extra"] = torch.zeros(1)
    del c["parent"]["norm.weight"]
    with pytest.raises(patch.MissingWeights):
        with patch.empty_parameters() as loads:
            m = Parent(c)
        patch.build(lambda: m)
    assert loads[0]["missing"] == ["norm.weight"] and loads[0]["unexpected"] == ["extra"]


def test_module_methods_are_restored_even_on_error(ckpts):
    register, load = nn.Module.register_parameter, nn.Module.load_state_dict
    with pytest.raises(ZeroDivisionError):
        patch.build(lambda: 1 / 0)
    assert nn.Module.register_parameter is register and nn.Module.load_state_dict is load
    assert not nn.Linear(2, 2).weight.is_meta
