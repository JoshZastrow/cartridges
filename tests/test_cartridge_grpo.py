import math
import threading

import pytest

torch = pytest.importorskip("torch")

from cartridges.cartridge.cache import Cartridge  # noqa: E402
from cartridges.cartridge.grpo import (  # noqa: E402
    GRPOConfig,
    Group,
    RolloutBuffer,
    group_advantages,
    grpo_loss,
    sequence_logprobs,
    train_grpo,
    trim_eos,
)
from test_cartridge_cache import P, CharTokenizer, ids, tiny_model  # noqa: E402


def test_group_advantages_are_standardized():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert adv == pytest.approx([1.0, -1.0, -1.0, 1.0], abs=1e-3)
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_loss_on_policy_is_minus_mean_advantage_and_gradient_is_policy_gradient():
    logp = torch.tensor([[-1.0, -2.0], [-0.5, 0.0]], requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]])
    adv = torch.tensor([2.0, -1.0])
    loss, stats = grpo_loss(logp, logp.detach(), adv, mask, 0.2, 0.28)
    # ratio = 1: loss = -sum(A * mask) / tokens = -(2 + 2 - 1) / 3
    assert loss.item() == pytest.approx(-1.0)
    assert stats == {"ratio_mean": pytest.approx(1.0), "kl": pytest.approx(0.0), "clip_frac": 0.0}
    loss.backward()
    # d/dlogp of -A * exp(logp - old) at logp == old is -A / n_tokens on unmasked tokens
    torch.testing.assert_close(logp.grad, torch.tensor([[-2 / 3, -2 / 3], [1 / 3, 0.0]]))


@pytest.mark.parametrize(
    "delta, adv, clipped",
    [
        (math.log(1.5), 1.0, True),  # ratio above 1 + clip_high with A > 0: clipped, no gradient
        (math.log(1.1), 1.0, False),  # inside the trust region
        (math.log(0.5), -1.0, True),  # ratio below 1 - clip_low with A < 0: clipped
        (math.log(0.5), 1.0, False),  # below the region but A > 0: min keeps the unclipped term
    ],
)
def test_importance_ratio_clipping(delta, adv, clipped):
    old = torch.tensor([[-1.0]])
    logp = (old + delta).clone().requires_grad_(True)
    loss, stats = grpo_loss(logp, old, torch.tensor([adv]), torch.ones(1, 1), 0.2, 0.28)
    loss.backward()
    ratio = math.exp(delta)
    if clipped:
        assert stats["clip_frac"] == 1.0
        assert logp.grad.item() == 0.0
        assert loss.item() == pytest.approx(-adv * min(max(ratio, 0.8), 1.28))
    else:
        assert stats["clip_frac"] == 0.0
        assert logp.grad.item() == pytest.approx(-adv * ratio)


def test_n_tokens_normalizes_across_groups():
    logp = torch.zeros(1, 4)
    loss, _ = grpo_loss(logp, logp, torch.tensor([1.0]), torch.ones(1, 4), 0.2, 0.2, n_tokens=8)
    assert loss.item() == pytest.approx(-0.5)


def test_buffer_drops_stale_groups():
    buf = RolloutBuffer(maxsize=8, max_staleness=2)
    stop = threading.Event()
    for v in (0, 1, 3):
        buf.put(Group(torch.zeros(1), [], torch.zeros(0), [], [], v), stop)
    assert buf.get(version=3).version == 1  # version 0 lags by 3 > 2
    assert buf.dropped == 1
    assert buf.get(version=3).version == 3


def test_trim_eos_keeps_first_eos():
    assert trim_eos(torch.tensor([4, 5, 2, 7, 2]), [2]).tolist() == [4, 5, 2]
    assert trim_eos(torch.tensor([4, 5]), [2]).tolist() == [4, 5]


def test_sequence_logprobs_match_unpadded_and_ratio_is_one_on_policy():
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    prompt = ids(5, seed=9)[0]
    comps = [ids(4, seed=10)[0], ids(2, seed=11)[0]]
    with torch.no_grad():
        logp, mask = sequence_logprobs(cart, model, prompt, comps, temperature=0.7)
        alone, _ = sequence_logprobs(cart, model, prompt, comps[1:], temperature=0.7)
        full = cart(model, torch.cat([prompt, comps[0]])[None]).logits[0, 4:-1] / 0.7
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    torch.testing.assert_close(logp[1, :2], alone[0], rtol=1e-4, atol=1e-5)
    ref = torch.log_softmax(full, -1).gather(-1, comps[0][:, None]).squeeze(-1)
    torch.testing.assert_close(logp[0], ref, rtol=1e-4, atol=1e-5)
    # Same tensors in the "trainer" and the "worker": the importance ratio is exactly 1.
    _, stats = grpo_loss(logp, logp, torch.tensor([1.0, -1.0]), mask, 0.2, 0.28)
    assert stats["ratio_mean"] == pytest.approx(1.0)


def test_async_grpo_end_to_end_raises_reward():
    # Reward: fraction of generated tokens that are even. The cartridge alone must learn it.
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    before = {k: v.clone() for k, v in cart.state_dict().items()}

    def reward(text, item):
        toks = text.split()
        return sum(int(t) % 2 == 0 for t in toks) / max(len(toks), 1)

    items = [{"question": q} for q in ("a", "bb", "ccc", "dddd")]
    config = GRPOConfig(group_size=8, groups_per_step=2, max_new_tokens=6, lr=0.5, max_steps=40, max_staleness=2)
    logged = []
    summary = train_grpo(model, CharTokenizer(), cart, items, reward, config, log=lambda m, s: logged.append(m))
    assert summary["steps"] == 40
    assert all(m["grpo/staleness_max"] <= 2 for m in logged)
    assert all(m["grpo/kl"] >= 0 and m["throughput/step_tok_s"] > 0 for m in logged)
    assert not torch.equal(before["trainable_keys.0"], cart.state_dict()["trainable_keys.0"])
    torch.testing.assert_close(before["frozen_keys"], cart.state_dict()["frozen_keys"])
    first = sum(m["grpo/reward_mean"] for m in logged[:5]) / 5
    last = sum(m["grpo/reward_mean"] for m in logged[-5:]) / 5
    assert last > first + 0.1, (first, last)
