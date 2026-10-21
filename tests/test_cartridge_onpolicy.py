import pytest

torch = pytest.importorskip("torch")

from cartridges.cartridge import onpolicy  # noqa: E402
from cartridges.cartridge.cache import Cartridge  # noqa: E402
from cartridges.cartridge.onpolicy import (  # noqa: E402
    OnPolicyConfig,
    chunked_attention,
    onpolicy_val,
    prefill_journal,
    reverse_kl_loss,
    split_batch,
    teacher_hidden,
    train_mixed,
)
from cartridges.cartridge.train_distill import prepare_frozen, train  # noqa: E402
from test_cartridge_train import rand_ids, teacher_examples, tiny_model  # noqa: E402

EOS = [0]


def journal(model, ids: list[int]):
    return prefill_journal(model, ids, chunk=32).cache


def setup(checkpointing: bool = True):
    model = tiny_model()
    context = rand_ids(64, 1)
    cache = journal(model, context)
    cart = Cartridge.from_prefill(model, torch.tensor([rand_ids(16, 2)]))  # starts off wrong
    prepare_frozen(model, gradient_checkpointing=checkpointing)
    return model, context, cache, cart


def brute_reverse_kl(model, cart, context, prompt, answer) -> float:
    with torch.no_grad():
        T = len(answer)
        s = cart(model, torch.tensor([prompt + answer])).logits[0, -T - 1 : -1]
        t = model(input_ids=torch.tensor([context + prompt + answer])).logits[0, -T - 1 : -1]
        s, t = torch.log_softmax(s.float(), -1), torch.log_softmax(t.float(), -1)
        return (s.exp() * (s - t)).sum(-1).mean().item()


def test_split_batch_rounds_to_whole_questions():
    c = OnPolicyConfig(batch_size=16, samples_per_question=2)
    assert split_batch(c, 0.5) == (4, 8)
    assert split_batch(c, 0.25) == (2, 12)
    assert split_batch(c, 0.0) == (0, 16)
    assert split_batch(c, 1.0) == (8, 0)


def test_reverse_kl_is_zero_when_the_cartridge_is_the_journal():
    model = tiny_model()
    context = rand_ids(16, 3)
    cart = Cartridge.from_prefill(model, torch.tensor([context]))
    cache = journal(model, context)
    prompt, answer = rand_ids(5, 4), rand_ids(7, 5)
    th = teacher_hidden(model, cache, prompt, answer)
    loss = reverse_kl_loss(model, cart, torch.tensor([prompt + answer]), th)
    loss.backward()
    assert abs(loss.item()) < 1e-5
    assert max(p.grad.abs().max().item() for p in cart.parameters() if p.grad is not None) < 1e-4


def test_chunked_reverse_kl_matches_brute_force_and_teacher_cache_is_untouched():
    model, context, cache, cart = setup()
    before = [layer.keys.clone() if getattr(layer, "keys", None) is not None else
              layer.recurrent_states[0].clone() for layer in cache.layers]
    prompt, answer = rand_ids(5, 6), rand_ids(11, 7)
    th = teacher_hidden(model, cache, prompt, answer)
    loss = reverse_kl_loss(model, cart, torch.tensor([prompt + answer]), th, chunk=3)
    assert loss.item() == pytest.approx(brute_reverse_kl(model, cart, context, prompt, answer), rel=1e-4)
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in cart.parameters())
    after = [layer.keys if getattr(layer, "keys", None) is not None else layer.recurrent_states[0]
             for layer in cache.layers]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_sampled_ids_reach_the_teacher_unchanged(monkeypatch):
    """Token in, token out: the ids the cartridge sampled are the ids the
    teacher scores. No tokenizer is involved anywhere in train_mixed."""
    model, _, cache, cart = setup()
    sampled, scored = [], []
    real_sample, real_teacher = onpolicy.sample, onpolicy.teacher_hidden

    def spy_sample(*args, **kwargs):
        out = real_sample(*args, **kwargs)
        sampled.extend(out)
        return out

    def spy_teacher(model, cache, prompt, completion):
        scored.append(list(completion))
        return real_teacher(model, cache, prompt, completion)

    monkeypatch.setattr(onpolicy, "sample", spy_sample)
    monkeypatch.setattr(onpolicy, "teacher_hidden", spy_teacher)
    c = OnPolicyConfig(steps=1, batch_size=4, lam=1.0, lam_start=1.0, samples_per_question=2, max_new_tokens=6)
    train_mixed(model, cart, [], [], [rand_ids(5, 8), rand_ids(5, 9)], cache, EOS, c)
    assert scored == sampled and len(scored) == 4


def test_lambda_zero_is_stage_one_training():
    model, context, cache, cart = setup()
    examples = teacher_examples(model, context, 8)
    twin = Cartridge.from_prefill(model, torch.tensor([rand_ids(16, 2)]))
    order = list(range(8)) * 2
    c = OnPolicyConfig(steps=2, batch_size=4, lam=0.0, lam_start=0.0, lr=1e-2, warmup_steps=0)
    train_mixed(model, cart, examples, order, [], cache, EOS, c)
    train(model, twin, examples, lr=1e-2, batch_size=4, max_steps=2, relative_lr=True, order=order)
    for a, b in zip(cart.parameters(), twin.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_on_policy_training_lowers_the_on_policy_kl():
    model, context, cache, cart = setup()
    questions = [rand_ids(5, 20 + i) for i in range(4)]
    eos = [96]  # rarely sampled: answers run to max_new_tokens
    start = onpolicy_val(model, cart, cache, questions, 8, eos)["val/onpolicy_kl"]
    c = OnPolicyConfig(steps=15, batch_size=4, lam=1.0, lam_start=1.0, samples_per_question=2, max_new_tokens=8,
                       lr=5e-2, warmup_steps=2)
    history = []
    train_mixed(model, cart, [], [], questions, cache, eos, c, log=lambda m, s: history.append(m))
    end = onpolicy_val(model, cart, cache, questions, 8, eos)["val/onpolicy_kl"]
    assert end < 0.7 * start, (start, end)
    assert history[-1]["train/onpolicy_kl"] < history[0]["train/onpolicy_kl"]


def test_chunked_attention_matches_eager_with_a_cache_prefix():
    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

    torch.manual_seed(0)
    module = type("M", (), {"num_key_value_groups": 2, "training": False})()
    q = torch.randn(1, 4, 7, 16)
    k, v = torch.randn(1, 2, 19, 16), torch.randn(1, 2, 19, 16)  # 12 cached keys, then the 7 new
    mask = torch.zeros(1, 1, 7, 19)
    mask[..., :, 12:] = torch.triu(torch.full((7, 7), float("-inf")), diagonal=1)
    want, _ = eager_attention_forward(module, q, k, v, mask, scaling=0.25)
    got, _ = chunked_attention(module, q, k, v, None, scaling=0.25, chunk=3)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
