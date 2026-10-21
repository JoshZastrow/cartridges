import pytest

torch = pytest.importorskip("torch")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from cartridges.cartridge.cache import Cartridge, merge_caches  # noqa: E402


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        initializer_range=0.2,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


def ids(n: int, seed: int = 1, batch: int = 1):
    return torch.randint(0, 97, (batch, n), generator=torch.Generator().manual_seed(seed))


class CharTokenizer:
    """A chat template plus character-level ids: enough for teacher.render."""

    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 90 for c in text]

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [" ".join(str(t) for t in s.tolist()) for s in seqs]


# 70 corpus tokens: the gated-delta chunk kernel runs in chunks of 64, so the
# prefill crosses a chunk boundary.
P = 70


def test_init_reproduces_prefill_logits():
    model = tiny_model()
    x, y = ids(P), ids(9, seed=2)
    cart = Cartridge.from_prefill(model, x)
    assert cart.p == P
    with torch.no_grad():
        full = model(input_ids=torch.cat([x, y], dim=1)).logits[:, P:]
        ours = cart(model, y).logits
        alone = model(input_ids=y).logits
    torch.testing.assert_close(ours, full, rtol=1e-4, atol=1e-4)
    assert (alone - full).abs().max() > 0.1  # the prefix matters, so the check is meaningful


def test_init_matches_teacher_truncated_to_p():
    # Layout invariant: teacher = context_block ids + prompt_ids; student = cartridge
    # (first p ids of that block) + the same prompt_ids at positions p...
    from cartridges.cartridge.teacher import render

    model, tok = tiny_model(), CharTokenizer()
    context = "2021-11-20 [Journal:p.1] walked to the lake. " * 10
    messages = [{"role": "user", "content": "Where did I walk?"}]
    prefix_ids, prompt_ids = render(tok, messages, context)
    assert len(prefix_ids) > P
    cart = Cartridge.init_from_text(model, tok, context, P)
    with torch.no_grad():
        teacher = model(input_ids=torch.tensor([prefix_ids[:P] + prompt_ids])).logits[:, P:]
        student = cart(model, torch.tensor([prompt_ids])).logits
    # fp32 noise from the chunked gated-delta scan over a longer prompt; a wrong state is off by > 0.1.
    torch.testing.assert_close(student, teacher, rtol=1e-3, atol=5e-4)


@pytest.mark.parametrize("group", ["trainable_keys", "conv_states", "recurrent_states"])
def test_every_state_matters(group):
    model = tiny_model()
    x, y = ids(P), ids(9, seed=2)
    cart = Cartridge.from_prefill(model, x)
    with torch.no_grad():
        before = cart(model, y).logits
        for t in getattr(cart, group):
            t.zero_()
        assert (cart(model, y).logits - before).abs().max() > 0.1


def test_generate_matches_greedy_prefill():
    model = tiny_model()
    x, y = ids(P), ids(5, seed=3)
    cart = Cartridge.from_prefill(model, x)
    new = cart.generate(model, y, max_new_tokens=6)
    seq = torch.cat([x, y], dim=1)
    with torch.no_grad():
        for _ in range(6):
            seq = torch.cat([seq, model(input_ids=seq).logits[:, -1:].argmax(-1)], dim=1)
    assert new.tolist() == seq[:, P + 5 :].tolist()


def test_grads_only_on_cartridge():
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    loss = cart(model, ids(8, seed=4, batch=2)).logits.logsumexp(-1).mean()
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    for group in (cart.trainable_keys, cart.trainable_values, cart.conv_states, cart.recurrent_states):
        for p in group:
            assert p.grad is not None and p.grad.abs().sum() > 0
    assert not cart.frozen_keys.requires_grad
    trainable = {n for n, p in cart.named_parameters() if p.requires_grad}
    assert not any(n.startswith("frozen") for n in trainable)


def test_cartridge_steers_the_model():
    # Training the cartridge alone must be able to lower a loss (gradients are real).
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    y = ids(6, seed=5)
    opt = torch.optim.Adam(cart.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        logits = cart(model, y[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.1


def test_right_padding_matches_unpadded():
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    a, b = ids(7, seed=6), ids(4, seed=7)
    batch = torch.cat([a, torch.cat([b, torch.zeros(1, 3, dtype=torch.long)], dim=1)])
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    with torch.no_grad():
        padded = cart(model, batch, attention_mask=mask).logits
        torch.testing.assert_close(padded[0], cart(model, a).logits[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(padded[1, :4], cart(model, b).logits[0], rtol=1e-4, atol=1e-4)


def test_save_load_round_trip(tmp_path):
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P), meta={"run_id": "abc"})
    with torch.no_grad():
        cart.trainable_keys[0].add_(0.5)
        cart.recurrent_states[1].mul_(2)
    cart.save(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"cartridge.safetensors", "cartridge.json"}
    back = Cartridge.load(tmp_path)
    assert back.config_dict() == cart.config_dict()
    assert back.meta["run_id"] == "abc"
    for (n1, t1), (n2, t2) in zip(cart.state_dict().items(), back.state_dict().items()):
        assert n1 == n2 and torch.equal(t1, t2)
    y = ids(5, seed=8)
    with torch.no_grad():
        torch.testing.assert_close(back(model, y).logits, cart(model, y).logits)


def test_gradient_checkpointing_gives_same_grads():
    # The read-only cache lets checkpointed layers replay the forward with the cartridge.
    grads = []
    for checkpointing in (False, True):
        model = tiny_model()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for layer in model.model.layers:
                layer._can_checkpoint_with_cache = True
            model.train()
        cart = Cartridge.from_prefill(model, ids(P))
        cart(model, ids(8, seed=4, batch=2)).logits.logsumexp(-1).mean().backward()
        grads.append([p.grad.clone() for p in cart.parameters()])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def _trim(row: list[int], eos: int) -> list[int]:
    return row[: row.index(eos) + 1] if eos in row else row


def test_generate_batch_matches_per_prompt_generate():
    """Prompts of different lengths decoded as one batch give the same greedy
    tokens as decoding each prompt alone (padding stays out of the result)."""
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    prompts = [ids(5, seed=10)[0].tolist(), ids(9, seed=11)[0].tolist(), ids(3, seed=12)[0].tolist()]
    batched = cart.generate_batch(model, prompts, n=2, max_new_tokens=8)
    assert batched.shape == (6, 8)
    for i, prompt in enumerate(prompts):
        alone = cart.generate(model, torch.tensor([prompt]), max_new_tokens=8)[0].tolist()
        assert batched[2 * i].tolist() == alone and batched[2 * i + 1].tolist() == alone


def test_generate_batch_stops_rows_independently():
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    prompts = [ids(5, seed=10)[0].tolist(), ids(9, seed=11)[0].tolist()]
    free = cart.generate(model, torch.tensor([prompts[0]]), max_new_tokens=8)[0].tolist()
    eos = free[2]  # prompt 0 stops at its third token; prompt 1 keeps going unless it emits it too
    batched = cart.generate_batch(model, prompts, n=1, max_new_tokens=8, eos_token_id=eos)
    for i, prompt in enumerate(prompts):
        alone = cart.generate(model, torch.tensor([prompt]), max_new_tokens=8, eos_token_id=eos)[0].tolist()
        assert _trim(batched[i].tolist(), eos) == _trim(alone, eos)
    assert _trim(batched[0].tolist(), eos) == free[:3]


def test_merged_cache_decode_step_matches_each_prompt_alone():
    """One decode step through the merged, padded cache gives each row the
    logits of running its own prompt + token after the cartridge, unbatched."""
    model = tiny_model()
    cart = Cartridge.from_prefill(model, ids(P))
    prompts = [ids(5, seed=10)[0].tolist(), ids(9, seed=11)[0].tolist(), ids(3, seed=12)[0].tolist()]
    nxt = torch.tensor([7, 8, 9])
    with torch.no_grad():
        caches = [cart(model, torch.tensor([q]), read_only=False).past_key_values for q in prompts]
        cache, prompt_mask = merge_caches(caches, [len(q) for q in prompts])
        mask = torch.cat([prompt_mask.new_ones(3, cart.p), prompt_mask, prompt_mask.new_ones(3, 1)], dim=1)
        pos = torch.tensor([cart.p + len(q) for q in prompts])[:, None]
        got = model(input_ids=nxt[:, None], position_ids=pos, attention_mask=mask, past_key_values=cache,
                    use_cache=True).logits[:, -1]
        for i, q in enumerate(prompts):
            want = cart(model, torch.tensor([q + [nxt[i].item()]])).logits[0, -1]
            torch.testing.assert_close(got[i], want, rtol=1e-4, atol=1e-4)
