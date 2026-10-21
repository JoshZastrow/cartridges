"""Stage 2b: mixed on-policy context distillation (docs/plans/onpolicy-distill.md).

Each optimizer step mixes two kinds of sequences:

- off-policy (1 - lam of the batch): stored teacher answers, stage 1's loss
  (train_distill.student_loss, KL(teacher top-k || student));
- on-policy (lam of the batch): the cartridge samples answers to training
  questions (no grad), the teacher reads those exact token ids with the
  journal in context (no grad), and the loss is the exact reverse KL
  KL(student || teacher) over the full vocabulary at every answer position.

Teacher and student are the same frozen model: the teacher is the model on a
fork of the prefilled journal cache (prefill_journal), the student the model on
the cartridge. Passes over the ~205k-token journal use chunked_attention. Sampled ids go from Cartridge.generate straight into both
forwards: nothing is decoded and re-tokenized (token in, token out).

The loop is synchronous: sample with the current cartridge, score, one
backward over the batch, one Adam step, then sample with the new cartridge.
Every on-policy sample comes from the policy being updated, so there is no
importance ratio or clipping (unlike grpo.py).
"""

from __future__ import annotations

import contextlib
import math
import random
import time
from typing import Callable

import torch
from pydantic import BaseModel, ConfigDict


class OnPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 30
    batch_size: int = 16  # sequences per optimizer step, both kinds
    lam: float = 0.5  # on-policy fraction of the batch
    lam_start: float = 0.25  # used for the first lam_warmup_steps steps
    lam_warmup_steps: int = 10
    samples_per_question: int = 2
    max_gen_rows: int = 16  # rows per batched generate call (Cartridge.generate_batch), train and val
    max_new_tokens: int = 768
    temperature: float = 1.0  # on-policy: sample the cartridge's own distribution
    lr: float = 5e-2  # relative (train_distill.train), the stage-1 sweep's best
    relative_lr: bool = True
    warmup_steps: int = 5  # Adam moments start fresh
    val_every: int = 5
    val_n: int = 16  # eval-split questions for the on-policy val KL (greedy answers)
    val_max_new_tokens: int = 512
    patience: int = 3  # on-policy val evals without a new best; 0: never stop
    seed: int = 0


def lam_at(c: OnPolicyConfig, step: int) -> float:
    return c.lam_start if step <= c.lam_warmup_steps else c.lam


def split_batch(c: OnPolicyConfig, lam: float) -> tuple[int, int]:
    """(on-policy questions, off-policy sequences) for a step: the on-policy
    share rounded to whole questions of samples_per_question rows."""
    n_q = round(lam * c.batch_size / c.samples_per_question)
    return n_q, c.batch_size - n_q * c.samples_per_question


@contextlib.contextmanager
def eval_mode(model):
    """Generation and teacher passes run without gradient checkpointing
    (HF turns the cache off when checkpointing in train mode)."""
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


# Registered attention name for the teacher passes. It must not contain
# "flash": transformers then looks for a flash-attn kernel (teacher.py).
LONG_ATTENTION = "chunked_causal"


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, chunk: int = 256, **kwargs):
    """Exact causal attention for batch 1 with no padding, `chunk` queries at a
    time, the new queries aligned to the end of the keys (the journal cache
    first). Grouped-query heads read the shared K/V without repeating them.
    Peak memory is chunk x heads x keys fp32 scores (~5 GB for 27B at 205k
    keys), where SDPA's math fallback at head_dim 256 materializes all queries
    at once (teacher.py: OOM). Inference only; the mask argument is ignored."""
    b, h, lq, d = query.shape
    hkv, lk = key.shape[1], key.shape[2]
    q = query.reshape(b, hkv, h // hkv, lq, d)
    k, v = key[:, :, None], value[:, :, None]
    scale = scaling if scaling is not None else d**-0.5
    cols = torch.arange(lk, device=query.device)
    out = torch.empty_like(q)
    for a in range(0, lq, chunk):
        qc = q[:, :, :, a : a + chunk]
        scores = torch.matmul(qc, k.transpose(-1, -2)).float() * scale
        rows = torch.arange(a, a + qc.shape[3], device=query.device) + (lk - lq)
        scores.masked_fill_(cols[None, :] > rows[:, None], float("-inf"))
        out[:, :, :, a : a + chunk] = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)
    return out.reshape(b, h, lq, d).transpose(1, 2), None  # [b, lq, h, d]


@contextlib.contextmanager
def long_context(model):
    """Eval mode and chunked_attention for passes over the journal cache; the
    student's own passes keep the model's attention implementation."""
    from transformers import AttentionInterface

    AttentionInterface.register(LONG_ATTENTION, chunked_attention)
    previous = model.config._attn_implementation
    model.set_attn_implementation(LONG_ATTENTION)
    try:
        with eval_mode(model):
            yield
    finally:
        model.set_attn_implementation(previous)


def prefill_journal(model, prefix_ids: list[int], chunk: int = 1024):
    """teacher.Scorer with the journal's context block prefilled, `chunk`
    tokens per forward, under long_context."""
    from cartridges.cartridge.teacher import Scorer

    scorer = Scorer(model, prefill_chunk=chunk)
    with long_context(model):
        scorer.prefill(prefix_ids)
    return scorer


@torch.no_grad()
def sample(model, cart, prompts: list[list[int]], n: int, max_new_tokens: int, temperature: float,
           eos_ids: list[int], generator: torch.Generator | None = None, max_rows: int = 16) -> list[list[int]]:
    """n answers to each prompt, prompt-major (prompt i's answers at i*n .. i*n+n-1),
    as token ids through the first EOS (kept: stopping is part of what the
    policy chose). Never decoded. Prompts are decoded together, up to
    `max_rows` rows per Cartridge.generate_batch call."""
    from cartridges.cartridge.grpo import trim_eos

    per_call = max(1, max_rows // n)
    out = []
    with eval_mode(model):
        for a in range(0, len(prompts), per_call):
            new = cart.generate_batch(model, prompts[a : a + per_call], n=n, max_new_tokens=max_new_tokens,
                                      temperature=temperature, eos_token_id=eos_ids, generator=generator)
            out += [trim_eos(row, eos_ids).tolist() for row in new]
    return [ids if ids else eos_ids[:1] for ids in out]


def teacher_hidden(model, journal_cache, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """The teacher's final hidden states [T, d] at the positions that predict
    each completion token, with the journal in context: one forward over
    prompt + completion on a fork of the prefilled journal cache (the cache
    itself is not changed). Hidden states, not logits: [T, d] instead of [T, V].
    Runs in inference mode like teacher.Scorer (its cache is made there); the
    clone outside it is a normal tensor the student's loss can use."""
    from cartridges.cartridge.teacher import fork_cache

    T = len(completion_ids)
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
    with torch.inference_mode(), long_context(model):
        hidden = model.model(input_ids=ids, past_key_values=fork_cache(journal_cache), use_cache=True).last_hidden_state
        hidden = hidden[0, -T - 1 : -1]
    return hidden.clone()


def reverse_kl_loss(model, cart, input_ids: torch.Tensor, teacher_h: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Mean over the last T = len(teacher_h) positions of the exact
    KL(student || teacher) over the vocabulary. The student reads the
    cartridge; lm_head and both log_softmaxes run `chunk` positions at a time,
    checkpointed, so no [T, V] tensor is kept for backward."""
    from torch.utils.checkpoint import checkpoint

    T, L = teacher_h.shape[0], input_ids.shape[1]
    position_ids = torch.arange(cart.p, cart.p + L, device=input_ids.device)[None]
    hidden = model.model(
        input_ids=input_ids, past_key_values=cart.as_past_key_values(1), position_ids=position_ids, use_cache=False,
    ).last_hidden_state[0, L - T - 1 : L - 1]

    def kl_sum(h, th):
        s = torch.log_softmax(model.lm_head(h).float(), dim=-1)
        t = torch.log_softmax(model.lm_head(th).float(), dim=-1)
        return (s.exp() * (s - t)).sum()

    total = 0.0
    for a in range(0, T, chunk):
        args = (hidden[a : a + chunk], teacher_h[a : a + chunk])
        total = total + (checkpoint(kl_sum, *args, use_reentrant=False) if torch.is_grad_enabled() else kl_sum(*args))
    return total / T


def onpolicy_val(model, cart, journal_cache, prompts: list[list[int]], max_new_tokens: int,
                 eos_ids: list[int], max_rows: int = 16) -> dict:
    """The cartridge's greedy answers to `prompts`, graded by the teacher:
    mean per-token reverse KL, answer length, and the share that hit EOS."""
    kls, lengths, stopped = [], [], 0
    answers = sample(model, cart, prompts, 1, max_new_tokens, 0.0, eos_ids, max_rows=max_rows)
    for prompt, answer in zip(prompts, answers, strict=True):
        th = teacher_hidden(model, journal_cache, prompt, answer)
        with torch.no_grad():
            kls.append(reverse_kl_loss(model, cart, torch.tensor([prompt + answer], device=model.device), th).item())
        lengths.append(len(answer))
        stopped += answer[-1] in eos_ids
    return {"val/onpolicy_kl": sum(kls) / len(kls), "val/onpolicy_tokens": sum(lengths) / len(lengths),
            "val/onpolicy_stop_rate": stopped / len(prompts)}


def train_mixed(
    model,
    cart,
    examples: list,
    order: list[int],
    questions: list[list[int]],
    journal_cache,
    eos_ids: list[int],
    config: OnPolicyConfig,
    *,
    max_seq_len: int = 3584,
    log: Callable[[dict, int], None] = lambda metrics, step: None,
    on_step: Callable[[int], bool | None] = lambda step: None,
) -> dict:
    """Optimize `cart` for config.steps mixed steps. Off-policy sequences are
    `examples` (train_distill.Example) in `order` (cycled); on-policy prompts
    are `questions` (prompt ids), shuffled and cycled. Calls log(metrics, step)
    and on_step(step) after each step; a truthy on_step stops. Raises
    FloatingPointError on a non-finite loss or gradient, before the update."""
    from cartridges.cartridge.train_distill import make_optimizer, student_loss, to_tensors

    c = config
    params, optimizer, warmup = make_optimizer(cart, c.lr, c.relative_lr, c.warmup_steps)
    device = params[0].device
    rng = random.Random(c.seed)
    generator = torch.Generator(device=device).manual_seed(c.seed)
    q_order: list[int] = []
    off_pos = 0
    metrics: dict = {}
    for step in range(1, c.steps + 1):
        start = time.monotonic()
        lam = lam_at(c, step)
        n_q, n_off = split_batch(c, lam)
        n_on = n_q * c.samples_per_question
        # Weights: (1 - lam) x mean off-policy loss + lam x mean on-policy loss
        # (each kind at full weight when the other is absent).
        w_off = (1.0 if n_on == 0 else 1 - lam) / max(n_off, 1)
        w_on = (1.0 if n_off == 0 else lam) / max(n_on, 1)

        off_loss = off_kl = 0.0
        for _ in range(n_off):
            ex = examples[order[off_pos % len(order)]]
            off_pos += 1
            ids, top, lp = to_tensors(ex, max_seq_len)
            loss, kl = student_loss(model, cart, ids.to(device), top.to(device), lp.to(device))
            (loss * w_off).backward()
            off_loss += loss.item()
            off_kl += kl.item()

        on_kl, on_tokens, on_stopped = 0.0, 0, 0
        prompts = []
        for _ in range(n_q):
            if not q_order:
                q_order = rng.sample(range(len(questions)), len(questions))
            prompts.append(questions[q_order.pop()])
        t0 = time.monotonic()
        answers = sample(model, cart, prompts, c.samples_per_question, c.max_new_tokens, c.temperature,
                         eos_ids, generator, max_rows=c.max_gen_rows) if prompts else []
        gen_s = time.monotonic() - t0
        for k, answer in enumerate(answers):
            prompt = prompts[k // c.samples_per_question]
            answer = answer[: max_seq_len - len(prompt)]
            th = teacher_hidden(model, journal_cache, prompt, answer)
            loss = reverse_kl_loss(model, cart, torch.tensor([prompt + answer], device=device), th)
            (loss * w_on).backward()
            on_kl += loss.item()
            on_tokens += len(answer)
            on_stopped += answer[-1] in eos_ids

        grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
        total = off_loss + on_kl
        if not (math.isfinite(total) and math.isfinite(grad_norm)):
            raise FloatingPointError(f"step {step}: loss {total}, grad norm {grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scale = warmup.get_last_lr()[0] / optimizer.param_groups[0]["initial_lr"]
        warmup.step()
        elapsed = time.monotonic() - start
        metrics = {
            "train/lambda": lam,
            "train/grad_norm": grad_norm,
            "train/lr": c.lr * scale,
            "time/step_s": elapsed,
            "time/generate_s": gen_s,
            **({"train/loss": off_loss / n_off, "train/kl": off_kl / n_off} if n_off else {}),
            **({"train/onpolicy_kl": on_kl / n_on, "train/onpolicy_tokens": on_tokens / n_on,
                "train/onpolicy_stop_rate": on_stopped / n_on,
                "throughput/generate_tok_s": on_tokens / max(gen_s, 1e-9)} if n_on else {}),
        }
        log(metrics, step)
        if on_step(step):
            break
    return metrics
