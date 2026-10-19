"""Stage 2: async GRPO on the cartridge's tensors.

Follows the structure of https://huggingface.co/blog/asyncgrpo-lora-hfjobs, with
the LoRA adapter replaced by the cartridge (cache.Cartridge) and the vLLM servers
replaced by a generation thread on the same GPU (vLLM cannot serve a trainable
KV prefix plus DeltaNet initial states):

- The generation worker holds its own copy of the cartridge at policy version v.
  For each prompt it samples a group of G answers, scores them with the reward
  fn, computes group-relative advantages and the behavior logprobs, and puts the
  group on a bounded rollout queue (a full queue blocks it: backpressure).
- The trainer pulls groups, drops any whose version lags the current policy by
  more than `max_staleness`, and takes one optimizer step per
  `groups_per_step` groups. Every `weight_sync_steps` steps it publishes the
  cartridge tensors (weight sync = copying a few tens of MB of cache tensors)
  and bumps the policy version; the worker loads them before its next group.
- Loss: token-level clipped importance ratio (PPO/DAPO style, no KL term),
  ratio = pi_theta / pi_behavior per completion token, clipped to
  [1 - clip_low, 1 + clip_high], normalized by the step's completion tokens.
  Advantages: (r - mean_group) / (std_group + eps); a group whose rewards are all
  equal carries no signal and is skipped.

The reward is evals.grounded_reward (owned by eval): the caller passes
`reward_fn(completion_text, item) -> float`.
"""

from __future__ import annotations

import copy
import logging
import queue
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict

from cartridges.cartridge.cache import Cartridge

logger = logging.getLogger(__name__)


class GRPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_size: int = 8  # G rollouts per prompt (blog: 8)
    groups_per_step: int = 4  # prompts per optimizer step
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    lr: float = 1e-4  # 4B smoke: 5e-4 already clips 20-40% of tokens; 1e-4 learns (reward 0.64 -> ~0.8 in 12 steps)
    clip_low: float = 0.2
    clip_high: float = 0.28
    max_staleness: int = 4  # policy versions (blog: 4)
    weight_sync_steps: int = 1  # optimizer steps between syncs (blog: 4; our sync is a local copy)
    queue_maxsize: int = 8  # groups
    max_steps: int = 50
    seed: int = 0


@dataclass
class Group:
    prompt_ids: torch.Tensor  # [L]
    completions: list[torch.Tensor]  # G x [T_i]
    old_logprobs: torch.Tensor  # [G, T] behavior policy, 0 past each completion
    rewards: list[float]
    advantages: list[float]
    version: int


# ---------------------------------------------------------------- math


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    r = torch.tensor(rewards, dtype=torch.float64)
    std = r.std(unbiased=False)
    return ((r - r.mean()) / (std + eps)).tolist()


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
    n_tokens: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Clipped surrogate summed over completion tokens [G, T], divided by `n_tokens`
    (default: this batch's tokens; pass the whole step's count to accumulate)."""
    mask = mask.to(logprobs.dtype)
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages[:, None].to(logprobs.dtype)
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    per_token = -torch.minimum(ratio * adv, clipped * adv)
    n = n_tokens if n_tokens is not None else mask.sum().clamp(min=1)
    loss = (per_token * mask).sum() / n
    with torch.no_grad():
        m = mask.bool()
        is_clipped = (ratio * adv != torch.minimum(ratio * adv, clipped * adv)) & m
        log_ratio = (logprobs - old_logprobs)[m]
        stats = {
            "ratio_mean": ratio[m].mean().item() if m.any() else 1.0,
            # k3 estimate of KL(behavior || current) on the sampled tokens (logged only; no KL term in the loss)
            "kl": (log_ratio.exp() - 1 - log_ratio).mean().item() if m.any() else 0.0,
            "clip_frac": (is_clipped.sum() / m.sum().clamp(min=1)).item(),
        }
    return loss, stats


def pad_completions(completions: list[torch.Tensor], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(c) for c in completions)
    ids = torch.full((len(completions), T), pad_id, dtype=torch.long, device=completions[0].device)
    mask = torch.zeros_like(ids)
    for i, c in enumerate(completions):
        ids[i, : len(c)], mask[i, : len(c)] = c, 1
    return ids, mask


def sequence_logprobs(
    cart: Cartridge, model, prompt_ids: torch.Tensor, completions: list[torch.Tensor], temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs [G, T] (and mask) of each completion after the shared prompt,
    from logits / temperature as sampled. Right padding keeps the DeltaNet states clean."""
    comp, mask = pad_completions(completions)
    G, L = comp.shape[0], prompt_ids.shape[0]
    input_ids = torch.cat([prompt_ids.expand(G, -1), comp], dim=1)
    attention_mask = torch.cat([mask.new_ones(G, L), mask], dim=1)
    logits = cart(model, input_ids, attention_mask=attention_mask, logits_to_keep=comp.shape[1] + 1).logits
    logits = logits[:, :-1].float() / max(temperature, 1e-6)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, comp[..., None]).squeeze(-1)
    return logp * mask, mask


def encode_prompt(tokenizer, question: str) -> list[int]:
    """The student's prompt_ids for a question (teacher.render, non-thinking)."""
    from cartridges.cartridge.evals import question_messages
    from cartridges.cartridge.teacher import render

    return render(tokenizer, question_messages(question))[1]


def trim_eos(tokens: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Keep tokens through the first EOS (the EOS is part of what the policy chose)."""
    hits = torch.isin(tokens, torch.tensor(eos_ids, device=tokens.device)).nonzero()
    return tokens[: hits[0, 0] + 1] if len(hits) else tokens


# ---------------------------------------------------------------- async loop


class RolloutBuffer:
    """Bounded queue of groups; `get` drops groups more than `max_staleness` versions old."""

    def __init__(self, maxsize: int, max_staleness: int):
        self.queue: queue.Queue[Group] = queue.Queue(maxsize=maxsize)
        self.max_staleness = max_staleness
        self.dropped = 0

    def put(self, group: Group, stop: threading.Event) -> bool:
        while not stop.is_set():
            try:
                self.queue.put(group, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get(self, version: int, timeout: float = 600.0) -> Group:
        while True:
            group = self.queue.get(timeout=timeout)
            if version - group.version <= self.max_staleness:
                return group
            self.dropped += 1


class GenerationWorker(threading.Thread):
    """Samples groups with its own cartridge copy; `publish` hands it new weights."""

    def __init__(self, model, tokenizer, cart: Cartridge, items: list[dict], reward_fn, buffer: RolloutBuffer, config: GRPOConfig):
        super().__init__(daemon=True)
        self.model, self.tokenizer, self.items, self.reward_fn = model, tokenizer, items, reward_fn
        self.buffer, self.config = buffer, config
        self.cart = copy.deepcopy(cart).requires_grad_(False)
        self.version = 0
        self._pending: tuple[dict, int] | None = None
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.rng = random.Random(config.seed)
        self.generator = torch.Generator(device=model.device).manual_seed(config.seed)
        eos = tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        self.gen_seconds = 0.0

    def publish(self, state_dict: dict, version: int) -> None:
        with self._lock:
            self._pending = ({k: v.detach().clone() for k, v in state_dict.items()}, version)

    def _sync(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, None
        if pending:
            self.cart.load_state_dict(pending[0])
            self.version = pending[1]

    def run(self) -> None:
        try:
            order: list[dict] = []
            while not self.stop.is_set():
                if not order:
                    order = self.rng.sample(self.items, len(self.items))
                self._sync()
                group = self.rollout(order.pop())
                if not self.buffer.put(group, self.stop):
                    return
        except BaseException as e:  # surfaced by the trainer loop
            self.error = e
            logger.exception("generation worker failed")

    @torch.no_grad()
    def rollout(self, item: dict) -> Group:
        c, device = self.config, self.model.device
        t0 = time.monotonic()
        prompt = torch.tensor(encode_prompt(self.tokenizer, item["question"]), device=device)
        new = self.cart.generate(
            self.model, prompt.expand(c.group_size, -1), max_new_tokens=c.max_new_tokens,
            temperature=c.temperature, top_p=c.top_p, eos_token_id=self.eos_ids or None, generator=self.generator,
        )
        completions = [trim_eos(row, self.eos_ids) if self.eos_ids else row for row in new]
        completions = [x if len(x) else new.new_tensor(self.eos_ids[:1] or [0]) for x in completions]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        rewards = [float(self.reward_fn(t, item)) for t in texts]
        old, _ = sequence_logprobs(self.cart, self.model, prompt, completions, c.temperature)
        self.gen_seconds += time.monotonic() - t0
        return Group(prompt, completions, old, rewards, group_advantages(rewards), self.version)


def train_grpo(
    model,
    tokenizer,
    cart: Cartridge,
    items: list[dict],
    reward_fn: Callable[[str, dict], float],
    config: GRPOConfig,
    log: Callable[[dict, int], None] | None = None,
    on_step: Callable[[int], None] | None = None,
) -> dict:
    """Async GRPO on `cart` in place. `items` need a "question" plus whatever `reward_fn` reads.
    `log(metrics, step)` receives per-step metrics; `on_step(step)` can save checkpoints."""
    model.requires_grad_(False)
    opt = torch.optim.AdamW(cart.parameters(), lr=config.lr, weight_decay=0.0)
    buffer = RolloutBuffer(config.queue_maxsize, config.max_staleness)
    worker = GenerationWorker(model, tokenizer, cart, items, reward_fn, buffer, config)
    worker.start()
    version, t_start = 0, time.monotonic()
    history: list[dict] = []
    try:
        for step in range(1, config.max_steps + 1):
            t0 = time.monotonic()
            groups = []
            while len(groups) < config.groups_per_step:
                if worker.error:
                    raise RuntimeError("generation worker failed") from worker.error
                try:
                    groups.append(buffer.get(version, timeout=1.0))
                except queue.Empty:
                    continue
            wait = time.monotonic() - t0
            staleness = [version - g.version for g in groups]
            useful = [g for g in groups if any(abs(a) > 0 for a in g.advantages)]
            n_tokens = sum(sum(len(c) for c in g.completions) for g in useful)
            opt.zero_grad(set_to_none=True)
            stats = []
            for g in useful:
                logp, mask = sequence_logprobs(cart, model, g.prompt_ids, g.completions, config.temperature)
                adv = torch.tensor(g.advantages, device=logp.device)
                loss, s = grpo_loss(logp, g.old_logprobs, adv, mask, config.clip_low, config.clip_high, n_tokens)
                loss.backward()
                stats.append({**s, "loss": loss.item()})
            if useful:
                opt.step()
                version += 1
            if version and version % config.weight_sync_steps == 0:
                worker.publish(cart.state_dict(), version)
            rewards = [r for g in groups for r in g.rewards]
            metrics = {
                "grpo/reward_mean": sum(rewards) / len(rewards),
                "grpo/loss": sum(s["loss"] for s in stats),
                "grpo/ratio_mean": _mean([s["ratio_mean"] for s in stats], 1.0),
                "grpo/clip_frac": _mean([s["clip_frac"] for s in stats], 0.0),
                "grpo/kl": _mean([s["kl"] for s in stats], 0.0),
                "grpo/staleness_mean": sum(staleness) / len(staleness),
                "grpo/staleness_max": max(staleness),
                "grpo/useful_groups": len(useful),
                "grpo/dropped_stale": buffer.dropped,
                "grpo/completion_tokens": sum(len(c) for g in groups for c in g.completions) / (len(groups) * config.group_size),
                "grpo/policy_version": version,
                "perf/rollout_wait_s": wait,
                "perf/step_s": time.monotonic() - t0,
                "throughput/step_tok_s": sum(len(c) for g in groups for c in g.completions) / (time.monotonic() - t0),
                "perf/generation_s_total": worker.gen_seconds,
            }
            history.append(metrics)
            if log:
                log(metrics, step)
            if on_step:
                on_step(step)
    finally:
        worker.stop.set()
        worker.join(timeout=60)
    return {
        "steps": len(history),
        "policy_version": version,
        "reward_first": history[0]["grpo/reward_mean"] if history else None,
        "reward_last": history[-1]["grpo/reward_mean"] if history else None,
        "dropped_stale": buffer.dropped,
        "seconds": time.monotonic() - t_start,
    }


def _mean(values: list[float], default: float) -> float:
    return sum(values) / len(values) if values else default
