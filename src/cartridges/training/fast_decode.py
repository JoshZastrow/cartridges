"""Faster single-stream decode for DeepSeek-OCR-2 under `model.infer()`.

Used by `profiling.py`; everything here plugs in underneath the model's own
`infer()` (prompt build, image tiling, output decoding stay untouched):

- `_IncrementalNoRepeatNGram`: exact, incremental replacement for HF's
  NoRepeatNGramLogitsProcessor.
- `_moe_one_sync`: DeepseekV2MoE.moe without a host sync per expert, plus a
  sync-free single-token path over stacked expert weights (LoRA unmerged).
- `CudaGraphDecoder`: static KV cache, the 1-token decode step replayed from a
  CUDA graph; greedy + EOS + no-repeat-ngram semantics of infer()'s generate.
- `RANGES`: record_function/NVTX stage labels shared with the profiler.

torch/transformers are imported lazily (this module is imported locally too).
"""

from __future__ import annotations

import time

from cartridges.training.finetune import _stage
from cartridges.training.perf_metrics import NoRepeatNGramBlocker


class _Ranges:
    """Named `record_function` + NVTX ranges opened/closed from hooks.

    `infer()` is remote code, so stages inside it are labelled from the
    outside: forward hooks on the vision modules and the language model,
    plus the generate()/tokenizer.decode wrappers. Ranges stay strictly
    nested (preprocess closes before the first LM call; each LM-call range
    closes before the next one opens), which both backends require. Cheap
    (~µs) when no profiler is attached, so they stay on for timed runs too.
    """

    def __init__(self):
        self._open: list = []
        self.lm_calls = 0
        self.call_times: list[float] = []  # host time at each LM call (per-step wall time)

    def reset_calls(self):
        self.lm_calls = 0
        self.call_times = []

    def push(self, name: str):
        import torch

        rf = torch.profiler.record_function(name)
        rf.__enter__()
        torch.cuda.nvtx.range_push(name)
        self._open.append((name, rf))

    def pop(self, name: str):
        import torch

        if self._open and self._open[-1][0] == name:
            torch.cuda.nvtx.range_pop()
            self._open.pop()[1].__exit__(None, None, None)

    def lm_call(self):
        """One language-model forward = prefill (first in a generate) or one decode step."""
        self.end_lm()
        self.call_times.append(time.perf_counter())
        self.push("prefill" if self.lm_calls == 0 else "decode_step")
        self.lm_calls += 1

    def end_lm(self):
        # whichever is innermost-open: generate setup (before the first LM
        # call), the prefill range, or the previous decode step
        self.pop("generate_setup")
        self.pop("prefill")
        self.pop("decode_step")


RANGES = _Ranges()  # process-wide: there is one model and one thread per container


class _IncrementalNoRepeatNGram:
    """Drop-in for HF's NoRepeatNGramLogitsProcessor (batch 1), backed by
    `NoRepeatNGramBlocker`'s incremental n-gram index."""

    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size
        self.blocker: NoRepeatNGramBlocker | None = None

    def __call__(self, input_ids, scores):
        if self.blocker is None:
            self.blocker = NoRepeatNGramBlocker(self.ngram_size, input_ids[0].tolist())
        else:
            for token in input_ids[0, len(self.blocker.tokens) :].tolist():
                self.blocker.append(token)
        banned = self.blocker.banned()
        if banned:
            scores[0, banned] = -float("inf")
        return scores



def _expert_linear(x, stacked: dict, name: str, idx):
    """One expert projection for one token, replicating what the eager path
    runs: `F.linear(x, W)` and, for an unmerged PEFT LoRA layer,
    `result + lora_B(lora_A(x)) * scaling`. `idx` is a 1-element CUDA index
    tensor; index_select (not `W[idx]`) keeps the gather free of host syncs."""
    import torch
    import torch.nn.functional as F

    out = F.linear(x, torch.index_select(stacked[name], 0, idx)[0])
    if name + ".A" in stacked:
        lora_a = torch.index_select(stacked[name + ".A"], 0, idx)[0]
        lora_b = torch.index_select(stacked[name + ".B"], 0, idx)[0]
        out = out + F.linear(F.linear(x, lora_a), lora_b) * stacked[name + ".scaling"]
    return out


def _batched_linear(x, stacked: dict, name: str, idx):
    """`_expert_linear` for all k routed experts at once: x [k, in] -> [k, out]."""
    import torch

    def bmm(weights, inputs):
        return torch.bmm(torch.index_select(weights, 0, idx), inputs[..., None])[..., 0]

    out = bmm(stacked[name], x)
    if name + ".A" in stacked:
        out = out + bmm(stacked[name + ".B"], bmm(stacked[name + ".A"], x)) * stacked[name + ".scaling"]
    return out


def _moe_one_sync(self, hidden_states, topk_ids, topk_weight):
    """transformers' DeepseekV2MoE.moe with two changes:

    1. `tokens_per_expert.tolist()` once, instead of iterating the CUDA tensor
       (`if num_tokens == 0` on each element = one blocking GPU->CPU sync per
       expert: 64 experts x 11 MoE layers ~ 700 syncs per decoded token).
       Same math, bit-identical outputs.
    2. If `CudaGraphDecoder` stacked the expert weights (`_stacked`) and this
       is a single token, run the top-k experts from gathered weights: same
       per-expert ops as eager (LoRA left unmerged), but no host sync and
       static shapes, so the decode step can be captured in a CUDA graph.
    """
    import torch

    stacked = getattr(self, "_stacked", None)
    if hidden_states.shape[0] == 1 and stacked is not None and getattr(self, "_batched", False):
        # All k routed experts at once: gather [k, out, in] weights, bmm.
        # ~6x fewer kernels than the per-expert path below; same math up to
        # matmul accumulation order (LoRA still unmerged).
        idx = topk_ids[0]
        x = hidden_states.expand(idx.shape[0], -1)
        act = self.experts[0].act_fn(_batched_linear(x, stacked, "gate_proj", idx))
        h = act * _batched_linear(x, stacked, "up_proj", idx)
        out = _batched_linear(h, stacked, "down_proj", idx)  # [k, hidden]
        return (out.type(topk_weight.dtype) * topk_weight[0, :, None]).sum(0, keepdim=True).type(out.dtype)

    if hidden_states.shape[0] == 1 and stacked is not None:
        act_fn = self.experts[0].act_fn
        outs = []
        for j in range(topk_ids.shape[1]):
            idx = topk_ids[0, j : j + 1]
            gate = _expert_linear(hidden_states, stacked, "gate_proj", idx)
            up = _expert_linear(hidden_states, stacked, "up_proj", idx)
            outs.append(_expert_linear(act_fn(gate) * up, stacked, "down_proj", idx))
        out = torch.cat(outs, dim=0)  # [k, hidden], in topk order like eager's new_x
        return (
            out.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(out.dtype)
        )

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0).tolist()
    indicies = topk_ids.view(-1).argsort()
    sorted_tokens = hidden_states[indicies // topk_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue
        end_idx = start_idx + num_tokens
        outputs.append(self.experts[i + self.ep_rank * self.experts_per_rank](sorted_tokens[start_idx:end_idx]))
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[indicies] = outs
    return (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def _moe_modules(model):
    return [m for m in model.modules() if hasattr(m, "experts") and hasattr(m, "moe")]


def _patch_moe(model) -> int:
    classes = {type(m) for m in _moe_modules(model)}
    for cls in classes:
        cls.moe = _moe_one_sync
    return len(classes)


class CudaGraphDecoder:
    """Greedy decode with a static KV cache and the 1-token step replayed
    from a CUDA graph. Prefill still runs the model's own forward (vision
    encoder + full prompt) eagerly, into the same static cache.

    Decoding semantics match `infer()`'s generate call: greedy, stop at
    eos_token_id (included in the output), max_new_tokens, and the same
    no-repeat-n-gram rule (checked on the CPU each step; the argmax token
    has to come back to the host anyway to check it and for EOS).
    """

    # Prompt is ~1,150 tokens max (257 global + up to 6x144 crop tokens + text)
    # plus max_new_tokens <= 1024; one fixed size = one graph for every page.
    MAX_CACHE_LEN = 2560

    def __init__(self, model):
        import torch
        from transformers import StaticCache

        self.model = model
        self.cache = StaticCache(config=model.config, max_cache_len=self.MAX_CACHE_LEN)
        self.ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.pos = torch.zeros((1,), dtype=torch.long, device="cuda")
        self.graph = None
        self.logits = None
        self._stack_expert_weights()

    def _stack_expert_weights(self):
        """Stack each MoE layer's per-expert weights ([E, out, in]), plus the
        LoRA A/B matrices if the adapter is still unmerged, so the decode
        step can gather the routed experts without leaving the GPU. Prefill
        keeps using the per-expert modules (copies, ~5 GB extra in bf16)."""
        import torch

        for moe in _moe_modules(self.model):
            stacked = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                layers = [getattr(e, name) for e in moe.experts]
                if hasattr(layers[0], "base_layer"):  # unmerged PEFT lora.Linear
                    stacked[name] = torch.stack([l.base_layer.weight for l in layers])
                    stacked[name + ".A"] = torch.stack([l.lora_A["default"].weight for l in layers])
                    stacked[name + ".B"] = torch.stack([l.lora_B["default"].weight for l in layers])
                    stacked[name + ".scaling"] = layers[0].scaling["default"]
                else:
                    stacked[name] = torch.stack([l.weight for l in layers])
            moe._stacked = stacked

    def _step(self):
        from transformers import DeepseekV2Model

        lm = self.model.model
        out = DeepseekV2Model.forward(
            lm,
            inputs_embeds=lm.embed_tokens(self.ids),
            position_ids=self.pos[None],
            past_key_values=self.cache,
            cache_position=self.pos,
            use_cache=True,
        )
        return self.model.lm_head(out.last_hidden_state[:, -1]).float()

    def _capture(self):
        import torch

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # also lazily allocates the static cache
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()
        self.cache.reset()

    def generate(self, input_ids, *, timer, state, images, images_seq_mask, images_spatial_crop,
                 eos_token_id, max_new_tokens, no_repeat_ngram_size=0, **_ignored):
        import torch

        if self.graph is None:
            # First call happens inside infer()'s autocast/no_grad context, so
            # the graph is captured under the same numerics as eager decode.
            with _stage("capture_cuda_graph", max_cache_len=self.MAX_CACHE_LEN):
                self._capture()
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.MAX_CACHE_LEN:
            raise ValueError(f"prompt {prompt_len} + {max_new_tokens} new tokens > {self.MAX_CACHE_LEN}")

        # Prefill exactly as HF generate does it (dynamic cache, all-ones
        # attention mask -> causal SDPA), then copy K/V into the static cache;
        # prefilling straight into the static cache would use a masked
        # attention kernel and already perturb the first token's logits.
        from transformers import DynamicCache

        dynamic = DynamicCache()
        prefill = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=torch.arange(prompt_len, device=input_ids.device)[None],
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            past_key_values=dynamic,
            use_cache=True,
        )
        logits = prefill.logits[:, -1]
        self.cache.reset()
        for static_layer, dynamic_layer in zip(self.cache.layers, dynamic.layers):
            static_layer.keys[:, :, :prompt_len].copy_(dynamic_layer.keys)
            static_layer.values[:, :, :prompt_len].copy_(dynamic_layer.values)
        del dynamic, prefill
        timer.mark()

        blocker = NoRepeatNGramBlocker(no_repeat_ngram_size, input_ids[0].tolist()) if no_repeat_ngram_size else None
        tokens: list[int] = []
        for step in range(max_new_tokens):
            # Same labels/timing as the HF path: the n-gram check plus the
            # argmax sync are this path's "logits processors".
            RANGES.push("logits_processors")
            start = time.perf_counter()
            if blocker is not None:
                banned = blocker.banned()
                if banned:
                    logits = logits.clone()
                    logits[0, banned] = -float("inf")
            token = int(logits.argmax(dim=-1))
            tokens.append(token)
            if blocker is not None:
                blocker.append(token)
            elapsed = time.perf_counter() - start
            state.processor_s.append(elapsed)
            state.processor_times.setdefault("graph_path.no_repeat_ngram+argmax", []).append(elapsed)
            RANGES.pop("logits_processors")
            if token == eos_token_id or step == max_new_tokens - 1:
                break
            RANGES.lm_call()  # one decode step: graph replay + next token check
            self.ids.fill_(token)
            self.pos.fill_(prompt_len + step)
            self.graph.replay()
            logits = self.logits
        new = torch.tensor([tokens], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, new], dim=1)


