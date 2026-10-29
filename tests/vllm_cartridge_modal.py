"""GPU parity check of the vLLM cartridge path (serve/vllm_cartridge.py) against
HF's Cartridge (FR1 of issue #4), and a mixed-cartridge batch (FR2).

uv run modal run tests/vllm_cartridge_modal.py [--model-id Qwen/Qwen3.8-27B --cartridges /vol/...]
(export the cartridges first: `cartridges cartridge_export_vllm --cartridge ...`).

One container holds the HF model (bf16, fla kernels) and an in-process vLLM
engine with the same arguments as CartridgeEngine. All requests run as one
vLLM batch: every prompt with each cartridge, plus each prompt with no
cartridge (base-model noise floor). Per request: greedy match of the first 32
tokens vs Cartridge.generate, top-1 agreement of HF teacher-forced on vLLM's
greedy tokens, and max |logprob| difference of the first token over vLLM's
top 20. Then a small benchmark on the same GPU (_bench). Tracked in W&B (job_type profile) so the spend is counted.
This module must not import pytest (the container imports it).
"""

import json
import os

import modal

from cartridges.cartridge import teacher
from cartridges.serve import vllm_cartridge

app = modal.App("cartridges-vllm-parity", image=teacher.image)

SMOKE_CARTRIDGES = "/vol/cartridges/exp-20261013-smoke/grpo,/vol/cartridges/check-Qwen3.5-4B"
PROMPTS = [
    "What topics come up most often?",
    "Summarize the main themes in two sentences.",
    "What did I write about work?",
    "How did my mood change over time?",
    "List three people who are mentioned often.",
    "What habits did I try to build?",
    "What was a difficult period, and why?",
    "Give one piece of advice based on what you know about me.",
]


@app.function(gpu=["L4", "L40S"], timeout=40 * 60, secrets=teacher.secrets,
              volumes={teacher.VOLUME_PATH: teacher.volume, teacher.COMPILE_CACHE_PATH: teacher.compile_cache})
def parity(model_id: str, cartridges: list[str], max_tokens: int, group: str) -> dict:
    from cartridges.cartridge import tracking

    with tracking.init_run("profile", model_id, group, config={"check": "vllm-parity", "cartridges": cartridges},
                           role="cartridge-vllm") as run:  # GPU detected: one of the fallbacks
        out = _parity(model_id, cartridges, max_tokens)
        run.summary.update({k: v for k, v in out["summary"].items() if isinstance(v, (int, float))})
        return {**out, "wandb_run": run.name, "cost": run.cost()}


def _parity(model_id: str, cartridges: list[str], max_tokens: int) -> dict:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # engine in this process: the connector is inspectable
    import time

    import torch
    import vllm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    from cartridges.cartridge.cache import Cartridge
    from cartridges.serve import vllm_connector

    tok = AutoTokenizer.from_pretrained(model_id)
    eos = tok.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    prompts = [teacher.render(tok, [{"role": "user", "content": q}])[1] for q in PROMPTS]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()  # no accelerate
    carts = {c: Cartridge.load(vllm_cartridge.engine_path(c), device="cuda") for c in cartridges}

    # Record what the worker sees on its first load (layouts, block ids).
    seen = {}
    original = vllm_connector.CartridgeConnector.start_load_kv

    def recording(self, forward_context, **kwargs):
        loads = self._get_connector_metadata().loads
        if loads and "groups" not in seen:
            layers = forward_context.no_compile_layers
            seen["groups"] = {n: g for n, g in self.groups.items() if n.endswith((".0.linear_attn", ".3.self_attn.attn"))}
            seen["kv_cache"] = {n: [list(t.shape) + [str(t.dtype)] for t in (layers[n].kv_cache if isinstance(layers[n].kv_cache, tuple) else [layers[n].kv_cache])]
                                for n in seen["groups"]}
            seen["block_ids"] = [[len(ids), ids[:3]] for ids in loads[0].block_ids]
        return original(self, forward_context, **kwargs)

    vllm_connector.CartridgeConnector.start_load_kv = recording
    llm = LLM(**vllm_cartridge.engine_kwargs(model_id, gpu_memory_utilization=0.55, max_model_len=8192))
    config = llm.llm_engine.vllm_config
    seen["block_size"] = config.cache_config.block_size
    seen["mamba_ssm_cache_dtype"] = config.cache_config.mamba_ssm_cache_dtype

    requests = [(c, i) for c in [*cartridges, None] for i in range(len(prompts))]
    vllm_prompts, params = [], []
    for c, i in requests:
        extra = None
        if c is not None:
            extra = {"kv_transfer_params": vllm_cartridge.transfer_params(c, model_id, vllm.__version__)}
        p = extra["kv_transfer_params"]["p"] if extra else 0
        vllm_prompts.append(TokensPrompt(prompt_token_ids=[vllm_cartridge.PLACEHOLDER] * p + prompts[i]))
        params.append(SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=20, extra_args=extra))
    t0 = time.perf_counter()
    outs = llm.generate(vllm_prompts, params)
    batch_s = time.perf_counter() - t0

    rows = []
    for (c, i), o in zip(requests, outs):
        v_ids = list(o.outputs[0].token_ids)
        x = torch.tensor([prompts[i]], device="cuda")
        cart = carts.get(c)
        with torch.no_grad():
            if cart is not None:
                hf_ids = cart.generate(model, x, max_new_tokens=max_tokens, eos_token_id=eos_ids)[0].tolist()
                logits = cart(model, torch.tensor([prompts[i] + v_ids], device="cuda")).logits[0].float()
            else:
                hf_ids = model.generate(x, max_new_tokens=max_tokens, do_sample=False)[0, x.shape[1]:].tolist()
                logits = model(torch.tensor([prompts[i] + v_ids], device="cuda")).logits[0].float()
        n = len(prompts[i])
        top1 = logits[n - 1 : n - 1 + len(v_ids)].argmax(-1).tolist()
        first = torch.log_softmax(logits[n - 1], -1)
        lp_diff = max(abs(lp.logprob - first[t].item()) for t, lp in o.outputs[0].logprobs[0].items())
        k = 32
        rows.append({
            "cartridge": c or "base", "prompt": i, "vllm_tokens": len(v_ids),
            "greedy32_match": hf_ids[:k] == v_ids[:k],
            "common_prefix": next((j for j, (a, b) in enumerate(zip(hf_ids, v_ids)) if a != b), min(len(hf_ids), len(v_ids))),
            "top1_agree": sum(a == b for a, b in zip(top1, v_ids)) / max(1, len(v_ids)),
            "first_logprob_max_abs_diff": lp_diff,
            "vllm_text": o.outputs[0].text[:60],
        })
    summary = {"batch_requests": len(requests), "batch_s": batch_s}
    for c in [*cartridges, "base"]:
        r = [row for row in rows if row["cartridge"] == c]
        summary[f"{c}/greedy32_match"] = sum(row["greedy32_match"] for row in r) / len(r)
        summary[f"{c}/top1_agree_min"] = min(row["top1_agree"] for row in r)
        summary[f"{c}/first_logprob_max_abs_diff"] = max(row["first_logprob_max_abs_diff"] for row in r)
    summary.update(_bench(llm, model, carts[cartridges[0]], prompts[0], cartridges[0], model_id))
    return {"summary": summary, "rows": rows, "internals": seen, "vllm": vllm.__version__}


def _bench(llm, model, cart, prompt: list[int], cartridge: str, model_id: str, n: int = 256) -> dict:
    """Batch-1 TTFT and decode tok/s on this GPU (fixed n tokens, EOS ignored):
    HF Cartridge.stream (CartridgeServer's path), vLLM with the cartridge, vLLM
    base model with no context; and vLLM batch-8 throughput with the cartridge."""
    import time

    import torch
    import vllm
    from vllm import SamplingParams, TokensPrompt

    out = {}
    x = torch.tensor([prompt], device="cuda")
    for _ in range(2):  # the first pass warms the kernels
        t0, first = time.perf_counter(), None
        for k, _token in enumerate(cart.stream(model, x, max_new_tokens=n)):
            first = first or time.perf_counter()
        end = time.perf_counter()
    out["bench/hf_ttft_s"], out["bench/hf_decode_tok_s"] = first - t0, (n - 1) / (end - first)

    extra = {"kv_transfer_params": vllm_cartridge.transfer_params(cartridge, model_id, vllm.__version__)}
    for name, kv in (("vllm", extra), ("vllm_base", None)):
        tokens = [vllm_cartridge.PLACEHOLDER] * extra["kv_transfer_params"]["p"] + prompt if kv else prompt

        def run(max_tokens, batch=1):
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, extra_args=kv)
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=tokens)] * batch, sp, use_tqdm=False)
            return time.perf_counter() - t0

        run(n)  # warm
        ttft = min(run(1) for _ in range(3))
        out[f"bench/{name}_ttft_s"] = ttft
        out[f"bench/{name}_decode_tok_s"] = (n - 1) / (run(n) - ttft)
        out[f"bench/{name}_batch8_tok_s"] = 8 * n / run(n, batch=8)
    return out


@app.local_entrypoint()
def main(model_id: str = teacher.SMOKE_MODEL_ID, cartridges: str = SMOKE_CARTRIDGES, max_tokens: int = 256, group: str = ""):
    from cartridges.cartridge.tracking import group_name

    gpu = vllm_cartridge.GPU_FOR_MODEL.get(model_id)
    fn = parity.with_options(gpu=gpu) if gpu else parity
    out = fn.remote(model_id, cartridges.split(","), max_tokens, group or group_name("vllm-parity"))
    print(json.dumps(out, indent=2, default=str))
