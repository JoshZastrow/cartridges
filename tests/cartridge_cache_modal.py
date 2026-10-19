"""GPU check of cache.Cartridge on the real hybrid model, with the fla kernels.

uv run modal run tests/cartridge_cache_modal.py [--model-id Qwen/Qwen3.5-4B] [--grpo-lrs 1e-4,5e-4]
Runs are tracked in W&B (job_type profile / grpo) so their spend is counted.

Checks that (1) a cartridge initialized from x reproduces the logits of
prefilling x (bf16 tolerance, argmax agreement), (2) fla's chunk kernel passes
gradients to the DeltaNet initial states and every cartridge tensor gets a
finite, nonzero gradient under gradient checkpointing, (3) generation runs.
This module must not import pytest (the container imports it).
"""

import json

from cartridges.cartridge.train_distill import VOLUME_PATH, image, secrets, volume

import modal

app = modal.App("cartridges-cache-check", image=image)


GPU = "A100-80GB"


@app.function(gpu=GPU, timeout=20 * 60, volumes={VOLUME_PATH: volume}, secrets=secrets)
def check(model_id: str, p: int, group: str) -> dict:
    from cartridges.cartridge import tracking

    with tracking.init_run("profile", model_id, group, config={"check": "cache", "p": p}, p=p, gpu=GPU) as run:
        out = _check(model_id, p)
        run.summary.update({k: v for k, v in out.items() if isinstance(v, (int, float, str))})
        return {**out, "wandb_run": run.name, "cost": run.cost()}


def _check(model_id: str, p: int) -> dict:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.grpo import encode_prompt

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.requires_grad_(False)
    text = " ".join(f"Entry {i}: on day {i} the author walked {i % 7} miles and wrote about it." for i in range(400))
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    x, y = ids[:, :p], ids[:, p : p + 64]
    out = {"model_id": model_id, "p": p, "transformers": __import__("transformers").__version__}
    try:
        import fla

        out["fla"] = fla.__version__
    except ImportError:
        out["fla"] = None

    cart = Cartridge.from_prefill(model, x)
    with torch.no_grad():
        full = model(input_ids=torch.cat([x, y], 1)).logits[:, p:].float()
        ours = cart(model, y).logits.float()
        alone = model(input_ids=y).logits.float()
        # Baseline for bf16/kernel noise: HF's own cache, prefilled with x, then y.
        from transformers import DynamicCache

        cache = DynamicCache(config=model.config)
        model(input_ids=x, past_key_values=cache, use_cache=True)
        split = model(input_ids=y, past_key_values=cache, use_cache=True).logits.float()
    out["max_abs_diff_hf_split_vs_prefill"] = (split - full).abs().max().item()
    out["max_abs_diff_vs_hf_split"] = (ours - split).abs().max().item()
    out["mean_abs_diff_vs_prefill"] = (ours - full).abs().mean().item()
    out["max_abs_diff_vs_prefill"] = (ours - full).abs().max().item()
    out["argmax_agree_vs_prefill"] = (ours.argmax(-1) == full.argmax(-1)).float().mean().item()
    out["max_abs_diff_no_prefix"] = (alone - full).abs().max().item()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for layer in model.model.layers:
        layer._can_checkpoint_with_cache = True
    model.train()
    t0 = time.monotonic()
    logits = cart(model, y[:, :-1]).logits.float()
    loss = torch.nn.functional.cross_entropy(logits[0], y[0, 1:])
    loss.backward()
    torch.cuda.synchronize()
    out["fwd_bwd_s"] = time.monotonic() - t0
    for name in ("trainable_keys", "trainable_values", "conv_states", "recurrent_states"):
        norms = [g.grad.norm().item() if g.grad is not None else None for g in getattr(cart, name)]
        out[f"grad_{name}"] = {"none": sum(n is None for n in norms), "min": min(n for n in norms if n is not None),
                               "finite": all(n is not None and n == n and n != float("inf") for n in norms)}
    out["model_grads"] = sum(p.grad is not None for p in model.parameters())

    model.eval()
    prompt = torch.tensor([encode_prompt(tok, "What did the author do on day 3?")], device="cuda")
    t0 = time.monotonic()
    new = cart.generate(model, prompt, max_new_tokens=40, eos_token_id=tok.eos_token_id)
    out["generate_s"] = time.monotonic() - t0
    out["answer"] = tok.decode(new[0], skip_special_tokens=True)
    out["cartridge_mb_at_p"] = cart.nbytes() / 2**20
    out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    # An untrained cartridge for smoke-testing `cartridges cartridge-ask`.
    out["saved"] = str(cart.save(f"{VOLUME_PATH}/cartridges/check-{model_id.split('/')[-1]}"))
    volume.commit()
    return out


@app.function(gpu=GPU, timeout=30 * 60, volumes={VOLUME_PATH: volume}, secrets=secrets)
def grpo_smoke(model_id: str, p: int, steps: int, lr: float, group: str) -> dict:
    """A few async GRPO steps on synthetic "miles on day N" QA, rewarded by token F1."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cartridges.cartridge import tracking
    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.evals import token_f1
    from cartridges.cartridge.grpo import GRPOConfig, train_grpo

    config = GRPOConfig(group_size=8, groups_per_step=2, max_new_tokens=32, max_steps=steps, lr=lr)
    # Started before the model load so the run's cost covers the whole container.
    run = tracking.init_run("grpo", model_id, group, config={"smoke": True, **config.model_dump()}, p=p, gpu=GPU,
                            tags=["smoke"])
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    text = " ".join(f"Entry {i}: on day {i} the author walked {i % 7} miles and wrote about it." for i in range(400))
    cart = Cartridge.init_from_text(model, tok, text, p)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for layer in model.model.layers:
        layer._can_checkpoint_with_cache = True
    model.train()
    items = [{"question": f"How many miles did the author walk on day {i}?", "answer": f"{i % 7} miles"} for i in range(30)]
    logged = []

    def log(metrics, step):
        logged.append(metrics)
        run.log(metrics, step=step)

    with run:
        summary = train_grpo(model, tok, cart, items, lambda t, item: token_f1(t, item["answer"]), config, log=log)
        summary.update(wandb_run=run.name, cost=run.cost())
    keys = ("grpo/reward_mean", "grpo/ratio_mean", "grpo/clip_frac", "grpo/staleness_mean")
    per_step = [" ".join(f"{k.split('/')[1]}={m[k]:.3f}" for k in keys) for m in logged]
    return {**summary, "lr": lr, "per_step": per_step, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen3.5-4B", p: int = 512, grpo_lrs: str = "", steps: int = 12, group: str = "") -> None:
    from cartridges.cartridge.tracking import group_name

    group = group or group_name("cache-check")
    if not grpo_lrs:
        print(json.dumps(check.remote(model_id, p, group), indent=2, default=str))
        return
    lrs = [float(x) for x in grpo_lrs.split(",")]
    for out in grpo_smoke.starmap([(model_id, p, steps, lr, group) for lr in lrs]):
        print(json.dumps(out, indent=2, default=str))
