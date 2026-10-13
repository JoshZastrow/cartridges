"""HF vs vLLM: hidden states at each stage boundary, same weights, same inputs.

Both sides load the same merged bf16 weights (/cache/models/<sha>), so only
the implementation differs. Per page, each side saves:
  sam/global, sam/local        SAM encoder output (global view, tiles)
  qwen2/global, qwen2/local    Qwen2-as-encoder output
  proj/global, proj/local      projector output (image embeddings)
  view_sep                     the view-separator parameter
  lm_in                        LM layer-0 input: full prompt embeddings
  final_norm_last              final-norm hidden state, last prompt position
  logits_last                  logits at the last prompt position
plus input_ids. HF runs twice: with autocast bf16 (the eval path) and plain
bf16 (no autocast, like vLLM). A third function compares.

uv run modal run scripts/parity/stage_parity.py --pages 63,52,44
"""

import modal

from cartridges.training.finetune import image as _training_image

SHA = "a2f23f486707d22fb735e48c9cd258a6660e57ee"
MERGED = f"/cache/models/{SHA}"
OUT = "/parity"

app = modal.App("cartridges-stage-parity")
model_cache = modal.Volume.from_name("cartridges-training-cache")
scratch = modal.Volume.from_name("cartridges-parity-scratch", create_if_missing=True)
hf_image = _training_image.add_local_python_source("cartridges")
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.29.0")
    .add_local_python_source("cartridges")
)
VOLUMES = {"/cache": model_cache, OUT: scratch}


def _first_tensor(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return _first_tensor(x[0])
    raise TypeError(type(x))


@app.function(image=hf_image, gpu="A100", timeout=30 * 60, volumes=VOLUMES)
def hf_side(pngs: dict[int, bytes]) -> list[str]:
    import io
    import sys

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, "/cache")
    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation
    from cartridges.training.config import TrainConfig

    config = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MERGED, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        use_safetensors=True,
    ).cuda().eval()
    inner = model.model
    collator = DeepSeekOCRDataCollator(tokenizer=tokenizer, model=model, image_size=config.image_size,
                                       base_size=config.base_size, crop_mode=config.crop_mode)
    store: dict = {}

    def view(x):  # SAM input: 1024 px = global view, 768 px = tiles
        return "global" if x.shape[-1] == 1024 else "local"

    for name, key in (("sam_model", "sam"), ("qwen2_model", "qwen2"), ("projector", "proj")):
        def hook(mod, args, out, key=key):
            store[f"{key}/{store['_view']}"] = _first_tensor(out).detach().to(torch.bfloat16).cpu()
        getattr(inner, name).register_forward_hook(hook)
    inner.sam_model.register_forward_pre_hook(lambda mod, args: store.__setitem__("_view", view(args[0])))
    inner.layers[0].register_forward_pre_hook(
        lambda mod, args, kwargs: store.__setitem__(
            "lm_in", (args[0] if args else kwargs["hidden_states"])[0].detach().to(torch.bfloat16).cpu()),
        with_kwargs=True)
    inner.norm.register_forward_hook(
        lambda mod, args, out: store.__setitem__("final_norm_last", out[0, -1].detach().to(torch.bfloat16).cpu()))

    saved = []
    for page, data in pngs.items():
        img = Image.open(io.BytesIO(data)).convert("RGB")
        t = collator.process_single_sample(to_conversation({"image": img, "text": ""})["messages"])
        n = t["prompt_token_count"]
        kwargs = dict(
            input_ids=t["input_ids"][:n].unsqueeze(0).cuda(),
            images=[(t["images_crop"].cuda(), t["images_ori"].cuda())],
            images_seq_mask=t["images_seq_mask"][:n].unsqueeze(0).cuda(),
            images_spatial_crop=t["images_spatial_crop"],
            use_cache=False,
        )
        for mode in ("autocast", "bf16"):
            store.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "autocast"):
                logits = model(**kwargs).logits
            store.pop("_view", None)
            store["logits_last"] = logits[0, -1].float().cpu()
            store["view_sep"] = inner.view_seperator.detach().to(torch.bfloat16).cpu()
            store["input_ids"] = t["input_ids"][:n].cpu()
            path = f"{OUT}/hf_{mode}_{page}.pt"
            torch.save(dict(store), path)
            saved.append(path)
    scratch.commit()
    return saved


@app.function(image=vllm_image, gpu="A100", timeout=30 * 60, volumes=VOLUMES)
def vllm_side(pngs: dict[int, bytes]) -> list[str]:
    import io
    import json
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # model in this process, so hooks see it
    import torch
    from PIL import Image
    from vllm import LLM, SamplingParams

    from cartridges.serve.recipe import OcrRecipe

    recipe = OcrRecipe(model_sha=SHA)
    llm = LLM(
        model=MERGED, trust_remote_code=True,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        mm_processor_kwargs=recipe.mm_processor_kwargs(),
        enable_prefix_caching=False, mm_processor_cache_gb=0, limit_mm_per_prompt={"image": 1},
        max_model_len=8192, max_num_seqs=1, enforce_eager=True, gpu_memory_utilization=0.85,
    )
    store: dict = {}

    def install(model):
        def view(x):
            return "global" if x.shape[-1] == 1024 else "local"

        model.sam_model.register_forward_pre_hook(lambda m, a: store.__setitem__("_view", view(a[0])))
        for name, key in (("sam_model", "sam"), ("qwen2_model", "qwen2"), ("projector", "proj")):
            def hook(mod, args, out, key=key):
                store[f"{key}/{store['_view']}"] = _first_tensor(out).detach().to(torch.bfloat16).cpu()
            getattr(model, name).register_forward_hook(hook)
        lm = model.language_model.model
        # vLLM layers: forward(positions, hidden_states, residual)
        lm.layers[0].register_forward_pre_hook(
            lambda m, a, k: store.__setitem__(
                "lm_in", (a[1] if len(a) > 1 else k["hidden_states"]).detach().to(torch.bfloat16).cpu()),
            with_kwargs=True)
        lm.norm.register_forward_hook(
            lambda m, a, out: store.__setitem__(
                "final_norm_last", _first_tensor(out)[-1].detach().to(torch.bfloat16).cpu()))
        return type(model).__name__

    print("hooked:", llm.apply_model(install))
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    saved = []
    for page, data in pngs.items():
        img = Image.open(io.BytesIO(data)).convert("RGB")
        store.clear()
        out = llm.chat(
            [{"role": "user", "content": [
                {"type": "image_pil", "image_pil": img},
                {"type": "text", "text": recipe.prompt}]}],
            params, use_tqdm=False,
        )[0]
        store.pop("_view", None)
        store["input_ids"] = torch.tensor(out.prompt_token_ids)
        store["view_sep"] = llm.apply_model(lambda m: m.view_seperator.detach().to(torch.bfloat16).cpu())[0]
        store["top_logprobs"] = json.dumps({int(k): v.logprob for k, v in out.outputs[0].logprobs[0].items()})
        path = f"{OUT}/vllm_{page}.pt"
        torch.save(dict(store), path)
        saved.append(path)
    scratch.commit()
    return saved


@app.function(image=hf_image, cpu=4, memory=16 * 1024, timeout=10 * 60, volumes=VOLUMES)
def compare(pages: list[int]) -> list[dict]:
    import json

    import torch

    scratch.reload()

    def stats(a, b) -> dict:
        a, b = a.float().flatten(), b.float().flatten()
        if a.shape != b.shape:
            return {"shape_mismatch": [list(a.shape), list(b.shape)]}
        d = (a - b).abs()
        return {
            "max_abs": round(d.max().item(), 5),
            "rel_l2": round(((a - b).norm() / a.norm().clamp_min(1e-12)).item(), 5),
            "cos": round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 6),
            "ref_absmax": round(a.abs().max().item(), 3),
        }

    rows = []
    for page in pages:
        v = torch.load(f"{OUT}/vllm_{page}.pt")
        for mode in ("autocast", "bf16"):
            h = torch.load(f"{OUT}/hf_{mode}_{page}.pt")
            row = {"page": page, "hf_mode": mode,
                   "input_ids_equal": torch.equal(h["input_ids"], v["input_ids"]),
                   "n_tokens": [len(h["input_ids"]), len(v["input_ids"])]}
            for key in ("sam/local", "sam/global", "qwen2/local", "qwen2/global", "proj/local",
                        "proj/global", "view_sep", "lm_in", "final_norm_last"):
                if key in h and key in v:
                    row[key] = stats(h[key], v[key])
                elif key in h or key in v:
                    row[key] = {"missing": "hf" if key not in h else "vllm"}
            if row["input_ids_equal"]:
                img = h["input_ids"] == 128815
                row["lm_in/image_pos"] = stats(h["lm_in"][img], v["lm_in"][img])
                row["lm_in/text_pos"] = stats(h["lm_in"][~img], v["lm_in"][~img])
            h_lp = torch.log_softmax(h["logits_last"].float(), -1)
            v_lp = json.loads(v["top_logprobs"])
            row["first_token_logprob_absdiff_top20"] = round(
                max(abs(h_lp[int(t)].item() - lp) for t, lp in v_lp.items()), 4)
            row["top1"] = [int(h_lp.argmax()), int(max(v_lp, key=v_lp.get))]
            rows.append(row)
    return rows


@app.local_entrypoint()
def main(pages: str = "63,52,44", out: str = "stage_parity.json"):
    import io
    import json

    from datasets import load_from_disk

    ids = [int(p) for p in pages.split(",")]
    test = load_from_disk("data/train/hf_dataset")["test"]
    pngs = {}
    for i in ids:
        buf = io.BytesIO()
        test[i]["image"].save(buf, format="PNG")
        pngs[i] = buf.getvalue()
    calls = [hf_side.spawn(pngs), vllm_side.spawn(pngs)]
    for c in calls:
        print(c.get())
    rows = compare.remote(ids)
    json.dump(rows, open(out, "w"), indent=1)
    print(json.dumps(rows, indent=1))
