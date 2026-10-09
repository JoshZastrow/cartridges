"""Merge a LoRA adapter into the base weights, for vLLM serving.

Why merge: the adapter targets q/k/v/o/gate/up/down_proj, and these names
can also occur in the vision encoder. vLLM LoRA support for vision layers is
not certain. A merged model has no such risk, and it is faster to serve.

The merge is a pure tensor operation on CPU: W += scale * (B @ A), with
scale = lora_alpha / r (or lora_alpha / sqrt(r) for rsLoRA). It does not
load unsloth or the model code. It reads and writes safetensors only, so the
base key names stay the same.

Run with:
uv run cartridges model_merge --revision prod
"""

from __future__ import annotations

import datetime
import json
import math

import modal

from cartridges.serve.registry import HF_REPO_ID

MINUTES = 60
BASE_MODEL_DIR = "/cache/deepseek_ocr"
MODELS_DIR = "/cache/models"
MARKER = "merged.json"

_PEFT_PREFIX = "base_model.model."
_LORA_A = ".lora_A.weight"
_LORA_B = ".lora_B.weight"


def lora_scale(adapter_config: dict) -> float:
    r = adapter_config["r"]
    alpha = adapter_config["lora_alpha"]
    return alpha / math.sqrt(r) if adapter_config.get("use_rslora") else alpha / r


def lora_merge_plan(adapter_keys: list[str], base_keys: set[str]) -> dict[str, tuple[str, str]]:
    """Map each base weight key to its (lora_A key, lora_B key).

    PEFT saves 'base_model.model.<module>.lora_A.weight'. The base weight is
    '<module>.weight'. Raise ValueError if an adapter key is not a LoRA
    A/B pair, or if a module has no base weight. A silent skip would serve a
    model that is only partly fine-tuned.
    """
    plan: dict[str, tuple[str, str]] = {}
    unknown = []
    for key in adapter_keys:
        if key.endswith(_LORA_A):
            module = key.removeprefix(_PEFT_PREFIX).removesuffix(_LORA_A)
            plan[f"{module}.weight"] = (key, key.removesuffix(_LORA_A) + _LORA_B)
        elif not key.endswith(_LORA_B):
            unknown.append(key)
    if unknown:
        raise ValueError(f"adapter has non-LoRA keys (not supported): {unknown[:5]}")
    adapter_set = set(adapter_keys)
    missing_b = [b for _, b in plan.values() if b not in adapter_set]
    if missing_b:
        raise ValueError(f"lora_A without lora_B: {missing_b[:5]}")
    no_base = [k for k in plan if k not in base_keys]
    if no_base:
        raise ValueError(f"adapter modules with no base weight: {no_base[:5]}")
    return plan


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    # safetensors' torch save_file goes through numpy.
    .pip_install("safetensors", "numpy", "huggingface_hub>=0.34.0", "pydantic")
    .add_local_python_source("cartridges")
)
app = modal.App("cartridges-merge", image=image)
model_cache = modal.Volume.from_name("cartridges-training-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("HF_TOKEN")


@app.function(
    cpu=8,
    memory=48 * 1024,
    timeout=60 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
def merge_adapter(sha: str, repo_id: str = HF_REPO_ID, overwrite: bool = False) -> dict:
    """Write the merged model to /cache/models/<sha>/. Skip the work if the
    marker file is there (unless overwrite=True)."""
    import shutil
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_dir = Path(BASE_MODEL_DIR)
    out_dir = Path(MODELS_DIR) / sha
    marker = out_dir / MARKER
    if marker.exists() and not overwrite:
        return json.loads(marker.read_text())

    adapter_dir = Path(
        snapshot_download(
            repo_id,
            revision=sha,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
    )
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = lora_scale(adapter_config)
    adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))

    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {base_dir}; run training download_model first")
    shard_keys = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            shard_keys[shard.name] = set(f.keys())
    plan = lora_merge_plan(list(adapter), set().union(*shard_keys.values()))

    tmp_dir = out_dir.with_name(f"{sha}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # Copy config, tokenizer, and trust_remote_code files; weights are
    # written below.
    shutil.copytree(base_dir, tmp_dir, ignore=shutil.ignore_patterns("*.safetensors"))

    merged = 0
    for shard in shards:
        tensors = load_file(str(shard))
        for base_key in shard_keys[shard.name] & plan.keys():
            a_key, b_key = plan[base_key]
            weight = tensors[base_key]
            delta = adapter[b_key].float() @ adapter[a_key].float()
            tensors[base_key] = (weight.float() + scale * delta).to(weight.dtype)
            merged += 1
        save_file(tensors, str(tmp_dir / shard.name), metadata={"format": "pt"})
        del tensors

    if merged != len(plan):
        raise RuntimeError(f"merged {merged} of {len(plan)} modules")
    record = {
        "sha": sha,
        "repo_id": repo_id,
        "scale": scale,
        "merged_modules": merged,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    (tmp_dir / MARKER).write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    model_cache.commit()
    return record
