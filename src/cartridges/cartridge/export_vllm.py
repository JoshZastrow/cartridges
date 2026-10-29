"""Export a cartridge for the vLLM CartridgeEngine (serve/vllm_cartridge.py).

The engine never reads `cartridge.safetensors` itself: an explicit export puts
each tensor in the layout and dtype of vLLM 0.29's caches, so the connector
(serve/vllm_connector.py) only copies them into the request's blocks:

- full_attention layer i: `layers.{i}.k` / `layers.{i}.v` [p, kv_heads, head_dim]
  in the model dtype (the frozen sink and trainable positions merged), still
  post-RoPE at positions 0..p-1;
- linear_attention layer i: `layers.{i}.conv` [kernel-1, conv_dim] (vLLM keeps
  the last kernel-1 inputs, "SD" layout; HF keeps all kernel columns, and the
  oldest one never reaches an output) in the model dtype, and `layers.{i}.ssm`
  [v_heads, v_dim, k_dim] (HF's [v_heads, k_dim, v_dim] transposed) in the
  config's mamba_ssm_dtype.

Written to `<cartridge>/vllm/{state.safetensors,manifest.json}`. The manifest's
cartridge_id ends in the state file's sha256, so a re-exported (retrained)
cartridge never hits a stale copy in the engine's GPU store.

    uv run cartridges cartridge_export_vllm --cartridge /vol/cartridges/<group>/<name>
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from cartridges.cartridge.train_distill import MINUTES, VOLUME_PATH, image, secrets, volume

if TYPE_CHECKING:
    import torch

    from cartridges.cartridge.cache import Cartridge

EXPORT_DIR = "vllm"
STATE = "state.safetensors"
MANIFEST = "manifest.json"
VLLM_VERSION = "0.29.0"  # teacher.image's pin; the engine refuses exports for another version
CONV_LAYOUT = "SD"  # vLLM 0.29's default VLLM_SSM_CONV_STATE_LAYOUT

app = modal.App("cartridges-export-vllm", image=image)


def state_dict(cart: Cartridge, text_config) -> dict[str, torch.Tensor]:
    """The cartridge's tensors in vLLM's per-layer layout (module docstring),
    checked against the base model's text config."""
    import torch

    c = text_config
    if list(c.layer_types) != cart.layer_types:
        raise ValueError(f"layer_types of the cartridge do not match {cart.meta.get('base_model')}")
    dtype = cart.dtype_
    ssm_dtype = getattr(torch, getattr(c, "mamba_ssm_dtype", None) or str(dtype).removeprefix("torch."))
    kernel = c.linear_conv_kernel_dim
    conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
    expected = {
        "k": (cart.p, c.num_key_value_heads, c.head_dim),
        "conv": (kernel - 1, conv_dim),
        "ssm": (c.linear_num_value_heads, c.linear_value_head_dim, c.linear_key_head_dim),
    }
    expected["v"] = expected["k"]
    out = {}
    for j, i in enumerate(cart.attn_layers):
        for name, frozen, trainable in (("k", cart.frozen_keys, cart.trainable_keys),
                                        ("v", cart.frozen_values, cart.trainable_values)):
            full = torch.cat([frozen[j], trainable[j]], dim=2)[0]  # [kv_heads, p, head_dim]
            out[f"layers.{i}.{name}"] = full.transpose(0, 1).to(dtype)
    for j, i in enumerate(cart.linear_layers):
        if cart.conv_states[j].shape[-1] != kernel:
            raise ValueError(f"layer {i}: conv state width {cart.conv_states[j].shape[-1]}, expected kernel {kernel}")
        out[f"layers.{i}.conv"] = cart.conv_states[j][0, :, -(kernel - 1):].T.to(dtype)
        out[f"layers.{i}.ssm"] = cart.recurrent_states[j][0].transpose(-1, -2).to(ssm_dtype)
    for key, t in out.items():
        if tuple(t.shape) != expected[key.rsplit(".", 1)[1]]:
            raise ValueError(f"{key}: shape {tuple(t.shape)}, expected {expected[key.rsplit('.', 1)[1]]}")
    return {k: t.detach().contiguous() for k, t in out.items()}


def export(cartridge: str | Path, text_config, revision: str = "main") -> dict:
    """Write `<cartridge>/vllm/` and return its manifest."""
    from safetensors.torch import save_file

    from cartridges.cartridge.cache import Cartridge

    path = Path(cartridge)
    cart = Cartridge.load(path)
    out = path / EXPORT_DIR
    out.mkdir(exist_ok=True)
    tensors = state_dict(cart, text_config)
    save_file(tensors, str(out / STATE))
    sha = hashlib.sha256((out / STATE).read_bytes()).hexdigest()
    manifest = {
        "cartridge_id": f"{path.parent.name}/{path.name}@{sha[:12]}",
        "sha256": sha,
        "base_model": cart.meta["base_model"],
        "revision": revision,
        "vllm_version": VLLM_VERSION,
        "p": cart.p,
        "conv_layout": CONV_LAYOUT,
        "layer_types": cart.layer_types,
        "tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")} for k, t in tensors.items()},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(cartridge: str | Path) -> dict:
    """The export's manifest; FileNotFoundError when the cartridge was never exported."""
    return json.loads((Path(cartridge) / EXPORT_DIR / MANIFEST).read_text())


@app.function(cpu=2, memory=8192, volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=10 * MINUTES)
def export_remote(cartridge: str, revision: str = "main") -> dict:
    """`export` on the volume (cartridge: /vol/cartridges/<group>/<name>)."""
    from transformers import AutoConfig

    volume.reload()
    base_model = json.loads((Path(cartridge) / "cartridge.json").read_text())["base_model"]
    text_config = AutoConfig.from_pretrained(base_model, revision=revision).get_text_config()
    manifest = export(cartridge, text_config, revision)
    volume.commit()
    return {k: v for k, v in manifest.items() if k not in ("tensors", "layer_types")}


def run(cartridge: str, revision: str = "main") -> dict:
    with app.run():
        return export_remote.remote(cartridge, revision)
