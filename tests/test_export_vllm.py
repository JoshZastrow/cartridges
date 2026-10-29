"""The vLLM export of a cartridge (cartridge/export_vllm.py) and the engine's
checks of it (serve/vllm_cartridge.py), on a tiny CPU cartridge."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from cartridges.cartridge import export_vllm
from cartridges.cartridge.cache import Cartridge
from cartridges.serve import vllm_cartridge

TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
P, KV_HEADS, HEAD_DIM = 4, 2, 8
K_HEADS, K_DIM, V_HEADS, V_DIM, KERNEL = 1, 4, 2, 3, 4
CONV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM


def text_config(**overrides):
    return SimpleNamespace(**{
        "layer_types": TYPES, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "linear_num_key_heads": K_HEADS, "linear_key_head_dim": K_DIM, "linear_num_value_heads": V_HEADS,
        "linear_value_head_dim": V_DIM, "linear_conv_kernel_dim": KERNEL, "mamba_ssm_dtype": "float32",
        **overrides,
    })


@pytest.fixture
def cartridge_dir(tmp_path):
    g = torch.Generator().manual_seed(0)
    cart = Cartridge(
        TYPES,
        keys=[torch.randn(1, KV_HEADS, P, HEAD_DIM, generator=g) for _ in range(2)],
        values=[torch.randn(1, KV_HEADS, P, HEAD_DIM, generator=g) for _ in range(2)],
        conv_states=[torch.randn(1, CONV_DIM, KERNEL, generator=g) for _ in range(2)],
        recurrent_states=[torch.randn(1, V_HEADS, K_DIM, V_DIM, generator=g) for _ in range(2)],
        meta={"base_model": "tiny/model", "group": "exp-20261028-test"},
    )
    return cart.save(tmp_path / "exp-20261028-test" / "best")


def test_export_writes_vllm_layouts(cartridge_dir):
    cart = Cartridge.load(cartridge_dir)
    manifest = export_vllm.export(cartridge_dir, text_config())
    state = load_file(str(cartridge_dir / "vllm" / "state.safetensors"))

    assert sorted(state) == sorted([f"layers.{i}.{n}" for i in (1, 3) for n in "kv"]
                                   + [f"layers.{i}.{n}" for i in (0, 2) for n in ("conv", "ssm")])
    keys = torch.cat([cart.frozen_keys[1], cart.trainable_keys[1]], dim=2)[0]  # attention layer 3
    assert state["layers.3.k"].shape == (P, KV_HEADS, HEAD_DIM) and state["layers.3.k"].dtype == torch.bfloat16
    assert torch.equal(state["layers.3.k"], keys.transpose(0, 1).bfloat16())
    assert torch.equal(state["layers.1.v"][0], cart.frozen_values[0][0, :, 0].bfloat16())  # the sink, position 0
    # conv: the last kernel-1 inputs, [kernel-1, conv_dim] ("SD"), model dtype
    assert torch.equal(state["layers.2.conv"], cart.conv_states[1][0, :, 1:].T.bfloat16())
    # ssm: [v_heads, v_dim, k_dim], mamba_ssm_dtype
    assert state["layers.0.ssm"].dtype == torch.float32
    assert torch.equal(state["layers.0.ssm"], cart.recurrent_states[0][0].transpose(-1, -2))

    assert manifest == json.loads((cartridge_dir / "vllm" / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256((cartridge_dir / "vllm" / "state.safetensors").read_bytes()).hexdigest()
    assert manifest["cartridge_id"] == f"exp-20261028-test/best@{manifest['sha256'][:12]}"
    assert (manifest["p"], manifest["base_model"], manifest["vllm_version"]) == (P, "tiny/model", "0.29.0")
    assert manifest["tensors"]["layers.0.conv"] == {"shape": [KERNEL - 1, CONV_DIM], "dtype": "bfloat16"}


def test_ssm_dtype_defaults_to_the_model_dtype(cartridge_dir):
    export_vllm.export(cartridge_dir, text_config(mamba_ssm_dtype=None))
    assert load_file(str(cartridge_dir / "vllm" / "state.safetensors"))["layers.0.ssm"].dtype == torch.bfloat16


def test_a_retrained_cartridge_gets_a_new_id(cartridge_dir):
    first = export_vllm.export(cartridge_dir, text_config())["cartridge_id"]
    cart = Cartridge.load(cartridge_dir)
    with torch.no_grad():
        cart.conv_states[0].add_(1.0)
    cart.save(cartridge_dir)
    assert export_vllm.export(cartridge_dir, text_config())["cartridge_id"] != first


@pytest.mark.parametrize("overrides", [
    {"layer_types": TYPES[::-1]},
    {"num_key_value_heads": KV_HEADS + 1},
    {"linear_value_head_dim": V_DIM + 1},
    {"linear_conv_kernel_dim": KERNEL + 1},
])
def test_export_rejects_a_mismatched_model(cartridge_dir, overrides):
    with pytest.raises(ValueError):
        export_vllm.export(cartridge_dir, text_config(**overrides))
    assert not (cartridge_dir / "vllm" / "manifest.json").exists()


def test_engine_refuses_other_models_and_versions(cartridge_dir):
    manifest = export_vllm.export(cartridge_dir, text_config())
    vllm_cartridge.check_manifest(manifest, "tiny/model", "0.29.0")
    with pytest.raises(ValueError, match="is for tiny/model"):
        vllm_cartridge.check_manifest(manifest, "Qwen/Qwen3.8-27B", "0.29.0")
    with pytest.raises(ValueError, match="re-export"):
        vllm_cartridge.check_manifest(manifest, "tiny/model", "0.30.0")


def test_engine_refuses_an_unexported_cartridge(cartridge_dir, monkeypatch):
    reloads = []
    monkeypatch.setattr(vllm_cartridge.teacher.volume, "reload", lambda: reloads.append(1))
    monkeypatch.setattr(vllm_cartridge, "engine_path", lambda c: c)
    with pytest.raises(ValueError, match="cartridge_export_vllm"):
        vllm_cartridge.transfer_params(str(cartridge_dir), "tiny/model", "0.29.0")
    assert reloads == [1]  # looked again after a volume reload

    manifest = export_vllm.export(cartridge_dir, text_config())
    assert vllm_cartridge.transfer_params(str(cartridge_dir), "tiny/model", "0.29.0") == {
        "cartridge_id": manifest["cartridge_id"], "path": str(cartridge_dir / "vllm" / "state.safetensors"), "p": P,
    }


def test_engine_path_maps_the_training_mount():
    assert vllm_cartridge.engine_path("/vol/cartridges/g/n") == "/cache/cartridges/g/n"
