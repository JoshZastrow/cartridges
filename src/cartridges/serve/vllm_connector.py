"""CartridgeConnector: a vLLM 0.29 KV connector that puts a cartridge in place
of a request's first `p` (placeholder) tokens.

A request that carries `kv_transfer_params = {"cartridge_id", "path", "p"}`
(path: the export's state.safetensors)
(SamplingParams.extra_args, which vLLM copies to Request.kv_transfer_params)
reports its first p tokens as externally computed, so vLLM never runs them
through the model. Before the step's forward, the worker copies the export
(cartridge/export_vllm.py) into the request's blocks:

- attention layers: K/V of positions 0..p-1 into the paged cache. vLLM 0.29
  binds each attention layer to a [kernel blocks, kv_heads, kernel block size,
  K|V] view; a scheduler (manager) block b covers kernel blocks
  b*ratio .. b*ratio+ratio-1 (worker/block_table.py).
- linear-attention (GDN) layers: conv/ssm state into the request's state block,
  block_ids[group][0], which the GDN kernels read as the initial state since
  the request has num_computed_tokens > 0 (mamba_cache_mode "none").

Loading is synchronous (HBM -> HBM, enqueued on the forward's stream), so the
layer-wise hooks are no-ops. Consumer only: nothing is saved. Requests without
a cartridge_id are untouched. There is no recompute fallback: vLLM's
load-error path does not support hybrid (HMA) block layouts in 0.29, so a
cartridge that cannot be loaded raises, and the engine checks the export
before submitting (vllm_cartridge.CartridgeEngine).

Prefix caching must be off (v1): placeholder blocks of different cartridges
would hash alike.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger

logger = init_logger(__name__)  # vLLM's handler: the worker process has no other

STORE_SIZE = 8  # cartridges kept on the GPU


@dataclass
class Load:
    req_id: str
    cartridge_id: str
    path: str  # the export's state.safetensors, as the engine container sees it
    p: int
    block_ids: tuple[list[int], ...]  # one list per KV cache group


@dataclass
class CartridgeMeta(KVConnectorMetadata):
    loads: list[Load] = field(default_factory=list)


class CartridgeStore:
    """Exports on the GPU, least recently used evicted past `max_items`."""

    def __init__(self, max_items: int = STORE_SIZE, device: str = "cuda"):
        self.max_items, self.device = max_items, device
        self.items: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def get(self, cartridge_id: str, path: str) -> dict[str, torch.Tensor]:
        if cartridge_id in self.items:
            self.items.move_to_end(cartridge_id)
            return self.items[cartridge_id]
        from safetensors.torch import load_file

        t0 = time.perf_counter()
        state = load_file(path, device=self.device)
        logger.info("cartridge %s loaded in %.0f ms", cartridge_id, 1000 * (time.perf_counter() - t0))
        self.items[cartridge_id] = state
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return state


def layer_index(layer_name: str) -> int:
    """`language_model.model.layers.12.self_attn.attn` -> 12."""
    parts = layer_name.split(".")
    return int(parts[parts.index("layers") + 1])


def attention_slots(block_ids: list[int], p: int, block_size: int, kernel_block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(kernel block, offset) of positions 0..p-1 for a request's manager blocks."""
    pos = torch.arange(p)
    ratio = block_size // kernel_block_size
    blocks = torch.tensor(block_ids)[pos // block_size] * ratio + (pos % block_size) // kernel_block_size
    return blocks, pos % kernel_block_size


def write_attention(cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_ids: list[int], block_size: int) -> None:
    """K/V [p, kv_heads, head_dim] into a [kernel blocks, kv_heads, kernel block size, K|V] cache view."""
    blocks, offsets = attention_slots(block_ids, k.shape[0], block_size, cache.shape[2])
    kv = torch.cat([k, v], dim=-1)
    if cache.shape[1] != kv.shape[1] or cache.shape[3] != kv.shape[2] or cache.dtype != kv.dtype:
        raise ValueError(f"cartridge K|V {tuple(kv.shape)} {kv.dtype} does not fit cache {tuple(cache.shape)} {cache.dtype}")
    cache[blocks.to(cache.device), :, offsets.to(cache.device)] = kv


def write_state(dst: torch.Tensor, block: int, src: torch.Tensor) -> None:
    """One GDN state (conv or ssm) into its state block."""
    if dst.shape[1:] != src.shape or dst.dtype != src.dtype:
        raise ValueError(f"cartridge state {tuple(src.shape)} {src.dtype} does not fit {tuple(dst.shape[1:])} {dst.dtype}")
    dst[block].copy_(src)


class CartridgeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.pending: dict[str, Load] = {}  # scheduler side
        self.store = CartridgeStore() if role == KVConnectorRole.WORKER else None
        # layer name -> (KV cache group, manager block size)
        self.groups = {
            name: (g, group.kv_cache_spec.block_size)
            for g, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }

    # ---- scheduler side ----

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params or {}
        if "cartridge_id" not in params:
            return 0, False
        return max(0, params["p"] - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int) -> None:
        if num_external_tokens:
            params = request.kv_transfer_params
            self.pending[request.request_id] = Load(
                request.request_id, params["cartridge_id"], params["path"], params["p"], blocks.get_block_ids()
            )

    def build_connector_meta(self, scheduler_output) -> CartridgeMeta:
        meta, self.pending = CartridgeMeta(list(self.pending.values())), {}
        return meta

    def request_finished_all_groups(self, request, block_ids) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    # ---- worker side ----

    def start_load_kv(self, forward_context, **kwargs) -> None:
        for load in self._get_connector_metadata().loads:
            state = self.store.get(load.cartridge_id, load.path)
            for name, layer in forward_context.no_compile_layers.items():
                if name not in self.groups:
                    continue
                group, block_size = self.groups[name]
                i = layer_index(name)
                if isinstance(layer.kv_cache, tuple):  # GDN: (conv, ssm)
                    block = load.block_ids[group][0]
                    write_state(layer.kv_cache[0], block, state[f"layers.{i}.conv"])
                    write_state(layer.kv_cache[1], block, state[f"layers.{i}.ssm"])
                else:
                    write_attention(layer.kv_cache, state[f"layers.{i}.k"], state[f"layers.{i}.v"],
                                    load.block_ids[group], block_size)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        pass
