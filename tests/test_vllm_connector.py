"""CartridgeConnector (serve/vllm_connector.py) on CPU, against stand-ins for
vLLM's connector base classes (vLLM is not installed locally) and fake
requests, blocks and cache views shaped like vLLM 0.29's."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

P, KV_HEADS, HEAD_DIM = 64, 2, 4
BLOCK, KERNEL_BLOCK = 32, 16  # manager block split into 2 kernel blocks
NUM_BLOCKS = 6
CONV, SSM = (3, 5), (2, 3, 4)


@pytest.fixture
def connector_module(monkeypatch):
    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config):
            self._kv_cache_config, self._role, self._connector_metadata = kv_cache_config, role, None

        def bind_connector_metadata(self, meta):
            self._connector_metadata = meta

        def _get_connector_metadata(self):
            return self._connector_metadata

    class KVConnectorMetadata:
        pass

    class SupportsHMA:
        pass

    base.KVConnectorBase_V1, base.KVConnectorMetadata, base.SupportsHMA = KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA
    base.KVConnectorRole = types.SimpleNamespace(SCHEDULER=0, WORKER=1)
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
                 "vllm.distributed.kv_transfer.kv_connector.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.delitem(sys.modules, "cartridges.serve.vllm_connector", raising=False)
    yield importlib.import_module("cartridges.serve.vllm_connector")
    sys.modules.pop("cartridges.serve.vllm_connector", None)


ATTN = "language_model.model.layers.1.self_attn.attn"
GDN = "language_model.model.layers.0.linear_attn"


def kv_cache_config():
    """Two groups, as vLLM's HMA makes them: attention (group 0), GDN (group 1)."""
    return SimpleNamespace(num_blocks=NUM_BLOCKS, kv_cache_groups=[
        SimpleNamespace(layer_names=[ATTN], kv_cache_spec=SimpleNamespace(block_size=BLOCK)),
        SimpleNamespace(layer_names=[GDN], kv_cache_spec=SimpleNamespace(block_size=4096)),
    ])


def make(module, role):
    return module.CartridgeConnector(SimpleNamespace(), role, kv_cache_config())


def request(req_id="r1", **params):
    return SimpleNamespace(request_id=req_id, kv_transfer_params=params or None)


CART = {"cartridge_id": "g/n@abc", "path": "/cache/cartridges/g/n/vllm/state.safetensors", "p": P}


def test_scheduler_reports_the_cartridge_positions_as_external(connector_module):
    conn = make(connector_module, connector_module.KVConnectorRole.SCHEDULER)
    assert conn.get_num_new_matched_tokens(request(**CART), 0) == (P, False)
    assert conn.get_num_new_matched_tokens(request(**CART), 32) == (P - 32, False)
    assert conn.get_num_new_matched_tokens(request(**CART), P + 16) == (0, False)
    assert conn.get_num_new_matched_tokens(request(), 0) == (0, False)  # no cartridge: untouched
    assert conn.get_num_new_matched_tokens(request(other="x"), 0) == (0, False)


def test_scheduler_sends_each_load_once_with_its_block_ids(connector_module):
    conn = make(connector_module, connector_module.KVConnectorRole.SCHEDULER)
    blocks = SimpleNamespace(get_block_ids=lambda: ([4, 1, 5], [2]))
    conn.update_state_after_alloc(request("r1", **CART), blocks, P)
    conn.update_state_after_alloc(request("r2"), blocks, 0)  # nothing external: no load
    meta = conn.build_connector_meta(SimpleNamespace())
    assert meta.loads == [connector_module.Load("r1", CART["cartridge_id"], CART["path"], P, ([4, 1, 5], [2]))]
    assert conn.build_connector_meta(SimpleNamespace()).loads == []  # pending cleared
    assert conn.request_finished_all_groups(request("r1", **CART), ([4, 1, 5], [2])) == (False, None)


def test_attention_slots_split_manager_blocks_into_kernel_blocks(connector_module):
    blocks, offsets = connector_module.attention_slots([4, 1], 40, BLOCK, KERNEL_BLOCK)
    # position t: manager block ids[t // 32]; kernel block = id * 2 + (t % 32) // 16
    assert blocks[:16].tolist() == [8] * 16 and blocks[16:32].tolist() == [9] * 16
    assert blocks[32:40].tolist() == [2] * 8
    assert offsets.tolist() == list(range(16)) * 2 + list(range(8))


def state():
    g = torch.Generator().manual_seed(0)
    return {
        "layers.1.k": torch.randn(P, KV_HEADS, HEAD_DIM, generator=g).bfloat16(),
        "layers.1.v": torch.randn(P, KV_HEADS, HEAD_DIM, generator=g).bfloat16(),
        "layers.0.conv": torch.randn(*CONV, generator=g).bfloat16(),
        "layers.0.ssm": torch.randn(*SSM, generator=g),
    }


def forward_context():
    """Layers bound as vLLM binds them: attention to a [kernel blocks, heads,
    kernel block size, K|V] view, GDN to (conv, ssm) per state block."""
    attn = torch.zeros(NUM_BLOCKS * BLOCK // KERNEL_BLOCK, KV_HEADS, KERNEL_BLOCK, 2 * HEAD_DIM, dtype=torch.bfloat16)
    gdn = (torch.zeros(NUM_BLOCKS, *CONV, dtype=torch.bfloat16), torch.zeros(NUM_BLOCKS, *SSM))
    return SimpleNamespace(no_compile_layers={
        ATTN: SimpleNamespace(kv_cache=attn), GDN: SimpleNamespace(kv_cache=gdn), "mlp": SimpleNamespace(),
    })


def test_worker_writes_the_cartridge_into_the_requests_blocks(connector_module):
    conn = make(connector_module, connector_module.KVConnectorRole.WORKER)
    conn.store.items[CART["cartridge_id"]] = s = state()
    conn.bind_connector_metadata(connector_module.CartridgeMeta(
        [connector_module.Load("r1", CART["cartridge_id"], CART["path"], P, ([4, 1, 5], [2]))]))
    ctx = forward_context()
    conn.start_load_kv(ctx)

    attn = ctx.no_compile_layers[ATTN].kv_cache
    paged = torch.cat([attn[8], attn[9], attn[2], attn[3]], dim=1)  # manager blocks 4, 1 -> kernel blocks 8, 9, 2, 3
    assert torch.equal(paged[..., :HEAD_DIM].transpose(0, 1), s["layers.1.k"])
    assert torch.equal(paged[..., HEAD_DIM:].transpose(0, 1), s["layers.1.v"])
    assert attn[10:].abs().sum() == 0 and attn[0].abs().sum() == 0  # block 5 (beyond p) and others untouched
    conv, ssm = ctx.no_compile_layers[GDN].kv_cache
    assert torch.equal(conv[2], s["layers.0.conv"]) and torch.equal(ssm[2], s["layers.0.ssm"])
    assert conv[[0, 1, 3, 4, 5]].abs().sum() == 0


def test_worker_refuses_a_state_that_does_not_fit(connector_module):
    conn = make(connector_module, connector_module.KVConnectorRole.WORKER)
    bad = state()
    bad["layers.0.ssm"] = bad["layers.0.ssm"].bfloat16()  # vLLM keeps it in fp32 here
    conn.store.items[CART["cartridge_id"]] = bad
    conn.bind_connector_metadata(connector_module.CartridgeMeta(
        [connector_module.Load("r1", CART["cartridge_id"], CART["path"], P, ([4, 1], [2]))]))
    with pytest.raises(ValueError, match="does not fit"):
        conn.start_load_kv(forward_context())


def test_store_keeps_the_most_recently_used(connector_module, tmp_path):
    from safetensors.torch import save_file

    store = connector_module.CartridgeStore(max_items=2, device="cpu")
    paths = {}
    for name in "abc":
        paths[name] = str(tmp_path / f"{name}.safetensors")
        save_file({"x": torch.zeros(1)}, paths[name])
    store.get("a", paths["a"])
    store.get("b", paths["b"])
    store.get("a", paths["a"])  # hit: a becomes the most recent
    store.get("c", paths["c"])  # evicts b
    assert list(store.items) == ["a", "c"]


def test_layer_index_parses_vllm_layer_names(connector_module):
    assert connector_module.layer_index(ATTN) == 1
    assert connector_module.layer_index("model.layers.47.linear_attn") == 47
