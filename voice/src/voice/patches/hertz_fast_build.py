"""Build hertz-dev without random-initialising it: parameters on meta, checkpoint tensors assigned.

Upstream's constructors build every module on the CPU in fp32, random-initialise
it, then overwrite it with `load_state_dict` from inside `__init__`. That took
293 s of a ~352 s cold start (H200 container, #27). `build` runs the same
constructors with two changes, and only while they run:

- `nn.Module.register_parameter` puts each new parameter on the meta device:
  no storage, so its random init is free. Buffers and plain tensors (FSQ
  levels, pad buffers, RoPE tables) are built exactly as upstream builds them;
- `nn.Module.load_state_dict` passes `assign=True`, so the checkpoint's own
  tensors become the parameters instead of being copied into them.

The weights come from the same checkpoints through the same `load_state_dict`
calls. Upstream's later `.bfloat16()` gives the same bf16 values either way.
What could differ is a parameter that no checkpoint fills. Upstream would keep
its CPU random init; here it stays on meta, and `build` raises `MissingWeights`
naming it rather than inventing a value. Every load's missing and unexpected
keys are logged (three of upstream's loads are `strict=False`) and returned.

    model, loads = build(get_hertz_dev_config(is_split=True))   # HertzBackend.load does this
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from collections.abc import Callable

from torch import nn

logger = logging.getLogger(__name__)


class MissingWeights(RuntimeError):
    """Parameters or buffers that no checkpoint filled (still on meta after the build)."""


@contextlib.contextmanager
def empty_parameters():
    """Within: new parameters go on meta and `load_state_dict` assigns. Yields
    the list of loads, one {module, missing, unexpected} per call."""
    register, load = nn.Module.register_parameter, nn.Module.load_state_dict
    loads, loading = [], []

    def register_parameter(self, name, param):
        if param is not None and not param.is_meta and not loading:  # assign=True re-registers checkpoint tensors
            param = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
        register(self, name, param)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        loading.append(self)
        try:
            result = load(self, state_dict, strict=strict, assign=True)
        finally:
            loading.pop()
        loads.append({"module": type(self).__name__, "missing": list(result.missing_keys),
                      "unexpected": list(result.unexpected_keys)})
        return result

    nn.Module.register_parameter, nn.Module.load_state_dict = register_parameter, load_state_dict
    try:
        yield loads
    finally:
        nn.Module.register_parameter, nn.Module.load_state_dict = register, load


def build(make: Callable[[], nn.Module]) -> tuple[nn.Module, list[dict]]:
    """`make()` under `empty_parameters`; raises `MissingWeights` if any tensor is left on meta."""
    with empty_parameters() as loads:
        model = make()
    for r in loads:
        logger.info("load_state_dict %s: %d missing %s, %d unexpected %s", r["module"], len(r["missing"]),
                    r["missing"][:8], len(r["unexpected"]), r["unexpected"][:8])
    empty = [n for n, t in itertools.chain(model.named_parameters(), model.named_buffers()) if t.is_meta]
    if empty:
        raise MissingWeights(f"{len(empty)} tensors were in no checkpoint (upstream would random-init them): "
                             f"{empty[:10]}")
    return model, loads
