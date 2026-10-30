"""Run hertz-dev's attention on the flash / memory-efficient SDPA kernels.

On torch 2.14 (H200, bf16) `F.scaled_dot_product_attention` in upstream's
`transformer.GQA._sdpa` picks a slower backend by default: the live step took
142 ms against 58 ms with flash/efficient forced (lm 65 -> 25 ms, resynthesizer
50 -> 12 ms; per-stage, CUDA-synced; measured on #21). torch 2.5.1 runs the same
code at 109 ms. The patch wraps `_sdpa` in `sdpa_kernel([FLASH, EFFICIENT])`;
the math is unchanged, only the kernel differs (bf16-level numerics, checked
against 2.5.1 on #21). Upstream's tree is not edited.

    apply()   # after `transformer` is importable (HertzBackend.load does this)
"""

from __future__ import annotations

import functools

BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION")


def apply() -> None:
    import transformer  # upstream hertz-dev module, on sys.path from HERTZ_DEV_DIR
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sdpa = transformer.GQA._sdpa
    if getattr(sdpa, "_voice_patched", False):
        return
    kernels = [getattr(SDPBackend, b) for b in BACKENDS]

    @functools.wraps(sdpa)
    def patched(self, q, k, v):
        with sdpa_kernel(kernels):
            return sdpa(self, q, k, v)

    patched._voice_patched = True
    transformer.GQA._sdpa = patched
