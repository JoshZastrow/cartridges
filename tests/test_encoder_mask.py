"""The vision encoder gets its design attention mask (encoder_mask.py).

Design mask over [144 image tokens, 144 query tokens]: image tokens attend to
all image tokens; query tokens attend to all image tokens and causally to
query tokens. Without the patch, transformers >= 4.53 gives a plain causal
mask (docs/hf-vllm-parity.md).

Runs on Modal (the model code is on the volume). Set CARTRIDGES_MODAL_TESTS=1:

CARTRIDGES_MODAL_TESTS=1 uv run pytest tests/test_encoder_mask.py -s
"""

import os

import pytest

N = 144


def design_mask() -> list[list[bool]]:
    def allow(q: int, k: int) -> bool:
        if k < N:
            return True  # every token sees all image tokens
        return q >= N and k <= q  # query tokens: causal among queries
    return [[allow(q, k) for k in range(2 * N)] for q in range(2 * N)]


def test_design_mask_shape():
    mask = design_mask()
    assert mask[0][N - 1] and not mask[0][N]  # image token: all images, no queries
    assert mask[N][N] and not mask[N][N + 1]  # first query: itself, not later queries
    assert all(mask[2 * N - 1])  # last query: everything


@pytest.mark.skipif(os.environ.get("CARTRIDGES_MODAL_TESTS") != "1", reason="needs Modal; set CARTRIDGES_MODAL_TESTS=1")
def test_patched_encoder_uses_the_design_mask():
    from encoder_mask_modal import app, captured_masks

    with app.run():
        result = captured_masks.remote()
    print("transformers", result["transformers"], "patched", result["n_patched"])
    expected = design_mask()
    assert result["before"] != expected, "unpatched encoder already has the design mask: is the patch still needed?"
    assert result["n_patched"] == 1 and result["n_patched_again"] == 0
    assert result["after"] == expected
    assert result["output_changed"]
