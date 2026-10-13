"""Property test: the eval path (finetune._generate_one) gives the model the
same inputs as training (DeepSeekOCRDataCollator).

History: eval used model.infer(), which strips the prompt and dropped the
trailing space token (223) of "<image>\nFree OCR. ". This test found it.

Property: for each image, the eval path and the collator make the same prompt token
ids, the same image-token mask, the same crop grid, and bit-identical bf16
pixel tensors (global view and tiles).

The test calls the real _generate_one() with a stub model whose generate()
records its inputs and stops. Thus no weights load.

The model code needs the training image (transformers 4.56, the model files
on the volume) and a GPU (the inputs go to .cuda()). The test runs on Modal; set
CARTRIDGES_MODAL_TESTS=1 to run it:

CARTRIDGES_MODAL_TESTS=1 uv run pytest tests/test_eval_preprocessing.py -s
"""

import os

import pytest

from eval_preprocessing_modal import app, compare_inputs


@pytest.mark.skipif(os.environ.get("CARTRIDGES_MODAL_TESTS") != "1", reason="needs Modal GPU; set CARTRIDGES_MODAL_TESTS=1")
def test_eval_inputs_match_training_inputs():
    with app.run():
        result = compare_inputs.remote()
    print("cases:", result["n_cases"], "| eval generate kwargs:", result["generate_kwargs"])
    for field, rows in result["mismatches"].items():
        print(f"{field}: {len(rows)} mismatches", rows[:2] if rows else "")
    failed = {f: len(rows) for f, rows in result["mismatches"].items() if rows}
    assert not failed, f"eval inputs differ from training inputs: {failed}"
