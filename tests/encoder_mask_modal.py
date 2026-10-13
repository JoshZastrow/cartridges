"""Modal side of tests/test_encoder_mask.py. This module must not import
pytest: the container imports it, and the training image has no pytest."""

import modal

from cartridges.training.finetune import image as _training_image

app = modal.App("cartridges-encoder-mask-test")
model_cache = modal.Volume.from_name("cartridges-training-cache")


@app.function(
    image=_training_image.add_local_python_source("cartridges"),
    cpu=4,
    memory=16 * 1024,
    timeout=15 * 60,
    volumes={"/cache": model_cache},
)
def captured_masks() -> dict:
    """Build the model's own Qwen2 encoder (random weights), run one tile,
    and return which (query, key) pairs the layer-0 attention mask allows,
    before and after patch_encoder_mask()."""
    import sys

    import torch
    import transformers

    sys.path.insert(0, "/cache")
    from deepseek_ocr.deepencoderv2 import build_qwen2_decoder_as_encoder

    from cartridges.training.encoder_mask import ENCODER_CLASS, patch_encoder_mask

    encoder = build_qwen2_decoder_as_encoder().eval()
    inner = next(m for m in encoder.modules() if type(m).__name__ == ENCODER_CLASS)
    seen = {}
    inner.layers[0].register_forward_pre_hook(
        lambda m, args, kwargs: seen.__setitem__("mask", kwargs.get("attention_mask")), with_kwargs=True
    )
    x = torch.randn(1, 896, 12, 12)  # SAM output for one 768 px tile: 144 image tokens

    def allowed():
        with torch.no_grad():
            out = encoder(x)
        mask = seen["mask"]
        return (None if mask is None else (mask[0, 0] == 0).tolist()), out

    before, out_before = allowed()
    n_patched = patch_encoder_mask(encoder)
    after, out_after = allowed()
    return {
        "transformers": transformers.__version__,
        "n_patched": n_patched,
        "n_patched_again": patch_encoder_mask(encoder),
        "before": before,
        "after": after,
        "output_changed": not torch.equal(out_before, out_after),
    }
