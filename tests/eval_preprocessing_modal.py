"""Modal side of tests/test_eval_preprocessing.py. This module must not import
pytest: the container imports it, and the training image has no pytest."""

import modal

from cartridges.training.finetune import image as _training_image

app = modal.App("cartridges-eval-preprocessing-test")
model_cache = modal.Volume.from_name("cartridges-training-cache")
FIELDS = ("prompt_ids", "seq_mask", "grid", "global", "crops")


def _cases(seed: int, n_random: int) -> list[tuple[str, int, int, str]]:
    """(name, width, height, mode). Edge cases at the 768 crop threshold and
    at extreme aspect ratios, then random sizes and modes."""
    import random

    cases = [
        ("small", 300, 200, "RGB"),
        ("at-threshold", 768, 768, "RGB"),
        ("over-width", 769, 768, "RGB"),
        ("over-height", 768, 769, "RGB"),
        ("tall-strip", 400, 3000, "RGB"),
        ("wide-strip", 3000, 400, "RGB"),
        ("letter-250dpi", 2125, 2750, "RGB"),
        ("gray", 1700, 2200, "L"),
        ("alpha", 1200, 1600, "RGBA"),
        ("palette", 1000, 1300, "P"),
    ]
    rng = random.Random(seed)
    for i in range(n_random):
        cases.append(
            (f"random-{i}", rng.randint(32, 3200), rng.randint(32, 3200), rng.choice(["RGB", "L", "RGBA", "P"]))
        )
    return cases


@app.function(
    image=_training_image.add_local_python_source("cartridges"),
    gpu=["L4", "A10G", "A100"],
    timeout=30 * 60,
    volumes={"/cache": model_cache},
)
def compare_inputs(seed: int = 0, n_random: int = 40, n_dataset: int = 20) -> dict:
    import hashlib
    import random
    import sys
    from types import SimpleNamespace

    import numpy as np
    import torch
    from datasets import load_from_disk
    from PIL import Image
    from transformers import AutoTokenizer

    sys.path.insert(0, "/cache")
    from cartridges.training.collator import DeepSeekOCRDataCollator, to_conversation
    from cartridges.training.finetune import _generate_one

    tokenizer = AutoTokenizer.from_pretrained("/cache/deepseek_ocr", trust_remote_code=True)
    collator = DeepSeekOCRDataCollator(tokenizer, SimpleNamespace(dtype=torch.bfloat16))

    def digest(t) -> str:
        t = t.to(torch.bfloat16).cpu().contiguous()
        return hashlib.sha256(t.view(torch.int16).numpy().tobytes()).hexdigest()[:16] + str(list(t.shape))

    class Captured(Exception):
        pass

    class Stub:
        def disable_torch_init(self):
            pass

        def generate(self, input_ids, images, images_seq_mask, images_spatial_crop, **kwargs):
            raise Captured(input_ids, images, images_seq_mask, images_spatial_crop, kwargs)

    def eval_side(img: Image.Image) -> tuple[dict, dict]:
        # The eval path: finetune._generate_one, with a stub model.
        try:
            _generate_one(Stub(), tokenizer, collator, img)
        except Captured as c:
            input_ids, images, seq_mask, grid, kwargs = c.args
        else:
            raise AssertionError("_generate_one() did not call generate()")
        crops, ori = images[0]
        cropped = grid.tolist() != [[1, 1]]
        return {
            "prompt_ids": input_ids[0].tolist(),
            "seq_mask": seq_mask[0].tolist(),
            "grid": grid.tolist(),
            "global": digest(ori),
            "crops": digest(crops) if cropped else None,
        }, {k: v for k, v in kwargs.items() if k != "streamer"}

    def train_side(img: Image.Image) -> dict:
        t = collator.process_single_sample(to_conversation({"image": img, "text": "x"})["messages"])
        n = t["prompt_token_count"]
        cropped = t["images_spatial_crop"].tolist() != [[1, 1]]
        return {
            "prompt_ids": t["input_ids"][:n].tolist(),
            "seq_mask": t["images_seq_mask"][:n].tolist(),
            "grid": t["images_spatial_crop"].tolist(),
            "global": digest(t["images_ori"]),
            "crops": digest(t["images_crop"]) if cropped else None,
        }

    rng = np.random.default_rng(seed)
    images = []
    for name, w, h, mode in _cases(seed, n_random):
        # Noise on white, like ink on paper; content does not change the
        # property, but it makes resize and pad errors visible in the hash.
        arr = np.where(rng.random((h, w, 3)) < 0.1, rng.integers(0, 255, (h, w, 3)), 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        if mode == "RGBA":
            img = img.convert("RGBA")
        elif mode != "RGB":
            img = img.convert(mode)
        images.append((name, img))
    test_split = list(load_from_disk("/data/hf_dataset")["test"])
    for i in random.Random(seed).sample(range(len(test_split)), min(n_dataset, len(test_split))):
        images.append((f"dataset-{i}", test_split[i]["image"]))

    mismatches = {f: [] for f in FIELDS}
    generate_kwargs = None
    for name, img in images:
        ev, generate_kwargs = eval_side(img)
        tr = train_side(img)
        for f in FIELDS:
            if ev[f] != tr[f]:
                a, b = ev[f], tr[f]
                if f in ("prompt_ids", "seq_mask"):
                    a, b = f"len {len(a)} tail {a[-6:]}", f"len {len(b)} tail {b[-6:]}"
                mismatches[f].append({"case": name, "size": list(img.size), "mode": img.mode,
                                      "eval": a, "train": b})
    return {"n_cases": len(images), "generate_kwargs": generate_kwargs, "mismatches": mismatches}

