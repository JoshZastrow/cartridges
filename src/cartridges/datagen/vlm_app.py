"""Modal-hosted VLM for handwriting transcription — fallback for when OCR fails.

DeepSeek-OCR-2 is a purpose-built OCR model (not a general VQA model), reported
strong on document/handwriting transcription (OmniDocBench).

Deploy: uv run modal deploy src/cartridges/datagen/vlm_app.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
QWEN_OCR_MODEL_NAME = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
MINUTES = 60

# "Free OCR" = raw text only, no layout/grounding markup — what we want for
# a plain transcription.
FREE_OCR_PROMPT = "<image>\nFree OCR. "
QWEN_OCR_PROMPT = (
    "Transcribe all text in this image exactly as written, verbatim. "
    "Output only the transcribed text, nothing else."
)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers==4.46.3",
    "tokenizers==0.20.3",
    "einops",
    "addict",
    "easydict",
    "accelerate",
    "pillow",
)

app = modal.App("cartridges-vlm", image=image)

# Model registry: model_name -> Modal class name serving it.
TRANSCRIBER_CLASSES = {
    MODEL_NAME: "Transcriber",
    QWEN_OCR_MODEL_NAME: "QwenOCRTranscriber",
}

model_cache = modal.Volume.from_name("cartridges-vlm-cache", create_if_missing=True)


@app.cls(
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/cache": model_cache},
)
class Transcriber:
    @modal.enter()
    def load(self):
        import warnings

        import torch  # pyright: ignore[reportMissingImports]
        import transformers
        from transformers import AutoModel, AutoTokenizer

        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=UserWarning)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, cache_dir="/cache"
        )
        # tokenizer.eos_token_id is None for this tokenizer — resolve the
        # actual end-of-sequence token id so infer()'s internal generate()
        # call (which reads tokenizer.eos_token_id directly) doesn't fall
        # back to an unset pad/eos id.
        eos_id = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = eos_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = eos_id

        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir="/cache",
            )
            .eval()
            .cuda()
            .to(torch.bfloat16)
        )

    @modal.method()
    def transcribe(self, image_bytes: bytes) -> str:
        import contextlib
        import io
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.png"
            image_path.write_bytes(image_bytes)
            # model.infer() prints internal debug shapes (BASE/PATCHES) via
            # plain print() — silence stdout for the duration of the call.
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.model.infer(
                    self.tokenizer,
                    prompt=FREE_OCR_PROMPT,
                    image_file=str(image_path),
                    output_path=tmpdir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    eval_mode=True,
                )
            return result.strip()


@app.cls(
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/cache": model_cache},
)
class QwenOCRTranscriber:
    @modal.enter()
    def load(self):
        import torch  # pyright: ignore[reportMissingImports]
        import transformers
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        transformers.logging.set_verbosity_error()

        self.processor = AutoProcessor.from_pretrained(
            QWEN_OCR_MODEL_NAME, cache_dir="/cache"
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_OCR_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir="/cache",
        )

    @modal.method()
    def transcribe(self, image_bytes: bytes) -> str:
        import io

        from PIL import Image

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QWEN_OCR_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_image], padding=True, return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
