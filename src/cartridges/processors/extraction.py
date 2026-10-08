"""Structured field extraction from page text using a Gemma model served with vLLM.

Deploy with:
uv run modal deploy src/cartridges/processors/extraction.py
"""

import hashlib

import modal

from cartridges.storage.cache import get_cached, put_cached

GEMMA_MODEL_NAME = "google/gemma-3-4b-it"
MINUTES = 60

DATE_EXTRACTION_PROMPT = (
    "Extract the date written on this journal page, if any. Reply with only "
    "the date in YYYY-MM-DD format, or the single word 'unknown' if no date "
    "is present. Do not explain your answer.\n\nPage text:\n{text}"
)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def extract_date(text: str, *, model_name: str = GEMMA_MODEL_NAME) -> str:
    """Extract a date from page text, checking the cache before calling Modal."""
    text_hash = hash_text(text)

    cached = get_cached(text_hash, model_name)
    if cached is not None:
        return cached

    extractor = modal.Cls.from_name("cartridges-extraction", "DateExtractor")
    date = extractor(model_name=model_name).extract.remote(text)
    put_cached(text_hash, model_name, date)
    return date


image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.6",
    "transformers==4.46.3",
)

app = modal.App("cartridges-extraction", image=image)

model_cache = modal.Volume.from_name("cartridges-extraction-cache", create_if_missing=True)


@app.cls(
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/cache": model_cache},
)
class DateExtractor:
    model_name: str = modal.parameter(default=GEMMA_MODEL_NAME)

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(model=self.model_name, download_dir="/cache", dtype="bfloat16")

    @modal.method()
    def extract(self, text: str) -> str:
        from vllm import SamplingParams

        prompt = DATE_EXTRACTION_PROMPT.format(text=text)
        params = SamplingParams(max_tokens=16, temperature=0)
        output = self.llm.generate([prompt], params)[0]
        return output.outputs[0].text.strip()
