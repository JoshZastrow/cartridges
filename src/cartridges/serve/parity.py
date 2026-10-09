"""Phase 0: numeric parity of vLLM serving against the training eval path.

The training eval path (training/finetune.py `_run_inference`) runs unsloth +
PEFT (adapter not merged) and `model.infer()`:
- prompt "<image>\\nFree OCR. " (plain, with BOS from the processor);
- base_size 1024, image_size 768, crop_mode True;
- greedy decoding, no_repeat_ngram_size 35 over the full output;
- max_new_tokens capped at TrainConfig.max_new_tokens.

PARITY_DECODING sets the vLLM request to the same values:
- the prompt: vLLM's default template gives BOS + image tokens + "\\n" +
  "Free OCR. ". Checked with /tokenize on the deployed server: token ids
  0, 128815 x N, 201, 21431, 126041, 16, 223 (the same as the HF prompt);
- ngram_size 35 with a window larger than max_model_len (8192) is equal to
  no_repeat_ngram_size 35 over the full output;
- no whitelist (HF has none).
vLLM's DeepSeek-OCR-2 processor defaults are already 1024 / 768 / crop on.

The outputs are not identical bit for bit: the kernels are different, and
merged bf16 weights round differently. Greedy decoding can then go a
different way after one token. Thus the check uses CER, not equality.

Pass condition (docs/plans/batch-ocr-pipeline.md, Phase 0):
- |mean CER vLLM - mean CER HF| <= 0.01 (1 point), against the reference;
- mean CER of vLLM against HF < 0.02.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from cartridges.batch.backends import OcrBackend
from cartridges.batch.keys import DecodingSettings
from cartridges.training.config import TrainConfig

logger = logging.getLogger(__name__)

PARITY_DECODING = DecodingSettings(
    max_tokens=TrainConfig().max_new_tokens,
    ngram_size=35,
    window_size=8192,
    whitelist_token_ids=(),
)
MAX_CER_GAP = 0.01
MAX_CROSS_CER = 0.02


class ParityExample(BaseModel):
    index: int
    reference: str
    hf: str | None
    vllm: str | None
    cer_hf: float | None
    cer_vllm: float | None
    cer_cross: float | None  # vLLM against HF
    exact: bool
    first_diff: int | None  # first character index where HF and vLLM differ


class ParityReport(BaseModel):
    revision: str
    model_sha: str
    n: int
    n_errors: int
    mean_cer_hf: float | None
    mean_cer_vllm: float | None
    cer_gap: float | None
    mean_cer_cross: float | None
    exact_rate: float
    passed: bool
    reasons: list[str]
    examples: list[ParityExample]


def _cer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)


def first_diff(a: str, b: str) -> int | None:
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare(
    references: list[str],
    hf: list[str | None],
    vllm: list[str | None],
    *,
    revision: str,
    model_sha: str,
    max_cer_gap: float = MAX_CER_GAP,
    max_cross_cer: float = MAX_CROSS_CER,
) -> ParityReport:
    if not (len(references) == len(hf) == len(vllm)):
        raise ValueError(f"length mismatch: {len(references)} / {len(hf)} / {len(vllm)}")

    examples = []
    for i, (ref, h, v) in enumerate(zip(references, hf, vllm)):
        both = h is not None and v is not None
        examples.append(
            ParityExample(
                index=i, reference=ref, hf=h, vllm=v,
                cer_hf=_cer(ref, h) if h is not None else None,
                cer_vllm=_cer(ref, v) if v is not None else None,
                cer_cross=_cer(h, v) if both else None,
                exact=both and h == v,
                first_diff=first_diff(h, v) if both else None,
            )
        )

    # Compare means over the same examples: those that both paths completed.
    scored = [e for e in examples if e.cer_cross is not None]
    mean_hf = _mean([e.cer_hf for e in scored])
    mean_vllm = _mean([e.cer_vllm for e in scored])
    mean_cross = _mean([e.cer_cross for e in scored])
    gap = abs(mean_vllm - mean_hf) if scored else None
    n_errors = len(examples) - len(scored)

    reasons = []
    if not scored:
        reasons.append("no example completed on both paths")
    if gap is not None and gap > max_cer_gap:
        reasons.append(f"cer_gap {gap:.4f} > {max_cer_gap}")
    if mean_cross is not None and mean_cross >= max_cross_cer:
        reasons.append(f"mean_cer_cross {mean_cross:.4f} >= {max_cross_cer}")
    if n_errors:
        reasons.append(f"{n_errors} example(s) failed on one path")

    return ParityReport(
        revision=revision, model_sha=model_sha, n=len(examples), n_errors=n_errors,
        mean_cer_hf=mean_hf, mean_cer_vllm=mean_vllm, cer_gap=gap, mean_cer_cross=mean_cross,
        exact_rate=sum(e.exact for e in examples) / len(examples) if examples else 0.0,
        passed=not reasons, reasons=reasons, examples=examples,
    )


async def vllm_predictions(
    images: list[bytes], backend: OcrBackend, concurrency: int = 16
) -> list[str | None]:
    """Send each image one time. A failed request gives None (no retry: a
    retry could hide a server fault that the parity check must show)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(png: bytes) -> str | None:
        async with semaphore:
            try:
                return await backend.transcribe(png)
            except Exception:
                logger.exception("vLLM request failed")
                return None

    return list(await asyncio.gather(*(one(png) for png in images)))
