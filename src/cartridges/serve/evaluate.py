"""Evaluate the served model: the labeled split through the OCR service.

The eval calls the same /ocr endpoint as production. Thus the eval measures
exactly the model that the pipeline uses: the same weights, recipe, engine,
and container image.

Metrics (same as training/finetune.py):
- corpus_cer: all character edits / all reference characters
  (training/metrics.corpus_cer); the headline number;
- median_cer: the median of per-page CER;
- loop_rate: the share of pages that end in a repetition loop
  (training/loops.loop_rate; loops that the reference has are forgiven).
"""

from __future__ import annotations

import asyncio
import logging
import statistics

from pydantic import BaseModel

from cartridges.batch.backends import OcrBackend
from cartridges.training.loops import loop_rate
from cartridges.training.metrics import corpus_cer

logger = logging.getLogger(__name__)


class Prediction(BaseModel):
    index: int
    reference: str
    hypothesis: str | None
    cer: float | None
    status: str  # "ok" | "error"


class EvalReport(BaseModel):
    model_id: str
    model_sha: str
    revision: str
    split: str
    n: int
    n_errors: int
    corpus_cer: float | None
    n_scored: int
    n_blank_refs: int
    median_cer: float | None
    loop_rate: float
    elapsed_s: float
    pages_per_s: float
    predictions: list[Prediction]


def _page_cer(reference: str, hypothesis: str | None) -> float | None:
    import jiwer

    if hypothesis is None or not reference.strip():
        return None
    return jiwer.cer(reference, hypothesis)


def score(references: list[str], hypotheses: list[str | None]) -> dict:
    """The metrics of EvalReport (all fields except identity and timing)."""
    predictions = [
        Prediction(
            index=i, reference=ref, hypothesis=hyp, cer=_page_cer(ref, hyp),
            status="ok" if hyp is not None else "error",
        )
        for i, (ref, hyp) in enumerate(zip(references, hypotheses, strict=True))
    ]
    page_cers = [p.cer for p in predictions if p.cer is not None]
    return {
        "n": len(predictions),
        "n_errors": sum(p.status == "error" for p in predictions),
        **corpus_cer(references, hypotheses),
        "median_cer": statistics.median(page_cers) if page_cers else None,
        "loop_rate": loop_rate(hypotheses, references),
        "predictions": predictions,
    }


async def transcribe_all(
    images: list[bytes], backend: OcrBackend, concurrency: int = 32
) -> list[str | None]:
    """Send each image one time. A failed request gives None and counts as
    an error in the report (no retry: the eval must show server faults)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(png: bytes) -> str | None:
        async with semaphore:
            try:
                return await backend.transcribe(png)
            except Exception:
                logger.exception("OCR request failed")
                return None

    return list(await asyncio.gather(*(one(png) for png in images)))
