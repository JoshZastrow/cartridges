"""Transcription quality metrics.

Corpus CER is the headline CER: all character edits (substitutions,
deletions, insertions) summed over every page, divided by the total number of
reference characters. Unlike the mean of per-page CER, long pages weigh more
than short ones, and one runaway page can't dominate the average.

Mean NED (normalized edit distance) is the per-page metric DeepSeek-OCR
reports on OmniDocBench: edits / max(len(reference), len(hypothesis)),
averaged over pages. It's bounded to [0, 1], so a runaway page counts at most 1.

Pages with an empty reference (blank journal pages) have no defined CER, so
they're excluded and counted separately; failed inferences (hypothesis None)
are excluded too.
"""

from __future__ import annotations


def corpus_cer(references: list[str], hypotheses: list[str | None]) -> dict:
    import jiwer

    pairs = [(r, h) for r, h in zip(references, hypotheses, strict=True) if h is not None]
    scored = [(r, h) for r, h in pairs if r.strip()]
    n_blank = len(pairs) - len(scored)
    if not scored:
        return {"corpus_cer": None, "n_scored": 0, "n_blank_refs": n_blank}
    # process_characters over lists aggregates edits and reference lengths
    # across all pages before dividing.
    out = jiwer.process_characters([r for r, _ in scored], [h for _, h in scored])
    return {"corpus_cer": out.cer, "n_scored": len(scored), "n_blank_refs": n_blank}


def mean_ned(references: list[str], hypotheses: list[str | None]) -> float | None:
    from rapidfuzz.distance import Levenshtein  # installed with jiwer

    scored = [(r, h) for r, h in zip(references, hypotheses, strict=True) if h is not None and r.strip()]
    if not scored:
        return None
    return sum(Levenshtein.normalized_distance(r, h) for r, h in scored) / len(scored)
