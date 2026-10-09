"""Detect the "repeats its last few words" failure in OCR transcriptions.

The single metric for the looping investigation is `loop_rate`: the share of
transcriptions that *end* in a loop -- a phrase of 1..`max_phrase_words`
words repeated back-to-back at least `min_repeats` times as the final words
of the output (a trailing partial copy, cut off by the token cap, is allowed).
Repetition elsewhere in a page doesn't count; only the tail does.

Words are compared lowercased with punctuation stripped, since greedy
decoding's loops often vary only in casing or punctuation.

Pure Python (no Modal/GPU deps) so it runs locally and in the container.
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w']+")


def _words(text: str) -> list[str]:
    return [w for w in (_PUNCT_RE.sub("", w.lower()) for w in text.split()) if w]


def ends_in_loop(text: str | None, *, min_repeats: int = 3, max_phrase_words: int = 12) -> bool:
    if not text:
        return False
    words = _words(text)
    for n in range(1, max_phrase_words + 1):
        for trim in range(n):
            body = words[: len(words) - trim]
            if len(body) < n * min_repeats:
                break
            phrase = body[-n:]
            if words[len(body) :] != phrase[:trim]:
                continue
            repeats = 1
            while body[-(repeats + 1) * n : len(body) - repeats * n] == phrase:
                repeats += 1
            if repeats >= min_repeats:
                return True
    return False


def loop_rate(texts: list[str | None], references: list[str] | None = None, **kwargs) -> float:
    """Share of transcriptions that end in a loop. `None` (a failed inference)
    is excluded, so errors don't masquerade as clean pages. With
    `references`, a loop the handwriting itself ends with (e.g. "Repair,
    repair, repair.") doesn't count against the model."""
    if references is None:
        references = [None] * len(texts)
    scored = [(t, r) for t, r in zip(texts, references, strict=True) if t is not None]
    if not scored:
        return 0.0
    looped = sum(ends_in_loop(t, **kwargs) and not ends_in_loop(r, **kwargs) for t, r in scored)
    return looped / len(scored)
