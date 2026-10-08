"""Pure-Python helpers for inference profiling (no torch/modal imports).

Kept separate from `profiling.py` so the timing math and the no-repeat-ngram
logic can be unit tested locally on a Mac without the CUDA stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationTiming:
    """CUDA-synchronized timestamps (seconds, same clock) around one generate().

    `t_first` is when the first token's logits were ready, i.e. right after the
    prefill forward pass; every later token is one decode step.
    """

    t_start: float
    t_first: float
    t_end: float
    n_output_tokens: int

    @property
    def prefill_s(self) -> float:
        return self.t_first - self.t_start

    @property
    def decode_s(self) -> float:
        return self.t_end - self.t_first

    @property
    def decode_tok_s(self) -> float | None:
        # The first output token comes out of the prefill pass, so only the
        # remaining n-1 tokens are produced by decode steps.
        n_decode = self.n_output_tokens - 1
        if n_decode <= 0 or self.decode_s <= 0:
            return None
        return n_decode / self.decode_s

    @property
    def generate_s(self) -> float:
        return self.t_end - self.t_start


def mean_in_window(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    """Mean of `value` over `(t, value)` samples with t0 <= t <= t1, or None."""
    values = [v for t, v in samples if t0 <= t <= t1]
    return sum(values) / len(values) if values else None


def parse_nvidia_smi_line(line: str) -> tuple[float, float] | None:
    """Parse one `--query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`
    line into (util_percent, mem_used_mib). Returns None for anything else
    (headers, blank lines, '[N/A]')."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class NoRepeatNGramBlocker:
    """Incremental, exact equivalent of HF's `NoRepeatNGramLogitsProcessor`
    for a single sequence.

    HF's version rebuilds a dict of every n-gram in the *whole* sequence
    (prompt included -- here ~1,000 identical image-placeholder tokens) on
    every decode step, in Python, with `list + [x]` concatenation. Same rule,
    but each n-gram is indexed once as it completes:

    an n-gram (prefix of n-1 tokens, next token) may occur only once, so the
    next token is banned if `(last n-1 tokens, token)` already occurred
    anywhere earlier in the sequence.
    """

    def __init__(self, ngram_size: int, tokens: list[int] | None = None):
        if ngram_size < 1:
            raise ValueError("ngram_size must be >= 1")
        self.n = ngram_size
        self.tokens: list[int] = []
        self._next: dict[tuple[int, ...], set[int]] = {}
        for token in tokens or []:
            self.append(token)

    def append(self, token: int) -> None:
        self.tokens.append(token)
        if len(self.tokens) >= self.n:
            ngram = self.tokens[-self.n :]
            self._next.setdefault(tuple(ngram[:-1]), set()).add(ngram[-1])

    def banned(self) -> list[int]:
        """Tokens that may not come next."""
        # HF: no bans until cur_len + 1 >= ngram_size.
        if len(self.tokens) + 1 < self.n:
            return []
        prefix = tuple(self.tokens[len(self.tokens) - self.n + 1 :]) if self.n > 1 else ()
        return sorted(self._next.get(prefix, ()))


def hf_reference_banned(tokens: list[int], ngram_size: int) -> list[int]:
    """Straight port of HF's `_calc_banned_ngram_tokens` for one hypothesis,
    used as the test oracle for `NoRepeatNGramBlocker`."""
    cur_len = len(tokens)
    if cur_len + 1 < ngram_size:
        return []
    ngrams: dict[tuple[int, ...], list[int]] = {}
    for ngram in zip(*[tokens[i:] for i in range(ngram_size)]):
        ngrams[tuple(ngram[:-1])] = ngrams.get(tuple(ngram[:-1]), []) + [ngram[-1]]
    start_idx = cur_len + 1 - ngram_size
    return sorted(set(ngrams.get(tuple(tokens[start_idx:cur_len]), [])))
