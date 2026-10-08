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


LAUNCH_CALLS = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch"}
SYNC_CALLS = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"}
# CPU ops that block on the GPU (.item() / .tolist() / .cpu() / nonzero)
SYNC_OPS = {"aten::item", "aten::_local_scalar_dense", "aten::nonzero"}
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _union_length(intervals: list[tuple[float, float]]) -> tuple[float, list[float]]:
    """Total covered length of possibly-overlapping intervals, plus the gaps between them."""
    covered, gaps = 0.0, []
    cur = None
    for start, end in sorted(intervals):
        if cur is None:
            cur = [start, end]
        elif start > cur[1]:
            covered += cur[1] - cur[0]
            gaps.append(start - cur[1])
            cur = [start, end]
        else:
            cur[1] = max(cur[1], end)
    if cur is not None:
        covered += cur[1] - cur[0]
    return covered, gaps


def trace_stats(events: list[dict], step_name: str = "decode_step", gap_us: float = 20.0) -> dict:
    """Per-decode-step host/GPU statistics from a torch.profiler Chrome trace.

    `events` is the trace's "traceEvents" list (timestamps in microseconds).
    Decode steps are the `record_function(step_name)` CPU ranges; launches,
    host syncs, and logits-processor time are attributed to the step whose
    range contains them. GPU busy/idle is measured over the whole decode
    window (first step start -> last step end).
    """
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
    steps = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in complete
        if e.get("cat") == "user_annotation" and e["name"] == step_name
    )
    stage_ms: dict[str, float] = {}
    for e in complete:
        if e.get("cat") == "user_annotation":
            stage_ms[e["name"]] = stage_ms.get(e["name"], 0.0) + e["dur"] / 1e3
    if not steps:
        return {"n_steps": 0, "stage_ms": stage_ms}

    def in_steps(e) -> bool:
        return any(s <= e["ts"] < t for s, t in steps)

    launches = syncs = memcpy_calls = dtoh = 0
    sync_ops: dict[str, int] = {}
    proc_us = 0.0
    for e in complete:
        name, cat = e["name"], e.get("cat")
        if cat == "cuda_runtime" and name in LAUNCH_CALLS and in_steps(e):
            launches += 1
        elif cat == "cuda_runtime" and name in SYNC_CALLS and in_steps(e):
            syncs += 1
        elif cat == "cuda_runtime" and name == "cudaMemcpyAsync" and in_steps(e):
            memcpy_calls += 1
        elif cat == "gpu_memcpy" and "DtoH" in name and in_steps(e):
            dtoh += 1  # GPU-side timestamp; profiler aligns the clocks
        elif cat == "cpu_op" and name in SYNC_OPS and in_steps(e):
            sync_ops[name] = sync_ops.get(name, 0) + 1
        elif cat == "user_annotation" and name.startswith("logits_processors") and in_steps(e):
            proc_us += e["dur"]

    window = (steps[0][0], steps[-1][1])
    gpu = [
        (max(e["ts"], window[0]), min(e["ts"] + e["dur"], window[1])) for e in complete
        if e.get("cat") in GPU_CATS and e["ts"] < window[1] and e["ts"] + e["dur"] > window[0]
    ]
    busy, gaps = _union_length(gpu)
    span = window[1] - window[0]
    n = len(steps)
    return {
        "n_steps": n,
        "step_ms": sum(t - s for s, t in steps) / n / 1e3,
        "launches_per_step": launches / n,
        "stream_syncs_per_step": syncs / n,
        "memcpy_calls_per_step": memcpy_calls / n,
        "dtoh_copies_per_step": dtoh / n,
        "sync_ops_per_step": {k: v / n for k, v in sync_ops.items()},
        "logits_processors_ms_per_step": proc_us / n / 1e3,
        "gpu_busy_frac": busy / span if span > 0 else None,
        "gpu_gaps_per_step": sum(1 for g in gaps if g > gap_us) / n,
        "gpu_idle_frac_in_gaps": sum(g for g in gaps if g > gap_us) / span if span > 0 else None,
        "stage_ms": stage_ms,
    }


def step_durations(call_times: list[float], t_end: float) -> list[float]:
    """Durations of consecutive steps that start at `call_times` (sorted) and
    end at the next start, the last one at `t_end`."""
    bounds = list(call_times) + [t_end]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


# Top-level stages of one model.infer() call, in order; nested labels
# (vision_encode.*, logits_processors.*, lm.layer*) are excluded so the sum
# is comparable to wall time.
TOP_LEVEL_STAGES = ("save_image", "preprocess", "generate_setup", "prefill", "decode_step", "decode_text")


def stage_coverage(stage_ms: dict[str, float], wall_ms: float) -> float | None:
    """Fraction of wall time covered by the top-level stage labels."""
    if wall_ms <= 0:
        return None
    return sum(stage_ms.get(s, 0.0) for s in TOP_LEVEL_STAGES) / wall_ms


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
