"""Pure-Python helpers for inference profiling (no torch/modal imports).

Kept separate from `profiling.py` so the timing math and the no-repeat-ngram
logic can be unit tested locally on a Mac without the CUDA stack.
"""

from __future__ import annotations

import bisect
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

    starts = [s for s, _ in steps]

    def in_steps(e) -> bool:  # steps don't overlap: check the last one starting before e
        i = bisect.bisect_right(starts, e["ts"]) - 1
        return i >= 0 and e["ts"] < steps[i][1]

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


def op_tables(events: list[dict], n: int = 30) -> str:
    """Top-n CPU ops by self time and top-n GPU kernels by time, from Chrome
    trace events (replaces prof.key_averages(), which builds a Python event
    tree and took ~8 min on a 64-step eager decode trace).

    CPU self time = duration minus time in directly nested cpu_op events on
    the same thread."""
    cpu: dict[str, list[float]] = {}  # name -> [self_us, total_us, count]
    by_thread: dict = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "cpu_op" and "dur" in e:
            by_thread.setdefault((e.get("pid"), e.get("tid")), []).append(e)
    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack: list[list] = []  # [end, event, child_us]
        def close(entry):
            end, ev, child = entry
            row = cpu.setdefault(ev["name"], [0.0, 0.0, 0])
            row[0] += ev["dur"] - child
            row[1] += ev["dur"]
            row[2] += 1
        for e in evs:
            while stack and e["ts"] >= stack[-1][0]:
                close(stack.pop())
            if stack:
                stack[-1][2] += e["dur"]
            stack.append([e["ts"] + e["dur"], e, 0.0])
        while stack:
            close(stack.pop())
    gpu: dict[str, list[float]] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in GPU_CATS and "dur" in e:
            row = gpu.setdefault(e["name"], [0.0, 0])
            row[0] += e["dur"]
            row[1] += 1

    def table(title, rows, headers):
        out = [title, "  ".join(headers)]
        out += ["  ".join(r) for r in rows]
        return "\n".join(out)

    cpu_total = sum(r[0] for r in cpu.values()) or 1.0
    cpu_rows = [
        (f"{v[0] / 1e3:12.3f}", f"{100 * v[0] / cpu_total:6.1f}%", f"{v[1] / 1e3:12.3f}", f"{int(v[2]):9d}", k[:90])
        for k, v in sorted(cpu.items(), key=lambda kv: -kv[1][0])[:n]
    ]
    gpu_total = sum(r[0] for r in gpu.values()) or 1.0
    gpu_rows = [
        (f"{v[0] / 1e3:12.3f}", f"{100 * v[0] / gpu_total:6.1f}%", f"{int(v[1]):9d}", k[:90])
        for k, v in sorted(gpu.items(), key=lambda kv: -kv[1][0])[:n]
    ]
    return (
        table(f"Top {n} CPU ops by self time", cpu_rows, ("self_ms".rjust(12), "share".rjust(7), "total_ms".rjust(12), "calls".rjust(9), "op"))
        + "\n\n"
        + table(f"Top {n} GPU kernels/copies by time", gpu_rows, ("gpu_ms".rjust(12), "share".rjust(7), "calls".rjust(9), "kernel"))
        + "\n"
    )


def collapse_folded(lines) -> list[str]:
    """Sum the counts of identical stacks in `export_stacks` output
    ("frame;frame;frame <count>" per event, hundreds of MB uncollapsed) into
    standard collapsed-stack lines, heaviest first."""
    totals: dict[str, int] = {}
    for line in lines:
        line = line.rstrip("\n")
        stack, _, count = line.rpartition(" ")
        if not stack or not count.lstrip("-").isdigit():
            continue
        totals[stack] = totals.get(stack, 0) + int(count)
    return [f"{stack} {count}" for stack, count in sorted(totals.items(), key=lambda kv: -kv[1])]


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
