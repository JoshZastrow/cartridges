import random

import pytest

from cartridges.training.perf_metrics import (
    GenerationTiming,
    NoRepeatNGramBlocker,
    collapse_folded,
    hf_reference_banned,
    op_tables,
    mean_in_window,
    parse_nvidia_smi_line,
    stage_coverage,
    step_durations,
    summarize,
    trace_stats,
)


def test_generation_timing_excludes_prefill_from_decode_rate():
    timing = GenerationTiming(t_start=10.0, t_first=10.5, t_end=12.5, n_output_tokens=201)
    assert timing.prefill_s == pytest.approx(0.5)
    assert timing.decode_s == pytest.approx(2.0)
    # 200 decode steps after the prefill-produced first token
    assert timing.decode_tok_s == pytest.approx(100.0)
    assert timing.generate_s == pytest.approx(2.5)


def test_generation_timing_single_token_has_no_decode_rate():
    timing = GenerationTiming(t_start=0.0, t_first=0.4, t_end=0.4, n_output_tokens=1)
    assert timing.decode_tok_s is None


def test_mean_in_window():
    samples = [(0.0, 10.0), (1.0, 50.0), (2.0, 70.0), (3.0, 0.0)]
    assert mean_in_window(samples, 1.0, 2.0) == pytest.approx(60.0)
    assert mean_in_window(samples, 5.0, 6.0) is None


def test_parse_nvidia_smi_line():
    assert parse_nvidia_smi_line("87, 14321") == (87.0, 14321.0)
    assert parse_nvidia_smi_line("[N/A], 100") is None
    assert parse_nvidia_smi_line("") is None


@pytest.mark.parametrize("ngram_size", [1, 2, 3, 5, 35])
def test_blocker_matches_hf_reference_on_random_sequences(ngram_size):
    rng = random.Random(ngram_size)
    for _ in range(20):
        # small vocab + a long run of one token (like image placeholders)
        # so repeats and bans actually happen
        prompt = [0] + [7] * rng.randint(0, 60) + [rng.randint(0, 4) for _ in range(5)]
        blocker = NoRepeatNGramBlocker(ngram_size, prompt)
        seq = list(prompt)
        assert blocker.banned() == hf_reference_banned(seq, ngram_size)
        for _ in range(80):
            token = rng.randint(0, 4)
            blocker.append(token)
            seq.append(token)
            assert blocker.banned() == hf_reference_banned(seq, ngram_size)


def _x(name, cat, ts, dur):
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur}


def test_trace_stats_attributes_launches_syncs_and_gpu_idle_to_steps():
    events = [
        _x("prefill", "user_annotation", 0, 100),
        _x("cudaLaunchKernel", "cuda_runtime", 10, 1),  # prefill: not counted
        # step 1: [100, 200)
        _x("decode_step", "user_annotation", 100, 100),
        _x("cudaLaunchKernel", "cuda_runtime", 110, 1),
        _x("cudaLaunchKernel", "cuda_runtime", 120, 1),
        _x("aten::item", "cpu_op", 150, 5),
        _x("cudaStreamSynchronize", "cuda_runtime", 151, 3),
        _x("logits_processors", "user_annotation", 160, 30),
        _x("gemv", "kernel", 100, 40),  # busy 100-140
        # step 2: [200, 300)
        _x("decode_step", "user_annotation", 200, 100),
        _x("cudaGraphLaunch", "cuda_runtime", 205, 1),
        _x("logits_processors", "user_annotation", 260, 10),
        _x("gemv", "kernel", 230, 30),  # busy 230-260
        _x("gemv", "kernel", 250, 20),  # overlaps -> busy 230-270
    ]
    stats = trace_stats(events)
    assert stats["n_steps"] == 2
    assert stats["step_ms"] == pytest.approx(0.1)
    assert stats["launches_per_step"] == pytest.approx(1.5)
    assert stats["stream_syncs_per_step"] == pytest.approx(0.5)
    assert stats["sync_ops_per_step"] == {"aten::item": 0.5}
    assert stats["logits_processors_ms_per_step"] == pytest.approx(0.02)
    # window 100-300 (200us); busy 40 + 40 = 80us
    assert stats["gpu_busy_frac"] == pytest.approx(0.4)
    # gaps: 140->230 (90us) and 270->300 isn't a gap between kernels
    assert stats["gpu_gaps_per_step"] == pytest.approx(0.5)
    assert stats["gpu_idle_frac_in_gaps"] == pytest.approx(90 / 200)
    assert stats["stage_ms"]["prefill"] == pytest.approx(0.1)


def test_trace_stats_counts_dtoh_copies_and_memcpy_calls():
    events = [
        _x("decode_step", "user_annotation", 0, 100),
        _x("cudaMemcpyAsync", "cuda_runtime", 10, 2),
        _x("Memcpy DtoH (Device -> Pinned)", "gpu_memcpy", 12, 1),
        _x("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", 20, 1),
    ]
    stats = trace_stats(events)
    assert stats["memcpy_calls_per_step"] == 1
    assert stats["dtoh_copies_per_step"] == 1


def test_op_tables_self_time_excludes_nested_ops():
    events = [
        dict(_x("aten::linear", "cpu_op", 0, 100), pid=1, tid=1),
        dict(_x("aten::mm", "cpu_op", 10, 60), pid=1, tid=1),  # nested in linear
        dict(_x("aten::item", "cpu_op", 200, 30), pid=1, tid=1),
        _x("gemv_kernel", "kernel", 15, 40),
        _x("gemv_kernel", "kernel", 120, 20),
    ]
    out = op_tables(events, n=5)
    lines = out.splitlines()
    cpu = {line.split()[-1]: float(line.split()[0]) for line in lines[2:5]}
    assert cpu == {"aten::mm": 0.06, "aten::linear": 0.04, "aten::item": 0.03}
    gpu_line = next(line for line in lines if line.endswith("gemv_kernel"))
    assert float(gpu_line.split()[0]) == pytest.approx(0.06) and gpu_line.split()[2] == "2"


def test_collapse_folded_sums_identical_stacks():
    lines = ["a;b 3\n", "a;b 4\n", "a;c 1\n", "garbage\n", "a;b x 2\n"]
    assert collapse_folded(lines) == ["a;b 7", "a;b x 2", "a;c 1"]


def test_step_durations_and_summary():
    assert step_durations([1.0, 1.5, 3.0], 3.5) == [0.5, 1.5, 0.5]
    s = summarize([0.5, 1.5, 0.5])
    assert s["n"] == 3 and s["mean"] == pytest.approx(2.5 / 3) and s["p50"] == 0.5 and s["max"] == 1.5
    assert summarize([]) == {"n": 0}


def test_stage_coverage_ignores_nested_labels():
    stage_ms = {"save_image": 10, "preprocess": 20, "prefill": 30, "decode_step": 30,
                "vision_encode.sam_model": 25, "logits_processors": 5}
    assert stage_coverage(stage_ms, 100) == pytest.approx(0.9)
    assert stage_coverage(stage_ms, 0) is None


def test_trace_stats_without_steps():
    assert trace_stats([_x("prefill", "user_annotation", 0, 10)])["n_steps"] == 0


def test_blocker_bans_repeat_of_seen_ngram():
    blocker = NoRepeatNGramBlocker(3, [1, 2, 3, 1, 2])
    assert blocker.banned() == [3]
