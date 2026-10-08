import random

import pytest

from cartridges.training.perf_metrics import (
    GenerationTiming,
    NoRepeatNGramBlocker,
    hf_reference_banned,
    mean_in_window,
    parse_nvidia_smi_line,
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


def test_blocker_bans_repeat_of_seen_ngram():
    blocker = NoRepeatNGramBlocker(3, [1, 2, 3, 1, 2])
    assert blocker.banned() == [3]
