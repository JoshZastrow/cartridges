from cartridges.training.profile_page_report import render_report


def _stages():
    return {
        "config": {"run_name": "t", "mode": "generate", "split": "test", "index": 0,
                   "model_source": "artifact", "hf_revision": "run-x", "merge_adapter": False},
        "wall_ms": 1000.0,
        "ttft_ms": 200.0,
        "output_tokens": 9,
        "stage_ms": {"preprocess": 100.0, "prefill": 200.0, "decode_step": 650.0,
                     "vision_encode.sam_model": 50.0},
        "stage_coverage": 0.95,
        "decode_step_ms": {"p50": 80.0, "p90": 90.0},
        "decode_step_ms_all": [80.0, 70.0, 90.0],
        "logits_processor_ms_per_step": {"NoRepeatNGramLogitsProcessor": 12.5},
        "trace_stats": {"gpu_busy_frac": 0.1, "launches_per_step": 3600, "stream_syncs_per_step": 850,
                        "sync_ops_per_step": {"aten::_local_scalar_dense": 850}},
    }


def test_render_report_includes_key_numbers_and_verdict():
    out = render_report(_stages(), [(0.0, 10.0, 8000.0), (0.05, 20.0, 8100.0)], "op table <x>")
    assert out.startswith("<!doctype html>")
    assert "80.0 ms of wall time" in out  # verdict: median step
    assert "8.0 ms of it (10%)" in out  # GPU busy share of the step
    assert "3,600 kernel launches" in out
    assert "NoRepeatNGramLogitsProcessor" in out
    assert "vision_encode.sam_model" in out  # nested labels table
    assert "op table &lt;x&gt;" in out  # escaped


def test_render_report_forward_mode_without_steps_or_gpu():
    stages = _stages()
    stages.update(decode_step_ms={}, decode_step_ms_all=[], trace_stats={})
    out = render_report(stages, [], "")
    assert "No decode steps recorded" in out
    assert "no data" in out
