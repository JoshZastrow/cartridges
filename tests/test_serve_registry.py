import pytest

from cartridges.serve.merge import lora_merge_plan, lora_scale
from cartridges.serve.parity import PARITY_DECODING, compare, first_diff
from cartridges.serve.registry import check_promotion
from cartridges.serve.vllm_server import vllm_command


def pred(cer, hypothesis="the cat sat", reference="the cat sat"):
    return {"cer": cer, "hypothesis": hypothesis, "reference": reference}


# --- check_promotion ----------------------------------------------------------


def test_promotion_passes_under_thresholds():
    result = check_promotion([pred(0.05), pred(0.07)], max_cer=0.08, max_loop_rate=0.0)
    assert result.passed and result.reasons == []
    assert result.mean_cer == pytest.approx(0.06)


def test_promotion_fails_on_cer():
    result = check_promotion([pred(0.2)], max_cer=0.08, max_loop_rate=1.0)
    assert not result.passed and "mean_cer" in result.reasons[0]


def test_promotion_fails_on_loops_but_forgives_loops_in_the_reference():
    looped = "and then and then and then and then"
    preds = [pred(0.01, hypothesis=looped), pred(0.01)]
    assert not check_promotion(preds, max_cer=0.08, max_loop_rate=0.1).passed

    forgiven = [pred(0.01, hypothesis=looped, reference=looped), pred(0.01)]
    assert check_promotion(forgiven, max_cer=0.08, max_loop_rate=0.1).passed


def test_promotion_fails_on_errors_and_on_no_data():
    result = check_promotion(
        [pred(0.01), pred(None, hypothesis=None)], max_cer=0.08, max_loop_rate=1.0
    )
    assert result.n_errors == 1 and not result.passed
    assert not check_promotion([], max_cer=1.0, max_loop_rate=1.0).passed


# --- merge plan ---------------------------------------------------------------

BASE_KEYS = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.embed_tokens.weight",
}


def adapter_keys(*modules):
    keys = []
    for m in modules:
        keys += [f"base_model.model.{m}.lora_A.weight", f"base_model.model.{m}.lora_B.weight"]
    return keys


def test_merge_plan_maps_adapter_to_base_keys():
    plan = lora_merge_plan(
        adapter_keys("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.up_proj"), BASE_KEYS
    )
    assert plan["model.layers.0.self_attn.q_proj.weight"] == (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    )
    assert len(plan) == 2


def test_merge_plan_rejects_a_module_with_no_base_weight():
    with pytest.raises(ValueError, match="no base weight"):
        lora_merge_plan(adapter_keys("vision.blocks.0.q_proj"), BASE_KEYS)


def test_merge_plan_rejects_non_lora_keys_and_lone_a():
    with pytest.raises(ValueError, match="non-LoRA"):
        lora_merge_plan(["base_model.model.lm_head.weight"], BASE_KEYS)
    with pytest.raises(ValueError, match="without lora_B"):
        lora_merge_plan(adapter_keys("model.layers.0.mlp.up_proj")[:1], BASE_KEYS)


def test_lora_scale():
    assert lora_scale({"r": 16, "lora_alpha": 16}) == 1.0
    assert lora_scale({"r": 16, "lora_alpha": 16, "use_rslora": True}) == 4.0


# --- parity -------------------------------------------------------------------


def test_parity_passes_when_outputs_agree():
    refs = ["the cat sat on the mat", "hello world"]
    report = compare(refs, ["the cat sat on the mat", "hello word"], ["the cat sat on the mat", "hello word"], revision="r", model_sha="s")
    assert report.passed and report.exact_rate == 1.0 and report.cer_gap == 0.0


def test_parity_fails_when_vllm_drifts_from_hf():
    refs = ["abcdefghij"] * 2
    report = compare(refs, ["abcdefghij"] * 2, ["abcdefghij", "abcdeXXXXX"], revision="r", model_sha="s")
    assert not report.passed
    assert any("cer_gap" in r for r in report.reasons)
    assert any("mean_cer_cross" in r for r in report.reasons)
    assert report.examples[1].first_diff == 5


def test_parity_fails_on_a_failed_request_and_scores_only_shared_examples():
    report = compare(["a b", "c d"], ["a b", "c d"], ["a b", None], revision="r", model_sha="s")
    assert report.n_errors == 1 and not report.passed
    assert report.mean_cer_hf == 0.0 and report.mean_cer_vllm == 0.0


def test_first_diff_on_prefix():
    assert first_diff("abc", "abc") is None
    assert first_diff("abc", "abcd") == 3


def test_parity_decoding_matches_the_hf_eval_path():
    d = PARITY_DECODING
    assert (d.ngram_size, d.whitelist_token_ids, d.temperature) == (35, (), 0.0)
    assert d.window_size >= 8192  # >= max_model_len: the full output
    # vLLM's default template gives "<image>\n" + prompt: the training prompt.
    from cartridges.training.collator import FREE_OCR_PROMPT

    assert "<image>\n" + d.prompt == FREE_OCR_PROMPT


# --- server command -----------------------------------------------------------


def test_vllm_command_serves_the_sha_from_its_merged_dir():
    command = vllm_command("abc123", 32)
    assert command[:3] == ["vllm", "serve", "/cache/models/abc123"]
    assert command[command.index("--served-model-name") + 1] == "abc123"
    assert "NGramPerReqLogitsProcessor" in command[command.index("--logits-processors") + 1]
    assert vllm_command("base", 8)[2] == "/cache/deepseek_ocr"
