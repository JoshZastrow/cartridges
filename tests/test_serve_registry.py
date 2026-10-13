import base64

import httpx
import pytest

from cartridges.serve.evaluate import score
from cartridges.serve.merge import lora_merge_plan, lora_scale
from cartridges.serve.recipe import END_OF_SENTENCE, OcrRecipe
from cartridges.serve.registry import check_promotion
from cartridges.serve.vllm_server import build_app, vllm_command

# --- eval scoring and promotion -----------------------------------------------


def test_score_uses_corpus_cer_and_counts_errors_and_loops():
    looped = "and then and then and then and then"
    metrics = score(
        ["abcd", "efgh", "ijkl", "the end"],
        ["abcd", "efXX", None, looped],
    )
    assert metrics["n"] == 4 and metrics["n_errors"] == 1
    # corpus CER over the 3 scored pages: (0 + 2 + len edits of the loop) / chars
    assert 0 < metrics["corpus_cer"] and metrics["n_scored"] == 3
    assert metrics["loop_rate"] == pytest.approx(1 / 3)
    assert [p.status for p in metrics["predictions"]] == ["ok", "ok", "error", "ok"]


def promotion(**overrides):
    args = dict(n=10, n_errors=0, corpus_cer=0.05, loop_rate=0.0, max_cer=0.08, max_loop_rate=0.02)
    return check_promotion(**{**args, **overrides})


def test_promotion_passes_under_thresholds():
    result = promotion()
    assert result.passed and result.reasons == []


def test_promotion_fails_on_cer_loops_errors_and_no_data():
    assert "corpus_cer" in promotion(corpus_cer=0.2).reasons[0]
    assert "loop_rate" in promotion(loop_rate=0.1).reasons[0]
    assert "error rate" in promotion(n_errors=1).reasons[0]
    assert not promotion(n=0, corpus_cer=None).passed


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


# --- recipe --------------------------------------------------------------------


def test_recipe_model_id_changes_with_weights_and_every_setting():
    base = OcrRecipe(model_sha="sha1")
    assert base.model_id == OcrRecipe(model_sha="sha1").model_id
    variants = [
        OcrRecipe(model_sha="sha2"),
        base.model_copy(update={"max_tokens": 2048}),
        base.model_copy(update={"prompt": "OCR."}),
        base.model_copy(update={"image_size": 1024}),
        base.model_copy(update={"ngram_size": 30}),
        base.model_copy(update={"repetition_penalty": 1.1}),
    ]
    assert len({base.model_id, *(v.model_id for v in variants)}) == 7


def test_recipe_prompt_is_the_training_prompt():
    # vLLM's template for this model gives "<image>\n" + prompt.
    from cartridges.training.collator import FREE_OCR_PROMPT

    recipe = OcrRecipe(model_sha="x")
    assert "<image>\n" + recipe.prompt == FREE_OCR_PROMPT
    # Greedy, no_repeat_ngram_size 10 over the full output, no whitelist.
    assert (recipe.temperature, recipe.ngram_size, recipe.whitelist_token_ids) == (0.0, 10, ())
    assert recipe.window_size >= 8192


# --- server ---------------------------------------------------------------------


def test_vllm_command_serves_the_recipe():
    recipe = OcrRecipe(model_sha="abc123")
    command = vllm_command(recipe, 32)
    assert command[:3] == ["vllm", "serve", "/cache/models/abc123"]
    assert command[command.index("--served-model-name") + 1] == recipe.model_id
    assert '"image_size": 768' in command[command.index("--mm-processor-kwargs") + 1]
    assert "NGramPerReqLogitsProcessor" in command[command.index("--logits-processors") + 1]
    assert vllm_command(OcrRecipe(model_sha="base"), 8)[2] == "/cache/deepseek_ocr"


def fake_vllm(seen: list):
    """An httpx client whose transport answers like vLLM chat completions."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"  page text{END_OF_SENTENCE}"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            },
        )

    return httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(handler))


def test_ocr_endpoint_applies_the_recipe():
    from fastapi.testclient import TestClient

    recipe = OcrRecipe(model_sha="abc123")
    seen: list = []
    client = TestClient(build_app(recipe, "key", fake_vllm(seen)))
    auth = {"Authorization": "Bearer key"}

    png = base64.b64encode(b"png-bytes").decode()
    response = client.post("/ocr", json={"image_base64": png}, headers=auth)
    assert response.status_code == 200
    assert response.json() == {
        "text": "page text", "model_id": recipe.model_id,
        "finish_reason": "stop", "completion_tokens": 3, "first_token_top_logprobs": None,
    }
    body = seen[0]
    assert body["model"] == recipe.model_id
    assert body["messages"][0]["content"][1] == {"type": "text", "text": "Free OCR. "}
    assert body["vllm_xargs"] == {"ngram_size": 10, "window_size": 8192, "whitelist_token_ids": []}
    assert body["repetition_penalty"] == 1.05
    assert "logprobs" not in body

    assert client.get("/info", headers=auth).json()["model_id"] == recipe.model_id
    assert client.get("/health").status_code == 200


def test_ocr_endpoint_returns_first_token_logprobs_on_request():
    from fastapi.testclient import TestClient

    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        top = [{"token": "token_id:42", "logprob": -0.1}, {"token": "token_id:7", "logprob": -2.5}]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop",
                         "logprobs": {"content": [{"token": "token_id:42", "logprob": -0.1, "top_logprobs": top}]}}],
            "usage": {"completion_tokens": 1},
        })

    vllm = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(handler))
    client = TestClient(build_app(OcrRecipe(model_sha="x"), "key", vllm))
    response = client.post("/ocr", json={"image_base64": "", "top_logprobs": 2}, headers={"Authorization": "Bearer key"})
    assert response.json()["first_token_top_logprobs"] == [[42, -0.1], [7, -2.5]]
    assert (seen[0]["logprobs"], seen[0]["top_logprobs"], seen[0]["return_tokens_as_token_ids"]) == (True, 2, True)


def test_ocr_endpoint_needs_the_api_key():
    from fastapi.testclient import TestClient

    client = TestClient(build_app(OcrRecipe(model_sha="x"), "key", fake_vllm([])))
    assert client.post("/ocr", json={"image_base64": ""}).status_code == 401
    assert client.get("/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
