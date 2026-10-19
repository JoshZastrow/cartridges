"""Teacher prompt rendering, vLLM logprob parsing, and the HF scorer's reused
prefix cache, on CPU: fakes, the cached Qwen3.5 tokenizer, a tiny qwen3_5 model."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from cartridges.cartridge import teacher


def test_topk_from_vllm_keeps_top_k_most_likely_first():
    lp = lambda x: SimpleNamespace(logprob=x)
    positions = [
        {5: lp(-0.1), 7: lp(-2.0), 9: lp(-3.0)},
        # sampled token 42 is outside the top 2: vLLM adds it as an extra entry
        {1: lp(-1.5), 2: lp(-0.5), 42: lp(-6.0)},
    ]
    ids, lps = teacher.topk_from_vllm(positions, k=2)
    assert ids == [[5, 7], [2, 1]]
    assert lps == [[-0.1, -2.0], [-0.5, -1.5]]


def test_sampling_kwargs():
    structured = lambda **kw: ("structured", kw)  # stands in for vllm's StructuredOutputsParams
    assert teacher.sampling_kwargs(1, 96, 0, 0.0, None, None, structured) == {
        "n": 1, "max_tokens": 96, "logprobs": None, "seed": None, "temperature": 0.0}
    sampled = teacher.sampling_kwargs(2, 64, 20, None, 7, None, structured)
    assert sampled == {"n": 2, "max_tokens": 64, "logprobs": 20, "seed": 7, **teacher.SAMPLING}
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    constrained = teacher.sampling_kwargs(1, 96, 0, 0.0, None, schema, structured)
    assert constrained["structured_outputs"] == ("structured", {"json": schema})


# ---------------------------------------------------------------- tokenizer


@pytest.fixture(scope="module", params=[teacher.SMOKE_MODEL_ID, teacher.MODEL_ID])
def tokenizer(request):
    """The real chat templates (4B and 27B differ: 3.8 adds reasoning_effort and
    preserve_thinking), from the local HF cache."""
    import transformers

    try:
        return transformers.AutoTokenizer.from_pretrained(request.param, local_files_only=True)
    except OSError:
        pytest.skip(f"{request.param} tokenizer not in the local HF cache")


MESSAGES = [
    {"role": "user", "content": "What did the author eat on 2021-11-19?"},
    {"role": "assistant", "content": "Leftovers [Auto biography:p.1]."},
    {"role": "user", "content": "And what were they listening to?"},
]


def test_render_splits_context_block_from_conversation(tokenizer):
    prefix, prompt = teacher.render(tokenizer, MESSAGES, context="## [Auto biography:p.1] date: 2021-11-19\n\nD: ...")
    text = tokenizer.decode(prefix)
    assert text.startswith("<|im_start|>system\n" + teacher.INSTRUCTIONS) and text.endswith("<|im_end|>\n")
    # The student's prompt does not depend on the context, and the teacher's
    # prompt is the context block followed by exactly those tokens.
    assert teacher.render(tokenizer, MESSAGES)[1] == prompt
    assert teacher.render(tokenizer, MESSAGES, context="other")[1] == prompt
    assert tokenizer.decode(prompt).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # Without a system turn in the conversation, this is what the chat template
    # makes of [system(context), *messages].
    full = tokenizer.apply_chat_template(
        [{"role": "system", "content": f"{teacher.INSTRUCTIONS}\n\nx"}, *MESSAGES],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert teacher.render(tokenizer, MESSAGES, context="x")[0] + prompt == tokenizer.encode(full, add_special_tokens=False)


def test_render_no_context_is_base_mode(tokenizer):
    prefix, prompt = teacher.render(tokenizer, MESSAGES[:1])
    assert prefix == []
    assert tokenizer.decode(prompt).startswith("<|im_start|>user\n")


def test_render_keeps_the_callers_system_turn_after_the_context(tokenizer):
    # The template sees only the caller's messages (one system turn, first), so
    # it never has to accept a second system turn: the context block is
    # prepended as tokens.
    messages = [{"role": "system", "content": "Answer briefly."}, *MESSAGES]
    with pytest.raises(Exception, match="System message must be at the beginning"):
        tokenizer.apply_chat_template([{"role": "system", "content": "x"}, *messages], tokenize=False)
    prefix, prompt = teacher.render(tokenizer, messages, context="x")
    assert tokenizer.decode(prompt).startswith("<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\n")
    assert prompt == teacher.render(tokenizer, messages)[1]
    assert tokenizer.decode(prefix + prompt).count("<|im_start|>system") == 2


def test_completion_ids_end_with_im_end(tokenizer):
    ids = teacher.completion_ids(tokenizer, "Cat Stevens [Auto biography:p.1].")
    assert tokenizer.decode(ids) == "Cat Stevens [Auto biography:p.1].<|im_end|>"
    assert teacher.completion_ids(tokenizer, [1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------- tiny model


@pytest.fixture(scope="module")
def tiny_model():
    import torch
    from transformers.models import qwen3_5
    torch.manual_seed(0)
    config = qwen3_5.Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        max_position_embeddings=512,
    )
    return qwen3_5.Qwen3_5ForCausalLM(config).to(torch.float32).eval()


def _reference(model, ids: list[int], T: int, k: int):
    """Top-k of the last T predictions from one uncached forward over `ids`."""
    import torch

    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -T - 1 : -1]
    top = torch.log_softmax(logits.float(), -1).topk(k, -1)
    return top.indices.tolist(), top.values


def test_scorer_with_prefilled_cache_matches_full_forward(tiny_model):
    import torch

    prefix, prompt, completion = list(range(3, 40)), [50, 51, 52], [60, 61, 62, 63]
    scorer = teacher.Scorer(tiny_model, prefill_chunk=8)  # several prefill chunks, the last one partial
    scorer.prefill(prefix)
    assert scorer.cache.get_seq_length() == len(prefix)

    ids, lps = scorer.score_ids(prompt, completion, top_k=5)
    ref_ids, ref_lps = _reference(tiny_model, prefix + prompt + completion, len(completion), 5)
    assert ids == ref_ids
    torch.testing.assert_close(torch.tensor(lps), ref_lps, atol=1e-4, rtol=1e-4)


def test_scoring_does_not_change_the_prefilled_cache(tiny_model):
    scorer = teacher.Scorer(tiny_model, prefill_chunk=16)
    scorer.prefill(list(range(3, 30)))
    first = scorer.score_ids([50, 51], [60, 61, 62], top_k=3)
    scorer.score_ids([70, 71, 72, 73], [80], top_k=3)  # a different conversation in between
    assert scorer.score_ids([50, 51], [60, 61, 62], top_k=3) == first
    assert scorer.cache.get_seq_length() == 27


def test_fork_cache_shares_attention_kv(tiny_model):
    scorer = teacher.Scorer(tiny_model)
    scorer.prefill(list(range(3, 20)))
    fork = teacher.fork_cache(scorer.cache)
    attention = [i for i, t in enumerate(tiny_model.config.layer_types) if t == "full_attention"]
    linear = [i for i, t in enumerate(tiny_model.config.layer_types) if t == "linear_attention"]
    assert all(fork.layers[i].keys is scorer.cache.layers[i].keys for i in attention)
    assert all(fork.layers[i].recurrent_states[0] is not scorer.cache.layers[i].recurrent_states[0] for i in linear)
    assert scorer.cache_bytes() > 0


def test_cache_bytes_matches_the_prefilled_cache(tiny_model):
    scorer = teacher.Scorer(tiny_model)
    scorer.prefill(list(range(3, 40)))
    # The tiny model is float32 throughout.
    assert teacher.cache_bytes(tiny_model.config, 37, kv_bytes=4, state_bytes=4) == scorer.cache_bytes()


def test_scoring_without_context_is_the_base_model(tiny_model):
    import torch

    scorer = teacher.Scorer(tiny_model)
    ids, lps = scorer.score_ids([50, 51, 52], [60, 61], top_k=4, with_context=False)
    ref_ids, ref_lps = _reference(tiny_model, [50, 51, 52, 60, 61], 2, 4)
    assert ids == ref_ids
    torch.testing.assert_close(torch.tensor(lps), ref_lps, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- Modal options


def test_27b_uses_the_decorator_options_so_every_caller_shares_one_pool():
    assert teacher.TEACHER_OPTIONS == {"gpu": ["H200", "H100"], "max_containers": 1, "scaledown_window": 300}
    assert teacher.teacher_options(teacher.MODEL_ID) == {}


def test_other_models_override_the_gpu():
    assert teacher.teacher_options(teacher.SMOKE_MODEL_ID) == {"gpu": "L4"}
    assert teacher.teacher_options(teacher.MODEL_ID, gpu="H100", scaledown_window=60) == {
        "gpu": "H100", "scaledown_window": 60}


# Captured from `modal container list --json` (modal 1.5.6.dev19) on
# 2026-10-15: one container starting ("Pending"), one running; app names and
# ids are the real ones except that the teacher rows were relabeled.
REAL_CONTAINER_LIST = """[
  {
    "container_id": "ta-01M510DBSTWR9MS9H9H5TNJYJR",
    "app_id": "ap-M2fht6tXbRgQSq9RwUclnJ",
    "app_name": "cartridges-teacher",
    "start_time": "Pending"
  },
  {
    "container_id": "ta-01M510E1A82FNTN01T6WY426WR",
    "app_id": "ap-OTHERAPPxxxxxxxxxxxxxx",
    "app_name": "cartridges-capture-sleeper",
    "start_time": "2026-10-15 17:05:21-07:00"
  },
  {
    "container_id": "ta-01M510F2C76MCK1G9QVHBKD76R",
    "app_id": "ap-M2fht6tXbRgQSq9RwUclnJ",
    "app_name": "cartridges-teacher",
    "start_time": "2026-10-15 17:04:58-07:00"
  }
]
"""


def _fake_cli(listing: str, calls: list):
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=listing if cmd[2] == "list" else "")

    return run


def test_stop_stops_only_this_apps_containers_from_real_cli_output():
    calls = []
    assert teacher.stop(run=_fake_cli(REAL_CONTAINER_LIST, calls), recheck_s=0) == [
        "ta-01M510DBSTWR9MS9H9H5TNJYJR", "ta-01M510F2C76MCK1G9QVHBKD76R"]
    assert calls[0] == ["modal", "container", "list", "--json"]
    # --yes (`modal container stop --help`: "-y, --yes  Run without pausing for
    # confirmation"); without it the prompt aborts when there is no TTY.
    assert calls[1:] == [["modal", "container", "stop", "--yes", "ta-01M510DBSTWR9MS9H9H5TNJYJR"],
                         ["modal", "container", "stop", "--yes", "ta-01M510F2C76MCK1G9QVHBKD76R"]]
    assert teacher.stop(run=_fake_cli("[]\n", [])) == []


def test_stop_warns_when_containers_run_but_none_are_the_teachers(caplog):
    only_other = '[{"container_id": "ta-x", "app_id": "ap-y", "app_name": "other", "start_time": "Pending"}]'
    with caplog.at_level("WARNING"):
        assert teacher.stop(run=_fake_cli(only_other, []), recheck_s=0) == []
    assert "none of app 'cartridges-teacher'" in caplog.text and "other" in caplog.text


def test_stop_raises_on_an_unexpected_cli_format():
    # The display column names, which the first stop() wrongly expected.
    old_shape = '[{"Container ID": "ta-1", "App Name": "cartridges-teacher"}]'
    with pytest.raises(RuntimeError, match="unexpected"):
        teacher.stop(run=_fake_cli(old_shape, []))


def test_teacher_cls_drops_group_so_every_caller_shares_one_pool(monkeypatch, caplog):
    made = []

    class FakeCls:
        def with_options(self, **options):
            made.append(("options", options))
            return self

        def with_concurrency(self, **kw):
            made.append(("concurrency", kw))
            return self

        def __call__(self, **params):
            made.append(("params", params))
            return params

    monkeypatch.setattr(teacher.modal.Cls, "from_name", staticmethod(lambda app, name: FakeCls()))
    with caplog.at_level("WARNING"):
        assert teacher.teacher_cls("Teacher", teacher.MODEL_ID, group="exp-20261015-x") == {"model_id": teacher.MODEL_ID}
    assert "group" in caplog.text
    assert made == [("params", {"model_id": teacher.MODEL_ID})]  # no options, no group: the one 27B pool


def test_stop_reports_a_container_rescheduled_after_the_stop(caplog):
    import json

    before = [{"container_id": "ta-1", "app_id": "a", "app_name": teacher.APP_NAME, "start_time": "x"}]
    after = [{"container_id": "ta-2", "app_id": "a", "app_name": teacher.APP_NAME, "start_time": "Pending"}]
    listings = iter([before, after])

    def run(cmd, **kw):
        return SimpleNamespace(stdout=json.dumps(next(listings)) if cmd[2] == "list" else "")

    slept = []
    with caplog.at_level("ERROR"):
        assert teacher.stop(run=run, recheck_s=15, sleep=slept.append) == ["ta-1"]
    assert slept == [15] and "ta-2" in caplog.text and "rescheduled" in caplog.text


class FakeCall:
    def __init__(self, result=None, error=None, block=False, delay=0.0):
        self.result, self.error, self.block, self.delay, self.cancelled = result, error, block, delay, False

    def get(self, timeout=None):
        if self.block:
            raise KeyboardInterrupt  # the user hits Ctrl-C while we wait on this call
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result

    def cancel(self):
        self.cancelled = True


def _fake_teacher(calls):
    """calls[i] answers conversation [i + 1] (spawns arrive in any order under concurrency)."""
    spawned = []
    lock = threading.Lock()

    def spawn(conversation, **kw):
        with lock:
            spawned.append((conversation, kw))
        return calls[conversation[0] - 1]

    return SimpleNamespace(generate=SimpleNamespace(spawn=spawn)), spawned


def test_generate_all_returns_results_in_order_and_none_for_failures():
    calls = [FakeCall(result=["a"]), FakeCall(error=RuntimeError("boom")), FakeCall(result=["c"])]
    fake, spawned = _fake_teacher(calls)
    assert teacher.generate_all(fake, [[1], [2], [3]], max_tokens=8) == [["a"], None, ["c"]]
    assert sorted(spawned) == [([1], {"max_tokens": 8}), ([2], {"max_tokens": 8}), ([3], {"max_tokens": 8})]
    assert not any(c.cancelled for c in calls)


def test_generate_all_runs_requests_concurrently():
    # Structural, not wall-clock (a loaded CI box made a timing bound flaky): count gets in flight at once.
    lock, live, peak = threading.Lock(), [0], [0]
    gate = threading.Barrier(4, timeout=5)  # 4 gets must be waiting together, or the barrier times out

    class Concurrent(FakeCall):
        def get(self, timeout=None):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            gate.wait()
            with lock:
                live[0] -= 1
            return self.result

    calls = [Concurrent(result=[i]) for i in range(1, 9)]
    fake, _ = _fake_teacher(calls)
    assert teacher.generate_all(fake, [[i] for i in range(1, 9)], window=4) == [[i] for i in range(1, 9)]
    assert peak[0] == 4  # exactly `window` requests were outstanding together


def test_generate_all_cancels_every_call_on_interrupt():
    calls = [FakeCall(result=["a"], delay=0.3), FakeCall(block=True), FakeCall(result=["c"], delay=0.3)]
    fake, _ = _fake_teacher(calls)
    with pytest.raises(KeyboardInterrupt):
        teacher.generate_all(fake, [[1], [2], [3]], window=3)
    assert all(c.cancelled for c in calls)  # cancelling the request cancels each spawned Modal call


def test_stop_can_target_only_the_sessions_container():
    calls = []
    assert teacher.stop(run=_fake_cli(REAL_CONTAINER_LIST, calls), recheck_s=0,
                        container_ids=["ta-01M510F2C76MCK1G9QVHBKD76R", "ta-not-running"]) == ["ta-01M510F2C76MCK1G9QVHBKD76R"]
    assert calls[1:] == [["modal", "container", "stop", "--yes", "ta-01M510F2C76MCK1G9QVHBKD76R"]]


def test_teacher_cls_max_inputs_sets_modal_concurrency_and_vllm_max_num_seqs(monkeypatch):
    made = []

    class FakeCls:
        def with_options(self, **options):
            made.append(("options", options))
            return self

        def with_concurrency(self, **kw):
            made.append(("concurrency", kw))
            return self

        def __call__(self, **params):
            made.append(("params", params))
            return params

    monkeypatch.setattr(teacher.modal.Cls, "from_name", staticmethod(lambda app, name: FakeCls()))
    teacher.teacher_cls("Teacher", teacher.SMOKE_MODEL_ID, gpu="H100", max_inputs=128)
    assert made == [("options", {"gpu": "H100"}), ("concurrency", {"max_inputs": 128}),
                    ("params", {"model_id": teacher.SMOKE_MODEL_ID, "max_num_seqs": 128})]


def test_instructions_require_bracketed_citations():
    s = teacher.INSTRUCTIONS
    # Concrete examples only: 4B copied "[<journal>:p.<N>]" placeholders verbatim (pilot vuuap1dl).
    assert "<journal>" not in s and "<N>" not in s
    assert "exactly as its heading names it, e.g. [Morning Pages:p.63] or [Auto biography:p.12]" in s
    assert "never a page number alone" in s
    assert "questionnaire" in s and "300-500 words" in s
    assert "If the journals don't show it, say so" in s
    assert "Never present the question's wording as a journal quote" in s


def test_score_returns_chosen_token_logprobs_matching_a_full_forward(tiny_model):
    import torch

    prefix, prompt, completion = list(range(3, 30)), [50, 51], [60, 61, 62, 63, 64]
    scorer = teacher.Scorer(tiny_model, prefill_chunk=8)
    scorer.prefill(prefix)
    top_ids, top_lps, chosen = scorer.score(prompt, completion, top_k=3, chunk=2)  # several chunks, last partial
    with torch.inference_mode():
        logits = tiny_model(input_ids=torch.tensor([prefix + prompt + completion])).logits[0, -len(completion) - 1 : -1]
    ref = torch.log_softmax(logits.float(), -1)
    torch.testing.assert_close(torch.tensor(chosen), ref[torch.arange(len(completion)), torch.tensor(completion)],
                               atol=1e-4, rtol=1e-4)
    assert len(top_ids) == len(top_lps) == len(completion) and scorer.score_ids(prompt, completion, top_k=3)[0] == top_ids
    base = scorer.score(prompt, completion, top_k=1, with_context=False)[2]
    with torch.inference_mode():
        blog = tiny_model(input_ids=torch.tensor([prompt + completion])).logits[0, -len(completion) - 1 : -1]
    bref = torch.log_softmax(blog.float(), -1)[torch.arange(len(completion)), torch.tensor(completion)]
    torch.testing.assert_close(torch.tensor(base), bref, atol=1e-4, rtol=1e-4)


def test_27b_scorer_defaults_to_an_h200(monkeypatch):
    made = []

    class FakeCls:
        def with_options(self, **options):
            made.append(options)
            return self

        def __call__(self, **params):
            return params

    monkeypatch.setattr(teacher.modal.Cls, "from_name", staticmethod(lambda app, name: FakeCls()))
    teacher.teacher_cls("TeacherScorer", teacher.MODEL_ID)
    teacher.teacher_cls("Teacher", teacher.MODEL_ID)  # the Teacher keeps its decorator options (no with_options)
    assert made == [{"gpu": "H200", "max_containers": 1}]  # one scorer container, never a fan-out


def test_scorer_attention_name_is_not_taken_for_a_flash_kernel():
    from transformers.utils.generic import is_flash_attention_requested

    assert not is_flash_attention_requested(requested_attention_implementation=teacher.SCORER_ATTENTION)
