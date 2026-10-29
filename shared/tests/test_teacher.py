"""shared.teacher's pure prompt helpers, with a character-level fake tokenizer."""

from types import SimpleNamespace

from shared import teacher as t


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "") + ("<nothink>" if kw.get("enable_thinking") is False else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return 0


def test_context_block_parameters():
    assert t.context_block("CTX") == t.QWEN_SYSTEM_TURN.format(content="CTX")
    assert t.context_block("CTX", "Cite.", system_turn="[{content}]") == "[Cite.\n\nCTX]"


def test_render_prompt_ids_do_not_depend_on_context():
    tok, msgs = FakeTokenizer(), [{"role": "user", "content": "hi"}]
    prefix0, prompt0 = t.render(tok, msgs)
    prefix1, prompt1 = t.render(tok, msgs, context="journal", instructions="", system_turn="{content}|")
    assert prefix0 == [] and prompt0 == prompt1
    assert "".join(map(chr, prefix1)) == "journal|"
    assert "".join(map(chr, prompt0)).endswith("<assistant><nothink>")


def test_completion_ids_appends_stop_token():
    assert t.completion_ids(FakeTokenizer(), "ab") == [97, 98, 0]
    assert t.completion_ids(FakeTokenizer(), [5, 6]) == [5, 6]


def test_sampling_kwargs_and_topk():
    assert t.sampling_kwargs(1, 8, 0, 0.0, None, None, None) == {
        "n": 1, "max_tokens": 8, "logprobs": None, "seed": None, "temperature": 0.0}
    kw = t.sampling_kwargs(2, 8, 5, None, 1, {"type": "object"}, lambda json: ("so", json))
    assert kw["top_k"] == 20 and kw["structured_outputs"] == ("so", {"type": "object"})
    lp = lambda x: SimpleNamespace(logprob=x)  # noqa: E731
    assert t.topk_from_vllm([{3: lp(-2.0), 4: lp(-0.5), 9: lp(-1.0)}], k=2) == ([[4, 9]], [[-0.5, -1.0]])
