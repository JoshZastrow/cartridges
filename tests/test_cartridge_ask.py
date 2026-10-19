from cartridges.cartridge.ask import prompt_ids
from cartridges.cartridge.tasks.label import CITATION_REMINDER, answer_messages
from cartridges.cartridge.teacher import INSTRUCTIONS, render


class CharTokenizer:
    """A chat template plus character-level ids: enough for teacher.render."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_ask_prompt_is_task_answer_format_without_context():
    tok, question = CharTokenizer(), "What did I write about the lake in 2021?"
    ids = prompt_ids(tok, question)
    assert ids == render(tok, answer_messages(question))[1]
    assert ids == render(tok, answer_messages(question), "some context")[1]  # context is never in the prompt
    text = decode(ids)
    assert text.startswith(f"<|im_start|>system\n{CITATION_REMINDER}<|im_end|>\n<|im_start|>user\n{question}")
    assert INSTRUCTIONS not in text
