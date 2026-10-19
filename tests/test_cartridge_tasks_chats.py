import json
from types import SimpleNamespace

from cartridges.cartridge.tasks.chats import (
    CLASSIFIER_SYSTEM,
    Verdict,
    classifier_messages,
    classify,
    extract,
    label_stats,
    load_conversations,
    parse_label,
    thread,
    validate,
)


def conversation(cid, parts_by_turn, created=1_700_000_000.0, branch=False):
    """A ChatGPT export conversation: a chain root -> turns; `branch` adds an
    abandoned sibling of the first user turn that is not on the current thread."""
    mapping = {"root": {"message": None, "parent": None}}
    parent = "root"
    for i, (role, parts) in enumerate(parts_by_turn):
        node = f"{cid}-{i}"
        mapping[node] = {"message": {"author": {"role": role}, "content": {"parts": parts}}, "parent": parent}
        parent = node
    if branch:
        mapping["stale"] = {"message": {"author": {"role": "user"}, "content": {"parts": ["edited away"]}},
                            "parent": "root"}
    return {"conversation_id": cid, "create_time": created, "title": f"t {cid}", "default_model_slug": "gpt-4o",
            "current_node": parent, "mapping": mapping}


VOICE = {"content_type": "audio_transcription", "text": "so I keep leaving early", "direction": "in"}
IMAGE = {"content_type": "image_asset_pointer", "asset_pointer": "file-1"}
CONVS = [
    conversation("a", [("system", [""]), ("user", ["Why do I push people away?"]), ("assistant", ["..."])],
                 branch=True),
    conversation("b", [("user", [IMAGE, VOICE]), ("assistant", ["ok"])]),
    conversation("c", [("user", [IMAGE]), ("user", ["describe it"])]),  # attachment-only first prompt
    conversation("d", [("assistant", ["hi"])]),  # no user turn
    conversation("e", [("user", ["why do I push  people away?"])], created=1_600_000_000.0),  # earlier duplicate
]


def test_thread_follows_current_node():
    assert [m["content"]["parts"] for m in thread(CONVS[0])] == [[""], ["Why do I push people away?"], ["..."]]


def test_extract_and_validate():
    prompts, empty = extract(CONVS)
    assert [p.conversation_id for p in prompts] == ["a", "b", "e"] and empty == 2
    assert prompts[1].text == "so I keep leaving early"  # voice transcription, image skipped
    kept, dups = validate(prompts)
    assert dups == 1 and sorted(p.conversation_id for p in kept) == ["b", "e"]  # the earlier of a/e


def test_load_conversations(tmp_path):
    (tmp_path / "conversations-000.json").write_text(json.dumps(CONVS[:2]))
    (tmp_path / "conversations-001.json").write_text(json.dumps(CONVS[2:]))
    assert [c["conversation_id"] for c in load_conversations(tmp_path)] == ["a", "b", "c", "d", "e"]


def test_classifier_messages_truncate_and_share_a_prefix():
    long, short = extract([conversation("x", [("user", ["x" * 5000])]), CONVS[0]])[0]
    m1, m2 = classifier_messages(long, max_chars=100), classifier_messages(short)
    assert m1[0] == m2[0] == {"role": "system", "content": CLASSIFIER_SYSTEM}
    assert m1[1]["content"].endswith("x" * 100 + " [...]")


def test_parse_label():
    ok = parse_label('```json\n{"category": "relationships", "task_kind": "self_synthesis", "relevant": true, '
                     '"reason": "asks why they push people away"}\n```')
    assert (ok.category, ok.task_kind, ok.relevant) == ("relationships", "self_synthesis", True)
    verdict = '{"category": "technical", "task_kind": null, "relevant": false, "reason": "%s"}'
    assert parse_label(verdict % "debug a script").category == "technical"
    for bad in ("not json", '{"category": "sports"}', '{"category": "unparsed"}', "[1]", None,
                '{"category": "technical", "task_kind": "debugging", "relevant": false, "reason": "x"}',
                '{"category": "technical", "task_kind": null, "relevant": "no", "reason": "x"}',
                verdict % ("x" * 121)):
        assert parse_label(bad).category == "unparsed"


def test_classify_writes_each_batch_and_stats():
    prompts = validate(extract(CONVS)[0])[0]
    calls, written = [], []

    def generate(batch, **kw):
        calls.append((len(batch), kw))
        return [SimpleNamespace(text='{"category": "relationships", "task_kind": null, "relevant": true, '
                                     '"reason": "x"}'), None][: len(batch)]

    labels = classify(prompts, generate, written.append, batch_size=1)
    kw = {"max_tokens": 96, "temperature": 0.0, "with_context": False, "json_schema": Verdict.model_json_schema()}
    assert calls == [(1, kw), (1, kw)] and [len(w) for w in written] == [1, 1]
    assert [lab.category for lab in labels] == ["relationships", "relationships"]
    stats = label_stats([*labels, parse_label(None)])
    assert stats["relevant"] == 2 and round(stats["unparsed_rate"], 2) == 0.33


def test_prompt_hash_changes_with_the_prompt_or_decoding(monkeypatch):
    from cartridges.cartridge.tasks import chats

    base = chats.prompt_hash()
    assert base == chats.prompt_hash()
    monkeypatch.setattr(chats, "CLASSIFIER_SYSTEM", chats.CLASSIFIER_SYSTEM + " Be brief.")
    assert chats.prompt_hash() != base
    monkeypatch.undo()
    monkeypatch.setattr(chats, "MAX_TOKENS", 128)
    assert chats.prompt_hash() != base


def test_promote_splits_by_kind_and_is_idempotent():
    from collections import Counter

    from cartridges.cartridge.tasks.chats import promote
    from cartridges.cartridge.tasks.types import ChatLabel, ChatPrompt
    from cartridges.cartridge.tasks.validate import text_hash

    def row(i, kind="self_synthesis", relevant=True, text=None):
        text = text or f"Why do I keep doing thing {i}?"
        p = ChatPrompt(conversation_id=f"c{i}", create_time=None, title=f"t{i}", model_slug=None, text=text,
                       text_hash=text_hash(text))
        return p, ChatLabel(category="self_reflection", task_kind=kind, relevant=relevant, reason="r",
                            raw_reply="{}"), "run-c"

    labeled = ([row(i) for i in range(10)] + [row(10 + i, "recall") for i in range(5)]
               + [row(20, relevant=False), row(21, kind=None), row(22, text="word " * 401)])
    tasks, dups = promote(labeled, existing_hashes=set(), seed=0)
    valid = [t for t in tasks if t.status == "validated"]
    assert dups == 0 and len(tasks) == 16 and len(valid) == 15  # not irrelevant, not kind-less
    assert Counter(t.split for t in valid if t.kind == "self_synthesis") == {"train": 9, "eval": 1}  # 90/10
    three_way = promote(labeled, set(), ratios={"train": 0.8, "eval": 0.1, "exemplar": 0.1})[0]
    assert Counter(t.split for t in three_way if t.kind == "self_synthesis" and t.status == "validated") == {
        "train": 8, "eval": 1, "exemplar": 1}
    assert [t.reject_reason for t in tasks if t.status == "rejected"] == ["too_long"]
    assert all(t.origin == "chat" and t.style is None and t.run_id == "run-c" for t in tasks)
    assert valid[0].metadata["conversation_id"].startswith("c") and valid[0].metadata["category"] == "self_reflection"
    assert promote(labeled, {t.text_hash for t in tasks}, seed=0) == ([], 16)  # re-running adds nothing
    assert promote(labeled, set(), seed=0)[0] == tasks  # deterministic
    custom = promote(labeled, set(), ratios={"train": 1, "eval": 0, "exemplar": 0})[0]
    assert {t.split for t in custom} == {"train"}
