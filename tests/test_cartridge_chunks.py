import json

import pytest

from cartridges.batch.outputs import collate
from cartridges.cartridge.chunks import approx_tokens, grounded_qa, qa_messages, sample_chunks, timeline_pages
from cartridges.cartridge.selfstudy import KINDS, parse_seed_reply, seed_prompts


def write_transcript(out, journal, pages):
    """pages: [(page number, date label, text)] in the transcript.md format."""
    d = out / journal
    d.mkdir(parents=True)
    sections = [f"## p. {n} - date: {label}\n\n{text}\n" for n, label, text in pages]
    (d / "transcript.md").write_text("\n---\n".join(s.rstrip() for s in sections) + "\n")


@pytest.fixture
def out(tmp_path):
    write_transcript(tmp_path, "A", [
        (n, f"2021-11-{n:02d}" if n % 3 == 1 else f"2021-11-{n - (n - 1) % 3:02d} (cont.)", f"M: content-{n}. " * 40)
        for n in range(1, 31)
    ])
    write_transcript(tmp_path, "B", [(n, f"2022-01-{n:02d}", f"S: walk {n}. " * 50) for n in range(1, 11)])
    collate(tmp_path)  # teacher_context.md
    return tmp_path


def test_chunks_are_spans_of_teacher_context_within_token_bounds(out):
    context = (out / "teacher_context.md").read_text()
    found = sample_chunks(out / "timeline.md", n=30, min_tokens=200, max_tokens=500, seed=3)
    for c in found:
        a, b = c.char_span
        assert context[a:b] == c.text
        assert c.text.startswith(f"## [{c.citations[0]}] date: ")
        assert len(c.citations) == len(c.pages)
        assert approx_tokens(c.text) <= 500
    first = found[0]
    assert first.date_range[0] <= first.date_range[1]


def test_sampled_token_share_follows_the_corpus(tmp_path):
    # Short pages in 2021 (A), long pages in 2022 (B): 75% of pages but a third of the tokens.
    write_transcript(tmp_path, "A", [(n, "2021-03-01", "short " * 50) for n in range(1, 121)])
    write_transcript(tmp_path, "B", [(n, "2022-03-01", "long page " * 120) for n in range(1, 41)])
    tokens = {p.citation: approx_tokens(p.section) for p in timeline_pages(tmp_path)}
    found = sample_chunks(tmp_path / "timeline.md", 2000, 512, 2048, seed=0)
    sampled = [c for ch in found for c in ch.citations]
    for journal in ("A", "B"):
        corpus = sum(t for c, t in tokens.items() if c.startswith(journal)) / sum(tokens.values())
        share = sum(tokens[c] for c in sampled if c.startswith(journal)) / sum(tokens[c] for c in sampled)
        assert abs(share / corpus - 1) < 0.2, (journal, share, corpus)


def test_sampling_is_deterministic_by_seed(out):
    def ids(seed):
        return [c.id for c in sample_chunks(out / "timeline.md", 10, 200, 500, seed=seed)]

    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_chunks_never_include_excluded_pages(out):
    held = {"A:p.13", "A:p.14", "B:p.2"}
    found = sample_chunks(out / "timeline.md", 100, 200, 2000, seed=0, exclude_pages=held)
    pages = {c for ch in found for c in ch.citations}
    assert not pages & held
    assert len(pages) == len(timeline_pages(out)) - len(held)  # every other page is reachable


def test_grounded_qa_keeps_verified_pairs_only(out):
    chunk = sample_chunks(out / "timeline.md", 1, 200, 500, seed=0)[0]
    assert chunk.citations[0] in qa_messages(chunk)[0]["content"]
    n = chunk.pages[0][0]
    good = {"question": "How content was I on 2021-11-13?", "answer": str(n), "answer_type": "number",
            "citations": [chunk.citations[0]], "evidence": chunk.pages[0][2][:20]}
    reply = json.dumps({"pairs": [good, {**good, "citations": ["C:p.1"]}, {**good, "evidence": "made up"}]})
    assert grounded_qa(chunk, reply) == [{**good, "chunk_id": chunk.id}]
    assert grounded_qa(chunk, "not json") == []
    wrapped = f"<think>\nLet me read the pages.\n</think>\n\n```json\n{reply}\n```"
    assert grounded_qa(chunk, wrapped) == [{**good, "chunk_id": chunk.id}]


def test_seed_prompts_cover_the_paper_families(out):
    chunk = sample_chunks(out / "timeline.md", 1, 200, 500, seed=0)[0]
    seeds = seed_prompts(chunk)
    assert [s.kind for s in seeds] == list(KINDS)
    for s in seeds:
        system, user = s.messages
        assert system["role"] == "system" and chunk.text in system["content"]
        assert user["role"] == "user" and user["content"]
    assert seed_prompts(chunk, seed=5) == seed_prompts(chunk, seed=5)
    assert "structuring" not in KINDS
    for seed in range(20):  # examples are placeholder templates, never named corpus events
        for s in seed_prompts(chunk, ("question", "summarization"), seed=seed):
            assert "{" in s.messages[1]["content"].split("pattern like", 1)[1].split("'")[1]
    for s in seeds:  # every seed asks for the question with its citations and evidence
        assert '"citations"' in s.messages[1]["content"] and '"evidence"' in s.messages[1]["content"]


def test_parse_seed_reply_checks_reference_against_chunk(out):
    chunk = sample_chunks(out / "timeline.md", 1, 200, 500, seed=0)[0]
    page, quote = chunk.citations[0], chunk.pages[0][2][:40]
    ref = {"question": "What did I do that morning?", "citations": [page], "evidence": [quote]}
    wrapped = f"<think>ok</think>\n```json\n{json.dumps(ref)}\n```"
    assert parse_seed_reply(chunk, wrapped) == {**ref, "grounded": True}
    assert parse_seed_reply(chunk, json.dumps({**ref, "citations": ["Nope:p.1"]}))["grounded"] is False
    assert parse_seed_reply(chunk, json.dumps({**ref, "evidence": ["made up entirely"]}))["grounded"] is False
    assert parse_seed_reply(chunk, json.dumps({**ref, "evidence": []}))["grounded"] is False
    assert parse_seed_reply(chunk, "What did I do?") is None  # not JSON
    assert parse_seed_reply(chunk, json.dumps({"citations": [page]})) is None  # no question
