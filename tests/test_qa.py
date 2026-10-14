import itertools
import pickle

import pytest

from cartridges.datagen.qa import Passage, Teacher, passages, verify

MOOD = "M: Passionate-9, warm-6, content-8."


def write_transcript(out, journal, pages):
    """pages: [(page number, date label, text)] in the transcript.md format."""
    d = out / journal
    d.mkdir(parents=True)
    sections = [f"## p. {n} - date: {label}\n\n{text}\n" for n, label, text in pages]
    (d / "transcript.md").write_text("\n---\n".join(s.rstrip() for s in sections) + "\n")


def test_passages_split_long_entries_into_runs_of_pages(tmp_path):
    write_transcript(tmp_path, "A", [
        (1, "2021-11-19 (est.)", "# Title"),  # too short alone: dropped
        (2, "2021-11-20", "x" * 400),
        (3, "2021-11-20 (cont.)", "y" * 400),
        (4, "2021-11-20 (cont.)", "z" * 400),
        (5, "2021-11-24", "w" * 400),
    ])
    found = passages(tmp_path, max_chars=900, min_chars=300)
    assert [p.id for p in found] == ["A:p.2-3", "A:p.4", "A:p.5"]
    assert found[0].citations == ["A:p.2", "A:p.3"]
    assert found[0].text.startswith("## [A:p.2] date: 2021-11-20\n\nxxx")
    assert "## [A:p.3] date: 2021-11-20 (cont.)" in found[0].text


PASSAGE = Passage("A", ((2, "2021-11-20", "S: Shopping for a couch.\n\n" + MOOD),))


def pair(**overrides):
    base = {
        "question": "How content did I feel the evening I shopped for a couch in Nov 2021?",
        "answer": "8",
        "answer_type": "number",
        "citations": ["A:p.2"],
        "evidence": "Passionate-9, warm-6,  Content-8",  # spacing and case differ
    }
    return {**base, **overrides}


@pytest.mark.parametrize(
    "overrides, ok",
    [
        ({}, True),
        ({"citations": ["A:p.3"]}, False),  # not a page of the passage
        ({"citations": []}, False),
        ({"evidence": "I felt calm and rested all day."}, False),  # made-up quote
        ({"answer": " "}, False),
        ({"answer_type": "essay"}, False),
    ],
)
def test_verify(overrides, ok):
    assert verify(pair(**overrides), PASSAGE) is ok


def test_teacher_pickles_without_its_client():
    teacher = Teacher("http://localhost:8000/v1", "teacher")
    teacher._client = object()  # stands in for an open client
    assert pickle.loads(pickle.dumps(teacher))._client is None


def test_dataset_yields_verified_pairs_only():
    pytest.importorskip("torch")
    from cartridges.datagen.qa_dataset import QAPairs

    def fake_teacher(passage):
        return [pair(), pair(evidence="not on the page at all"), pair(answer_type="number", answer="9")]

    out = list(itertools.islice(iter(QAPairs([PASSAGE], fake_teacher)), 4))
    assert [p["answer"] for p in out] == ["8", "9", "8", "9"]
    assert out[0]["passage_id"] == "A:p.2" and out[0]["date"] == "2021-11-20"
