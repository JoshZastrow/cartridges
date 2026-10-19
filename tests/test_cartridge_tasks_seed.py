from collections import Counter
from types import SimpleNamespace

import pytest

from cartridges.batch.outputs import collate
from cartridges.cartridge.tasks.seed import human_tasks, load_corpus, seed, to_template

ROWS = (
    [{"question": f"What does my framework {i} say about me?", "kind_v2": "framework_lens", "kind": "creative"}
     for i in range(9)]
    + [{"question": f"How did I change since year {i}?", "kind": "change_over_time"} for i in range(6)]
    + [{"question": f"Help me with situation {i}", "kind": "situation_help"} for i in range(6)]
    + [{"question": "What is my attachment style?", "kind": "framework_lens", "split": "eval"}]
)


def test_human_tasks_split_by_kind_deterministically():
    tasks, errors = human_tasks(ROWS, n_eval=8, seed=0)
    assert errors == []
    assert tasks == human_tasks(list(reversed(ROWS)), n_eval=8, seed=0)[0][::-1]
    assert Counter(t.split for t in tasks) == {"eval": 8, "exemplar": 14}
    evals = Counter(t.kind for t in tasks if t.split == "eval")
    assert evals == {"framework_lens": 1 + 3, "change_over_time": 2, "situation_help": 2}  # 7 left: 9:6:6 -> 3:2:2
    assert all(t.origin == "human" and t.status == "validated" and t.grounded is None for t in tasks)
    assert next(t for t in tasks if t.text == "What is my attachment style?").split == "eval"


def test_human_tasks_report_bad_rows():
    rows = [{"question": " ", "kind": "recall"}, {"question": "a", "kind": "creative"},
            {"question": "b", "kind": "recall", "split": "train"}, {"question": "c", "kind": "recall"},
            {"question": "C ", "kind": "recall"}]
    tasks, errors = human_tasks(rows)
    assert [t.text for t in tasks] == ["c"]
    assert [e.split(":")[1].strip() for e in errors] == [
        "empty question", "unknown kind 'creative'", "unknown split 'train'", "duplicate question"]


def test_to_template_links_its_original():
    original = human_tasks([{"question": "What did Sam say in Lisbon?", "kind": "recall"}], n_eval=0)[0][0]
    tpl = to_template(original, "What did {person} say in {place}?")
    assert (tpl.origin, tpl.split, tpl.parent_ids, tpl.kind) == ("human_template", "exemplar", (original.task_id,),
                                                                "recall")
    assert to_template(original, "what did sam say in  lisbon?") is None  # nothing replaced
    assert to_template(original, "") is None


def test_seed_inserts_humans_then_templates():
    inserted = []

    def generate(prompts, **kw):
        assert kw["with_context"] is False
        return [SimpleNamespace(text=p[0]["content"].split("Message:\n")[1].replace("framework", "{framework}"))
                for p in prompts]

    report = seed(ROWS, run_id="seed-1", generate=generate, insert=lambda ts: inserted.extend(ts) or len(ts))
    assert report["inserted_human"] == 22 and report["errors"] == []
    templates = [t for t in inserted if t.origin == "human_template"]
    assert len(templates) == report["inserted_templates"] == 6  # the framework_lens exemplars
    assert report["untemplated"] == 14 - 6
    assert all(t.run_id == "seed-1" for t in inserted)


def test_load_corpus_pages(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    (d / "transcript.md").write_text(
        "## p. 1 - date: 2021-11-19 (?)\n\nfirst\n\n---\n## p. 2 - date: 2021-11-19 (cont.)\n\nsecond\n"
        "\n---\n## p. 3 - date: 2021-11-24\n\nthird\n"
    )
    collate(tmp_path)
    corpus, pages = load_corpus(tmp_path / "teacher_context.md", heldout={"A:p.3"})
    assert len(corpus.corpus_id) == 12 and corpus.name == "teacher_context.md"
    assert [(p.citation, p.date_uncertain, p.heldout, p.text) for p in pages] == [
        ("A:p.1", True, False, "first"), ("A:p.2", True, False, "second"), ("A:p.3", False, True, "third")]
    assert str(pages[1].date) == "2021-11-19"


def test_seed_inserts_humans_even_if_templating_fails():
    inserted = []

    def generate(prompts, **kw):
        raise TimeoutError("teacher hung")

    with pytest.raises(TimeoutError):
        seed(ROWS, run_id="seed-1", generate=generate, insert=lambda ts: inserted.extend(ts) or len(ts))
    assert len(inserted) == 22 and {t.origin for t in inserted} == {"human"}


def test_seed_without_templates_makes_no_teacher_call():
    inserted = []
    report = seed(ROWS, run_id="seed-1", generate=None, insert=lambda ts: inserted.extend(ts) or len(ts))
    assert (report["inserted_human"], report["inserted_templates"], report["untemplated"]) == (22, 0, 14)
