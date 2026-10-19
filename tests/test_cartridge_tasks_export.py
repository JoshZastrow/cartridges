from cartridges.cartridge.tasks.export import render_md
from cartridges.cartridge.tasks.types import Task
from cartridges.cartridge.tasks.validate import task_id, text_hash


def task(text, kind, **kw):
    h = text_hash(text)
    return Task(task_id=task_id(h), text=text, text_hash=h, kind=kind, **{"origin": "synthetic", "split": "train",
                                                                          "status": "validated", **kw})


def test_render_md_groups_by_kind_with_provenance():
    parent = task("What does my {framework} say about me?", "framework_lens", origin="human_template",
                  split="exemplar")
    tasks = [
        task("Apply Stoicism to my week.", "framework_lens", style="typed", framework="Stoicism",
             source_citations=("A:p.1",), ref_citations=("A:p.1",), ref_evidence=("a quote",),
             parent_ids=(parent.task_id,)),
        task("What did I do in May?", "recall", status="rejected", reject_reason="ungrounded"),
    ]
    md = render_md(tasks, {parent.task_id: parent}, "Tasks: test")
    assert md.index("## framework_lens (1)") < md.index("## recall (1)")
    assert "### 1. validated · synthetic · train · typed · Stoicism" in md
    assert "> Apply Stoicism to my week." in md and "- focus pages: A:p.1" in md
    assert '- quote: "a quote"' in md and "- exemplar: What does my {framework} say about me?" in md
    assert "### 1. rejected: ungrounded" in md
