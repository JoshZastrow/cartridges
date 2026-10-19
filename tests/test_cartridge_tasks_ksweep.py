import datetime as dt

from cartridges.cartridge.tasks.ksweep import (
    kind_mix,
    real_pool,
    reference_rows,
    split_halves,
    table_rows,
    with_exemplars,
)
from cartridges.cartridge.tasks.types import Page, Recipe, Task
from cartridges.cartridge.tasks.validate import task_id, text_hash


def task(text, kind="self_synthesis", origin="chat", split="train", status="validated", parents=()):
    h = text_hash(text)
    return Task(task_id=task_id(h), text=text, text_hash=h, kind=kind, origin=origin, split=split, status=status,
                reject_reason=None if status == "validated" else "empty", parent_ids=parents)


HUMAN = task("What does my enneagram say about me?", "framework_lens", origin="human", split="exemplar")
TEMPLATE = task("What does my {framework} say about me?", "framework_lens", origin="human_template",
                split="exemplar", parents=(HUMAN.task_id,))
CHATS = [task(f"chat question {i}", "recall" if i % 3 else "self_synthesis") for i in range(12)]
TASKS = [HUMAN, TEMPLATE, *CHATS,
         task("eval question", split="eval"), task("synthetic", origin="synthetic"),
         task("rejected chat", status="rejected")]


def test_real_pool_and_halves():
    pool = real_pool(TASKS)
    assert len(pool) == 14 and {t.origin for t in pool} == {"human", "human_template", "chat"}
    a, b = split_halves(pool, seed=0)
    assert len(a) + len(b) == 14 and not {t.task_id for t in a} & {t.task_id for t in b}
    assert (HUMAN in a) == (TEMPLATE in a)  # a template stays with its original
    assert {t.task_id for t in split_halves(pool[::-1], seed=0)[0]} == {t.task_id for t in a}  # order-free


def test_kind_mix_has_a_floor():
    mix = kind_mix(CHATS, floor=0.05)
    assert abs(sum(mix.values()) - 1) < 1e-9 and min(mix.values()) > 0.03
    assert mix["recall"] > mix["self_synthesis"] > mix["build_artifact"]


def test_with_exemplars_is_paired_and_kind_first():
    page = Page(citation="A:p.1", journal="A", page_no=1, date=dt.date(2022, 1, 1), date_uncertain=False,
                heldout=False, text="x")
    recipes = [Recipe(kind="recall", style="typed", citations=(page.citation,)) for _ in range(3)]
    a = [t for t in CHATS]
    two = with_exemplars(recipes, a, 2, seed=0)
    assert [r.citations for r in two] == [r.citations for r in recipes]  # same recipes for every k
    by_id = {t.task_id: t for t in a}
    assert all(len(r.exemplar_ids) == 2 and all(by_id[i].kind == "recall" for i in r.exemplar_ids) for r in two)
    sixteen = with_exemplars(recipes, a, 16, seed=0)
    assert all(len(r.exemplar_ids) == 12 for r in sixteen)  # 8 recall, then the other kind; all there is
    assert with_exemplars(recipes, a, 0, seed=0)[0].exemplar_ids == ()
    assert with_exemplars(recipes, a, 2, seed=0) == two

    made = [task(f"draft {i}", "recall", origin="synthetic").model_copy(update={"source_citations": ("A:p.1",)})
            for i in range(3)]
    rows = table_rows(made, two, by_id, k=2)
    assert rows[0]["k"] == 2 and rows[0]["focus_pages"] == "A:p.1" and rows[0]["exemplars"].count("\n---\n") == 1
    assert reference_rows(CHATS[:2], [TASKS[-3]])[-1]["split"] == "eval"
