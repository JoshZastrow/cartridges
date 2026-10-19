import random
from collections import Counter

import pytest

from cartridges.cartridge.tasks.sampler import quotas, sample_tasks, select
from cartridges.cartridge.tasks.types import Task, TaskQuery
from cartridges.cartridge.tasks.validate import task_id, text_hash


def task(i, kind="recall", split="train", status="validated", origin="synthetic", style="typed"):
    text = f"task {i} {kind} {split} {status}"
    h = text_hash(text)
    return Task(
        task_id=task_id(h), text=text, text_hash=h, kind=kind, style=style, origin=origin, split=split,
        status=status, reject_reason=None if status == "validated" else "empty",
    )


POOL = (
    [task(i, "recall") for i in range(10)]
    + [task(i, "situation_help") for i in range(3)]
    + [task(i, "recall", split="eval", origin="human") for i in range(5)]
    + [task(i, "recall", status="rejected") for i in range(5)]
)


def test_quotas_use_largest_remainder():
    assert quotas(10, {"recall": 0.66, "situation_help": 0.34}) == {"recall": 7, "situation_help": 3}
    assert sum(quotas(7, {"recall": 1 / 3, "situation_help": 1 / 3, "self_synthesis": 1 / 3}).values()) == 7


def test_select_keeps_split_and_validated_only():
    s = select(POOL, TaskQuery(split="train", n=100, seed=0))
    assert len(s.tasks) == 13
    assert all(t.split == "train" and t.status == "validated" for t in s.tasks)
    assert len({t.task_id for t in s.tasks}) == 13  # without replacement


def test_select_reports_shortfall_without_refilling():
    s = select(POOL, TaskQuery(split="train", n=10, seed=0, mix={"recall": 0.5, "situation_help": 0.5}))
    kinds = [t.kind for t in s.tasks]
    assert kinds.count("recall") == 5 and kinds.count("situation_help") == 3
    assert s.shortfall == {"recall": 0, "situation_help": 2}


def test_select_is_deterministic_and_order_free():
    q = TaskQuery(split="train", n=6, seed=3, mix={"recall": 0.5, "situation_help": 0.5})
    shuffled = POOL[:]
    random.Random(1).shuffle(shuffled)
    assert select(POOL, q) == select(shuffled, q)
    assert select(POOL, q).sample_hash != select(POOL, q.model_copy(update={"seed": 4})).sample_hash


def test_select_applies_filters():
    first = select(POOL, TaskQuery(split="train", n=1, seed=0)).tasks[0]
    s = select(POOL, TaskQuery(split="train", n=100, seed=0, exclude=frozenset({first.task_id})))
    assert first not in s.tasks and len(s.tasks) == 12
    assert select(POOL, TaskQuery(split="eval", n=100, seed=0, origins=frozenset({"synthetic"}))).tasks == ()
    assert select(POOL, TaskQuery(split="train", n=100, seed=0, styles=frozenset({"dictated"}))).tasks == ()
    run = POOL[0].model_copy(update={"run_id": "r1"})
    assert select([run, *POOL[1:]], TaskQuery(split="train", n=100, seed=0, run_ids=frozenset({"r1"}))).tasks == (run,)


@pytest.mark.parametrize("bad", [{"n": 0}, {"mix": {"recall": 0.5}}, {"mix": {"gossip": 1.0}}])
def test_invalid_query_raises_value_error(bad):
    with pytest.raises(ValueError):
        TaskQuery(**{"split": "train", "n": 5, "seed": 0, **bad})


def test_sample_tasks_rechecks_the_fetch():
    s = sample_tasks(lambda q: POOL, TaskQuery(split="eval", n=100, seed=0))
    assert {t.split for t in s.tasks} == {"eval"}


def psych(i, framework, instrument):
    text = f"psych item {i} {instrument}"
    h = text_hash(text)
    return Task(task_id=task_id(h), text=text, text_hash=h, kind="framework_lens", framework=framework,
                origin="psych_test", split="train", status="validated", metadata={"instrument_id": instrument})


def test_origin_mix_then_kind_mix_and_framework_spread():
    big = [psych(i, "VIA", "ipip_via") for i in range(50)]  # one huge instrument
    small = [psych(100 + i, "attachment", inst) for i, inst in enumerate(["ecr"] * 5 + ["rsq"] * 5)]
    chats = [task(i, "recall", origin="chat") for i in range(5)] + [task(i, "situation_help", origin="chat")
                                                                     for i in range(5)]
    q = TaskQuery(split="train", n=20, seed=0, origin_mix={"chat": 0.5, "psych_test": 0.5},
                  mix={"recall": 0.5, "situation_help": 0.5})
    s = select([*big, *small, *chats, *POOL], q)
    by_origin = Counter(t.origin for t in s.tasks)
    assert by_origin == {"chat": 10, "psych_test": 10}
    assert Counter(t.kind for t in s.tasks if t.origin == "chat") == {"recall": 5, "situation_help": 5}
    psy = [t for t in s.tasks if t.origin == "psych_test"]
    assert Counter(t.framework for t in psy) == {"VIA": 5, "attachment": 5}  # even over frameworks
    per_instrument = Counter(t.metadata["instrument_id"] for t in psy if t.framework == "attachment")
    assert sorted(per_instrument.values()) == [2, 3]  # even over a framework's instruments
    assert sum(s.shortfall.values()) == 0 and set(s.shortfall) >= {"chat/recall", "psych_test/framework_lens"}
    short = select(chats, q.model_copy(update={"n": 40}))
    assert short.shortfall["psych_test/framework_lens"] == 20 and short.shortfall["chat/recall"] == 5
    assert select([*big, *small, *chats], q) == select([*chats, *small, *big][::-1], q)  # order-free
