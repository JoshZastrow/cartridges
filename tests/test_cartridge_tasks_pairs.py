"""Question-generator eval pairs (tasks/pairs.py) with fakes for the Teacher and
the Weave client: no GPU, no network."""

from __future__ import annotations

import datetime as dt
import uuid
from types import SimpleNamespace

import pytest

from cartridges.cartridge.tasks import pairs as pr
from cartridges.cartridge.tasks import types


def make_task(i: int, run_id: str = "run1", status: str = "validated", **kw) -> types.Task:
    fields = {
        "task_id": uuid.uuid5(uuid.NAMESPACE_URL, f"t{i}"), "text": f"Question {i}?", "text_hash": f"h{i}",
        "kind": "recall", "style": "typed", "origin": "synthetic", "split": "train", "status": status,
        "reject_reason": None if status == "validated" else "empty", "run_id": run_id,
        "source_citations": ("Auto biography:p.1",),
    }
    return types.Task(**{**fields, **kw})


def fake_generate(fail_bare: set[str] = frozenset()):
    calls = []

    def generate(batch, **kw):
        calls.append(kw)
        tag = "journal" if kw["with_context"] else "bare"
        out = []
        for m in batch:
            q = m[-1]["content"]  # the task is the last turn (label.task_messages may prepend a system turn)
            out.append(None if tag == "bare" and q in fail_bare else SimpleNamespace(text=f"{tag}: {q}"))
        return out

    generate.calls = calls
    return generate


def test_run_query_asks_for_the_runs_synthetic_tasks_in_the_kind_mix():
    from cartridges.cartridge.tasks.taxonomy import KIND_MIX

    q = pr.run_query(["run1", "run2"], 100, seed=3)
    assert (q.split, q.origins, q.run_ids, q.n, q.seed) == ("train", {"synthetic"}, {"run1", "run2"}, 100, 3)
    assert dict(q.mix) == KIND_MIX


def test_run_refuses_a_sample_with_shortfall():
    def sample(query):
        return types.Sample(query=query, tasks=(), shortfall={"recall": 3}, sample_hash="x")

    with pytest.raises(ValueError, match="shortfall"):
        pr.run(sample, ["run1"], fake_generate(), n=10)


def test_make_pairs_sends_a_bare_user_turn_to_both_systems():
    seen = []

    def generate(batch, **kw):
        seen.extend(batch)
        return [SimpleNamespace(text="x") for _ in batch]

    pr.make_pairs([make_task(0)], generate, seed=0)
    assert seen == [[{"role": "user", "content": "Question 0?"}]] * 2


def test_make_pairs_uses_same_budget_greedy_and_drops_failures():
    tasks = [make_task(i) for i in range(40)]
    gen = fake_generate(fail_bare={"Question 3?"})
    got, dropped = pr.make_pairs(tasks, gen, seed=0)
    assert dropped == 1 and len(got) == 39
    assert [c["with_context"] for c in gen.calls] == [True, False]
    assert all(c["max_tokens"] == 1536 and c["temperature"] == 0.0 for c in gen.calls)
    p = got[0]
    assert p.journal_answer == "journal: Question 0?" and p.no_journal_answer == "bare: Question 0?"
    assert {p.answer_a, p.answer_b} == {p.journal_answer, p.no_journal_answer}
    assert 10 < sum(p.journal_is_a for p in got) < 30  # both orders occur
    assert [p.journal_is_a for p in got] == [p.journal_is_a for p in pr.make_pairs(tasks, gen, seed=0)[0]]


def test_blinded_output_hides_the_mapping():
    (p,) = pr.blind([make_task(0)], ["J"], ["N"], seed=0)
    visible = {"answer_a": p.answer_a, "answer_b": p.answer_b}
    assert set(visible.values()) == {"J", "N"} and "journal" not in str(visible)


@pytest.mark.parametrize("journal_is_a, pref, errors, want_pref, want_errors", [
    (True, "A much better", "A", 2, "journal"),
    (False, "A much better", "A", -2, "no_journal"),
    (False, "B better", "B", 1, "journal"),
    (True, "tie", "both", 0, "both"),
    (True, None, "none", None, "none"),
])
def test_unblind(journal_is_a, pref, errors, want_pref, want_errors):
    p = pr.blind([make_task(0)], ["J"], ["N"], seed=0)[0].model_copy(update={"journal_is_a": journal_is_a})
    values = {"question_realism": "plausible", "factual_errors": errors, "notes": "why"}
    if pref:
        values["preference"] = pref
    row = pr.unblind(p, values)
    assert (row["preference"], row["factual_errors"], row["realism"]) == (want_pref, want_errors, "plausible")
    assert (row["journal_answer"], row["no_journal_answer"]) == ("J", "N")


def test_pairs_round_trip(tmp_path):
    got = pr.blind([make_task(i) for i in range(3)], ["J"] * 3, ["N"] * 3, seed=1)
    got = [p.model_copy(update={"call_id": f"c{i}"}) for i, p in enumerate(got)]
    assert pr.read_pairs(pr.write_pairs(tmp_path / "x.pairs.jsonl", got)) == got


class FakeClient:
    def __init__(self, queues=(), feedback=None):
        self.queues, self.feedback, self.added = list(queues), feedback or {}, []

    def list_annotation_queues(self, name):
        return [SimpleNamespace(id=q) for q in self.queues if q == name]

    def create_annotation_queue(self, name, scorer_refs, description=""):
        self.queues.append(name)
        return name

    def add_calls_to_annotation_queue(self, queue_id, call_ids, display_fields):
        self.added.append((queue_id, call_ids, display_fields))

    def get_call(self, call_id):
        return SimpleNamespace(feedback=self.feedback.get(call_id, []))


def fb(name, value, minute):
    return SimpleNamespace(feedback_type=f"wandb.annotation.{name}", payload={"value": value},
                           created_at=dt.datetime(2026, 10, 15, 12, minute))


def test_create_queue_creates_once_and_adds_calls():
    client = FakeClient()
    assert pr.create_queue(client, "q", ["ref"], ["c1"]) == "q"
    assert pr.create_queue(client, "q", ["ref"], ["c2"]) == "q"
    assert client.queues == ["q"] and [a[1] for a in client.added] == [["c1"], ["c2"]]
    assert client.added[0][2] == ["inputs.question", "output.answer_a", "output.answer_b"]


def test_read_feedback_keeps_latest_annotation_per_spec():
    client = FakeClient(feedback={"c1": [
        fb("preference", "A better", 1), fb("preference", "B better", 5), fb("notes", "ok", 2),
        SimpleNamespace(feedback_type="wandb.reaction.1", payload={"emoji": "+1"}, created_at=dt.datetime(2026, 1, 1)),
    ]})
    assert pr.read_feedback(client, ["c1", "c2"]) == {"c1": {"preference": "B better", "notes": "ok"}, "c2": {}}


def test_spec_enums_match_the_unblind_scale():
    assert set(pr.SPECS["preference"][1]["enum"]) == set(pr._PREFERENCE_A)
    assert pr.SPECS["factual_errors"][1]["enum"] == ["none", "A", "B", "both"]


def test_log_pairs_runs_the_op_offline():
    """Without weave.init the op still runs (no trace, no call id)."""
    got = pr.blind([make_task(0)], ["J"], ["N"], seed=0)
    assert pr.log_pairs(got) == got


# ---------------------------------------------------------------- run: persist before Weave, resume


def fixed_sample(n: int = 5, sample_hash: str = "s1"):
    tasks = tuple(make_task(i) for i in range(n))

    def sample(query):
        return types.Sample(query=query, tasks=tasks, shortfall={}, sample_hash=sample_hash)

    return sample


def fake_log(fail_after: int | None = None, logged: list | None = None):
    """log_pairs stand-in: gives call ids and saves after each, failing after `fail_after` calls."""
    def log(pairs, save=lambda ps: None):
        out = list(pairs)
        for i, p in enumerate(out):
            if p.call_id:
                continue
            if fail_after is not None and len([q for q in out if q.call_id]) >= fail_after:
                raise ConnectionError("weave down")
            out[i] = p.model_copy(update={"call_id": f"c-{p.task_id[:4]}"})
            if logged is not None:
                logged.append(p.task_id)
            save(out)
        return out
    return log


@pytest.fixture
def weave_fakes(monkeypatch):
    import weave

    client = FakeClient()
    monkeypatch.setattr(weave, "init", lambda project: client)
    monkeypatch.setattr(pr, "publish_specs", lambda: ["ref"])
    return client


def test_weave_failure_keeps_answers_and_resume_skips_generation(tmp_path, monkeypatch, weave_fakes):
    monkeypatch.setattr(pr, "log_pairs", fake_log(fail_after=2))
    with pytest.raises(ConnectionError):
        pr.run(fixed_sample(), ["run1"], fake_generate(), n=5, out_dir=tmp_path)
    saved = pr.read_pairs(tmp_path / "qgen-eval-v1.pairs.jsonl")
    assert len(saved) == 5 and sum(bool(p.call_id) for p in saved) == 2  # answers + key kept, 2 calls logged
    assert all(p.journal_answer.startswith("journal:") and p.sample_hash == "s1" for p in saved)

    def no_gpu(batch, **kw):
        raise AssertionError("resume must not regenerate")

    logged = []
    monkeypatch.setattr(pr, "log_pairs", fake_log(logged=logged))
    report = pr.run(fixed_sample(), ["run1"], no_gpu, n=5, out_dir=tmp_path)
    assert report["resumed"] and report["pairs"] == 5 and len(logged) == 3  # only the unlogged ones
    final = pr.read_pairs(tmp_path / "qgen-eval-v1.pairs.jsonl")
    assert all(p.call_id for p in final) and [p.journal_is_a for p in final] == [p.journal_is_a for p in saved]
    assert weave_fakes.added[0][1] == [p.call_id for p in final]


def test_failure_before_weave_still_writes_pairs(tmp_path, monkeypatch):
    import weave

    def down(project):
        raise ConnectionError("no network")

    monkeypatch.setattr(weave, "init", down)
    with pytest.raises(ConnectionError):
        pr.run(fixed_sample(3), ["run1"], fake_generate(), n=3, out_dir=tmp_path)
    assert len(pr.read_pairs(tmp_path / "qgen-eval-v1.pairs.jsonl")) == 3


def test_pairs_file_of_another_sample_is_never_overwritten(tmp_path, monkeypatch, weave_fakes):
    monkeypatch.setattr(pr, "log_pairs", fake_log())
    pr.run(fixed_sample(sample_hash="old"), ["run1"], fake_generate(), n=5, out_dir=tmp_path)
    before = (tmp_path / "qgen-eval-v1.pairs.jsonl").read_text()
    with pytest.raises(FileExistsError):
        pr.run(fixed_sample(sample_hash="new"), ["run2"], fake_generate(), n=5, out_dir=tmp_path)
    assert (tmp_path / "qgen-eval-v1.pairs.jsonl").read_text() == before


# ---------------------------------------------------------------- batch order (cascade attention)


def test_journal_batch_finishes_before_the_no_journal_batch_starts():
    events = []

    def generate(batch, **kw):
        tag = "journal" if kw["with_context"] else "bare"
        events.append(("start", tag))
        out = [SimpleNamespace(text=f"{tag}: {m[0]['content']}") for m in batch]
        events.append(("end", tag))
        return out

    pr.make_pairs([make_task(i) for i in range(4)], generate, seed=0)
    assert events == [("start", "journal"), ("end", "journal"), ("start", "bare"), ("end", "bare")]


def test_modal_generate_drains_every_call_before_returning():
    """With the real wrapper (teacher.generate_all: spawn all, then get all): no
    no-journal call is spawned until every journal call has returned."""
    from cartridges.cartridge.data import modal_generate

    log = []

    class FakeMethod:
        def spawn(self, conv, **kw):
            tag = "journal" if kw["with_context"] else "bare"
            log.append(("spawn", tag))

            def get(*args, **kwargs):  # generate_all may pass timeout=
                log.append(("get", tag))
                return [SimpleNamespace(text=tag)]

            return SimpleNamespace(get=get, cancel=lambda: None)

    gen = modal_generate(SimpleNamespace(generate=FakeMethod()))
    pr.make_pairs([make_task(i) for i in range(3)], gen, seed=0)
    last_journal_get = max(i for i, e in enumerate(log) if e == ("get", "journal"))
    first_bare_spawn = log.index(("spawn", "bare"))
    assert last_journal_get < first_bare_spawn and log.count(("get", "bare")) == 3