"""Teacher labeling (tasks/label.py) and the Target Store (tasks/targets.py),
with fakes for the sampler, the Teacher and the store: no database, no GPU."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import pytest

from cartridges.cartridge import teacher
from cartridges.cartridge.tasks import label as lab
from cartridges.cartridge.tasks import targets, types

PAGES = ["Morning Pages:p.63", "Auto biography:p.1"]


def make_task(text: str = "How have I changed since 2021?", kind: str = "change_over_time", **kw) -> types.Task:
    fields = {
        "task_id": uuid.uuid5(uuid.NAMESPACE_URL, text),
        "text": text,
        "text_hash": text,
        "kind": kind,
        "style": "typed",
        "origin": "synthetic",
        "split": "train",
        "status": "validated",
    }
    return types.Task(**{**fields, **kw})


def make_completion(text: str = "You started journaling [Auto biography:p.1].", T: int = 3, k: int = 2,
                    finish_reason: str = "stop", **kw) -> teacher.Completion:
    fields = {
        "text": text,
        "prompt_ids": [1, 2, 3],
        "completion_ids": list(range(10, 10 + T)),
        "topk_ids": [[10 + t, 99][:k] for t in range(T)],
        "topk_logprobs": [[-0.1, -2.5][:k] for _ in range(T)],
        "finish_reason": finish_reason,
    }
    return teacher.Completion(**{**fields, **kw})


# ---------------------------------------------------------------- validate_label


def test_validate_label_accepts_a_cited_answer():
    assert lab.validate_label(make_task(), make_completion(), PAGES) == ("validated", None)


def test_validate_label_keeps_truncated_answers():
    assert lab.validate_label(make_task(), make_completion(finish_reason="length"), PAGES) == ("validated", None)


@pytest.mark.parametrize("completion, reason", [
    (make_completion(text="   "), "no_reply"),
    (make_completion(completion_ids=[]), "no_reply"),
    (make_completion(topk_ids=[[1, 2]]), "topk_mismatch"),
    (make_completion(topk_logprobs=[[-0.1, -1.0]] * 2), "topk_mismatch"),
    (make_completion(text="I'm sorry, but I can't help with that."), "refusal"),
    (make_completion(text="As an AI, I don't have feelings."), "refusal"),
    (make_completion(text="You wrote that [Morning Pages:p.999]."), "invalid_citation"),
])
def test_validate_label_rejects(completion, reason):
    assert lab.validate_label(make_task(), completion, PAGES) == ("rejected", reason)


def test_saying_the_journals_dont_say_is_not_a_refusal():
    text = "Your journals don't say what you ate that day."
    assert lab.validate_label(make_task(), make_completion(text=text), PAGES) == ("validated", None)


def test_validate_label_takes_page_objects():
    pages = [types.Page(citation=c, journal=c.split(":")[0], page_no=1, date=None, date_uncertain=False,
                        heldout=False, text="") for c in PAGES]
    assert lab.validate_label(make_task(), make_completion(), pages) == ("validated", None)


# ---------------------------------------------------------------- rows


def test_to_target_row_copies_the_completion():
    task, completion = make_task(), make_completion()
    row = lab.to_target_row(task, completion, "L1")
    assert row.label_id == "L1" and row.task_id == str(task.task_id)
    assert row.prompt_ids == completion.prompt_ids and row.completion_ids == completion.completion_ids
    assert row.topk_ids == completion.topk_ids and row.topk_logprobs == completion.topk_logprobs


def test_label_ids_are_reproducible():
    task = make_task()
    assert lab.label_id("run1", task.task_id) == lab.label_id("run1", task.task_id)
    assert lab.label_id("run1", task.task_id) != lab.label_id("run2", task.task_id)


def test_to_prompt_is_the_teacher_prompt_without_the_journal():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(teacher.SMOKE_MODEL_ID, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")
    task = make_task()
    assert lab.to_prompt(task, tok) == tuple(teacher.render(tok, lab.task_messages(task))[1])
    text = tok.decode(lab.to_prompt(task, tok))
    assert text.startswith("<|im_start|>system\n" + lab.CITATION_REMINDER + "<|im_end|>\n<|im_start|>user\nHow have I changed")


# ---------------------------------------------------------------- Target Store


def test_shard_round_trip_to_examples(tmp_path):
    rows = [lab.to_target_row(make_task(f"q{i}"), make_completion(T=2 + i), f"L{i}") for i in range(3)]
    uri = targets.write_shard(rows, targets.shard_uri("run1", 0, tmp_path))
    assert uri.endswith("run1/part-00000.parquet")
    back = list(targets.read_rows([uri]))
    assert [r.label_id for r in back] == ["L0", "L1", "L2"]
    assert back[2].completion_ids == rows[2].completion_ids
    examples = list(targets.read_examples([uri]))
    assert examples[1].id == "L1" and len(examples[1].topk_ids) == 3 == len(examples[1].completion_ids)
    # float32 storage
    assert examples[0].topk_logprobs[0] == pytest.approx([-0.1, -2.5], abs=1e-6)


def test_examples_feed_train_distill_to_tensors(tmp_path):
    from cartridges.cartridge.train_distill import to_tensors

    uri = targets.write_shard([lab.to_target_row(make_task(), make_completion(T=4), "L0")],
                              targets.shard_uri("r", 0, tmp_path))
    example = next(targets.read_examples([uri]))
    input_ids, topk_ids, topk_logprobs = to_tensors(example, max_seq_len=64)
    assert input_ids.shape == (1, 3 + 4) and topk_ids.shape == (4, 2) == topk_logprobs.shape


def _label_for(row, uri, i, status="validated"):
    return types.Label(
        label_id=uuid.UUID(row.label_id), task_id=uuid.UUID(row.task_id), run_id="r", teacher_model="m",
        corpus_id="c", instructions_hash="h", answer="a", finish_reason="stop", n_prompt_tokens=3,
        n_completion_tokens=len(row.completion_ids), status=status,
        reject_reason=None if status == "validated" else "refusal", targets_uri=uri, targets_row=i,
    )


def test_read_labeled_reads_only_rows_that_labels_point_at(tmp_path):
    # Row 1 is an orphan: written to the shard, but the store kept no labels row
    # for it (a re-label skipped on conflict). Row 2's label was later rejected.
    rows = [lab.to_target_row(make_task(f"q{i}"), make_completion(T=2 + i), uuid.uuid4()) for i in range(4)]
    uri = targets.write_shard(rows, targets.shard_uri("run1", 0, tmp_path))
    labels = [_label_for(rows[3], uri, 3), _label_for(rows[0], uri, 0), _label_for(rows[2], uri, 2, "rejected")]
    examples = list(targets.read_labeled(labels))
    assert [e.id for e in examples] == [rows[3].label_id, rows[0].label_id]
    assert examples[0].completion_ids == rows[3].completion_ids
    assert rows[1].label_id not in {e.id for e in examples}
    assert len(list(targets.read_examples([uri]))) == 4  # the inspection read still sees everything


def test_read_labeled_rejects_a_pointer_to_another_labels_row(tmp_path):
    rows = [lab.to_target_row(make_task(f"q{i}"), make_completion(), uuid.uuid4()) for i in range(2)]
    uri = targets.write_shard(rows, targets.shard_uri("run1", 0, tmp_path))
    with pytest.raises(ValueError):
        list(targets.read_labeled([_label_for(rows[0], uri, 1)]))


def test_label_output_reads_back_through_the_labels(tmp_path):
    tasks = [make_task("a"), make_task("b")]
    store = FakeStore(tasks)
    lab.label(lab.LabelConfig(run_id="r", corpus_id="c", n=2, seed=0, targets_root=str(tmp_path)), PAGES,
              sample=store.sample, generate=lambda msgs, **kw: [make_completion(T=2), make_completion(T=4)],
              write_shard=targets.write_shard, insert_labels=store.insert_labels)
    assert [len(e.completion_ids) for e in targets.read_labeled(store.inserted)] == [2, 4]


class FakeVolume:
    def __init__(self, root):
        self.root, self.puts = root, []

    def batch_upload(self, force=False):
        volume = self

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put_file(self, local, remote):
                volume.puts.append(remote)
                dest = volume.root / remote.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(Path(local).read_bytes())

        return Batch()


def test_volume_writer_uploads_the_shard_under_the_mount(tmp_path):
    volume = FakeVolume(tmp_path)
    write = targets.volume_writer(volume, mount="/cache")
    rows = [lab.to_target_row(make_task(), make_completion(), uuid.uuid4())]
    uri = targets.shard_uri("run1", 0, "/cache/targets")
    assert write(rows, uri) == "targets/run1/part-00000.parquet"  # Volume-relative
    assert volume.puts == ["/targets/run1/part-00000.parquet"]
    assert [r.label_id for r in targets.read_rows([tmp_path / "targets/run1/part-00000.parquet"])] == [rows[0].label_id]
    with pytest.raises(ValueError):
        write(rows, "/elsewhere/part.parquet")


# ---------------------------------------------------------------- orchestrator


class FakeStore:
    def __init__(self, tasks):
        self.tasks, self.queries, self.inserted, self.shards = tasks, [], [], {}

    def sample(self, query):
        # Like the store: skip labeled tasks (unlabeled_by) and excluded ones, at most n.
        self.queries.append(query)
        labeled = {lbl.task_id for lbl in self.inserted}
        left = [t for t in self.tasks if t.task_id not in labeled and t.task_id not in query.exclude
                and t.split == query.split and (query.origins is None or t.origin in query.origins)][: query.n]
        return types.Sample(query=query, tasks=tuple(left), shortfall={}, sample_hash=f"h{len(self.queries)}")

    def write_shard(self, rows, uri):
        self.shards[uri] = list(rows)
        return uri

    def insert_labels(self, labels):
        self.inserted.extend(labels)


def test_label_writes_validated_targets_and_every_label(tmp_path):
    tasks = [make_task("a", "recall"), make_task("b", "recall"), make_task("c", "self_synthesis")]
    answers = {
        "a": make_completion(T=3),
        "b": make_completion(text="I can't help with that."),
        "c": make_completion(T=5, finish_reason="length", text="No citations here."),
    }
    calls = []

    def generate(messages_list, **kw):
        calls.append(kw)
        return [answers[m[-1]["content"]] for m in messages_list]

    store = FakeStore(tasks)
    config = lab.LabelConfig(run_id="run1", corpus_id="c0", n=3, seed=7, kind_mix={"recall": 0.5, "self_synthesis": 0.5},
                             targets_root=str(tmp_path))
    report = lab.label(config, PAGES, sample=store.sample, generate=generate,
                       write_shard=store.write_shard, insert_labels=store.insert_labels)

    (query,) = store.queries
    assert (query.split, query.n, query.seed) == ("train", 3, 7)
    assert query.unlabeled_by == (teacher.MODEL_ID, "c0", lab.instructions_hash())
    assert calls == [{"max_tokens": 3072, "logprobs": 20, "temperature": 0.0}]

    uri = targets.shard_uri("run1", "h1", tmp_path)
    # Every answer with tokens keeps a targets row (the refusal too); status decides what training reads.
    assert [r.task_id for r in store.shards[uri]] == [str(t.task_id) for t in tasks]
    by_task = {lbl.task_id: lbl for lbl in store.inserted}
    assert store.shards[uri][0].label_id == str(by_task[tasks[0].task_id].label_id)
    assert len(store.inserted) == 3
    assert (by_task[tasks[0].task_id].targets_uri, by_task[tasks[0].task_id].targets_row) == (uri, 0)
    assert (by_task[tasks[2].task_id].targets_uri, by_task[tasks[2].task_id].targets_row) == (uri, 2)
    rejected = by_task[tasks[1].task_id]
    assert (rejected.status, rejected.reject_reason, rejected.targets_row) == ("rejected", "refusal", 1)
    assert rejected.instructions_hash == lab.instructions_hash() and rejected.corpus_id == "c0"

    stats = report.overall
    assert stats.n_tasks == 3 and stats.n_validated == 2
    assert stats.reject_reasons == {"refusal": 1}
    assert stats.truncation_rate == pytest.approx(1 / 3)
    assert stats.refusal_rate == pytest.approx(1 / 3)
    assert stats.cite_rate == pytest.approx(1 / 3)
    assert stats.citation_validity == 1.0
    assert stats.tokens_by_kind == {"recall": 3, "self_synthesis": 5}
    assert report.targets_uris == [uri] and report.sample_hashes == ["h1"]


def test_label_with_nothing_valid_writes_no_shard():
    store = FakeStore([make_task()])
    report = lab.label(
        lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0), PAGES, sample=store.sample,
        generate=lambda msgs, **kw: [make_completion(text="")],
        write_shard=store.write_shard, insert_labels=store.insert_labels,
    )
    assert store.shards == {} and report.targets_uris == []
    assert [(x.status, x.reject_reason, x.targets_row) for x in store.inserted] == [("rejected", "no_reply", None)]


def test_label_treats_a_failed_call_as_no_reply(tmp_path):
    store = FakeStore([make_task("a"), make_task("b")])
    report = lab.label(
        lab.LabelConfig(run_id="r", corpus_id="c", n=2, seed=0, targets_root=str(tmp_path)), PAGES,
        sample=store.sample, generate=lambda msgs, **kw: [None, make_completion()],
        write_shard=store.write_shard, insert_labels=store.insert_labels,
    )
    failed = store.inserted[0]
    assert (failed.status, failed.reject_reason, failed.finish_reason, failed.targets_row) == (
        "rejected", "no_reply", "error", None)
    assert store.inserted[1].targets_row == 0 and report.overall.n_validated == 1


def test_label_rejects_a_generate_that_drops_tasks():
    store = FakeStore([make_task("a"), make_task("b")])
    with pytest.raises(ValueError):
        lab.label(lab.LabelConfig(run_id="r", corpus_id="c", n=2, seed=0), PAGES, sample=store.sample,
                  generate=lambda msgs, **kw: [make_completion()],
                  write_shard=store.write_shard, insert_labels=store.insert_labels)


def test_label_passes_the_origin_mix_to_the_sampler():
    store = FakeStore([make_task()])
    mix = {"chat": 0.4, "synthetic": 0.35, "psych_test": 0.25}
    lab.label(lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0, origin_mix=mix), PAGES,
              sample=store.sample, generate=lambda msgs, **kw: [make_completion()],
              write_shard=store.write_shard, insert_labels=store.insert_labels)
    (query,) = store.queries
    assert dict(query.origin_mix) == mix and query.split == "train"


def test_label_config_defaults_to_the_approved_origin_mix():
    from cartridges.cartridge.tasks.taxonomy import LABEL_ORIGIN_MIX

    assert LABEL_ORIGIN_MIX == {"chat": 0.40, "synthetic": 0.35, "psych_test": 0.25}
    assert dict(lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0).origin_mix) == LABEL_ORIGIN_MIX


def test_label_runs_in_batches_with_one_shard_and_insert_per_batch(tmp_path):
    tasks = [make_task(f"q{i}", kind="recall" if i % 2 else "self_synthesis", origin="synthetic" if i < 3 else "chat",
                       split="train") for i in range(5)]
    store, logged = FakeStore(tasks), []
    inserts = []
    report = lab.label(
        lab.LabelConfig(run_id="r", corpus_id="c", n=5, seed=0, batch_size=2, origin_mix=None,
                        targets_root=str(tmp_path)),
        PAGES, sample=store.sample, generate=lambda msgs, **kw: [make_completion(T=2) for _ in msgs],
        write_shard=store.write_shard, insert_labels=lambda ls: (inserts.append(len(ls)), store.insert_labels(ls)),
        log=logged.append,
    )
    assert [q.n for q in store.queries] == [2, 2, 1] and inserts == [2, 2, 1]
    assert [q.seed for q in store.queries] == [0, 1, 2] and all(q.origin_mix is None for q in store.queries)
    assert len(report.targets_uris) == 3 == len(set(report.targets_uris)) == len(store.shards)
    assert report.overall.n_tasks == 5 and set(report.by_origin) == {"synthetic", "chat"}
    assert report.by_origin["chat"].n_tasks == 2 and report.by_origin["synthetic"].completion_tokens == 6
    # Cumulative W&B scalars after each batch.
    assert [x["label/all/n_tasks"] for x in logged] == [2, 4, 5]
    assert logged[-1]["label/chat/cite_rate"] == 1.0 and "label/synthetic/tokens/recall" in logged[-1]


def test_label_resumes_by_skipping_labeled_tasks(tmp_path):
    tasks = [make_task(f"q{i}") for i in range(3)]
    store = FakeStore(tasks)
    config = lab.LabelConfig(run_id="r", corpus_id="c", n=10, seed=0, batch_size=2, origin_mix=None,
                             targets_root=str(tmp_path))
    gen = lambda msgs, **kw: [make_completion() for _ in msgs]

    def interrupted(msgs, **kw):
        if len(store.inserted) >= 2:
            raise KeyboardInterrupt
        return gen(msgs)

    with pytest.raises(KeyboardInterrupt):
        lab.label(config, PAGES, sample=store.sample, generate=interrupted,
                  write_shard=store.write_shard, insert_labels=store.insert_labels)
    assert len(store.inserted) == 2  # the first batch is safe
    report = lab.label(config, PAGES, sample=store.sample, generate=gen,
                       write_shard=store.write_shard, insert_labels=store.insert_labels)
    assert report.overall.n_tasks == 1  # only the task not yet labeled
    assert sorted(lbl.task_id for lbl in store.inserted) == sorted(t.task_id for t in tasks)


def test_eval_labels_are_a_separate_validation_set(tmp_path):
    tasks = [make_task("t-train", origin="chat", split="train"),
             make_task("t-chat", origin="chat", split="eval"),
             make_task("t-psych", origin="psych_test", split="eval"),
             make_task("t-human", origin="human", split="eval")]
    store = FakeStore(tasks)
    report = lab.label(
        lab.LabelConfig(run_id="val-run", corpus_id="c", n=100, seed=0, split="eval", origin_mix=None,
                        origins=frozenset({"chat", "psych_test"}), targets_root=str(tmp_path)),
        PAGES, sample=store.sample, generate=lambda msgs, **kw: [make_completion() for _ in msgs],
        write_shard=store.write_shard, insert_labels=store.insert_labels,
    )
    assert store.queries[0].split == "eval" and store.queries[0].origins == {"chat", "psych_test"}
    assert {lbl.task_id for lbl in store.inserted} == {tasks[1].task_id, tasks[2].task_id}
    assert all(lbl.run_id == "val-run" for lbl in store.inserted) and report.overall.n_tasks == 2
    with pytest.raises(ValueError):  # never for training
        lab.label(lab.LabelConfig(run_id="x", corpus_id="c", n=1, seed=0, origins=frozenset({"human"})), PAGES,
                  sample=store.sample, generate=lambda m, **kw: [], insert_labels=store.insert_labels)


def test_project_cost_scales_the_pilot_pace():
    pilot = lab.LabelStats(n_tasks=100, n_validated=98, reject_reasons={}, truncation_rate=0.05, refusal_rate=0.0,
                           cite_rate=0.9, citation_validity=1.0, abstention_rate=0.0, completion_tokens=98 * 700,
                           tokens_by_kind={})
    p = lab.project_cost(pilot, pilot_wall_s=300, n_total=3158, usd_per_hour=3.95, overhead_s=600)
    assert p["per_task_s"] == 3.0 and p["tokens_per_task"] == 700
    assert p["projected_wall_s"] == 3.0 * 3158 + 600
    assert p["projected_usd"] == pytest.approx((3.0 * 3158 + 600) / 3600 * 3.95)


@pytest.mark.parametrize("text", [
    "You wrote this on p.251 (Jan 28, 2026).",
    "See (p.1, p.4, p.110) for the pattern.",
    "On page 260 you said so.",
    "Across pp. 3-5 you describe it [Auto biography:p.1].",
])
def test_bare_page_references_are_unanchored(text):
    assert lab.validate_label(make_task(), make_completion(text=text), PAGES) == ("rejected", "unanchored_citation")


def test_bracketed_citations_and_times_are_not_unanchored():
    text = "At 7:46 p.m. you sat on the couch [Auto biography:p.1], and later [Morning Pages:p.63]."
    assert lab.validate_label(make_task(), make_completion(text=text), PAGES) == ("validated", None)


PSYCH_TASK = ('On the Systemizing Quotient (SQ-R), how would I answer this item? "When I am walking in the '
              'country, I am curious about how the various kinds of trees differ." Options: ...')


def test_psych_answer_quoting_the_item_as_a_journal_line_is_rejected():
    task = make_task(PSYCH_TASK, kind="framework_lens", origin="psych_test")
    text = ('Strongly Agree. You explicitly state, "I am curious about how the various kinds of trees differ" '
            "[Auto biography:p.1].")
    assert lab.validate_label(task, make_completion(text=text), PAGES) == ("rejected", "item_as_evidence")


def test_psych_answer_citing_real_entries_is_kept():
    task = make_task(PSYCH_TASK, kind="framework_lens", origin="psych_test")
    text = ('Agree: you describe "listening to Cat Stevens while making broth in the kitchen" with close '
            "attention to detail [Auto biography:p.1]. Your journals don't show nature walks directly.")
    assert lab.validate_label(task, make_completion(text=text), PAGES) == ("validated", None)
    stats = lab.label_stats([task], [make_completion(text=text)], [lab.to_label(task, make_completion(text=text),
                            lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0), "validated", None)], PAGES)
    assert stats.abstention_rate == 1.0


def test_item_echo_only_applies_to_psych_tasks():
    task = make_task('Help me with this: "When I am walking in the country, I am curious about trees."')
    text = 'You asked about "When I am walking in the country, I am curious about trees." [Auto biography:p.1]'
    assert lab.validate_label(task, make_completion(text=text), PAGES) == ("validated", None)


def test_task_prompt_ends_with_the_citation_reminder_then_the_task():
    msgs = lab.task_messages(make_task("What did I do in 2021?"))
    assert msgs == [{"role": "system", "content": lab.CITATION_REMINDER}, {"role": "user", "content": "What did I do in 2021?"}]
    assert lab.answer_messages("What did I do in 2021?") == msgs
    r = lab.CITATION_REMINDER
    assert "<journal>" not in r and "<N>" not in r
    assert "the journal named in the page heading" in r and "[Morning Pages:p.63] or [Auto biography:p.12]" in r
    assert "never a page number alone" in r and "say so" in r


def test_instructions_hash_covers_the_reminder():
    assert lab.instructions_hash() != lab.instructions_hash(reminder="")
    assert lab.instructions_hash() == lab.instructions_hash(teacher.INSTRUCTIONS, lab.CITATION_REMINDER)


def test_angle_bracket_citations_count_and_are_validated():
    assert lab.parse_citations("x [<Auto biography:p.1>] y [Morning Pages:p.63] z [<Auto biography:p.1>]") == [
        "Auto biography:p.1", "Morning Pages:p.63"]
    ok = make_completion(text="You wrote it [<Auto biography:p.1>].")
    assert lab.validate_label(make_task(), ok, PAGES) == ("validated", None)
    bad = make_completion(text="You wrote it [<There He Is:p.269>].")
    assert lab.validate_label(make_task(), bad, PAGES) == ("rejected", "invalid_citation")
    placeholder = make_completion(text="You wrote it [<journal>:p.127].")
    assert lab.validate_label(make_task(), placeholder, PAGES) == ("rejected", "invalid_citation")


@pytest.mark.parametrize("text", [
    # The 27B pilot's restatements of the item: allowed.
    ('You would likely answer **Strongly Disagree** to the statement "When I am walking in the country, I am curious '
     'about how the various kinds of trees differ." Your writing shows otherwise [Auto biography:p.1].'),
    ('On the item "When I am walking in the country, I am curious about how the various kinds of trees differ" you '
     "would pick Agree [Auto biography:p.1]."),
])
def test_restating_the_item_is_not_item_as_evidence(text):
    task = make_task(PSYCH_TASK, kind="framework_lens", origin="psych_test")
    assert lab.validate_label(task, make_completion(text=text), PAGES) == ("validated", None)


def test_item_quoted_with_a_citation_after_it_is_item_as_evidence():
    task = make_task(PSYCH_TASK, kind="framework_lens", origin="psych_test")
    text = ('Agree: "I am curious about how the various kinds of trees differ" [Auto biography:p.1] shows it.')
    assert lab.validate_label(task, make_completion(text=text), PAGES) == ("rejected", "item_as_evidence")


def test_human_tasks_are_labeled_only_as_eval_references(tmp_path):
    tasks = [make_task("h-eval", origin="human", split="eval"), make_task("h-train", origin="human", split="train")]
    store = FakeStore(tasks)
    report = lab.label(
        lab.LabelConfig(run_id="val-human", corpus_id="c", n=10, seed=0, split="eval", origin_mix=None,
                        origins=frozenset({"human"}), targets_root=str(tmp_path)),
        PAGES, sample=store.sample, generate=lambda msgs, **kw: [make_completion() for _ in msgs],
        write_shard=store.write_shard, insert_labels=store.insert_labels,
    )
    assert [lbl.task_id for lbl in store.inserted] == [tasks[0].task_id] and report.overall.n_tasks == 1
    train_store = FakeStore([tasks[1]])  # a train sample that (wrongly) returns a human task is still filtered
    train_store.sample = lambda q: types.Sample(query=q, tasks=(tasks[1],), shortfall={}, sample_hash="h")
    lab.label(lab.LabelConfig(run_id="t", corpus_id="c", n=1, seed=0, origin_mix=None), PAGES,
              sample=train_store.sample, generate=lambda msgs, **kw: [make_completion() for _ in msgs],
              write_shard=train_store.write_shard, insert_labels=train_store.insert_labels)
    assert train_store.inserted == []


def test_rejected_answers_keep_their_targets_row_but_are_not_read(tmp_path):
    tasks = [make_task("ok"), make_task("refused")]
    answers = {"ok": make_completion(T=2), "refused": make_completion(text="I can't help with that.", T=3)}
    store = FakeStore(tasks)
    lab.label(lab.LabelConfig(run_id="r", corpus_id="c", n=2, seed=0, origin_mix=None, targets_root=str(tmp_path)),
              PAGES, sample=store.sample, generate=lambda msgs, **kw: [answers[m[-1]["content"]] for m in msgs],
              write_shard=targets.write_shard, insert_labels=store.insert_labels)
    rejected = next(lbl for lbl in store.inserted if lbl.status == "rejected")
    assert rejected.targets_uri is not None and rejected.targets_row == 1
    assert [len(e.completion_ids) for e in targets.read_labeled(store.inserted)] == [2]  # only the validated one


def test_revalidate_updates_status_and_flags_labels_without_targets():
    psych = make_task(PSYCH_TASK, kind="framework_lens", origin="psych_test")
    restated = ('You would answer **Agree** to the statement "When I am walking in the country, I am curious about '
                'how the various kinds of trees differ." Your entries show it [Auto biography:p.1].')
    config = lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0)
    comp = make_completion(text=restated)
    # Rejected by the old guard; one kept its targets row, one (old run) did not.
    with_targets = lab.to_label(psych, comp, config, "rejected", "item_as_evidence", "uri", 0)
    without = lab.to_label(make_task(PSYCH_TASK + " ", kind="framework_lens", origin="psych_test"), comp, config,
                           "rejected", "item_as_evidence", None, None)
    now_bad = lab.to_label(make_task("x"), make_completion(text="On page 260 you said so."), config, "validated", None,
                           "uri", 1)
    unchanged = lab.to_label(make_task("y"), make_completion(), config, "validated", None, "uri", 2)
    tasks = {psych.task_id: psych}
    res = lab.revalidate([(psych, with_targets), (make_task(PSYCH_TASK + " ", kind="framework_lens", origin="psych_test"), without),
                          (make_task("x"), now_bad), (make_task("y"), unchanged)], PAGES)
    assert [(u.label_id, u.status, u.reject_reason) for u in res.updated] == [
        (with_targets.label_id, "validated", None), (now_bad.label_id, "rejected", "unanchored_citation")]
    assert [n.label_id for n in res.needs_relabel] == [without.label_id]
    assert tasks  # (kept for readability)


def test_stored_uris_resolve_against_the_readers_mount(tmp_path):
    assert targets.volume_relative("/cache/targets/r/part-a.parquet") == "targets/r/part-a.parquet"
    assert targets.volume_relative("/vol/targets/r/part-a.parquet") == "targets/r/part-a.parquet"
    assert targets.volume_relative("targets/r/part-a.parquet") == "targets/r/part-a.parquet"
    assert targets.volume_relative("/private/tmp/x.parquet") is None
    assert targets.resolve("/cache/targets/r/p.parquet", "/vol") == "/vol/targets/r/p.parquet"
    assert targets.resolve("targets/r/p.parquet", "/vol") == "/vol/targets/r/p.parquet"
    assert targets.resolve("/cache/targets/r/p.parquet") == "/cache/targets/r/p.parquet"
    assert targets.resolve("/private/tmp/x.parquet", "/vol") == "/private/tmp/x.parquet"

    # A shard written under a local "Volume" root, read back via an old /cache/… label and a new relative one.
    row = lab.to_target_row(make_task(), make_completion(T=3), uuid.uuid4())
    targets.write_shard([row], tmp_path / "targets/r/part-a.parquet")
    old = _label_for(row, "/cache/targets/r/part-a.parquet", 0)
    new = _label_for(row, "targets/r/part-a.parquet", 0)
    assert [len(e.completion_ids) for e in targets.read_labeled([old, new], root=tmp_path)] == [3, 3]


def test_label_stores_the_uri_the_writer_returns(tmp_path):
    store = FakeStore([make_task("a")])
    lab.label(lab.LabelConfig(run_id="r", corpus_id="c", n=1, seed=0, origin_mix=None, targets_root="/cache/targets"),
              PAGES, sample=store.sample, generate=lambda msgs, **kw: [make_completion()],
              write_shard=targets.volume_writer(FakeVolume(tmp_path), mount="/cache"), insert_labels=store.insert_labels)
    (stored,) = store.inserted
    assert stored.targets_uri == "targets/r/part-h1.parquet"
    assert len(list(targets.read_labeled(store.inserted, root=tmp_path))) == 1


# ---------------------------------------------------------------- streaming


class StreamCall:
    """A spawned teacher call: done after `polls` get() calls, or held until released."""

    def __init__(self, result, polls=1, held=False):
        self.result, self.polls, self.held, self.cancelled = result, polls, held, False

    def get(self, timeout=None):
        if self.held or self.polls > 0:
            self.polls -= 1
            time.sleep(0.001)  # like Modal's get(timeout): don't return instantly while running
            raise TimeoutError
        return [self.result]

    def cancel(self):
        self.cancelled = True


def _stream_store(n_tasks):
    tasks = [make_task(f"s{i}", origin="psych_test", kind="framework_lens", split="train") for i in range(n_tasks)]
    return tasks, FakeStore(tasks)


def test_label_stream_has_no_batch_barrier_behind_a_slow_call(tmp_path):
    tasks, store = _stream_store(20)
    lock, calls, live, peak, order = threading.Lock(), [], [0], [0], []

    class Tracked(StreamCall):
        def get(self, timeout=None):
            result = super().get(timeout)
            with lock:
                live[0] -= 1
                order.append(self.task)
            return result

    def spawn(messages, **kw):
        with lock:
            call = Tracked(make_completion(T=2), polls=1, held=not calls)  # the first call is the slow one
            call.task = messages[-1]["content"]
            calls.append(call)
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            if len(calls) == len(tasks):  # every other task submitted: release the slow one
                calls[0].held = False
            return call

    report = lab.label_stream(
        lab.LabelConfig(run_id="r", corpus_id="c", n=20, seed=0, origin_mix=None, targets_root=str(tmp_path)),
        PAGES, sample=store.sample, spawn=spawn, write_shard=store.write_shard,
        insert_labels=store.insert_labels, window=4, flush_every=5, poll_s=0.01,
    )
    assert report.overall.n_tasks == 20 and len(store.inserted) == 20
    # The window never exceeded 4, and while call 0 was stuck the other slots kept working: it finished among
    # the last window, after at least 16 of the 19 others (a per-batch barrier would have stalled them all).
    assert peak[0] == 4 and len(order) == 20 and order.index(calls[0].task) >= 16
    assert {lbl.task_id for lbl in store.inserted} == {t.task_id for t in tasks}  # each task labeled once


def test_label_stream_flushes_in_groups_and_logs_stats(tmp_path):
    _, store = _stream_store(10)
    logged = []
    report = lab.label_stream(
        lab.LabelConfig(run_id="r", corpus_id="c", n=10, seed=0, origin_mix=None, targets_root=str(tmp_path)),
        PAGES, sample=store.sample, spawn=lambda m, **kw: StreamCall(make_completion(T=3), polls=0),
        write_shard=store.write_shard, insert_labels=store.insert_labels, window=3, flush_every=4,
        sleep=lambda _: None, log=logged.append,
    )
    assert report.overall.n_tasks == 10 and len(store.shards) == len(report.targets_uris) >= 2
    assert [x["label/all/n_tasks"] for x in logged][-1] == 10 and logged[0]["label/all/n_tasks"] >= 4


def test_label_stream_cancels_in_flight_calls_and_keeps_landed_answers_on_interrupt(tmp_path):
    _, store = _stream_store(10)
    calls = []

    def spawn(messages, **kw):
        call = StreamCall(make_completion(T=2), polls=0, held=len(calls) >= 2)  # two land, the rest hang
        calls.append(call)
        return call

    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) > 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        lab.label_stream(
            lab.LabelConfig(run_id="r", corpus_id="c", n=10, seed=0, origin_mix=None, targets_root=str(tmp_path)),
            PAGES, sample=store.sample, spawn=spawn, write_shard=store.write_shard,
            insert_labels=store.insert_labels, window=4, flush_every=100, sleep=sleep,
        )
    assert len(store.inserted) == 2  # the landed answers were flushed
    assert all(c.cancelled for c in calls[2:]) and not any(c.cancelled for c in calls[:2])


def test_label_stream_stops_new_calls_when_the_guard_trips(tmp_path):
    _, store = _stream_store(30)
    spawned = []

    def spawn(messages, **kw):
        spawned.append(1)
        return StreamCall(make_completion(T=2), polls=0)

    report = lab.label_stream(
        lab.LabelConfig(run_id="r", corpus_id="c", n=30, seed=0, origin_mix=None, targets_root=str(tmp_path)),
        PAGES, sample=store.sample, spawn=spawn, write_shard=store.write_shard, insert_labels=store.insert_labels,
        window=4, flush_every=4, sleep=lambda _: None, should_stop=lambda: len(spawned) >= 8,
    )
    # No new calls once the guard trips; the at most one window already submitted drains and is kept.
    assert 8 <= len(spawned) <= 8 + 4 and report.overall.n_tasks == len(spawned) == len(store.inserted)


def test_window_for_short_answer_origins():
    assert lab.window_for({"psych_test"}) == 128
    assert lab.window_for({"chat"}) == 96 and lab.window_for({"psych_test", "chat"}) == 96 and lab.window_for(None) == 96

