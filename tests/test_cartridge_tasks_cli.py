"""`cartridges tasks_*` end to end on CPU: an in-memory store, a fake teacher,
W&B disabled."""

import contextlib
import json
import re
from collections import Counter
from types import SimpleNamespace

import pytest

from cartridges import cli
from cartridges.batch.outputs import collate
from cartridges.cartridge import teacher
from cartridges.cartridge.tasks import store, targets


class FakeStore:
    def __init__(self):
        self.pages, self.runs, self.tasks, self.labels, self.chats = {}, [], {}, [], {}
        self.chat_labels = {}

    def connect(self, database_url=None):
        return contextlib.nullcontext(self)

    def init(self, conn):
        pass

    def insert_corpus(self, conn, corpus, pages):
        self.pages[corpus.corpus_id] = list(pages)

    def fetch_pages(self, conn, corpus_id):
        return self.pages[corpus_id]

    def insert_run(self, conn, run):
        self.runs.append(run)

    def insert_tasks(self, conn, tasks):
        new = [t for t in tasks if t.text_hash not in {u.text_hash for u in self.tasks.values()}]
        self.tasks.update({t.task_id: t for t in new})
        return len(new)

    def delete_human_tasks(self, conn):
        human = [i for i, t in self.tasks.items() if t.origin in ("human", "human_template")]
        for i in human:
            del self.tasks[i]
        return len(human)

    def fetch_hashes(self, conn):
        return {t.text_hash for t in self.tasks.values()}

    def fetch_tasks(self, conn, query):
        key = query.unlabeled_by if isinstance(query.unlabeled_by, tuple) else (query.unlabeled_by,)
        labeled = {lab.task_id for lab in self.labels
                   if (lab.teacher_model, lab.corpus_id, lab.instructions_hash)[: len(key)] == key}
        return [t for t in self.tasks.values() if t.split == query.split and t.task_id not in labeled]

    def insert_labels(self, conn, labels):
        self.labels += labels
        return len(labels)

    def upsert_chat_prompts(self, conn, prompts):
        self.chats.update({p.conversation_id: p for p in prompts})
        return len(prompts)

    def fetch_unlabeled_chats(self, conn, model, prompt_hash):
        texts = {p.text_hash: p for p in self.chats.values()}
        done = {h for h, m, ph in self.chat_labels if (m, ph) == (model, prompt_hash)}
        return [p for h, p in texts.items() if h not in done], len(set(texts) & done)

    def insert_chat_labels(self, conn, rows):
        new = {(r.prompt.text_hash, r.classifier_model, r.prompt_hash): r for r in rows
               if r.label.category != "unparsed"}
        self.chat_labels.update({k: v for k, v in new.items() if k not in self.chat_labels})
        return len(new)

    def fetch_promotable_chats(self, conn, model, prompt_hash):
        by_hash = {p.text_hash: p for p in self.chats.values()}
        return [(by_hash[h], r.label, r.run_id) for (h, m, ph), r in self.chat_labels.items()
                if (m, ph) == (model, prompt_hash) and r.label.relevant and r.label.task_kind and h in by_hash]

    def list_tasks(self, conn, split=None, origin=None, run_ids=(), status=None):
        return [t for t in self.tasks.values() if (split is None or t.split == split)
                and (origin is None or t.origin == origin) and (not run_ids or t.run_id in run_ids)
                and (status is None or t.status == status)]

    def update_validation(self, conn, tasks):
        self.tasks.update({t.task_id: t for t in tasks})
        return len(tasks)

    def fetch_labels(self, conn, run_ids, status="validated", split="train"):
        return [lab for lab in self.labels if lab.run_id in run_ids and lab.status == status
                and self.tasks[lab.task_id].split == split]

    def fetch_tasks_by_id(self, conn, ids):
        return {i: self.tasks[i] for i in ids}

    def targets_manifest(self, conn, run_id):
        labs = [lab for lab in self.labels if lab.run_id == run_id]
        return {"run_id": run_id, "shards": sorted({lab.targets_uri for lab in labs if lab.targets_uri}),
                "labels": dict(Counter(lab.status for lab in labs))}

    def resplit_chat_exemplars(self, conn):
        moved = [t for t in self.tasks.values() if t.origin == "chat" and t.split == "exemplar"]
        self.tasks.update({t.task_id: t.model_copy(update={"split": "train"}) for t in moved})
        return len(moved)

    def stats(self, conn):
        return {"tasks": len(self.tasks), "labels": len(self.labels)}


def fake_generate(model, gpu=None, max_inputs=None):
    def generate(batch, **kw):
        out = []
        for conv in batch:
            text = conv[-1]["content"]  # the user turn (labels add a system reminder first)
            if text.startswith("Rewrite this message as a reusable template"):
                out.append(SimpleNamespace(text=text.split("Message:\n")[1].replace("Sam", "{person}")))
            elif '{"task": "...", "citations"' in text:
                c = re.search(r"## \[([^\]]+)\] date: [^\n]*\n\n([^\n]+)", text)
                task = f"Why do I keep coming back to {c.group(2)[:30]}, {len(out)}?"
                out.append(SimpleNamespace(text=json.dumps({"task": task, "citations": [c.group(1)],
                                                            "evidence": [c.group(2)[:25]]})))
            else:
                assert kw["logprobs"] == 20 and kw["temperature"] == 0.0
                out.append(None if not out and len(batch) > 1 else teacher.Completion(text="You hiked with Sam [A:p.1].", prompt_ids=[1, 2],
                                              completion_ids=[3, 4], topk_ids=[[3], [4]],
                                              topk_logprobs=[[-0.1], [-0.2]], finish_reason="stop"))
        return out

    return generate


class FakeTeacher:
    """teacher_cls(...) for label_stream: generate.spawn(messages, **kw) -> a call
    whose get() answers at once; the first call fails (the no_reply path)."""

    def __init__(self, on_spawn=None):
        self.spawned, self.on_spawn = 0, on_spawn
        self.generate = SimpleNamespace(spawn=self.spawn)

    def spawn(self, messages, **kw):
        self.spawned += 1
        if self.on_spawn:
            self.on_spawn()
        if self.spawned == 1:
            def fail(timeout=None):
                raise RuntimeError("container gone")
            return SimpleNamespace(get=fail, cancel=lambda: None)
        [reply] = fake_generate("m")([messages], **kw)
        return SimpleNamespace(get=lambda timeout=None: [reply], cancel=lambda: None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    fake = FakeStore()
    for name in ("connect", "init", "insert_corpus", "fetch_pages", "insert_run", "insert_tasks", "delete_human_tasks", "fetch_hashes",
                 "fetch_tasks", "insert_labels", "upsert_chat_prompts", "fetch_unlabeled_chats", "insert_chat_labels", "list_tasks", "update_validation", "fetch_promotable_chats", "resplit_chat_exemplars", "fetch_labels", "fetch_tasks_by_id", "targets_manifest", "stats"):
        monkeypatch.setattr(store, name, getattr(fake, name))
    monkeypatch.setattr(cli, "_tasks_generate", fake_generate)
    monkeypatch.setattr(teacher, "teacher_cls", lambda *a, **kw: FakeTeacher())
    monkeypatch.setattr(cli, "_tasks_write_shard", lambda: targets.write_shard)  # local, not the Volume

    d = tmp_path / "A"
    d.mkdir()
    pages = [f"## p. {n} - date: {2021 + n // 4}-0{1 + n % 4}-01\n\nHiked with Sam to lake number {n} and talked."
             for n in range(1, 17)]
    (d / "transcript.md").write_text("\n\n---\n".join(pages) + "\n")
    collate(tmp_path)
    (tmp_path / "heldout.json").write_text(json.dumps({"pages": ["A:p.2"]}))
    rows = [{"question": f"What did Sam tell me about plan {i}?", "kind_v2": k}
            for i, k in enumerate(["recall", "situation_help", "self_synthesis", "framework_lens"] * 3)]
    (tmp_path / "q.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return fake, tmp_path


def test_tasks_commands_end_to_end(env, capsys, monkeypatch):
    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    c = cli.Cli()

    c.tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    [pages] = fake.pages.values()
    assert len(pages) == 16 and [p.citation for p in pages if p.heldout] == ["A:p.2"]

    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4)
    humans = [t for t in fake.tasks.values() if t.origin == "human"]
    templates = [t for t in fake.tasks.values() if t.origin == "human_template"]
    assert len(humans) == 12 and sum(t.split == "eval" for t in humans) == 4
    assert len(templates) == 8 and all("{person}" in t.text for t in templates)

    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4, replace=True)  # re-seeding gives the same rows
    assert sorted(t.task_id for t in fake.tasks.values()) == sorted(t.task_id for t in humans + templates)

    c.tasks_synthesize(n=10, corpus=corpus)
    synthetic = [t for t in fake.tasks.values() if t.origin == "synthetic"]
    assert len(synthetic) == 10 and any(t.status == "validated" for t in synthetic)

    # label()'s batching belongs to teacher (label.py); here: the session wiring.
    from cartridges.cartridge import tracking

    stopped = []
    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_session_container", lambda model, gpu=None, max_inputs=None: f"ta-{gpu}")
    monkeypatch.setattr(cli, "_tasks_stop", lambda ids=None: stopped.append(ids))
    monkeypatch.setattr(store, "label_summary", lambda conn, run_ids: {"synthetic": {"n": len(fake.labels)}})
    monkeypatch.setattr(store, "label_samples", lambda conn, run_id, per_origin: [])
    capsys.readouterr()
    c.tasks_label(corpus=corpus, targets_root=str(tmp / "targets"))
    report = json.loads(capsys.readouterr().out)
    labeled = [lab.task_id for lab in fake.labels]
    synthetic_ok = {t.task_id for t in fake.tasks.values() if t.origin == "synthetic" and t.status == "validated"}
    # Phases in priority order; empty ones (no chat or psych tasks here) are skipped.
    human_eval = {t.task_id for t in fake.tasks.values() if t.origin == "human" and t.split == "eval"}
    assert [p["phase"] for p in report["phases"]] == ["train:synthetic", "eval:human"]  # empty phases skipped
    assert all(p["status"] == "done" for p in report["phases"])
    assert report["run_ids"] == {"train": [report["phases"][0]["run_id"]], "eval": [report["phases"][1]["run_id"]]}
    assert stopped == [["ta-H100"]]  # only this session's container, by id
    assert "train:synthetic=" + str(len(synthetic_ok)) in report["plan"] and report["session_usd"] < 0.01
    # every train task once, and the human questions only as eval references
    assert len(set(labeled)) == len(labeled) and set(labeled) == synthetic_ok | human_eval
    reasons = Counter(lab.reject_reason for lab in fake.labels)
    assert reasons["no_reply"] >= 1 and set(reasons) <= {None, "no_reply"}  # the fake fails one call per batch
    assert (tmp / "targets").exists()

    monkeypatch.chdir(tmp)
    c.tasks_log_targets(report["run_ids"]["train"][0])
    capsys.readouterr()
    manifest = json.loads((tmp / f"data/cartridge/targets/targets-{report['run_ids']['train'][0]}.json").read_text())
    assert manifest["shards"] and sum(manifest["labels"].values()) == len(synthetic_ok)
    c.tasks_export_qa(train_runs=report["run_ids"]["train"][0], name="qa-test")
    exported = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in (tmp / "data/cartridge/qa/qa-test-train.jsonl").read_text().splitlines()]
    train_run = report["run_ids"]["train"][0]
    validated = sum(lab.status == "validated" and lab.run_id == train_run for lab in fake.labels)
    assert exported["train"]["rows"] == len(rows) == validated
    assert {r["origin"] for r in rows} == {"synthetic"} and all(r["answer"] and r["question"] for r in rows)

    assert [r.component for r in fake.runs] == ["seed", "seed", "synthesize", "label", "label"]  # two phases ran
    assert {t.run_id for t in synthetic} == {fake.runs[2].run_id}

    capsys.readouterr()
    c.tasks_sample("eval", n=2)
    out = json.loads(capsys.readouterr().out)
    assert len(out["tasks"]) == 2 and out["sample_hash"]

    c.tasks_stats()
    assert json.loads(capsys.readouterr().out) == {"tasks": 30, "labels": len(fake.labels)}


def test_tasks_qgen_eval_wires_the_steps(env, capsys, monkeypatch):
    from cartridges.cartridge import tracking
    from cartridges.cartridge.tasks import pairs

    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    cli.Cli().tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    stopped, seen = [], {}
    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_tasks_stop", lambda: stopped.append(True))

    def fake_pairs(sample, run_ids, generate, n, queue):
        seen["sample"] = sample(pairs.run_query(run_ids, n, 0))
        return {"run_ids": run_ids, "n": n}

    monkeypatch.setattr(pairs, "run", fake_pairs)
    capsys.readouterr()
    cli.Cli().tasks_qgen_eval("exp-20261015-qgen-eval", file=str(tmp / "q.jsonl"), n=40, topup_n=20, pairs_n=10,
                              corpus=corpus)
    report = json.loads(capsys.readouterr().out)
    assert report["probe"] == "OK" and report["seed"]["inserted_templates"] > 0
    run_ids = [r["run_id"] for r in report["synthesize"]]
    assert report["pairs"] == {"run_ids": run_ids, "n": 10}
    assert set(t.run_id for t in seen["sample"].tasks) <= set(run_ids)
    assert stopped == [True]


def test_tasks_import_chats(env, capsys, monkeypatch, tmp_path):
    from cartridges.cartridge import tracking
    from cartridges.cartridge.tasks import chats

    fake, _ = env
    convs = [
        {"conversation_id": f"c{i}", "create_time": 1_700_000_000.0 + i, "title": None, "default_model_slug": None,
         "current_node": f"n{i}",
         "mapping": {f"n{i}": {"message": {"author": {"role": "user"}, "content": {"parts": [f"question {i}"]}},
                               "parent": None}}}
        for i in range(6)
    ]
    export = tmp_path / "export"
    export.mkdir()
    (export / "conversations-000.json").write_text(json.dumps(convs))
    stopped, batches = [], []
    replies = {"text": None}  # None: valid JSON; else the literal reply

    def generate(model, gpu=None, max_inputs=None):
        def run(batch, **kw):
            batches.append(len(batch))
            if replies["text"] is not None:
                return [SimpleNamespace(text=replies["text"]) for _ in batch]
            relevant = lambda i: "true" if i % 2 else "false"  # noqa: E731
            return [SimpleNamespace(text=f'{{"category": "relationships", "task_kind": "situation_help", '
                                         f'"relevant": {relevant(i)}, "reason": "r"}}') for i in range(len(batch))]

        return run

    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_tasks_stop", lambda: stopped.append(True))
    monkeypatch.setattr(cli, "_tasks_generate", generate)

    def run(**kw):
        cli.Cli().tasks_import_chats(**kw)
        return json.loads(capsys.readouterr().out)

    capsys.readouterr()
    assert run(dir=str(export), dry_run=True)["kept"] == 6 and fake.chats == {} and stopped == []

    report = run(dir=str(export), extract_only=True)  # stage 1: stored, no labels, no GPU
    assert report["stored"] == 6 and report["cache_misses"] == 6 and batches == [] and stopped == []

    report = run(limit=4)  # stage 2 from the database
    assert batches == [4] and report["cached"] == 4 and report["relevant"] == 2 and stopped == [True]
    assert (report["cache_hits"], report["cache_misses"]) == (0, 4)
    assert len(report["sample_relevant"]) == 2 and len(report["sample_discarded"]) == 2

    replies["text"] = "not json"  # the other 2 come back unparsed: not cached
    report = run(dir=str(export))
    assert batches == [4, 2] and report["cached"] == 0 and report["unparsed_rate"] == 1.0

    replies["text"] = None  # retried, then everything is cached
    assert run()["cache_misses"] == 2 and batches == [4, 2, 2]
    report = run()
    assert (report["cache_hits"], report["cache_misses"]) == (6, 0) and batches == [4, 2, 2]  # no generate call
    assert stopped == [True, True, True]  # no GPU session for a full cache hit
    chat_tasks = [t for t in fake.tasks.values() if t.origin == "chat"]
    assert len(chat_tasks) == 3 and report["promote"]["inserted"] == 0  # promoted earlier, nothing new now
    assert {t.kind for t in chat_tasks} == {"situation_help"}

    monkeypatch.setattr(chats, "CLASSIFIER_SYSTEM", chats.CLASSIFIER_SYSTEM + " Be brief.")  # a new prompt
    report = run()
    assert (report["cache_hits"], report["cache_misses"]) == (0, 6) and batches[-1] == 6


def test_tasks_revalidate_and_export_md(env, capsys):
    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    c = cli.Cli()
    c.tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4)
    c.tasks_synthesize(n=10, corpus=corpus)
    run_id = fake.runs[-1].run_id
    capsys.readouterr()

    c.tasks_revalidate(run_id)
    report = json.loads(capsys.readouterr().out)
    assert report["drafts"] == 10 and report["changed"] == 0  # the current validator already applied
    assert set(report["validated_vs_quota"]) >= {"recall", "situation_help"}

    out = tmp / "tasks.md"
    c.tasks_export_md(out=str(out), origin="synthetic", run_id=run_id)
    assert capsys.readouterr().out.strip() == str(out)
    md = out.read_text()
    assert md.startswith("# Tasks: synthetic-" + run_id) and "- exemplar: " in md


def test_tasks_ksweep(env, capsys, monkeypatch):
    from cartridges.cartridge import tracking
    from cartridges.cartridge.tasks.validate import task_id, text_hash
    from cartridges.cartridge.tasks.types import Task

    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    c = cli.Cli()
    c.tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4)
    for i, split in enumerate(["exemplar", "train", "eval", "train"]):
        text = f"How did Sam make me feel at the lake, part {i}?"
        t = Task(task_id=task_id(text_hash(text)), text=text, text_hash=text_hash(text), kind="recall", origin="chat",
                 split=split, status="validated")
        fake.tasks[t.task_id] = t
    stopped, prompts_seen = [], []
    real = fake_generate("m")

    def generate(model, gpu=None, max_inputs=None):
        def run(batch, **kw):
            assert kw["with_context"] is True  # no no-context calls in the sweep
            out = []
            for reply in real(batch, **kw):  # a distinct text per prompt, so no draft is a duplicate
                draft = json.loads(reply.text)
                out.append(SimpleNamespace(text=json.dumps({**draft, "task": f"{draft['task']} #{len(prompts_seen)}"})))
                prompts_seen.append(batch[len(out) - 1])
            return out

        return run

    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_tasks_stop", lambda: stopped.append(True))
    monkeypatch.setattr(cli, "_tasks_generate", generate)
    from cartridges.cartridge.tasks import neighbors

    monkeypatch.setattr(neighbors, "local_embedder", lambda: "embed")
    monkeypatch.setattr(neighbors, "add_neighbor_columns",
                        lambda rows, ref, embed: [{**r, "nn_sim_generated": 0.5} for r in rows])
    monkeypatch.setattr(neighbors, "summarize_k", lambda rows, ref, embed: {"qgen/n_gen": len(rows)})
    monkeypatch.chdir(tmp)
    capsys.readouterr()

    c.tasks_ksweep(ks="0,2", n=6, corpus=corpus)
    report = json.loads(capsys.readouterr().out)
    assert report["chat_exemplars_moved_to_train"] == 1 and stopped == [True]
    assert set(report["runs"]) == {"0", "2"} and len(prompts_seen) == 12
    focus = [re.findall(r"## \[([^\]]+)\]", p[0]["content"]) for p in prompts_seen]
    assert focus[:6] == focus[6:]  # the same recipes for every k
    assert "- (none)" in prompts_seen[0][0]["content"] and "- (none)" not in prompts_seen[6][0]["content"]
    rows = [json.loads(line) for line in (tmp / "data/cartridge/ksweep/qgen-k2.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and {r["k"] for r in rows} == {2} and all(len(r["exemplars_shown"]) == 2 for r in rows)
    assert all(r["nn_sim_generated"] == 0.5 and r["task_id"] for r in rows)
    ref = [json.loads(line) for line in (tmp / "data/cartridge/ksweep/qgen-real-reference.jsonl").read_text().splitlines()]
    assert {r["split"] for r in ref} == {"B", "eval"}
    sweep = [t for t in fake.tasks.values() if t.metadata.get("k") == 2]
    assert len(sweep) == 6 and all(len(t.metadata["exemplar_ids"]) == 2 for t in sweep)


def test_tasks_label_stops_before_passing_the_cap(env, capsys, monkeypatch):
    from cartridges.cartridge import tracking

    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    c = cli.Cli()
    c.tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4)
    c.tasks_synthesize(n=10, corpus=corpus)
    stopped = []
    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_session_container", lambda model, gpu=None, max_inputs=None: None)
    monkeypatch.setattr(cli, "_tasks_stop", lambda ids=None: stopped.append(ids))
    monkeypatch.setattr(store, "label_summary", lambda conn, run_ids: {})
    monkeypatch.setattr(store, "label_samples", lambda conn, run_id, per_origin: [])
    # No wall clock: the session's "elapsed time" is half an hour of H100 ($1.98) per
    # call started so far, read from the fake's own count.
    fake_teacher = FakeTeacher()
    monkeypatch.setattr(teacher, "teacher_cls", lambda *a, **kw: fake_teacher)
    monkeypatch.setattr("time.monotonic", lambda: 1800.0 * fake_teacher.spawned)
    capsys.readouterr()
    c.tasks_label(max_usd=5.0, drain_min=0, max_inputs=2, corpus=corpus, targets_root=str(tmp / "targets"))
    report = json.loads(capsys.readouterr().out)
    n_tasks = sum(t.origin == "synthetic" and t.status == "validated" for t in fake.tasks.values())
    assert [p["status"][:7] for p in report["phases"]] == ["stopped"] and report["window"] == 2
    # The guard trips once a 3rd call has started ($5.93 > $5). Before that, with at most 2
    # calls in flight, at most 4 can have started; after it, none. Every started call drains.
    assert 3 <= fake_teacher.spawned <= 4 < n_tasks
    assert len(fake.labels) == fake_teacher.spawned and stopped == [[]]


def test_tasks_label_pilot(env, capsys, monkeypatch):
    from cartridges.cartridge import tracking

    fake, tmp = env
    corpus = str(tmp / "teacher_context.md")
    c = cli.Cli()
    c.tasks_init(corpus=corpus, heldout=str(tmp / "heldout.json"))
    c.tasks_seed(str(tmp / "q.jsonl"), n_eval=4)
    c.tasks_synthesize(n=10, corpus=corpus)
    monkeypatch.setattr(tracking, "check_budget", lambda usd: {})
    monkeypatch.setattr(cli, "_tasks_probe", lambda model, gpu=None, max_inputs=None: SimpleNamespace(get=lambda timeout: "OK"))
    monkeypatch.setattr(cli, "_session_container", lambda model, gpu=None, max_inputs=None: "ta-1")
    monkeypatch.setattr(cli, "_tasks_stop", lambda ids=None: None)
    monkeypatch.setattr(store, "label_summary", lambda conn, run_ids: {})
    monkeypatch.setattr(store, "label_samples", lambda conn, run_id, per_origin: [{"origin": "synthetic"}])
    capsys.readouterr()
    c.tasks_label(pilot=3, corpus=corpus, targets_root=str(tmp / "targets"))
    report = json.loads(capsys.readouterr().out)
    assert report["plan"] == ["train:mix=3"] and report["run_ids"]["eval"] == [] and report["samples"]
    assert 0 < len(fake.labels) <= 3
