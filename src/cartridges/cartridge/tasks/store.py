"""The Task Store: Postgres schema `cartridge` (docs/plans/task-db.md, section 5).

The only module that talks to Postgres. SCHEMA is idempotent (IF NOT
EXISTS), as in storage/cache.py. Every function takes an open connection;
DATABASE_URL is read only in connect(). Two columns beyond the design doc:
corpus_pages.position (timeline order, for focus chunks of consecutive pages)
and corpus_pages.text (so validation needs no local corpus file).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from psycopg.types.json import Jsonb

from cartridges.cartridge.tasks.types import (
    ChatLabel, ChatPrompt, ChatRow, Corpus, GenRun, Label, Page, Split, Task, TaskQuery,
)

SCHEMA = """
CREATE SCHEMA IF NOT EXISTS cartridge;

CREATE TABLE IF NOT EXISTS cartridge.corpora (
    corpus_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    n_tokens    INT  NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cartridge.corpus_pages (
    corpus_id      TEXT NOT NULL REFERENCES cartridge.corpora,
    citation       TEXT NOT NULL,
    position       INT  NOT NULL,
    journal        TEXT NOT NULL,
    page_no        INT  NOT NULL,
    date           DATE,
    date_uncertain BOOL NOT NULL,
    heldout        BOOL NOT NULL DEFAULT false,
    text           TEXT NOT NULL,
    PRIMARY KEY (corpus_id, citation)
);

CREATE TABLE IF NOT EXISTS cartridge.gen_runs (
    run_id         TEXT PRIMARY KEY,
    component      TEXT NOT NULL CHECK (component IN ('seed', 'synthesize', 'label')),
    model          TEXT,
    prompt_version TEXT,
    config         JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cartridge.tasks (
    task_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text             TEXT NOT NULL,
    text_hash        TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL CHECK (kind IN ('situation_help', 'self_synthesis', 'framework_lens',
                         'change_over_time', 'role_request', 'build_artifact', 'recall')),
    style            TEXT CHECK (style IN ('dictated', 'typed', 'narrative')),
    framework        TEXT,
    role             TEXT,
    origin           TEXT NOT NULL CHECK (origin IN ('human', 'human_template', 'synthetic', 'chat', 'psych_test')),
    split            TEXT NOT NULL CHECK (split IN ('exemplar', 'train', 'eval')),
    status           TEXT NOT NULL CHECK (status IN ('validated', 'rejected')),
    reject_reason    TEXT,
    corpus_id        TEXT REFERENCES cartridge.corpora,
    source_citations TEXT[] NOT NULL DEFAULT '{}',
    ref_citations    TEXT[] NOT NULL DEFAULT '{}',
    ref_evidence     TEXT[] NOT NULL DEFAULT '{}',
    grounded         BOOL,
    parent_ids       UUID[] NOT NULL DEFAULT '{}',
    run_id           TEXT REFERENCES cartridge.gen_runs,
    metadata         JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (status = 'validated' OR reject_reason IS NOT NULL),
    CHECK (origin <> 'synthetic' OR split = 'train')
);
CREATE INDEX IF NOT EXISTS tasks_sample_idx ON cartridge.tasks (split, status, kind);

CREATE TABLE IF NOT EXISTS cartridge.labels (
    label_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id             UUID NOT NULL REFERENCES cartridge.tasks,
    run_id              TEXT NOT NULL REFERENCES cartridge.gen_runs,
    teacher_model       TEXT NOT NULL,
    corpus_id           TEXT NOT NULL REFERENCES cartridge.corpora,
    instructions_hash   TEXT NOT NULL,
    answer              TEXT NOT NULL,
    finish_reason       TEXT NOT NULL,
    n_prompt_tokens     INT  NOT NULL,
    n_completion_tokens INT  NOT NULL,
    citations           TEXT[] NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL CHECK (status IN ('validated', 'rejected')),
    reject_reason       TEXT,
    targets_uri         TEXT,
    targets_row         INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, teacher_model, corpus_id, instructions_hash)
);

-- Columns and CHECKs added after the first deploy, idempotently.
ALTER TABLE cartridge.tasks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';
ALTER TABLE cartridge.tasks DROP CONSTRAINT IF EXISTS tasks_origin_check;
ALTER TABLE cartridge.tasks ADD CONSTRAINT tasks_origin_check
    CHECK (origin IN ('human', 'human_template', 'synthetic', 'chat', 'psych_test'));
-- gen_runs predates the 'classify' component: widen its CHECK idempotently.
ALTER TABLE cartridge.gen_runs DROP CONSTRAINT IF EXISTS gen_runs_component_check;
ALTER TABLE cartridge.gen_runs ADD CONSTRAINT gen_runs_component_check
    CHECK (component IN ('seed', 'synthesize', 'label', 'classify'));

CREATE TABLE IF NOT EXISTS cartridge.chat_prompts (
    conversation_id  TEXT PRIMARY KEY,
    create_time      TIMESTAMPTZ,
    title            TEXT,
    model_slug       TEXT,
    text             TEXT NOT NULL,
    text_hash        TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_prompts_text_hash_idx ON cartridge.chat_prompts (text_hash);
-- Labels moved to chat_labels, and an older chat_prompts had them as columns.
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS category;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS task_kind;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS relevant;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS reason;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS classifier_model;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS prompt_version;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS run_id;
ALTER TABLE cartridge.chat_prompts DROP COLUMN IF EXISTS raw_reply;

-- Content-addressed classification cache, as storage/cache.py: one label per
-- (text, model, prompt), so no text is classified twice by the same model and prompt.
CREATE TABLE IF NOT EXISTS cartridge.chat_labels (
    text_hash        TEXT NOT NULL,
    classifier_model TEXT NOT NULL,
    prompt_hash      TEXT NOT NULL,
    category         TEXT NOT NULL CHECK (category IN ('self_reflection', 'relationships', 'emotions_situation',
                         'personal_growth', 'health_body', 'career_work', 'creative', 'learning_ideas', 'technical',
                         'factual_lookup', 'logistics', 'other')),
    task_kind        TEXT CHECK (task_kind IN ('situation_help', 'self_synthesis', 'framework_lens',
                         'change_over_time', 'role_request', 'build_artifact', 'recall')),
    relevant         BOOL NOT NULL,
    reason           TEXT,
    raw_reply        TEXT NOT NULL,
    run_id           TEXT REFERENCES cartridge.gen_runs,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (text_hash, classifier_model, prompt_hash)
);
"""

_TASK_COLUMNS = tuple(Task.model_fields)
_PAGE_COLUMNS = tuple(Page.model_fields)
_LABEL_COLUMNS = tuple(Label.model_fields)


def connect(database_url: str | None = None):
    import psycopg

    return psycopg.connect(database_url or os.environ["DATABASE_URL"], autocommit=True)


def init(conn) -> None:
    conn.execute(SCHEMA)


def _values(model, columns: tuple[str, ...]) -> tuple:
    def adapt(v):
        return list(v) if isinstance(v, tuple) else Jsonb(v) if isinstance(v, dict) else v

    return tuple(adapt(getattr(model, c)) for c in columns)


def _insert(conn, table: str, columns: tuple[str, ...], rows: Iterable[tuple], conflict: str) -> int:
    """Insert rows in one transaction; the number actually inserted."""
    sql = (
        f"INSERT INTO cartridge.{table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(['%s'] * len(columns))}) {conflict}"
    )
    rows = list(rows)
    if not rows:
        return 0
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(sql, rows)  # pipelined: one round trip per batch, not per row
        return cur.rowcount


def insert_corpus(conn, corpus: Corpus, pages: list[Page]) -> None:
    """The corpus and its pages in timeline order. Re-running updates the
    pages (e.g. a new held-out split)."""
    conn.execute(
        "INSERT INTO cartridge.corpora (corpus_id, name, n_tokens) VALUES (%s, %s, %s) "
        "ON CONFLICT (corpus_id) DO NOTHING",
        (corpus.corpus_id, corpus.name, corpus.n_tokens),
    )
    columns = ("corpus_id", "position", *_PAGE_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns[1:] if c != "citation")
    _insert(
        conn, "corpus_pages", columns,
        ((corpus.corpus_id, i, *_values(p, _PAGE_COLUMNS)) for i, p in enumerate(pages)),
        f"ON CONFLICT (corpus_id, citation) DO UPDATE SET {updates}",
    )


def fetch_pages(conn, corpus_id: str) -> list[Page]:
    """The corpus pages in timeline order."""
    rows = conn.execute(
        f"SELECT {', '.join(_PAGE_COLUMNS)} FROM cartridge.corpus_pages WHERE corpus_id = %s ORDER BY position",
        (corpus_id,),
    ).fetchall()
    return [Page(**dict(zip(_PAGE_COLUMNS, r))) for r in rows]


def insert_run(conn, run: GenRun) -> None:
    conn.execute(
        "INSERT INTO cartridge.gen_runs (run_id, component, model, prompt_version, config) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
        (run.run_id, run.component, run.model, run.prompt_version, Jsonb(run.config)),
    )


def insert_tasks(conn, tasks: Iterable[Task]) -> int:
    """Insert tasks; a text_hash already in the store is skipped. Returns the
    number inserted."""
    rows = (_values(t, _TASK_COLUMNS) for t in tasks)
    return _insert(conn, "tasks", _TASK_COLUMNS, rows, "ON CONFLICT (text_hash) DO NOTHING")


def fetch_tasks(conn, query: TaskQuery) -> list[Task]:
    """Candidate tasks for sampler.select: the query's SQL filters, unordered."""
    where, args = ["split = %s", "status = 'validated'"], [query.split]
    if query.mix is not None and query.origin_mix is None:
        where.append("kind = ANY(%s)")
        args.append(list(query.mix))
    if query.origin_mix is not None:
        where.append("origin = ANY(%s)")
        args.append(sorted(query.origin_mix))
    if query.origins is not None:
        where.append("origin = ANY(%s)")
        args.append(sorted(query.origins))
    if query.styles is not None:
        where.append("style = ANY(%s)")
        args.append(sorted(query.styles))
    if query.run_ids is not None:
        where.append("run_id = ANY(%s)")
        args.append(sorted(query.run_ids))
    if query.exclude:
        where.append("NOT (task_id = ANY(%s))")
        args.append(sorted(query.exclude))
    if query.unlabeled_by is not None:
        key = (query.unlabeled_by,) if isinstance(query.unlabeled_by, str) else tuple(query.unlabeled_by)
        match = " AND ".join(f"l.{c} = %s" for c in ("teacher_model", "corpus_id", "instructions_hash")[: len(key)])
        where.append(f"NOT EXISTS (SELECT 1 FROM cartridge.labels l WHERE l.task_id = t.task_id AND {match})")
        args.extend(key)
    rows = conn.execute(
        f"SELECT {', '.join(_TASK_COLUMNS)} FROM cartridge.tasks t WHERE {' AND '.join(where)}", args
    ).fetchall()
    return [Task(**dict(zip(_TASK_COLUMNS, r))) for r in rows]


def list_tasks(conn, split: str | None = None, origin: str | None = None, run_ids: Sequence[str] = (),
               status: str | None = None) -> list[Task]:
    """Tasks of any status matching the given filters, in creation order."""
    where, args = ["true"], []
    for column, value in (("split", split), ("origin", origin), ("status", status)):
        if value:
            where.append(f"{column} = %s")
            args.append(value)
    if run_ids:
        where.append("run_id = ANY(%s)")
        args.append(list(run_ids))
    rows = conn.execute(
        f"SELECT {', '.join(_TASK_COLUMNS)} FROM cartridge.tasks WHERE {' AND '.join(where)} "
        "ORDER BY created_at, task_id", args
    ).fetchall()
    return [Task(**dict(zip(_TASK_COLUMNS, r))) for r in rows]


def update_validation(conn, tasks: Iterable[Task]) -> int:
    """Write re-validated fields (text, text_hash, status, reject_reason,
    citations, quotes, grounded) onto stored tasks by task_id."""
    columns = ("text", "text_hash", "status", "reject_reason", "ref_citations", "ref_evidence", "grounded")
    values = [(*_values(t, columns), t.task_id) for t in tasks]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            f"UPDATE cartridge.tasks SET {', '.join(f'{c} = %s' for c in columns)} WHERE task_id = %s", values
        )
    return len(values)


def fetch_promotable_chats(conn, classifier_model: str, prompt_hash: str) -> list[tuple[ChatPrompt, ChatLabel, str]]:
    """(prompt, label, label run_id) for stored chat prompts this model and
    prompt labelled relevant with a task kind, oldest first."""
    rows = conn.execute(
        f"SELECT {', '.join('p.' + c for c in _CHAT_COLUMNS)}, l.category, l.task_kind, l.relevant, l.reason, "
        "l.raw_reply, l.run_id FROM cartridge.chat_prompts p JOIN cartridge.chat_labels l "
        "ON l.text_hash = p.text_hash AND l.classifier_model = %s AND l.prompt_hash = %s "
        "WHERE l.relevant AND l.task_kind IS NOT NULL ORDER BY p.create_time NULLS LAST, p.conversation_id",
        (classifier_model, prompt_hash),
    ).fetchall()
    n = len(_CHAT_COLUMNS)
    return [
        (ChatPrompt(**dict(zip(_CHAT_COLUMNS, r[:n]))),
         ChatLabel(**dict(zip(("category", "task_kind", "relevant", "reason", "raw_reply"), r[n : n + 5]))),
         r[n + 5])
        for r in rows
    ]


def resplit_chat_exemplars(conn) -> int:
    """Chat tasks in the exemplar split move to train (eval stays eval)."""
    return conn.execute("UPDATE cartridge.tasks SET split = 'train' WHERE origin = 'chat' AND split = 'exemplar'").rowcount


def fetch_tasks_by_id(conn, task_ids: Iterable) -> dict:
    """{task_id: Task} for these ids (any split or status)."""
    rows = conn.execute(
        f"SELECT {', '.join(_TASK_COLUMNS)} FROM cartridge.tasks WHERE task_id = ANY(%s)", (list(task_ids),)
    ).fetchall()
    return {r[0]: Task(**dict(zip(_TASK_COLUMNS, r))) for r in rows}


def fetch_hashes(conn) -> set[str]:
    """text_hash of every task, validated or rejected (dedupe for the Synthesizer)."""
    return {r[0] for r in conn.execute("SELECT text_hash FROM cartridge.tasks").fetchall()}


def insert_labels(conn, labels: Iterable[Label]) -> int:
    """Insert teacher labels; one per (task, teacher, corpus, instructions)."""
    rows = (_values(lab, _LABEL_COLUMNS) for lab in labels)
    return _insert(
        conn, "labels", _LABEL_COLUMNS, rows,
        "ON CONFLICT (task_id, teacher_model, corpus_id, instructions_hash) DO NOTHING",
    )


def targets_manifest(conn, run_id: str) -> dict:
    """The run's Parquet shards and label counts, for a targets-<run_id> artifact."""
    shards = [r[0] for r in conn.execute(
        "SELECT DISTINCT targets_uri FROM cartridge.labels WHERE run_id = %s AND targets_uri IS NOT NULL ORDER BY 1",
        (run_id,),
    ).fetchall()]
    counts = dict(conn.execute(
        "SELECT status, count(*) FROM cartridge.labels WHERE run_id = %s GROUP BY 1", (run_id,)
    ).fetchall())
    return {"run_id": run_id, "shards": shards, "labels": counts}


def update_labels(conn, labels: Iterable[Label]) -> int:
    """Write re-validated status and reject_reason onto stored labels by
    label_id (answers and targets untouched)."""
    values = [(lab.status, lab.reject_reason, lab.label_id) for lab in labels]
    if not values:
        return 0
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("UPDATE cartridge.labels SET status = %s, reject_reason = %s WHERE label_id = %s", values)
        return cur.rowcount


def delete_labels(conn, label_ids: Iterable) -> int:
    """Delete labels so their tasks can be labeled again (the labels' unique
    key and unlabeled_by would otherwise skip them). Their Parquet rows stay."""
    return conn.execute("DELETE FROM cartridge.labels WHERE label_id = ANY(%s)", (list(label_ids),)).rowcount


def delete_human_tasks(conn) -> int:
    """Delete the human tasks and their templates, so a changed question file can
    be re-seeded (ids follow from the text, so unchanged questions keep theirs)."""
    return conn.execute("DELETE FROM cartridge.tasks WHERE origin IN ('human', 'human_template')").rowcount


def fetch_labels(conn, run_ids: Sequence[str], status: str = "validated", split: Split = "train") -> list[Label]:
    """The labels of these runs whose task is in `split`, ordered by label_id.
    The default, train, is what training reads: eval and exemplar tasks never
    become training targets, nor do human tasks (a second guard: label() never
    labels them for train); split="eval" gives a held-out validation set."""
    rows = conn.execute(
        f"SELECT {', '.join('l.' + c for c in _LABEL_COLUMNS)} FROM cartridge.labels l "
        "JOIN cartridge.tasks t USING (task_id) "
        "WHERE l.run_id = ANY(%s) AND l.status = %s AND t.split = %s "
        "AND (t.split <> 'train' OR t.origin <> 'human') ORDER BY l.label_id",
        (list(run_ids), status, split),
    ).fetchall()
    return [Label(**dict(zip(_LABEL_COLUMNS, r))) for r in rows]


_CHAT_COLUMNS = ("conversation_id", "create_time", "title", "model_slug", "text", "text_hash")


def upsert_chat_prompts(conn, prompts: Iterable[ChatPrompt]) -> int:
    """Store extracted prompts; re-running updates them in place."""
    values = [tuple(getattr(p, c) for c in _CHAT_COLUMNS) for p in prompts]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _CHAT_COLUMNS[1:])
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO cartridge.chat_prompts ({', '.join(_CHAT_COLUMNS)}) "
            f"VALUES ({', '.join(['%s'] * len(_CHAT_COLUMNS))}) ON CONFLICT (conversation_id) DO UPDATE SET {updates}",
            values,
        )
    return len(values)


def fetch_unlabeled_chats(conn, classifier_model: str, prompt_hash: str) -> tuple[list[ChatPrompt], int]:
    """(one stored prompt per text with no label for this model and prompt,
    oldest first; the number of texts that have one: the cache hits)."""
    rows = conn.execute(
        f"SELECT DISTINCT ON (p.text_hash) {', '.join('p.' + c for c in _CHAT_COLUMNS)} FROM cartridge.chat_prompts p "
        "WHERE NOT EXISTS (SELECT 1 FROM cartridge.chat_labels l WHERE l.text_hash = p.text_hash "
        "AND l.classifier_model = %s AND l.prompt_hash = %s) "
        "ORDER BY p.text_hash, p.create_time NULLS LAST, p.conversation_id",
        (classifier_model, prompt_hash),
    ).fetchall()
    hits = conn.execute(
        "SELECT count(DISTINCT p.text_hash) FROM cartridge.chat_prompts p JOIN cartridge.chat_labels l "
        "ON l.text_hash = p.text_hash AND l.classifier_model = %s AND l.prompt_hash = %s",
        (classifier_model, prompt_hash),
    ).fetchone()[0]
    prompts = sorted((ChatPrompt(**dict(zip(_CHAT_COLUMNS, r))) for r in rows),
                     key=lambda p: (p.create_time is None, p.create_time, p.conversation_id))
    return prompts, hits


def insert_chat_labels(conn, rows: Iterable[ChatRow]) -> int:
    """Cache parsed labels (unparsed ones are left out, so they are retried);
    an existing (text, model, prompt) label is kept. Returns the rows sent."""
    values = [
        (r.prompt.text_hash, r.classifier_model, r.prompt_hash, r.label.category, r.label.task_kind,
         r.label.relevant, r.label.reason, r.label.raw_reply, r.run_id)
        for r in rows if r.label.category != "unparsed"
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO cartridge.chat_labels (text_hash, classifier_model, prompt_hash, category, task_kind, "
            "relevant, reason, raw_reply, run_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (text_hash, classifier_model, prompt_hash) DO NOTHING",
            values,
        )
    return len(values)


def label_summary(conn, run_ids: Sequence[str]) -> dict[str, dict]:
    """Per task origin over these label runs: tasks, validated, truncated
    (finish_reason length), cited (at least one citation), refusals,
    completion tokens and reject reasons; plus the rates."""
    rows = conn.execute(
        "SELECT t.origin, count(*), count(*) FILTER (WHERE l.status = 'validated'), "
        "count(*) FILTER (WHERE l.finish_reason = 'length'), count(*) FILTER (WHERE cardinality(l.citations) > 0), "
        "count(*) FILTER (WHERE l.reject_reason = 'refusal'), coalesce(sum(l.n_completion_tokens), 0) "
        "FROM cartridge.labels l JOIN cartridge.tasks t USING (task_id) WHERE l.run_id = ANY(%s) GROUP BY 1 ORDER BY 1",
        (list(run_ids),),
    ).fetchall()
    reasons = conn.execute(
        "SELECT t.origin, l.reject_reason, count(*) FROM cartridge.labels l JOIN cartridge.tasks t USING (task_id) "
        "WHERE l.run_id = ANY(%s) AND l.reject_reason IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 3 DESC",
        (list(run_ids),),
    ).fetchall()
    out = {}
    for origin, n, valid, truncated, cited, refused, tokens in rows:
        out[origin] = {"n": n, "validated": valid, "truncated": truncated, "cited": cited, "refused": refused,
                       "completion_tokens": tokens, "truncation_rate": truncated / n, "cite_rate": cited / n,
                       "reject_reasons": {r: k for o, r, k in reasons if o == origin}}
    return out


def label_samples(conn, run_id: str, per_origin: int = 1) -> list[dict]:
    """Validated labels of a run, `per_origin` per task origin (random): the
    task and the teacher's answer, for review."""
    rows = conn.execute(
        "SELECT origin, kind, text, answer, finish_reason, citations FROM ("
        "SELECT t.origin, t.kind, t.text, l.answer, l.finish_reason, l.citations, "
        "row_number() OVER (PARTITION BY t.origin ORDER BY random()) AS i "
        "FROM cartridge.labels l JOIN cartridge.tasks t USING (task_id) "
        "WHERE l.run_id = %s AND l.status = 'validated') s WHERE i <= %s ORDER BY origin",
        (run_id, per_origin),
    ).fetchall()
    return [dict(zip(("origin", "kind", "task", "answer", "finish_reason", "citations"), r)) for r in rows]


def stats(conn) -> dict:
    """Task counts by split, origin, kind and status; reject reasons; labels by
    teacher and status."""
    tasks = conn.execute(
        "SELECT split, origin, kind, status, count(*) FROM cartridge.tasks GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"
    ).fetchall()
    reasons = conn.execute(
        "SELECT kind, reject_reason, count(*) FROM cartridge.tasks WHERE status = 'rejected' GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    labels = conn.execute(
        "SELECT teacher_model, status, count(*) FROM cartridge.labels GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    return {
        "tasks": [dict(zip(("split", "origin", "kind", "status", "n"), r)) for r in tasks],
        "reject_reasons": [dict(zip(("kind", "reason", "n"), r)) for r in reasons],
        "labels": [dict(zip(("teacher_model", "status", "n"), r)) for r in labels],
    }
