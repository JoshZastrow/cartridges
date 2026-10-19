# Task DB, Task Synthesizer, Teacher: design

Status: proposal, 2026-10-15. Supersedes the self-study path (`selfstudy.py`,
`data.build_selfstudy`, `selfstudy/train.jsonl`). The GRPO QA path stays paused.

## 1. Components and names

| Component | Was | Role | Module |
|---|---|---|---|
| **Task Store** | `selfstudy/train.jsonl` | Postgres tables of tasks (user messages), their provenance, and teacher labels | `tasks/store.py` (the only module that talks to Postgres) |
| **Task Sampler** | (implicit) | Pure, deterministic, stratified selection of tasks from the store | `tasks/sampler.py` |
| **Task Synthesizer** | "Bot A" | Writes new tasks: the served model **with** the full journal (same cached prefix as the Teacher), pointed at a few focus pages and shown human exemplars | `tasks/synthesize.py` |
| **Teacher** | "Bot B" | Labels tasks: the served model **with** the full journal (ICL), greedy, top-20 logprobs per token → distillation targets | `tasks/label.py` |
| **Target Store** | inside `train.jsonl` | Token ids + top-k targets as Parquet shards (Modal Volume + W&B artifact); Postgres holds a pointer | `tasks/targets.py` |
| **Teacher service** | `teacher.py` | The vLLM server both call with the same journal prefix; they differ only in instructions (write a task vs answer it) | unchanged |

Data flow:

```
human JSONL ──seed──► Task Store (exemplar / eval)
                         │ sample(split=exemplar)
                         ▼
   corpus pages ──► Task Synthesizer ──► Task Store (train, synthetic)
                                            │ sample(split=train, unlabeled_by=teacher)
                                            ▼
                                         Teacher ──► Target Store (Parquet) + labels row
                                                          │
                                                          ▼
                                       train_distill: Example -> to_tensors -> KL loss
```

## 2. Code rules (composability)

Every component is the same five stages, each a separate function:

| Stage | Kind | Example |
|---|---|---|
| **load** | IO | `store.fetch_tasks(conn, query)`, `store.fetch_pages(conn, corpus_id)` |
| **validate** | pure, returns a value (never raises on bad data) | `validate_draft(draft, recipe, pages, seen) -> Task` with `status`, `reject_reason` |
| **preprocess** | pure | `render_prompt(recipe, pages, exemplars) -> Messages`, `teacher.render(...) -> ids` |
| **transform** | pure | `parse_draft(text) -> Draft \| None`, `to_target_row(task, completion) -> TargetRow` |
| **write** | IO | `store.insert_tasks(conn, tasks)`, `targets.write_shard(rows, uri)` |

- Data types are frozen pydantic models (`ConfigDict(frozen=True)`), as in `storage/cache.py` and `train_distill.py`.
- Orchestrators (`synthesize`, `label`) take their IO as arguments: `fetch`, `generate`, `write`. Tests pass fakes and need no database or GPU.
- No module-level connections or globals. `DATABASE_URL` is read only in `store.connect()`.
- Randomness only through an explicit `seed`. The same inputs and seed give the same output.

## 3. Package layout

```
src/cartridges/cartridge/tasks/
  types.py       enums + frozen models: Task, Recipe, Draft, Label, TargetRow, TaskQuery, Sample
  taxonomy.py    data only: KIND_MIX, STYLE_MIX, FRAMEWORKS, ROLES, PAGES_PER_KIND, PROMPT_VERSION
  store.py       IO: SCHEMA, connect, init, fetch_*/insert_*; psycopg only here
  sampler.py     pure select(); sample_tasks(fetch, query)
  validate.py    pure validate_draft(), validate_label(), normalize_text(), text_hash()
  synthesize.py  pure plan_recipes(), render_prompt(), parse_draft(); orchestrator synthesize()
  label.py       pure to_prompt(), to_target_row(); orchestrator label()
  targets.py     IO: write_shard(), read_examples() (-> train_distill.Example)
  seed.py        load human JSONL -> validate -> assign split -> insert
```

CLI (`cartridges tasks ...`, also the agent runbook skill):
`init` · `seed FILE` · `synthesize --n --seed` · `label --teacher --n` · `sample --split --n --seed` · `stats`.

## 4. Taxonomy (task kinds)

| `kind` | Share | Focus pages (the full journal is always in context) |
|---|---|---|
| `situation_help`: a new in-the-moment situation + "help me understand / change this" | 30% | 1–2 chunks with recurring people/themes |
| `self_synthesis`: patterns, strengths, beliefs, self-sabotage | 15% | 4–6 chunks across years |
| `framework_lens`: a framework applied to me | 15% | 3–4 chunks + one `framework` |
| `change_over_time`: how I've changed since … | 10% | chunks ≥ 2 years apart |
| `role_request`: "as a therapist / coach …" | 10% | 3–4 chunks + one `role` |
| `build_artifact`: values hierarchy, belief system, growth plan | 10% | 3–4 chunks |
| `recall`: what I did / said at a place and time | 10% | 1 chunk, held-out pages excluded |

`style`: `dictated` 40% · `typed` 40% · `narrative` 20%.

`FRAMEWORKS`: enneagram, attachment theory (Bowlby), Gottman, Esther Perel, Michelangelo
phenomenon, start/stop/keep, *Becoming the One* (Sheleana Aiyana), Internal Family Systems,
CBT cognitive distortions, schema therapy, ACT, Jungian shadow work, Stoicism, nonviolent
communication, Big Five, self-determination theory, polyvagal theory, Brené Brown (shame,
vulnerability).

## 5. Task Store schema (Postgres, schema `cartridge`)

TEXT with CHECK constraints rather than enum types, so new values are a one-line change.

```sql
CREATE SCHEMA IF NOT EXISTS cartridge;

CREATE TABLE IF NOT EXISTS cartridge.corpora (
    corpus_id   TEXT PRIMARY KEY,              -- sha256[:12] of teacher_context.md
    name        TEXT NOT NULL,                 -- 'teacher_context.md'
    n_tokens    INT  NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cartridge.corpus_pages (
    corpus_id      TEXT NOT NULL REFERENCES cartridge.corpora,
    citation       TEXT NOT NULL,              -- 'Morning Pages:p.63'
    journal        TEXT NOT NULL,
    page_no        INT  NOT NULL,
    date           DATE,
    date_uncertain BOOL NOT NULL,              -- '(?)' / '(est.)'
    heldout        BOOL NOT NULL DEFAULT false,-- from heldout_pages.json
    PRIMARY KEY (corpus_id, citation)
);

CREATE TABLE IF NOT EXISTS cartridge.gen_runs (
    run_id         TEXT PRIMARY KEY,           -- W&B run id (lineage both ways)
    component      TEXT NOT NULL CHECK (component IN ('seed', 'synthesize', 'label')),
    model          TEXT,
    prompt_version TEXT,
    config         JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cartridge.tasks (
    task_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text             TEXT NOT NULL,            -- the user message, verbatim
    text_hash        TEXT NOT NULL UNIQUE,     -- sha256 of normalize_text(text): exact dedupe
    kind             TEXT NOT NULL CHECK (kind IN ('situation_help', 'self_synthesis', 'framework_lens',
                         'change_over_time', 'role_request', 'build_artifact', 'recall')),
    style            TEXT CHECK (style IN ('dictated', 'typed', 'narrative')),
    framework        TEXT,
    role             TEXT,
    origin           TEXT NOT NULL CHECK (origin IN ('human', 'human_template', 'synthetic')),
    split            TEXT NOT NULL CHECK (split IN ('exemplar', 'train', 'eval')),
    status           TEXT NOT NULL CHECK (status IN ('validated', 'rejected')),
    reject_reason    TEXT,
    corpus_id        TEXT REFERENCES cartridge.corpora,
    source_citations TEXT[] NOT NULL DEFAULT '{}',  -- pages the Synthesizer was shown
    ref_citations    TEXT[] NOT NULL DEFAULT '{}',  -- pages the answer should draw on
    ref_evidence     TEXT[] NOT NULL DEFAULT '{}',  -- one verbatim quote per ref citation
    grounded         BOOL,                          -- refs check out against the pages; NULL for human
    parent_ids       UUID[] NOT NULL DEFAULT '{}',  -- exemplars shown, or the original of a template
    run_id           TEXT REFERENCES cartridge.gen_runs,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (status = 'validated' OR reject_reason IS NOT NULL),
    CHECK (origin <> 'synthetic' OR split = 'train')    -- generated tasks never become eval/exemplars
);
CREATE INDEX IF NOT EXISTS tasks_sample_idx ON cartridge.tasks (split, status, kind);

CREATE TABLE IF NOT EXISTS cartridge.labels (
    label_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id             UUID NOT NULL REFERENCES cartridge.tasks,
    run_id              TEXT NOT NULL REFERENCES cartridge.gen_runs,
    teacher_model       TEXT NOT NULL,
    corpus_id           TEXT NOT NULL REFERENCES cartridge.corpora,
    instructions_hash   TEXT NOT NULL,         -- sha256[:12] of teacher.INSTRUCTIONS
    answer              TEXT NOT NULL,
    finish_reason       TEXT NOT NULL,
    n_prompt_tokens     INT  NOT NULL,
    n_completion_tokens INT  NOT NULL,
    citations           TEXT[] NOT NULL DEFAULT '{}',  -- parsed from answer
    status              TEXT NOT NULL CHECK (status IN ('validated', 'rejected')),
    reject_reason       TEXT,
    targets_uri         TEXT,                  -- /cache/targets/<run_id>/part-00000.parquet
    targets_row         INT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, teacher_model, corpus_id, instructions_hash)
);
```

Rejected tasks and labels stay in the store: they are the datagen quality metrics
(reject rate by reason and kind) and stop the Synthesizer from re-proposing them (`text_hash`).

## 6. Target Store schema (Parquet)

Full-vocabulary logits are not stored: 151k vocab × ~1M target tokens × 2 bytes ≈ 300 GB.
The Teacher stores its **top-20 logprobs** per completion token (vLLM's maximum; the paper
also distills from top-k), ≈ 160 MB per 1M tokens.

| Column | Type | |
|---|---|---|
| `label_id` | string | → `cartridge.labels` |
| `task_id` | string | |
| `prompt_ids` | list<int32> | chat-templated task, no journal (student input) |
| `completion_ids` | list<int32> | teacher answer, ends with `<|im_end|>` |
| `topk_ids` | list<list<int32>> | [len(completion), 20] |
| `topk_logprobs` | list<list<float32>> | [len(completion), 20] |

Shards: `/cache/targets/<run_id>/part-NNNNN.parquet` on `cartridges-cartridge-cache`,
logged as W&B artifact `targets-<run_id>` (type `dataset`, with a browsable table of task
text + answer). `targets.read_examples(uris) -> Iterator[Example]` feeds the existing
`train_distill.to_tensors`.

## 7. Task Sampler: API contract

```python
class TaskQuery(BaseModel, frozen=True):
    split: Split                          # required: no default, so eval can't leak by omission
    n: int                                # > 0
    seed: int
    mix: Mapping[Kind, float] | None = None   # stratify by kind; weights sum to 1; None = no stratification
    origins: frozenset[Origin] | None = None  # None = any
    styles: frozenset[Style] | None = None
    unlabeled_by: str | None = None       # teacher model: skip tasks it has already labeled
    exclude: frozenset[UUID] = frozenset()

class Sample(BaseModel, frozen=True):
    query: TaskQuery
    tasks: tuple[Task, ...]
    shortfall: Mapping[Kind, int]         # requested minus returned, per kind (0 if met)
    sample_hash: str                      # sha256 of query + ordered task_ids: logged to W&B

def select(candidates: Sequence[Task], query: TaskQuery) -> Sample: ...          # pure
def sample_tasks(fetch: Callable[[TaskQuery], list[Task]], query: TaskQuery) -> Sample:
    return select(fetch(query), query)
```

Guarantees of `select`:
1. Returns only `status = 'validated'` tasks of `query.split` matching every filter.
2. Without replacement; never a task in `exclude`.
3. Per-kind quotas by largest remainder of `n × mix[kind]`; uniform within a kind via
   `random.Random(seed)`. A short kind is reported in `shortfall` and not refilled from other
   kinds, so the mix stays honest.
4. Deterministic: same candidates (in any order) + same query → same `Sample`. Candidates are
   sorted by `task_id` before sampling.
5. Raises `ValueError` only for an invalid query (unknown kind, weights not summing to 1, n ≤ 0).

`store.fetch_tasks(conn, query)` applies the SQL filters (split, status, origins, styles,
`unlabeled_by` via `NOT EXISTS` on labels); `select` re-checks them, so a wrong fetch can't
leak eval tasks.

Callers:
- Synthesizer: `TaskQuery(split='exemplar', origins={'human_template'}, n=3, seed=…, mix={kind: 1})`.
- Teacher: `TaskQuery(split='train', unlabeled_by=MODEL_ID, n=…, seed=…, mix=KIND_MIX)`.
- Eval: `TaskQuery(split='eval', n=…, seed=0)`.

## 8. Task Synthesizer: contract

```python
class Recipe(BaseModel, frozen=True):
    kind: Kind; style: Style; framework: str | None; role: str | None
    citations: tuple[str, ...]            # focus pages
    exemplar_ids: tuple[UUID, ...]

class Draft(BaseModel, frozen=True):      # parsed model reply
    text: str; ref_citations: tuple[str, ...]; ref_evidence: tuple[str, ...]

def plan_recipes(n, kind_mix, style_mix, pages, exemplars_by_kind, seed) -> list[Recipe]   # pure
def render_prompt(recipe, focus_texts, exemplar_texts) -> Messages                         # pure; journal prefix added by teacher.render
def parse_draft(reply: str) -> Draft | None                                                # pure
def validate_draft(draft, recipe, pages, seen_hashes) -> Task                              # pure
def synthesize(config, *, fetch_pages, sample, generate, insert) -> SynthReport            # orchestrator
```

Input `SynthConfig`: `n`, `seed`, `kind_mix`, `style_mix`, `exemplars_per_prompt=3`,
`temperature=0.6`, `max_tokens=512`, `model`, `prompt_version`.
Output: inserted `Task` rows (validated and rejected) + `SynthReport` (counts per kind,
reject reasons, grounded rate, distinct-n), logged to the W&B run.

`validate_draft` reject reasons: `unparseable`, `empty`, `duplicate` (text_hash seen),
`ungrounded` (a ref citation not in `recipe.citations` or a quote not found on a cited page),
`uncertain_date` (recall on a `date_uncertain` page), `heldout_page` (recall citing a held-out page),
`mentions_document` ("the journal", "page", "the author"), `too_long` (> 400 words),
`near_eval` (word-set Jaccard ≥ 0.5 with any `eval` task).

Why the full journal: questions like "how have I changed since four years ago" or "what does my
enneagram say about me" need the whole arc, and real names, places and courses make tasks read
like the user's. vLLM prefix caching means the journal is prefilled once per container, so each
task costs only its own tokens. To keep that prefix byte-identical, the synthesis instructions,
focus pages and exemplars go in the user turn (`teacher.generate(..., with_context=True)`), the
same way self-study seed prompts work today; no change to `teacher.py`. Focus pages steer coverage so tasks don't all cluster on the
most memorable episodes.

Held-out policy: the journal in context includes held-out pages, so the guards are on the output:
`recall` focus pages never include held-out pages, a `recall` draft citing a held-out page is
rejected (`heldout_page`), and any draft close to an `eval` task is rejected (`near_eval`).

## 9. Teacher: contract

```python
def to_prompt(task: Task, tokenizer) -> PromptIds                                  # pure, teacher.render
def validate_label(task, completion, pages) -> tuple[Status, str | None]           # pure
def to_target_row(task, completion, label_id) -> TargetRow                         # pure
def label(config, *, sample, generate, write_shard, insert_labels) -> LabelReport  # orchestrator
```

Input `LabelConfig`: `teacher_model`, `n`, `seed`, `kind_mix`, `max_answer_tokens=1536`,
`top_k=20`, `temperature=0.0`.
Output: one Parquet shard per run + one `labels` row per task + `LabelReport`
(truncation rate, refusal rate, cite rate, citation validity, tokens per kind).

`validate_label` reject reasons: `no_reply`, `topk_mismatch` (len(topk) ≠ len(completion)),
`refusal`, `invalid_citation` (cites a page that doesn't exist). `truncated` is kept but flagged:
it still carries useful targets, and the truncation rate is reported.

## 10. Human seed

`seed.py`: load `user_questions.jsonl` → validate (non-empty, `kind` known after mapping the
old families onto the taxonomy) → assign `split` deterministically, stratified by kind
(8 `eval`, 15 `exemplar`) → insert `origin='human'`. Then one Synthesizer-style pass rewrites
each exemplar into a placeholder template (`origin='human_template'`, `parent_ids=[original]`),
which the user reviews before any synthesis. Human tasks are never labeled for training.

## 11. Ownership

| Team member | Files |
|---|---|
| datagen | `types.py`, `taxonomy.py`, `store.py`, `sampler.py`, `validate.py`, `synthesize.py`, `seed.py` |
| teacher | `label.py`, `targets.py` |
| trainer | `train_distill` reads `targets.read_examples` instead of `train.jsonl` |
| eval | eval consumes `split='eval'` tasks; v2 judge |
| monitoring | `gen_runs` ↔ W&B run ids, sample/target artifacts, datagen metrics |

## 12. Open decisions

1. Database: the existing Neon project (`lingering-violet-55692519`, already holding the OCR
   pages via `DATABASE_URL`), new schema `cartridge`. Recommended.
2. Modal containers need `DATABASE_URL` as a Modal secret (`cartridges-db`).
3. Mapping the 23 human questions onto the seven kinds (draft for user review).
