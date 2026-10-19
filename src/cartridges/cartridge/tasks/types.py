"""Data types of the Task Store (docs/plans/task-db.md). All frozen.

Kinds, styles, origins, splits and statuses are Literals, checked again by the
Postgres CHECK constraints in store.SCHEMA; the tuples list their values.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Literal, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

Kind = Literal[
    "situation_help", "self_synthesis", "framework_lens", "change_over_time", "role_request", "build_artifact", "recall"
]
Style = Literal["dictated", "typed", "narrative"]
Origin = Literal["human", "human_template", "synthetic", "chat", "psych_test"]
Split = Literal["exemplar", "train", "eval"]
Status = Literal["validated", "rejected"]
Component = Literal["seed", "synthesize", "label", "classify"]

Category = Literal[
    "self_reflection", "relationships", "emotions_situation", "personal_growth", "health_body", "career_work",
    "creative", "learning_ideas", "technical", "factual_lookup", "logistics", "other", "unparsed",
]

KINDS: tuple[Kind, ...] = get_args(Kind)
CATEGORIES: tuple[Category, ...] = get_args(Category)
STYLES: tuple[Style, ...] = get_args(Style)

Messages = list[dict]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Corpus(_Frozen):
    corpus_id: str  # sha256[:12] of the corpus text
    name: str
    n_tokens: int


class Page(_Frozen):
    """One page of the corpus, as in teacher_context.md."""

    citation: str  # "Morning Pages:p.63"
    journal: str
    page_no: int
    date: dt.date | None
    date_uncertain: bool  # the entry's date is marked "(?)" or "(est.)"
    heldout: bool  # in eval's heldout_pages.json
    text: str


class GenRun(_Frozen):
    run_id: str  # the W&B run id
    component: Component
    model: str | None = None
    prompt_version: str | None = None
    config: dict = {}


class Task(_Frozen):
    """A user message (the task), its provenance and, for synthetic tasks,
    the reference pages its answer should draw on."""

    task_id: UUID
    text: str
    text_hash: str
    kind: Kind
    style: Style | None = None
    framework: str | None = None
    role: str | None = None
    origin: Origin
    split: Split
    status: Status
    reject_reason: str | None = None
    corpus_id: str | None = None
    source_citations: tuple[str, ...] = ()  # pages the Synthesizer was shown
    ref_citations: tuple[str, ...] = ()  # pages the answer should draw on
    ref_evidence: tuple[str, ...] = ()  # one verbatim quote per ref citation
    grounded: bool | None = None  # None for human tasks
    parent_ids: tuple[UUID, ...] = ()  # exemplars shown, or the original of a template
    run_id: str | None = None
    metadata: dict = {}  # provenance outside the task store, e.g. a chat's conversation_id


class Recipe(_Frozen):
    """What one Synthesizer call should write."""

    kind: Kind
    style: Style
    framework: str | None = None
    role: str | None = None
    citations: tuple[str, ...]  # focus pages
    exemplar_ids: tuple[UUID, ...] = ()


class Draft(_Frozen):
    """The Synthesizer's parsed reply."""

    text: str
    ref_citations: tuple[str, ...]
    ref_evidence: tuple[str, ...]


class Label(_Frozen):
    """A teacher answer to a task; its top-k targets live in a Parquet shard."""

    label_id: UUID
    task_id: UUID
    run_id: str
    teacher_model: str
    corpus_id: str
    instructions_hash: str
    answer: str
    finish_reason: str
    n_prompt_tokens: int
    n_completion_tokens: int
    citations: tuple[str, ...] = ()
    status: Status
    reject_reason: str | None = None
    targets_uri: str | None = None
    targets_row: int | None = None


class TargetRow(_Frozen):
    """One row of a Target Store shard (train_distill.Example plus ids)."""

    label_id: str
    task_id: str
    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]]
    topk_logprobs: list[list[float]]


class TaskQuery(_Frozen):
    split: Split  # required: no default, so eval can't leak by omission
    n: int
    seed: int
    mix: Mapping[Kind, float] | None = None  # stratify by kind; weights sum to 1
    origin_mix: Mapping[Origin, float] | None = None  # a quota per origin first, then `mix` within each
    origins: frozenset[Origin] | None = None  # None = any
    styles: frozenset[Style] | None = None
    run_ids: frozenset[str] | None = None  # None = any run
    # Skip tasks already labeled under this key (applied in SQL): (teacher_model,
    # corpus_id, instructions_hash), the labels' unique key, so a new corpus or new
    # instructions relabel; a bare model name matches any corpus and instructions.
    unlabeled_by: str | tuple[str, str, str] | None = None
    exclude: frozenset[UUID] = frozenset()

    @model_validator(mode="after")
    def _check(self) -> TaskQuery:
        if self.n <= 0:
            raise ValueError(f"n must be > 0, got {self.n}")
        for name, weights in (("mix", self.mix), ("origin_mix", self.origin_mix)):
            if weights is not None and not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
                raise ValueError(f"{name} weights sum to {sum(weights.values())}, not 1")
        return self


class Sample(_Frozen):
    query: TaskQuery
    tasks: tuple[Task, ...]
    shortfall: Mapping[str, int]  # requested minus returned, per kind, or per "origin/kind" with an origin_mix
    sample_hash: str  # sha256 of the query and the ordered task ids


class SynthConfig(_Frozen):
    n: int
    seed: int
    model: str
    run_id: str
    corpus_id: str
    kind_mix: Mapping[Kind, float]
    style_mix: Mapping[Style, float]
    exemplars_per_prompt: int = 3
    temperature: float = 0.6
    max_tokens: int = 512  # with DraftReply's limits; questions run ~110 tokens, replies ~230
    batch_size: int = 64  # prompts per generate call; each batch is inserted as it arrives
    prompt_version: str


class SynthReport(_Frozen):
    n_drafts: int
    n_validated: int
    n_inserted: int  # new rows (an existing text_hash is skipped)
    by_kind: Mapping[str, int]  # validated per kind
    reject_reasons: Mapping[str, int]
    grounded_rate: float  # over parsed drafts
    distinct_2: float  # distinct word bigrams / all bigrams, over validated texts


class ChatPrompt(_Frozen):
    """The first user message of one ChatGPT conversation."""

    conversation_id: str
    create_time: dt.datetime | None
    title: str | None
    model_slug: str | None
    text: str
    text_hash: str


class ChatLabel(_Frozen):
    """The classifier's verdict on a ChatPrompt ("unparsed" if its reply was not
    the expected JSON)."""

    category: Category
    task_kind: Kind | None = None
    relevant: bool | None = None
    reason: str | None = None
    raw_reply: str


class ChatRow(_Frozen):
    """A prompt and its label, for cartridge.chat_labels (keyed by the prompt's
    text_hash, the model and the prompt hash)."""

    prompt: ChatPrompt
    label: ChatLabel
    classifier_model: str
    prompt_hash: str
    run_id: str
