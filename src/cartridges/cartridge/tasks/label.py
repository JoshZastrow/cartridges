"""Teacher labeling: answer sampled train tasks with the full journal in context
and keep the answers' top-k logprobs as distillation targets
(docs/plans/task-db.md §9).

Pure stages:
    task_messages(task)                        the task as a single user turn
    to_prompt(task, tokenizer)                 its prompt_ids (teacher.render, no journal)
    validate_label(task, completion, pages)    (status, reject_reason)
    to_target_row(task, completion, label_id)  one Target Store row
    to_label(...)                              one cartridge.labels row
Orchestrator:
    label(config, pages=..., *, sample, generate, write_shard, insert_labels) -> LabelReport

The IO is injected: `sample(TaskQuery) -> Sample`, `generate(messages_list,
max_tokens=, logprobs=, temperature=) -> list[teacher.Completion | None]` (e.g. a
wrapper over Teacher.generate.map; None = a failed call), `write_shard(rows, uri)
-> uri` (targets.write_shard locally, or targets.volume_writer(...) to put the
shard on a Modal Volume) and `insert_labels(labels)` (store). Tests pass fakes.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

from cartridges.cartridge import teacher
from cartridges.cartridge.evals import (
    parse_citations,  # [J:p.N], [<J:p.N>]; a literal placeholder parses (and is invalid)
)
from cartridges.cartridge.tasks import targets
from cartridges.cartridge.tasks.taxonomy import LABEL_ORIGIN_MIX
from cartridges.cartridge.tasks.types import Label, Sample, TargetRow, Task, TaskQuery

logger = logging.getLogger(__name__)

PromptIds = tuple[int, ...]
# The model declining to answer at all. "The journals don't say" is a valid
# answer, not a refusal (evals.is_abstention covers that).
_REFUSAL = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,)?[^.]{0,40}\b(i can(no|')t|i cannot|i'?m (not able|unable)|"
    r"i am (not able|unable)|i won'?t) (help|assist|provide|answer|comply|do that)|\bas an ai\b",
    re.IGNORECASE,
)
# A page reference outside the [<journal>:p.<N>] form ("p.251", "pp. 3-5", "page 260"):
# unverifiable with three journals, and not what the cartridge should learn.
_BRACKETED = re.compile(r"\[[^\[\]]*\]")
_BARE_PAGE = re.compile(r"\b(?:pp?\.|pages?)\s*\d+", re.IGNORECASE)
# Quoted spans of an answer, straight or curly quotes, long enough to be a claim.
_QUOTED = re.compile(r'["\u201c]([^"\u201c\u201d]{20,})["\u201d]')
ITEM_ECHO_MIN_SCORE = 90  # rapidfuzz partial_ratio between a quoted span and the item
# An echoed item only counts as evidence when the answer attributes it to the
# journals: a "you wrote/state/note..." verb just before the quote, or a page
# citation right after it. Restating the item ('you would answer X to the
# statement "..."') is fine: all 6 hits in 27B pilot dfi8msjw were restatements.
_ATTRIBUTION_BEFORE = re.compile(
    r"\b(you (?:wrote|write|writes|said|say|state|stated|note|noted|explicitly|mention|mentioned|describe|"
    r"described|admit|admitted|confess|reflect)|journals? (?:say|says|said|state|show|note)|on p(?:age|\.))\b[^.\n]{0,40}$",
    re.IGNORECASE,
)
_CITATION_AFTER = re.compile(r"^[\s,.;:)\]*_]{0,6}\[<?[^\[\]<>]+?:p\.\d+>?\]")
# The answer says the journals don't settle it (psych abstention rate).
_ABSTAIN = re.compile(
    r"(journals?|entries|pages) (do not|don't|does not|doesn't) (say|mention|show|indicate|reveal)|"
    r"not enough (evidence|information)|no (clear |direct )?evidence|(can't|cannot) (tell|determine)",
    re.IGNORECASE,
)
# Repeated as a short system turn right before the task: INSTRUCTIONS sit at the
# top of a ~205k-token context, and 4B drifted to bare "p.N" refs (re-pilot
# zaerq80o: 59/60 unanchored answers had no bracketed citation at all).
CITATION_REMINDER = (
    "Cite each claim about the user's life with the journal named in the page heading and the page, "
    "e.g. [Morning Pages:p.63] or [Auto biography:p.12]; never a page number alone. "
    "If the journals don't show it, say so."
)


# Fixed namespace so label ids are reproducible: uuid5(namespace, run_id:task_id).
_LABEL_NAMESPACE = uuid.UUID("6f1d2c1e-5b8a-4c4e-9a43-1f4a2b7d9e10")


_FAILED = teacher.Completion(text="", prompt_ids=[], completion_ids=[], finish_reason="error")


class LabelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str  # W&B run id, also the gen_runs row and the shard directory
    corpus_id: str
    n: int
    seed: int
    teacher_model: str = teacher.MODEL_ID
    # "train" labels are distillation targets; "eval" labels (run under their own
    # run_id) are the validation set: trainer reads them for validation KL only,
    # and store.fetch_labels' split="train" filter keeps them out of training.
    split: Literal["train", "eval"] = "train"
    origins: frozenset[str] | None = None  # e.g. {"chat", "psych_test"}; None = any (human only with split="eval")
    kind_mix: Mapping[str, float] | None = None
    # A quota per task origin first, then kind_mix within each (TaskQuery.origin_mix).
    # Default: the user-approved label budget (2026-10-16). None: no origin quotas.
    origin_mix: Mapping[str, float] | None = LABEL_ORIGIN_MIX
    # 3072 (user, 2026-10-16): INSTRUCTIONS ask for 300-500 words, but timelines and
    # multi-period answers need room; at 1536, 49% of 4B answers were truncated.
    max_answer_tokens: int = 3072
    top_k: int = 20
    temperature: float = 0.0
    targets_root: str = targets.ROOT
    # Tasks per batch: each batch is sampled, answered, written (one shard + its
    # labels rows) before the next, so an interruption loses one batch at most
    # and a rerun resumes (unlabeled_by skips labeled tasks).
    batch_size: int = 256


class LabelStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_tasks: int
    n_validated: int
    reject_reasons: dict[str, int]
    truncation_rate: float  # finish_reason == "length", among all answers
    refusal_rate: float
    cite_rate: float  # answers with at least one citation
    citation_validity: float  # cited pages that exist, over all citations
    abstention_rate: float  # answers saying the journals don't settle it
    completion_tokens: int  # of validated labels
    tokens_by_kind: dict[str, int]  # completion tokens of validated labels, per kind


class LabelReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    sample_hashes: list[str]  # one per batch
    targets_uris: list[str]  # one shard per batch with validated labels
    shortfall: dict[str, int]  # of the last batch's sample
    overall: LabelStats
    by_origin: dict[str, LabelStats]


def instructions_hash(instructions: str = teacher.INSTRUCTIONS, reminder: str = CITATION_REMINDER) -> str:
    """Identifies a label set: the Teacher's INSTRUCTIONS plus the prompt-side
    citation reminder, so changing either starts a new set (unlabeled_by) and
    trainers can filter on it."""
    return hashlib.sha256(f"{instructions}\n{reminder}".encode()).hexdigest()[:12]


def label_id(run_id: str, task_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(_LABEL_NAMESPACE, f"{run_id}:{task_id}")


def answer_messages(text: str) -> list[dict]:
    """The conversation for answering a task: the citation reminder, then the
    task as the user turn. The one prompt format for teacher labels, student
    training, eval answers and cartridge-ask, so their tokens match."""
    return [{"role": "system", "content": CITATION_REMINDER}, {"role": "user", "content": text}]


def task_messages(task: Task) -> list[dict]:
    return answer_messages(task.text)


def to_prompt(task: Task, tokenizer) -> PromptIds:
    """The student's input for this task: the chat-templated user turn without
    the journal, exactly as the Teacher's prompt after its context block."""
    return tuple(teacher.render(tokenizer, task_messages(task))[1])


def _citations(pages: Collection) -> set[str]:
    return {p if isinstance(p, str) else p.citation for p in pages}


def validate_label(task: Task, completion: teacher.Completion, pages: Collection) -> tuple[str, str | None]:
    """("validated", None) or ("rejected", reason). `pages` are the corpus pages
    (citation strings, or types.Page). A truncated answer
    (finish_reason "length") stays validated: its targets are still useful, and
    the truncation rate is reported instead."""
    if not completion.text.strip() or not completion.completion_ids:
        return "rejected", "no_reply"
    T = len(completion.completion_ids)
    if len(completion.topk_ids) != T or len(completion.topk_logprobs) != T:
        return "rejected", "topk_mismatch"
    if _REFUSAL.search(completion.text):
        return "rejected", "refusal"
    if any(c not in _citations(pages) for c in parse_citations(completion.text)):
        return "rejected", "invalid_citation"
    if _BARE_PAGE.search(_BRACKETED.sub(" ", completion.text)):
        return "rejected", "unanchored_citation"
    if task.origin == "psych_test" and _quotes_item(task.text, completion.text):
        return "rejected", "item_as_evidence"
    return "validated", None


def _quotes_item(task_text: str, answer: str) -> bool:
    """A quoted span of the answer that is (nearly) the questionnaire item: the
    item's own wording presented as a journal quote. The item is the longest
    quoted span of the task text."""
    items = _QUOTED.findall(task_text)
    if not items:
        return False
    item = max(items, key=len).lower()
    for match in _QUOTED.finditer(answer):
        if fuzz.partial_ratio(match.group(1).lower(), item) < ITEM_ECHO_MIN_SCORE:
            continue
        before, after = answer[max(0, match.start() - 80):match.start()], answer[match.end():match.end() + 40]
        if _ATTRIBUTION_BEFORE.search(before) or _CITATION_AFTER.match(after):
            return True
    return False


def to_target_row(task: Task, completion: teacher.Completion, label_id: uuid.UUID | str) -> TargetRow:
    return TargetRow(
        label_id=str(label_id),
        task_id=str(task.task_id),
        prompt_ids=list(completion.prompt_ids),
        completion_ids=list(completion.completion_ids),
        topk_ids=[list(r) for r in completion.topk_ids],
        topk_logprobs=[list(r) for r in completion.topk_logprobs],
    )


def to_label(
    task: Task,
    completion: teacher.Completion,
    config: LabelConfig,
    status: str,
    reject_reason: str | None,
    targets_uri: str | None = None,
    targets_row: int | None = None,
) -> Label:
    return Label(
        label_id=label_id(config.run_id, task.task_id),
        task_id=task.task_id,
        run_id=config.run_id,
        teacher_model=config.teacher_model,
        corpus_id=config.corpus_id,
        instructions_hash=instructions_hash(),
        answer=completion.text,
        finish_reason=completion.finish_reason or "",
        n_prompt_tokens=len(completion.prompt_ids),
        n_completion_tokens=len(completion.completion_ids),
        citations=tuple(parse_citations(completion.text)),
        status=status,
        reject_reason=reject_reason,
        targets_uri=targets_uri,
        targets_row=targets_row,
    )


def label(
    config: LabelConfig,
    pages: Collection,
    *,
    sample: Callable[[TaskQuery], Sample],
    generate: Callable[..., Sequence[teacher.Completion | None]],
    write_shard: Callable[[Sequence[TargetRow], str], str] = targets.write_shard,
    insert_labels: Callable[[Sequence[Label]], object],
    log: Callable[[dict], object] | None = None,
) -> LabelReport:
    """Label up to config.n unlabeled train tasks, in batches of
    config.batch_size. Each batch: sample (unlabeled_by excludes everything
    already labeled, including earlier batches), answer greedily with the
    journal in context, write the validated answers' targets to one shard, and
    insert a labels row for every task (rejected ones without a shard pointer).
    Stops at n tasks or when the sampler has nothing left. `log` (e.g. a W&B
    run's log) gets the cumulative stats after every batch, overall and per
    origin. Wire `generate` through teacher.generate_all, so an interrupt
    cancels the batch's in-flight calls."""
    if config.split != "eval" and config.origins is not None and "human" in config.origins:
        raise ValueError("human tasks are labeled only as eval references (split='eval'), never for training")
    tasks: list[Task] = []
    completions: list[teacher.Completion] = []
    labels: list[Label] = []
    hashes: list[str] = []
    uris: list[str] = []
    shortfall: dict[str, int] = {}
    while len(tasks) < config.n:
        query = TaskQuery(
            split=config.split, n=min(config.batch_size, config.n - len(tasks)), seed=config.seed + len(hashes),
            mix=config.kind_mix, origin_mix=config.origin_mix, origins=config.origins,
            # The labels' unique key: new INSTRUCTIONS or a new corpus relabel a task.
            unlabeled_by=(config.teacher_model, config.corpus_id, instructions_hash()),
            exclude=frozenset(t.task_id for t in tasks),
        )
        picked = sample(query)
        # Human tasks only as eval references (CGC metric, judge), never as training targets.
        batch = [t for t in picked.tasks if t.origin != "human" or config.split == "eval"]
        shortfall = {str(k): v for k, v in picked.shortfall.items()}
        if not batch:
            break
        done = generate(
            [task_messages(t) for t in batch],
            max_tokens=config.max_answer_tokens,
            logprobs=config.top_k,
            temperature=config.temperature,
        )
        if len(done) != len(batch):
            raise ValueError(f"generate returned {len(done)} completions for {len(batch)} tasks")
        # A failed call (None) becomes an empty answer: rejected as no_reply, finish_reason "error".
        done = [c if c is not None else _FAILED for c in done]
        written, batch_labels = _write(config, pages, batch, done, picked.sample_hash[:12], write_shard, insert_labels)
        if written:
            uris.append(written)
        tasks += batch
        completions += done
        labels += batch_labels
        hashes.append(picked.sample_hash)
        report = _report(config, hashes, uris, shortfall, tasks, completions, labels, pages)
        if log is not None:
            log(flatten_stats(report))
    return _report(config, hashes, uris, shortfall, tasks, completions, labels, pages)


def _write(config, pages, batch, done, part, write_shard, insert_labels) -> tuple[str | None, list[Label]]:
    """Validate a group of answers, write their targets as one shard named by
    `part`, and insert one labels row each. Returns (shard uri or None, labels)."""
    uri = targets.shard_uri(config.run_id, part, config.targets_root)
    rows: list[TargetRow] = []
    verdicts, row_of = [], []
    for task, completion in zip(batch, done):
        status, reason = validate_label(task, completion, pages)
        verdicts.append((status, reason))
        row_of.append(None)
        # Every answer with usable tokens gets its targets row, rejected ones too:
        # read_labeled only reads validated labels, and revalidate() can later
        # promote a label whose rejection was a guard's false positive.
        if reason not in ("no_reply", "topk_mismatch"):
            row_of[-1] = len(rows)
            rows.append(to_target_row(task, completion, label_id(config.run_id, task.task_id)))
    # Labels store the uri the writer returns (Volume-relative for volume_writer).
    written = write_shard(rows, uri) if rows else None
    labels = [
        to_label(task, completion, config, status, reason, written if row is not None else None, row)
        for task, completion, (status, reason), row in zip(batch, done, verdicts, row_of)
    ]
    insert_labels(labels)
    return written, labels


def label_stream(
    config: LabelConfig,
    pages: Collection,
    *,
    sample: Callable[[TaskQuery], Sample],
    spawn: Callable[..., object],
    write_shard: Callable[[Sequence[TargetRow], str], str] = targets.write_shard,
    insert_labels: Callable[[Sequence[Label]], object],
    window: int = 96,
    flush_every: int = 128,
    flush_s: float = 60.0,
    log: Callable[[dict], object] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    poll_s: float = 0.2,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> LabelReport:
    """label() without the per-batch barrier: keeps `window` teacher calls in
    flight, refilling a slot as soon as an answer lands, while sampling (queries
    of 2 x window, started while 2 x window tasks are still queued) and writing (a shard + labels rows every
    `flush_every` answers or `flush_s` seconds, in order) run in background
    threads. `spawn(messages, **decoding)` starts one call and returns a handle
    with .get(timeout) -> [Completion] (raising TimeoutError while running) and
    .cancel(), e.g. Teacher.generate.spawn. Each in-flight slot runs in its own
    worker thread (spawn, then a blocking get), since every Modal spawn/get is
    a ~150 ms round trip: serving slots one at a time would starve the window.

    Guarantees, as in label(): an interruption loses only the answers not yet
    flushed (a rerun resumes via unlabeled_by) and cancels every in-flight call;
    `should_stop()` (e.g. the $ guard) stops new calls and lets the window
    drain; `log` gets the cumulative per-origin stats after every flush."""
    import threading
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if config.split != "eval" and config.origins is not None and "human" in config.origins:
        raise ValueError("human tasks are labeled only as eval references (split='eval'), never for training")
    decoding = {"max_tokens": config.max_answer_tokens, "logprobs": config.top_k, "temperature": config.temperature}
    seen: set = set()  # every task sampled so far: excluded from later queries
    pending: list[Task] = []  # sampled, not yet submitted
    inflight: dict = {}  # future -> task, one worker thread per slot
    live_calls: dict = {}  # task_id -> spawned call, for cancellation
    cancelled = threading.Event()
    buffer: list[tuple[Task, teacher.Completion]] = []
    tasks: list[Task] = []
    completions: list[teacher.Completion] = []
    labels: list[Label] = []
    hashes: list[str] = []
    uris: list[str] = []
    shortfall: dict[str, int] = {}
    state = {"exhausted": False, "sampled": 0, "last_flush": clock()}

    def query() -> TaskQuery:
        return TaskQuery(
            split=config.split, n=min(2 * window, config.n - state["sampled"]), seed=config.seed + len(hashes),
            mix=config.kind_mix, origin_mix=config.origin_mix, origins=config.origins,
            unlabeled_by=(config.teacher_model, config.corpus_id, instructions_hash()),
            exclude=frozenset(seen),
        )

    def take(picked: Sample) -> None:
        nonlocal shortfall
        new = [t for t in picked.tasks if (t.origin != "human" or config.split == "eval") and t.task_id not in seen]
        shortfall = {str(k): v for k, v in picked.shortfall.items()}
        hashes.append(picked.sample_hash)
        seen.update(t.task_id for t in new)
        state["sampled"] += len(new)
        pending.extend(new)
        if not new or state["sampled"] >= config.n:
            state["exhausted"] = True

    def flush_job(group: list[tuple[Task, teacher.Completion]]) -> None:
        batch, done = [t for t, _ in group], [c for _, c in group]
        part = hashlib.sha256("".join(sorted(str(t.task_id) for t in batch)).encode()).hexdigest()[:12]
        written, batch_labels = _write(config, pages, batch, done, part, write_shard, insert_labels)
        if written:
            uris.append(written)
        tasks.extend(batch)
        completions.extend(done)
        labels.extend(batch_labels)
        if log is not None:
            log(flatten_stats(_report(config, hashes, uris, shortfall, tasks, completions, labels, pages)))

    def run_one(task: Task):
        """One slot: spawn the call, then block in get. A Modal spawn/get is a
        ~150 ms round trip, so slots must not be served one at a time."""
        call = spawn(task_messages(task), **decoding)
        live_calls[task.task_id] = call
        try:
            while True:
                if cancelled.is_set():
                    return None
                try:
                    return call.get(timeout=30)
                except TimeoutError:
                    continue
        finally:
            live_calls.pop(task.task_id, None)

    sampler, writer, slots = ThreadPoolExecutor(1), ThreadPoolExecutor(1), ThreadPoolExecutor(window)
    prefetch = sampler.submit(sample, query())
    flushes = []
    try:
        while True:
            # Take a finished sample; or wait for it when a slot is free and nothing is queued.
            if prefetch is not None and (prefetch.done() or (not pending and len(inflight) < window)):
                take(prefetch.result())
                prefetch = None
            stopping = should_stop()
            # Keep a full window of tasks queued beyond the one in flight, so a refill never waits on the sampler.
            if prefetch is None and not state["exhausted"] and not stopping and len(pending) < 2 * window:
                prefetch = sampler.submit(sample, query())
            while pending and len(inflight) < window and not stopping:
                task = pending.pop(0)
                inflight[slots.submit(run_one, task)] = task
            landed = False
            if inflight:
                done, _ = wait(list(inflight), timeout=poll_s, return_when=FIRST_COMPLETED)
                for future in done:
                    task = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception:  # a failed call: rejected as no_reply, finish_reason "error"
                        logger.exception("label_stream: call for task %s failed", task.task_id)
                        result = None
                    buffer.append((task, (result[0] if result else None) or _FAILED))
                    landed = True
            # Done when the window has drained and either the guard stopped us (pending
            # tasks are simply not labeled) or there is nothing left to sample.
            idle = not inflight and (stopping or (not pending and prefetch is None and state["exhausted"]))
            if buffer and (len(buffer) >= flush_every or clock() - state["last_flush"] >= flush_s or idle):
                flushes.append(writer.submit(flush_job, buffer[:]))
                buffer.clear()
                state["last_flush"] = clock()
            if idle:
                break
            if not landed:
                sleep(0 if inflight else poll_s)  # wait() already blocked up to poll_s
        for f in flushes:
            f.result()  # surface write errors
    except BaseException:
        cancelled.set()
        n_cancelled = 0
        for call in list(live_calls.values()):
            try:
                call.cancel()
                n_cancelled += 1
            except Exception:
                logger.exception("label_stream: could not cancel a call")
        if buffer:  # keep what already landed
            flushes.append(writer.submit(flush_job, buffer[:]))
        logger.warning("label_stream: interrupted; cancelled %d in-flight calls", n_cancelled)
        raise
    finally:
        cancelled.set()
        slots.shutdown(wait=False, cancel_futures=True)
        sampler.shutdown(wait=False, cancel_futures=True)
        writer.shutdown(wait=True)
    return _report(config, hashes, uris, shortfall, tasks, completions, labels, pages)


# Short answers need less KV per request: psych answers average ~475 tokens on 27B.
SHORT_ANSWER_ORIGINS = frozenset({"psych_test"})


def window_for(origins: Collection[str] | None) -> int:
    """Calls in flight (= Teacher max_inputs) for a label session: 128 when
    every origin has short answers (KV ~46 of 75 GiB on a 27B H200), else 96.
    One session uses one value for its whole handle, so the pool never splits."""
    return 128 if origins and set(origins) <= SHORT_ANSWER_ORIGINS else 96


def project_cost(pilot: LabelStats, pilot_wall_s: float, n_total: int, usd_per_hour: float,
                 overhead_s: float = 0.0) -> dict:
    """Cost of labeling n_total tasks at the pilot's measured pace: wall seconds
    per task (the pilot's tasks were answered concurrently, so this already
    includes batching) times n_total, plus fixed overhead (cold start, idle
    tail), at the GPU's hourly rate."""
    per_task_s = pilot_wall_s / max(pilot.n_tasks, 1)
    wall_s = per_task_s * n_total + overhead_s
    return {
        "per_task_s": per_task_s,
        "tokens_per_task": pilot.completion_tokens / max(pilot.n_validated, 1),
        "projected_wall_s": wall_s,
        "projected_usd": wall_s / 3600 * usd_per_hour,
    }


class Revalidation(BaseModel):
    model_config = ConfigDict(frozen=True)

    updated: list[Label]  # status/reject_reason changed; safe to write back (targets present or not needed)
    needs_relabel: list[Label]  # now valid, but labeled before targets were kept for rejects: relabel the task


def revalidate(pairs: Sequence[tuple[Task, Label]], pages: Collection) -> Revalidation:
    """Re-run validate_label on stored answers (after a guard changes), without
    regenerating. Targets are never touched. A label that becomes valid but
    has no targets row (labeled before rejected answers kept their targets)
    can't be promoted: it is returned in needs_relabel instead. The token
    checks (no_reply, topk_mismatch) are kept as stored."""
    updated, needs_relabel = [], []
    for task, lab in pairs:
        if lab.reject_reason in ("no_reply", "topk_mismatch"):
            continue
        n = max(lab.n_completion_tokens, 1)
        stored = teacher.Completion(text=lab.answer, prompt_ids=[], completion_ids=[0] * n,
                                    topk_ids=[[0]] * n, topk_logprobs=[[0.0]] * n, finish_reason=lab.finish_reason)
        status, reason = validate_label(task, stored, pages)
        if (status, reason) == (lab.status, lab.reject_reason):
            continue
        if status == "validated" and lab.targets_uri is None:
            needs_relabel.append(lab)
        else:
            updated.append(lab.model_copy(update={"status": status, "reject_reason": reason}))
    return Revalidation(updated=updated, needs_relabel=needs_relabel)


def label_stats(tasks, completions, labels, pages) -> LabelStats:
    n = len(labels)
    valid_pages = _citations(pages)
    cited = [parse_citations(c.text) for c in completions]
    all_cites = [c for cs in cited for c in cs]
    tokens: Counter = Counter()
    for task, lab in zip(tasks, labels):
        if lab.status == "validated":
            tokens[str(task.kind)] += lab.n_completion_tokens
    return LabelStats(
        n_tasks=n,
        n_validated=sum(lab.status == "validated" for lab in labels),
        reject_reasons=dict(Counter(lab.reject_reason for lab in labels if lab.reject_reason)),
        truncation_rate=sum(c.finish_reason == "length" for c in completions) / n if n else 0.0,
        refusal_rate=sum(lab.reject_reason == "refusal" for lab in labels) / n if n else 0.0,
        cite_rate=sum(bool(cs) for cs in cited) / n if n else 0.0,
        citation_validity=sum(c in valid_pages for c in all_cites) / len(all_cites) if all_cites else 0.0,
        abstention_rate=sum(bool(_ABSTAIN.search(c.text)) for c in completions) / n if n else 0.0,
        completion_tokens=sum(tokens.values()),
        tokens_by_kind=dict(tokens),
    )


def _report(config, hashes, uris, shortfall, tasks, completions, labels, pages) -> LabelReport:
    by_origin: dict[str, list[int]] = {}
    for i, task in enumerate(tasks):
        by_origin.setdefault(str(task.origin), []).append(i)
    return LabelReport(
        run_id=config.run_id,
        sample_hashes=list(hashes),
        targets_uris=list(uris),
        shortfall=shortfall,
        overall=label_stats(tasks, completions, labels, pages),
        by_origin={
            origin: label_stats([tasks[i] for i in idx], [completions[i] for i in idx], [labels[i] for i in idx], pages)
            for origin, idx in by_origin.items()
        },
    )


def flatten_stats(report: LabelReport) -> dict:
    """label/<origin|all>/<stat> scalars for W&B (tokens as label/<origin>/tokens/<kind>)."""
    out = {}
    for origin, stats in [("all", report.overall), *report.by_origin.items()]:
        for name in ("n_tasks", "n_validated", "truncation_rate", "refusal_rate", "cite_rate",
                     "citation_validity", "abstention_rate", "completion_tokens"):
            out[f"label/{origin}/{name}"] = getattr(stats, name)
        for kind, tokens in stats.tokens_by_kind.items():
            out[f"label/{origin}/tokens/{kind}"] = tokens
        for reason, count in stats.reject_reasons.items():
            out[f"label/{origin}/rejected/{reason}"] = count
    return out
