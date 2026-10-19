"""ChatGPT history import: the first user message of each conversation,
classified by a small model into a category, an optional task kind and a
journal-relevance flag, for cartridge.chat_prompts.

load (IO) -> extract, validate, classifier_messages, parse_label (pure) ->
classify (injected generate) -> store.insert_chat_labels (IO), a cache keyed
by (text_hash, classifier model, prompt_hash()). A conversation
is its thread: parent links from current_node back to the root, reversed. A
message's text is its string parts plus voice transcriptions; images and other
attachments are skipped, so an attachment-only first prompt is dropped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cartridges.cartridge.chunks import _FENCE, _THINK
from cartridges.cartridge.tasks.types import CATEGORIES, KINDS, ChatLabel, ChatPrompt, Kind, Messages, Task
from cartridges.cartridge.tasks.validate import MAX_WORDS, task_id, text_hash

PROMPT_VERSION = "chats-v2"  # a readable label; the cache key is prompt_hash()
MAX_CHARS = 2000  # of a prompt sent to the classifier; the DB keeps the full text
MAX_TOKENS = 96
TEMPERATURE = 0.0
USER_TEMPLATE = "Message:\n{text}"
BATCH_SIZE = 256  # prompts per generate call; each batch's labels are cached as it returns

CLASSIFIER_SYSTEM = """You classify the first message a user sent to an AI assistant. Reply with JSON only:
{"category": "...", "task_kind": "..." or null, "relevant": true or false, "reason": "..."}

category, one of:
- self_reflection: the user's own patterns, identity, beliefs, motives
- relationships: partners, dating, family, friends
- emotions_situation: how to feel about or handle something happening now
- personal_growth: habits, self-improvement, courses, spiritual practice
- health_body: health, fitness, food, sleep, substances
- career_work: jobs, business, money decisions about work
- creative: writing, art, music, stories, names
- learning_ideas: books, concepts, explanations of ideas
- technical: code, software, data, tools
- factual_lookup: facts, news, web search, definitions
- logistics: booking, travel, shopping, admin, forms
- other: anything else

task_kind: the kind of request, if a personal assistant that knows the user's journal could take it on, else null:
- situation_help: a situation the user is in now, asking for help with it
- self_synthesis: patterns, strengths, beliefs or self-sabotage across their life
- framework_lens: a named framework (enneagram, attachment theory, ...) applied to them
- change_over_time: how they have changed over the years
- role_request: asks the assistant to act as a therapist, coach or similar
- build_artifact: build something personal (values, belief system, growth plan)
- recall: what they did, said or felt at a time or place

relevant: true if a good answer would benefit from knowing the user's life, history, relationships or journal; false \
for generic, technical or factual requests.
reason: at most 15 words."""


# ---------------------------------------------------------------- load and extract


def load_conversations(directory: Path) -> list[dict]:
    """Every conversation in <directory>/conversations-*.json."""
    out: list[dict] = []
    for path in sorted(Path(directory).glob("conversations-*.json")):
        out += json.loads(path.read_text())
    return out


def thread(conversation: dict) -> list[dict]:
    """The messages from the root to current_node."""
    mapping, node, out = conversation["mapping"], conversation.get("current_node"), []
    while node:
        entry = mapping[node]
        if entry.get("message"):
            out.append(entry["message"])
        node = entry.get("parent")
    return out[::-1]


def message_text(message: dict) -> str:
    """String parts and voice transcriptions, joined; attachments are skipped."""
    texts = []
    for part in (message.get("content") or {}).get("parts") or []:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("content_type") == "audio_transcription":
            texts.append(part.get("text") or "")
    return "\n".join(t for t in texts if t.strip()).strip()


def first_prompt(conversation: dict) -> ChatPrompt | None:
    """The thread's first user message; None if there is none or it has no
    text (attachments only)."""
    message = next((m for m in thread(conversation) if m["author"]["role"] == "user"), None)
    text = message_text(message) if message else ""
    if not text:
        return None
    created = conversation.get("create_time")
    return ChatPrompt(
        conversation_id=conversation["conversation_id"],
        create_time=dt.datetime.fromtimestamp(created, dt.UTC) if created else None,
        title=conversation.get("title"),
        model_slug=conversation.get("default_model_slug"),
        text=text,
        text_hash=text_hash(text),
    )


def extract(conversations: Sequence[dict]) -> tuple[list[ChatPrompt], int]:
    """(first prompts, conversations without a user message with text)."""
    prompts = [first_prompt(c) for c in conversations]
    return [p for p in prompts if p], sum(p is None for p in prompts)


def validate(prompts: Sequence[ChatPrompt]) -> tuple[list[ChatPrompt], int]:
    """Prompts deduplicated by text_hash, the earliest kept; (kept, duplicates)."""
    first: dict[str, ChatPrompt] = {}
    for p in sorted(prompts, key=lambda p: (p.create_time or dt.datetime.max.replace(tzinfo=dt.UTC), p.conversation_id)):
        first.setdefault(p.text_hash, p)
    return list(first.values()), len(prompts) - len(first)


# ---------------------------------------------------------------- classify


def classifier_messages(prompt: ChatPrompt, max_chars: int = MAX_CHARS) -> Messages:
    """A fixed system turn (prefix-cached) and the prompt, truncated."""
    text = prompt.text if len(prompt.text) <= max_chars else prompt.text[:max_chars] + " [...]"
    return [{"role": "system", "content": CLASSIFIER_SYSTEM}, {"role": "user", "content": USER_TEMPLATE.format(text=text)}]


class Verdict(BaseModel):
    """The classifier's reply; its JSON schema constrains decoding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: Literal[tuple(c for c in CATEGORIES if c != "unparsed")]
    task_kind: Kind | None
    relevant: bool
    reason: str = Field(max_length=120)


def parse_label(reply: str | None) -> ChatLabel:
    """The reply as a ChatLabel; category "unparsed" unless it is a valid Verdict."""
    raw = reply or ""
    try:
        verdict = Verdict.model_validate_json(_FENCE.sub("", _THINK.sub("", raw).strip()))
    except ValidationError:
        return ChatLabel(category="unparsed", raw_reply=raw)
    return ChatLabel(**verdict.model_dump(), raw_reply=raw)


def prompt_hash() -> str:
    """sha256 of everything that decides a label besides the text and model:
    any edit to the prompt, schema or decoding is a cache miss."""
    key = json.dumps({"system": CLASSIFIER_SYSTEM, "user": USER_TEMPLATE, "schema": Verdict.model_json_schema(), "max_chars": MAX_CHARS,
                      "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def classify(
    prompts: Sequence[ChatPrompt],
    generate: Callable[..., list],
    write: Callable[[list[tuple[ChatPrompt, ChatLabel]]], object],
    batch_size: int = BATCH_SIZE,
) -> list[ChatLabel]:
    """Label the prompts batch by batch (greedy, no journal, decoding
    constrained to the Verdict schema), passing each batch to `write` as it
    returns. A failed call is unparsed."""
    labels: list[ChatLabel] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        replies = generate([classifier_messages(p) for p in batch], max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                           with_context=False, json_schema=Verdict.model_json_schema())
        new = [parse_label(r.text if r is not None else None) for r in replies]
        write(list(zip(batch, new, strict=True)))
        labels += new
    return labels


def label_stats(labels: Sequence[ChatLabel]) -> dict:
    """Counts per category, relevance and parse-failure rates."""
    n = len(labels) or 1
    return {
        "categories": dict(Counter(lab.category for lab in labels).most_common()),
        "task_kinds": dict(Counter(lab.task_kind for lab in labels if lab.task_kind).most_common()),
        "relevant": sum(lab.relevant is True for lab in labels),
        "relevant_rate": sum(lab.relevant is True for lab in labels) / n,
        "unparsed_rate": sum(lab.category == "unparsed" for lab in labels) / n,
    }


# ---------------------------------------------------------------- promote

SPLIT_RATIOS = {"train": 0.9, "eval": 0.1, "exemplar": 0.0}


def promote(
    labeled: Sequence[tuple[ChatPrompt, ChatLabel, str | None]],
    existing_hashes: Collection[str],
    ratios: Mapping[str, float] = SPLIT_RATIOS,
    seed: int = 0,
) -> tuple[list[Task], int]:
    """Relevant chat prompts with a task kind as origin "chat" tasks:
    (new tasks, duplicates skipped). A text already in the store is skipped, so
    re-running adds nothing. Each kind is split train/eval/exemplar by `ratios`
    (largest remainder), uniformly within the kind for the seed. Only the
    checks that apply to a real question are run: empty and too_long."""
    fresh: dict[str, tuple[ChatPrompt, ChatLabel, str | None]] = {}
    for prompt, label, run_id in labeled:
        if label.relevant and label.task_kind and prompt.text_hash not in existing_hashes:
            fresh.setdefault(prompt.text_hash, (prompt, label, run_id))
    dups = sum(1 for p, lab, _ in labeled if lab.relevant and lab.task_kind) - len(fresh)
    rng = random.Random(seed)
    tasks = []
    for kind in KINDS:
        of_kind = sorted((v for v in fresh.values() if v[1].task_kind == kind), key=lambda v: v[0].text_hash)
        rng.shuffle(of_kind)
        splits = [s for s, k in _largest_remainder(len(of_kind), ratios).items() for _ in range(k)]
        for (prompt, label, run_id), split in zip(of_kind, splits, strict=True):
            text = prompt.text.strip()
            reason = "empty" if not text else "too_long" if len(text.split()) > MAX_WORDS else None
            tasks.append(Task(
                task_id=task_id(prompt.text_hash), text=text, text_hash=prompt.text_hash, kind=kind, origin="chat",
                split=split, status="rejected" if reason else "validated", reject_reason=reason, run_id=run_id,
                metadata={"conversation_id": prompt.conversation_id, "create_time": str(prompt.create_time),
                          "title": prompt.title, "category": label.category, "reason": label.reason},
            ))
    return tasks, dups


def _largest_remainder(n: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {k: n * w / sum(ratios.values()) for k, w in ratios.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - out[k]))[: n - sum(out.values())]:
        out[k] += 1
    return out
