"""Self-study seed prompts (Cartridges, arXiv 2506.06266, section 4.1).

A self-study conversation starts from a chunk of the corpus and a seed prompt:
the model, with the chunk in its system prompt, answers the seed prompt with a
first user message (a question or request about the chunk); the conversation
then continues and is distilled into the cartridge.

The system prompt and the seed families (summarization, question, use_case,
creative) are adapted from HazyResearch/cartridges:
cartridges/synthesizers/self_study.py (SYSTEM_PROMPT_TEMPLATE) and
cartridges/data/resources.py (*_seed_prompt, SEED_PROMPT_REGISTRY). Changes:
the paper's structuring family is dropped; question and summarization ask, in
the author's own voice, what I did, said or who I was with at a place and time
(not quiz questions about pages); every seed asks for JSON with the question,
the page citations its answer rests on and a verbatim quote from each, which
`parse_seed_reply` checks against the chunk (`grounded`), so each conversation
carries a reference for grounding checks; and a variant is drawn from a seeded
rng instead of the global one.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from cartridges.cartridge.chunks import _FENCE, _THINK, Chunk
from cartridges.datagen.qa import EVIDENCE_MIN_SCORE

KINDS = ("summarization", "question", "use_case", "creative")

SYSTEM_PROMPT = """You are in a conversation about the following user information: \
pages of a handwritten journal, transcribed by OCR. Each page starts with a heading \
"## [<journal>:p.<N>] date: <date>".

<info>
{chunk}
</info>"""

_FIRST_PERSON = (
    "Write it as the journal's author asking about their own life, in the first person (I, me, my). "
    "Say when and where: a month and year, a season, a named event or trip, and a place, precise enough to "
    "pick out this moment among years of entries. Never mention pages, entries, \"the journal\" or \"the author\". "
    "Do not ask about a date marked \"(?)\" or \"(est.)\"; name the event or place instead."
)

_OUTPUT = (
    'Reply with JSON only: {"question": "...", "citations": ["<journal>:p.<N>", ...], '
    '"evidence": ["...", ...]}. citations: the page anchors, exactly as in the page headings, that the answer '
    "comes from. evidence: one short verbatim quote (a sentence or line) from each cited page that supports the answer."
)


def _summarization(rng: random.Random) -> str:
    # Placeholder templates, never real events: a concrete example would be copied
    # into questions about whatever it names, including held-out eval pages.
    example = rng.choice([
        "What did I do while I was in {place}?",
        "How did I spend the week after {event} in {month year}?",
        "What happened at {event} in {month year}, from start to finish?",
    ])
    return (
        "Please generate a single message asking an LLM to recount what I did, saw and felt over a stretch of time "
        "or at a place covered by the pages above, such as a trip, an event, a week or a season. "
        f"{_FIRST_PERSON} Follow a pattern like '{example}', filling the braces from the pages above. {_OUTPUT}"
    )


def _question(rng: random.Random) -> str:
    example = rng.choice([
        "What was the name of the {person} I was talking to at {event} in {month year}?",
        "What did I tell {person} about {topic} when we were at {place} in {month year}?",
        "Who was I with at {place} in {month year}, and what did we talk about?",
    ])
    return (
        "Please generate a single question about something specific I did or said, or someone I was with, at a "
        f"particular place and time in the pages above. {_FIRST_PERSON} Follow a pattern like '{example}', "
        f"filling the braces from the pages above. {_OUTPUT}"
    )


def _use_case(rng: random.Random) -> str:
    return (
        "You are working to train a language model on the information in the following corpus. "
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve "
        "using the knowledge contained within this corpus. "
        "Consider how a user might want to apply this information, not just recall it. "
        "After considering potential use cases, your task will be to generate a sample question that reflects one of "
        "these downstream applications. "
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when "
        f"trying to accomplish their specific goal. {_OUTPUT}"
    )


def _creative(rng: random.Random) -> str:
    return (
        "You are having a creative conversation inspired by the information in the corpus. "
        f"Please generate a question for your conversation partner to start off the discussion. {_OUTPUT}"
    )


_SEEDS = {
    "summarization": _summarization,
    "question": _question,
    "use_case": _use_case,
    "creative": _creative,
}


@dataclass(frozen=True)
class SeedPrompt:
    kind: str
    chunk_id: str
    messages: list[dict]  # system (the chunk) + user (the seed); the reply is the conversation's first user turn


def seed_prompts(chunk: Chunk, kinds: tuple[str, ...] = KINDS, seed: int = 0) -> list[SeedPrompt]:
    """One seed prompt per kind for the chunk; the variant within a kind is
    deterministic for (seed, chunk.id)."""
    rng = random.Random(f"{seed}:{chunk.id}")
    system = {"role": "system", "content": SYSTEM_PROMPT.format(chunk=chunk.text)}
    return [
        SeedPrompt(kind, chunk.id, [system, {"role": "user", "content": _SEEDS[kind](rng)}])
        for kind in kinds
    ]


def parse_seed_reply(chunk: Chunk, text: str) -> dict | None:
    """Bot A's JSON reply to a seed prompt as {question, citations, evidence,
    grounded}, or None without a question. grounded: every citation is a page of
    the chunk and every evidence quote is found (fuzzily) on a cited page."""
    try:
        reply = json.loads(_FENCE.sub("", _THINK.sub("", text).strip()))
    except json.JSONDecodeError:
        return None
    question = reply.get("question") if isinstance(reply, dict) else None
    if not isinstance(question, str) or not question.strip():
        return None
    cited, evidence = reply.get("citations"), reply.get("evidence")
    cited = [c for c in cited if isinstance(c, str)] if isinstance(cited, list) else []
    evidence = [e for e in evidence if isinstance(e, str) and e.strip()] if isinstance(evidence, list) else []
    pages = {c: _normalize(t) for c, (_, _, t) in zip(chunk.citations, chunk.pages)}
    grounded = bool(cited and evidence) and all(c in pages for c in cited) and all(
        any(fuzz.partial_ratio(_normalize(e), pages[c]) >= EVIDENCE_MIN_SCORE for c in cited) for e in evidence
    )
    return {"question": question.strip(), "citations": cited, "evidence": evidence, "grounded": grounded}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
