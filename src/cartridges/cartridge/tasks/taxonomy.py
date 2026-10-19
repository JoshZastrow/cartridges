"""The task taxonomy (docs/plans/task-db.md, section 4). Data only."""

from __future__ import annotations

from cartridges.cartridge.tasks.types import Kind, Origin, Style

PROMPT_VERSION = "synth-v3"  # v3: JSON-schema constrained drafts with list/quote limits, max_tokens 512

KIND_MIX: dict[Kind, float] = {
    "situation_help": 0.30,
    "self_synthesis": 0.15,
    "framework_lens": 0.15,
    "change_over_time": 0.10,
    "role_request": 0.10,
    "build_artifact": 0.10,
    "recall": 0.10,
}

STYLE_MIX: dict[Style, float] = {"dictated": 0.4, "typed": 0.4, "narrative": 0.2}

# Teacher label budget by origin (confirmed by the user); human tasks are never labeled.
LABEL_ORIGIN_MIX: dict[Origin, float] = {"chat": 0.40, "synthetic": 0.35, "psych_test": 0.25}

# (min, max) focus chunks per task; a chunk is up to CHUNK_PAGES consecutive pages.
CHUNKS_PER_KIND: dict[Kind, tuple[int, int]] = {
    "situation_help": (1, 2),
    "self_synthesis": (4, 6),
    "framework_lens": (3, 4),
    "change_over_time": (2, 2),
    "role_request": (3, 4),
    "build_artifact": (3, 4),
    "recall": (1, 1),
}
CHUNK_PAGES = 3
MIN_YEARS_APART = 2  # change_over_time: its two chunks are at least this far apart

KIND_INSTRUCTIONS: dict[Kind, str] = {
    "situation_help": (
        "Describe a new, in-the-moment situation I am in right now, one that echoes the people or themes in the "
        "focus pages, and ask for help understanding it or changing how I respond."
    ),
    "self_synthesis": (
        "Ask about a pattern across my life: my strengths, core beliefs, recurring behaviors or ways I sabotage "
        "myself, as the focus pages show them across the years."
    ),
    "framework_lens": "Ask to apply {framework} to me, using what my life shows.",
    "change_over_time": (
        "Ask how I have changed between the times of the focus pages (they are years apart) and what has stayed "
        "the same."
    ),
    "role_request": "Ask the assistant to act as {role} and help me with something the focus pages show I struggle with.",
    "build_artifact": (
        "Ask the assistant to build something personal and concrete from my life: a hierarchy of values, a belief "
        "system, a growth plan, a set of practices or similar."
    ),
    "recall": (
        "Ask what I did or said, or who I was with, at a specific place and time in the focus pages. Name the "
        "month and year, event or place, never a page."
    ),
}

STYLE_INSTRUCTIONS: dict[Style, str] = {
    "dictated": (
        "Write it as speech-to-text dictation: run-on, lower-case or loosely punctuated, with fillers "
        "(uh, okay, so, like) and self-corrections."
    ),
    "typed": "Write it as a short typed message: one or two sentences, casual.",
    "narrative": "Write it as a longer message: a few sentences of backstory about what happened, then the ask.",
}

FRAMEWORKS: tuple[str, ...] = (
    "the Enneagram",
    "attachment theory (John Bowlby)",
    "the Gottman method",
    "Esther Perel's work on desire and intimacy",
    "the Michelangelo phenomenon",
    "start / stop / keep",
    "Becoming the One (Sheleana Aiyana)",
    "Internal Family Systems",
    "CBT cognitive distortions",
    "schema therapy",
    "acceptance and commitment therapy (ACT)",
    "Jungian shadow work",
    "Stoicism",
    "nonviolent communication",
    "the Big Five personality traits",
    "self-determination theory",
    "polyvagal theory",
    "Brené Brown on shame and vulnerability",
)

ROLES: tuple[str, ...] = (
    "a therapist",
    "a psychiatrist",
    "a couples therapist",
    "a life coach",
    "a dating and relationship coach",
    "a mentor who has known me for years",
    "a career coach",
    "a meditation teacher",
)
