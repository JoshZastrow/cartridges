import datetime as dt

import pytest

from cartridges.cartridge.tasks.types import Draft, Page, Recipe
from cartridges.cartridge.tasks.validate import jaccard, normalize_text, text_hash, validate_draft, word_set


def page(n, text, heldout=False, uncertain=False):
    return Page(citation=f"A:p.{n}", journal="A", page_no=n, date=dt.date(2022, 5, n), date_uncertain=uncertain,
                heldout=heldout, text=text)


PAGES = {
    p.citation: p
    for p in [
        page(1, "We hiked to the lake with Sam and talked about moving to Denver."),
        page(2, "Sam said I always leave before things get hard.", heldout=True),
        page(3, "Dinner with mom, she asked about the job.", uncertain=True),
    ]
}
RECIPE = Recipe(kind="situation_help", style="typed", citations=("A:p.1", "A:p.2", "A:p.3"))
RECALL = RECIPE.model_copy(update={"kind": "recall"})
GOOD = Draft(
    text="Sam just told me again that I bail when it gets hard. Why do I keep doing that?",
    ref_citations=("A:p.1", "A:p.2"),
    ref_evidence=("talked about moving to Denver", "I always leave before things get hard"),
)


def test_text_hash_ignores_case_and_spacing():
    assert normalize_text("  Hello\n  World ") == "hello world"
    assert text_hash("Hello  world") == text_hash("hello world")


def test_good_draft_is_validated_train_task():
    t = validate_draft(GOOD, RECIPE, PAGES, seen_hashes=set())
    assert (t.status, t.reject_reason, t.grounded) == ("validated", None, True)
    assert (t.origin, t.split, t.source_citations) == ("synthetic", "train", RECIPE.citations)


@pytest.mark.parametrize(
    "draft, recipe, reason",
    [
        (None, RECIPE, "unparseable"),
        (GOOD.model_copy(update={"text": "  "}), RECIPE, "empty"),
        (GOOD.model_copy(update={"text": "why " * 401}), RECIPE, "too_long"),
        (GOOD.model_copy(update={"text": "What does the journal say about Sam?"}), RECIPE, "mentions_document"),
        (GOOD.model_copy(update={"text": "On page 3 I met Sam."}), RECIPE, "mentions_document"),
        (GOOD.model_copy(update={"ref_citations": ("A:p.9", "A:p.2")}), RECIPE, "ungrounded"),
        (GOOD.model_copy(update={"ref_evidence": ("a made up quote here", "I always leave")}), RECIPE, "ungrounded"),
        (GOOD.model_copy(update={"ref_evidence": ()}), RECIPE, "ungrounded"),
        (GOOD.model_copy(update={"ref_citations": ("A:p.3",), "ref_evidence": ("Dinner with mom",)}), RECALL,
         "uncertain_date"),
        (GOOD, RECALL, "heldout_page"),
    ],
)
def test_reject_reasons(draft, recipe, reason):
    t = validate_draft(draft, recipe, PAGES, seen_hashes=set(), raw="not json")
    assert (t.status, t.reject_reason) == ("rejected", reason)


def test_duplicate_and_near_eval():
    assert validate_draft(GOOD, RECIPE, PAGES, {text_hash(GOOD.text)}).reject_reason == "duplicate"
    near = word_set("Sam just told me that I bail when it gets hard. Why do I do that?")
    assert jaccard(word_set(GOOD.text), near) >= 0.5
    assert validate_draft(GOOD, RECIPE, PAGES, set(), eval_words=[near]).reject_reason == "near_eval"
    far = word_set("What is my attachment style?")
    assert validate_draft(GOOD, RECIPE, PAGES, set(), eval_words=[far]).status == "validated"


def test_quotes_may_come_in_any_order():
    swapped = GOOD.model_copy(update={"ref_evidence": GOOD.ref_evidence[::-1]})
    assert validate_draft(swapped, RECIPE, PAGES, set()).status == "validated"
    one_quote = GOOD.model_copy(update={"ref_evidence": GOOD.ref_evidence[:1]})
    assert validate_draft(one_quote, RECIPE, PAGES, set()).status == "validated"


def test_heldout_only_guards_recall():
    # Non-recall drafts may cite held-out pages: the journal in context has them anyway.
    assert validate_draft(GOOD, RECIPE, PAGES, set()).status == "validated"


def test_near_exemplar():
    exemplar = word_set("Sam just told me again that I bail when it gets hard, why do I keep doing that")
    assert validate_draft(GOOD, RECIPE, PAGES, set(), exemplar_words=[exemplar]).reject_reason == "near_exemplar"


def test_instruction_echo():
    lens = RECIPE.model_copy(update={"kind": "framework_lens", "framework": "self-determination theory"})
    echo = GOOD.model_copy(update={"text": "Apply self-determination theory to me, using what my life shows."})
    assert validate_draft(echo, lens, PAGES, set()).reject_reason == "instruction_echo"
    assert validate_draft(GOOD, lens, PAGES, set()).status == "validated"
