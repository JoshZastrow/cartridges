import pytest

from cartridges.training.metrics import corpus_cer


def test_corpus_cer_sums_edits_over_total_reference_characters():
    # 1 edit in 3 chars + 0 edits in 2 chars = 1 / 5. (Mean of per-page CER would be 1/6.)
    result = corpus_cer(["abc", "de"], ["abd", "de"])
    assert result["corpus_cer"] == pytest.approx(1 / 5)
    assert result["n_scored"] == 2
    assert result["n_blank_refs"] == 0


def test_long_pages_outweigh_short_ones():
    # A fully wrong 2-char page next to a perfect 98-char page: 2 / 100, not 50%.
    result = corpus_cer(["xy", "a" * 98], ["zz", "a" * 98])
    assert result["corpus_cer"] == pytest.approx(2 / 100)


def test_blank_references_are_excluded_and_counted():
    result = corpus_cer(["", "abc"], ["The image contains no text.", "abc"])
    assert result["corpus_cer"] == 0.0
    assert result["n_scored"] == 1
    assert result["n_blank_refs"] == 1


def test_failed_inferences_are_excluded():
    result = corpus_cer(["abc", "abc"], [None, "abc"])
    assert result["corpus_cer"] == 0.0
    assert result["n_scored"] == 1


def test_nothing_to_score():
    assert corpus_cer([""], ["x"]) == {"corpus_cer": None, "n_scored": 0, "n_blank_refs": 1}
