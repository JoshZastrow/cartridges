import pytest

torch = pytest.importorskip("torch")
from transformers import RepetitionPenaltyLogitsProcessor  # noqa: E402

from cartridges.training.fast_decode import apply_repetition_penalty  # noqa: E402


@pytest.mark.parametrize("penalty", [1.05, 1.3, 0.9])
def test_repetition_penalty_matches_hf(penalty):
    torch.manual_seed(0)
    vocab = 50
    input_ids = torch.randint(0, vocab, (1, 30))  # repeats included
    logits = torch.randn(1, vocab) * 5  # positive and negative scores
    seen = torch.zeros(vocab, dtype=torch.bool)
    seen[input_ids[0]] = True

    expected = RepetitionPenaltyLogitsProcessor(penalty)(input_ids, logits.clone())
    assert torch.equal(apply_repetition_penalty(logits, seen, penalty), expected)


def test_unseen_tokens_are_untouched():
    logits = torch.tensor([[2.0, -2.0, 3.0]])
    seen = torch.tensor([True, True, False])
    assert apply_repetition_penalty(logits, seen, 2.0).tolist() == [[1.0, -4.0, 3.0]]
