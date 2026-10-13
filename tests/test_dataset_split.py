import pytest

from cartridges.training.dataset import ordered_split


def test_keeps_the_frozen_order_of_each_split():
    keys = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    spec = {"train": [["B", 2], ["A", 1]], "test": [["B", 1], ["A", 2]]}
    train, test = ordered_split(keys, spec)
    assert [keys[i] for i in train] == [("B", 2), ("A", 1)]
    assert [keys[i] for i in test] == [("B", 1), ("A", 2)]


def test_unlisted_pages_go_to_train_after_the_listed_ones():
    keys = [("A", 1), ("A", 9), ("A", 2)]
    spec = {"train": [["A", 1]], "test": [["A", 2]]}
    train, test = ordered_split(keys, spec)
    assert [keys[i] for i in train] == [("A", 1), ("A", 9)]
    assert [keys[i] for i in test] == [("A", 2)]


def test_listed_page_missing_from_the_data_is_an_error():
    with pytest.raises(ValueError, match="not in the staged data"):
        ordered_split([("A", 1)], {"train": [["A", 1]], "test": [["A", 5]]})


def test_page_in_both_splits_is_an_error():
    with pytest.raises(ValueError, match="both"):
        ordered_split([("A", 1)], {"train": [["A", 1]], "test": [["A", 1]]})
