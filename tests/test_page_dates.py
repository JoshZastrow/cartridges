import pytest

from cartridges.batch.dates import WrittenDate, page_dates, read_date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Saturday, August 23, 2025\n\nThere is a sand storm.", WrittenDate(8, 23, 2025, 5)),
        ("Monday, 9:55am\nOctober 21st, 2024\n\nWriting the date", WrittenDate(10, 21, 2024, 0)),
        ("## Wednesday, Oct. 2nd, 2024\nTahoe, on the deck", WrittenDate(10, 2, 2024, 2)),
        ("D: 12/5/21 //10am\nS: I was with Allie.", WrittenDate(12, 5, 2021, None)),
        ("Meditation - Wednesday, Sept. 5th, 2023", WrittenDate(9, 5, 2023, 2)),
        ("23rd of August 2025", WrittenDate(8, 23, 2025, None)),
        ("## Monday, June 13th\nbody", WrittenDate(6, 13, None, 0)),
        ("No date here.\nJust writing.", None),
        ("May I go outside?", None),  # "May" without a day is not a date
        # a date inside a sentence is body text, not the page's date
        ("It's been three days since I last talked with Alex. On Dec 3, 2023 we met.", None),
    ],
)
def test_read_date(text, expected):
    assert read_date(text) == expected


def test_only_the_first_lines_count():
    assert read_date("line\n" * 5 + "October 21st, 2024") is None


def test_page_dates_carry_the_entry_date_forward():
    texts = ["# Morning Pages", "October 21st, 2024\nentry", "more of the entry", "Oct. 24, 2024\nnext"]
    assert page_dates(texts) == ["unknown", "2024-10-21", "2024-10-21 (cont.)", "2024-10-24"]


def test_written_weekday_beats_a_misread_year():
    # Nov 28 was a Sunday in 2021, a Thursday in 2024.
    texts = ["12/5/21", "Sunday, Nov. 28, 2024", "12/10/21"]
    assert page_dates(texts) == ["2021-12-05", "2021-11-28", "2021-12-10"]


def test_timeline_beats_a_misread_year():
    texts = ["Friday, Oct. 20th, 2023", "October 21st, 2025", "December 2nd, 2023"]
    assert page_dates(texts) == ["2023-10-20", "2023-10-21", "2023-12-02"]


def test_a_misread_first_year_does_not_drag_the_rest():
    texts = ["D: 11/19/2024", "D: 12/5/21", "D: 12/10/21", "D: 12/13/21"]
    assert page_dates(texts) == ["2021-11-19", "2021-12-05", "2021-12-10", "2021-12-13"]


def test_missing_year_comes_from_the_timeline_and_weekday():
    # May 3 was a Wednesday in 2023.
    texts = ["Monday, May 22nd, 2023", "Wednesday, May 3rd", "Thursday, June 1st, 2023"]
    assert page_dates(texts)[1] == "2023-05-03"


def test_unresolvable_conflicts_are_marked():
    # Feb 1 is not a Monday in any nearby year, so the weekday can't match: flag it.
    texts = ["Jan 30, 2024", "Monday, Feb 1st", "Feb 5, 2024"]  # Feb 1 2024 = Thursday
    labels = page_dates(texts)
    assert labels[1].startswith("2024-02-01") and labels[1].endswith("(?)")


def test_no_year_anywhere_stays_unknown():
    assert page_dates(["Monday, June 13th", "body"]) == ["unknown", "unknown"]
