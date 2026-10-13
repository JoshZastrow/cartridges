import datetime

import pytest

from cartridges.batch.dates import page_dates, written_date

D = datetime.date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Saturday, August 23, 2025\n\nThere is a sand storm.", D(2025, 8, 23)),
        ("Monday, 9:55am\nOctober 21st, 2024\n\nWriting the date", D(2024, 10, 21)),
        ("Wednesday, Oct. 2nd, 2024\nTahoe, on the deck", D(2024, 10, 2)),
        ("## June 21st, 2025\nbody", D(2025, 6, 21)),
        ("23rd of August 2025", D(2025, 8, 23)),
        ("10/24/24 morning", D(2024, 10, 24)),
        ("No date here.\nJust writing.", None),
        ("May I go outside?", None),  # "May" without a day is not a date
        ("Feb 30, 2024", None),  # not a real day
    ],
)
def test_written_date(text, expected):
    assert written_date(text) == expected


def test_only_the_first_lines_count():
    body = "line\n" * 5 + "October 21st, 2024"
    assert written_date(body) is None


def test_missing_year_comes_from_the_previous_date():
    assert written_date("Tuesday, Oct 22", last=D(2024, 10, 21)) == D(2024, 10, 22)
    assert written_date("Jan 3", last=D(2024, 12, 30)) == D(2025, 1, 3)  # year rolls over
    assert written_date("Oct 22") is None  # no year anywhere yet


def test_missing_year_matches_a_written_weekday():
    # June 13 was a Monday in 2022, April 7 a Friday in 2023.
    assert written_date("## Monday, June 13th", last=D(2023, 1, 5)) == D(2022, 6, 13)
    assert written_date("Friday, April 7th", last=D(2024, 3, 1)) == D(2023, 4, 7)


def test_page_dates_carry_the_entry_date_forward():
    texts = ["# Morning Pages", "October 21st, 2024\nentry", "more of the entry", "Oct. 24, 2024\nnext"]
    assert page_dates(texts) == ["unknown", "2024-10-21", "2024-10-21 (cont.)", "2024-10-24"]
