"""Page dates for the batch outputs, read from the OCR text.

A journal entry starts with a handwritten date ("Monday, 9:55am / October
21st, 2024"); the pages after it continue the entry without one. So:
- written_date() looks for a date in the first few lines of a page;
- page_dates() gives every page a header label: the written date
  ("2024-10-21"), the last written date carried forward ("2024-10-21 (cont.)"),
  or "unknown" before the first date.

Rules only, no model: deterministic and free. A date the OCR misread passes
through as written.
"""

from __future__ import annotations

import datetime
import re

UNKNOWN_DATE = "unknown"
CONTINUED = " (cont.)"
HEAD_LINES = 4  # a page's date is in its first lines; deeper matches are body text

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_WEEKDAY = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\b", re.I)
_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s*(?P<year>\d{4}))?"
_PATTERNS = [
    # October 21st, 2024 / Oct. 2nd 2024 / Aug 23
    re.compile(rf"\b{_MONTH}\s+(?P<day>\d{{1,2}}){_ORD}\b{_YEAR}", re.I),
    # 23 August 2025 / 23rd of August
    re.compile(rf"\b(?P<day>\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MONTH}\b{_YEAR}", re.I),
    # 10/24/24, 10-24-2024 (US month first)
    re.compile(r"\b(?P<nmonth>\d{1,2})[/.-](?P<day>\d{1,2})[/.-](?P<year>\d{4}|\d{2})\b"),
]


def written_date(text: str, last: datetime.date | None = None) -> datetime.date | None:
    """The date written in the first lines of `text`, or None. A date without
    a year gets one from `last` (the previous written date): if a weekday is
    written too ("Monday, June 13th"), the year within one of `last`'s where
    that weekday matches; otherwise `last`'s year, one later if it would go
    back in time (December -> January)."""
    head = "\n".join(text.strip().splitlines()[:HEAD_LINES])
    weekday = _WEEKDAY.search(head)
    for pattern in _PATTERNS:
        for m in pattern.finditer(head):
            parts = m.groupdict()
            month = int(parts["nmonth"]) if parts.get("nmonth") else _MONTHS[parts["month"][:3].lower()]
            year = parts.get("year")
            if year is None:
                if last is None:
                    continue
                day = int(parts["day"])
                if weekday is not None:
                    want = _WEEKDAYS.index(weekday.group(1).lower())
                    years = sorted(range(last.year - 1, last.year + 2), key=lambda y: abs(y - last.year))
                    match = next((d for y in years if (d := _date(y, month, day)) and d.weekday() == want), None)
                    if match is not None:
                        return match
                guess = _date(last.year, month, day)
                if guess is not None and guess < last:
                    guess = _date(last.year + 1, month, day)
                if guess is not None:
                    return guess
                continue
            year = int(year) + (2000 if len(year) == 2 else 0)
            found = _date(year, month, int(parts["day"]))
            if found is not None and 1990 <= found.year <= 2100:
                return found
    return None


def page_dates(texts: list[str]) -> list[str]:
    """One header date label per page, in page order."""
    labels = []
    last: datetime.date | None = None
    for text in texts:
        found = written_date(text, last)
        if found is not None:
            last = found
            labels.append(found.isoformat())
        elif last is not None:
            labels.append(last.isoformat() + CONTINUED)
        else:
            labels.append(UNKNOWN_DATE)
    return labels


def _date(year: int, month: int, day: int) -> datetime.date | None:
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None
