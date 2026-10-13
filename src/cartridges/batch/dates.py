"""Page dates for the batch outputs, read from the OCR text.

A journal entry starts with a handwritten date ("Monday, 9:55am / October
21st, 2024"); the pages after it continue the entry without one. The journals
are written in order, so the timeline is evidence too:

- read_date() finds the date on a page's header line: month and day (trusted),
  plus the year and weekday if written (evidence only: the OCR misreads years,
  e.g. 2021 -> 2024, 2023 -> 2025).
- resolve_years() picks one year per dated page for the whole document at
  once (dynamic programming), favouring dates that move forward, small gaps,
  a matching written weekday, and the written year, in that order of weight.
- A numeric month may be a misread too ("11/7/22" for "1/7/22"): resolve_years
  also considers months one plausible misread away (_month_variants) at a
  cost, so a date far outside the timeline gets its month replaced when that fits.
- page_dates() labels every page: "2024-10-21" (dated), "2024-10-21 (cont.)"
  (carried from the last dated page), "2024-10-21 (?)" (month replaced, or the
  chosen date still conflicts: weekday or a jump back), "2024-10-21 (est.)"
  (a page before the first dated one: estimated from the next date), or
  "unknown" if the document has no date at all.

Rules only, no model: deterministic and free.
"""

from __future__ import annotations

import datetime
import re
from typing import NamedTuple

UNKNOWN_DATE = "unknown"
CONTINUED = " (cont.)"
UNSURE = " (?)"
ESTIMATED = " (est.)"
HEAD_LINES = 4  # a page's date is in its first lines
HEADER_MAX_CHARS = 70  # a longer line is a sentence, not a header
DATE_MAX_OFFSET = 25  # the date starts near the start of a header line

# Timeline costs (resolve_years). A written year is weak evidence (the OCR
# misreads it); the written weekday is strong; a big jump back is very unlikely.
COST_YEAR_NOT_AS_WRITTEN = 1.0
COST_WEEKDAY_MISMATCH = 3.0
COST_SMALL_STEP_BACK = 1.0  # a few weeks back: entries written out of order
COST_BIG_STEP_BACK = 10.0
COST_PER_YEAR_FORWARD = 1.0
COST_MONTH_MISREAD = 2.0  # > a small step back, < a big one
SAME_DAY_DAYS = 3
SMALL_STEP_BACK_DAYS = 45

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
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
_WEEKDAY = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\b", re.I)
_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_MARKUP = re.compile(r"^[\s#>*\-]+")


class WrittenDate(NamedTuple):
    month: int
    day: int
    year: int | None  # as read; may be wrong
    weekday: int | None  # 0 = Monday, as read
    month_text: str = ""  # the month as written ("11", "Nov."), for misread repair


def read_date(text: str) -> WrittenDate | None:
    """The date on the first header-like line of the page, or None. A header
    line is short and has the date near its start ("## Sunday, March 26,
    2023", "D: 12/5/21 //10am", "Meditation - Wednesday, Sept. 5th, 2023").
    The weekday counts if it is on that line or on a short line just above."""
    lines = [_MARKUP.sub("", line).strip() for line in text.strip().splitlines()[:HEAD_LINES]]
    for i, line in enumerate(lines):
        if not line or len(line) > HEADER_MAX_CHARS:
            continue
        for pattern in _PATTERNS:
            m = pattern.search(line)
            if m is None or m.start() > DATE_MAX_OFFSET:
                continue
            parts = m.groupdict()
            month = int(parts["nmonth"]) if parts.get("nmonth") else _MONTHS[parts["month"][:3].lower()]
            day = int(parts["day"])
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            year = parts.get("year")
            year = None if year is None else int(year) + (2000 if len(year) == 2 else 0)
            weekday = _WEEKDAY.search(line)
            if weekday is None and i > 0 and len(lines[i - 1]) <= DATE_MAX_OFFSET:
                weekday = _WEEKDAY.search(lines[i - 1])
            return WrittenDate(
                month, day, year,
                _WEEKDAYS.index(weekday.group(1).lower()) if weekday else None,
                parts.get("nmonth") or parts["month"],
            )
    return None


def resolve_years(found: list[WrittenDate]) -> list[tuple[datetime.date, bool] | None]:
    """One (date, unsure) per written date, choosing years for the whole
    sequence at once. None if no year fits (no year written anywhere, or an
    impossible day). unsure: the month was replaced, or the chosen date's
    weekday or step back conflicts."""
    written_years = [w.year for w in found if w.year is not None]
    if not written_years:
        return [None] * len(found)
    years = range(min(written_years) - 1, max(written_years) + 2)

    def own_cost(w: WrittenDate, d: datetime.date) -> float:
        cost = COST_YEAR_NOT_AS_WRITTEN if w.year is not None and d.year != w.year else 0.0
        if d.month != w.month:
            cost += COST_MONTH_MISREAD
        if w.weekday is not None and d.weekday() != w.weekday:
            cost += COST_WEEKDAY_MISMATCH
        return cost

    # best[i][d] = (cost of the best timeline ending with page i on date d, previous date)
    best: list[dict[datetime.date, tuple[float, datetime.date | None]]] = []
    for w in found:
        months = {w.month} | _month_variants(w.month_text, w.month)
        options = [d for y in years for m in months if (d := _date(y, m, w.day)) is not None]
        step: dict[datetime.date, tuple[float, datetime.date | None]] = {}
        prev = next((b for b in reversed(best) if b), None)
        for d in options:
            if prev is None:
                step[d] = (own_cost(w, d), None)
            else:
                step[d] = min(
                    ((c + _step_cost(p, d) + own_cost(w, d), p) for p, (c, _) in prev.items()),
                    key=lambda t: t[0],
                )
        best.append(step)

    # Walk back from the cheapest end.
    chosen: list[datetime.date | None] = [None] * len(found)
    last = next((i for i in reversed(range(len(best))) if best[i]), None)
    if last is None:
        return [None] * len(found)
    d = min(best[last], key=lambda k: best[last][k][0])
    for i in range(last, -1, -1):
        if not best[i]:
            continue
        chosen[i] = d
        d = best[i][d][1]

    out: list[tuple[datetime.date, bool] | None] = []
    prev_date = None
    for w, d in zip(found, chosen):
        if d is None:
            out.append(None)
            continue
        unsure = (
            d.month != w.month
            or (w.weekday is not None and d.weekday() != w.weekday)
            or (prev_date is not None and (prev_date - d).days > SMALL_STEP_BACK_DAYS)
        )
        out.append((d, unsure))
        prev_date = d
    return out


def page_dates(texts: list[str]) -> list[str]:
    """One header date label per page, in page order."""
    found = [read_date(text) for text in texts]
    dated = [i for i, w in enumerate(found) if w is not None]
    resolved = dict(zip(dated, resolve_years([found[i] for i in dated])))
    labels = []
    last: datetime.date | None = None
    for i in range(len(texts)):
        hit = resolved.get(i)
        if hit is not None:
            last, unsure = hit
            labels.append(last.isoformat() + (UNSURE if unsure else ""))
        elif last is not None:
            labels.append(last.isoformat() + CONTINUED)
        else:
            labels.append(UNKNOWN_DATE)
    # Pages before the first date (title pages, an undated first entry) take
    # the first date as an estimate, so every page has a place in the timeline.
    first = next((label for label in labels if label != UNKNOWN_DATE), None)
    if first is not None:
        estimate = first[:10] + ESTIMATED
        labels = [estimate if label == UNKNOWN_DATE else label for label in labels]
    return labels


# Handwritten digits the OCR confuses, both ways.
_CONFUSED_DIGITS = {"1": "7", "7": "1", "3": "8", "8": "3", "4": "9", "9": "4", "5": "6", "6": "5"}


def _month_variants(text: str, month: int) -> set[int]:
    """Months one plausible misread from a numeric month: drop or add a
    leading 1 ("11" <-> "1", "12" <-> "2") or swap a confusable digit
    ("1" <-> "7"). A written month name is not varied: on real pages the
    names read reliably, and an out-of-place named date is usually a misread
    day, which this can't repair (it stays flagged)."""
    if not text.isdigit():
        return set()
    digits = text.lstrip("0") or text
    out = set()
    if len(digits) == 2 and digits[0] == "1":
        out.add(int(digits[1]))
    if len(digits) == 1:
        out.add(int("1" + digits))
    for i, c in enumerate(digits):
        if c in _CONFUSED_DIGITS:
            out.add(int(digits[:i] + _CONFUSED_DIGITS[c] + digits[i + 1:]))
    return {m for m in out if 1 <= m <= 12 and m != month}


def _step_cost(prev: datetime.date, cur: datetime.date) -> float:
    days = (cur - prev).days
    if days >= -SAME_DAY_DAYS:
        return COST_PER_YEAR_FORWARD * max(days, 0) / 365
    if -days <= SMALL_STEP_BACK_DAYS:
        return COST_SMALL_STEP_BACK
    return COST_BIG_STEP_BACK


def _date(year: int, month: int, day: int) -> datetime.date | None:
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None
