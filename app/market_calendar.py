"""NYSE full-day closures. COPY of meridian-clock/market_calendar.py (the clock is the authority).

The clock skips every dispatch on these days, so the page must not expect a run on them.
Keep the two files in step: an unscheduled closure added to the clock's EXTRA_CLOSED goes here
too, the same day. tests/test_schedule.py pins 2026-2028 to the clock's output as of Nov 26, 2026.
Only change from the original: the Easter variable `l` is `w` (ruff E741).

Computed from the exchange's holiday rules (stdlib only).

Holidays: New Year's Day, Martin Luther King Jr. Day (3rd Mon Jan), Washington's
Birthday (3rd Mon Feb), Good Friday, Memorial Day (last Mon May), Juneteenth (Jun 19,
from 2022), Independence Day (Jul 4), Labor Day (1st Mon Sep), Thanksgiving (4th Thu
Nov), Christmas (Dec 25).

Fixed-date holidays on a Saturday close the Friday before, on a Sunday the Monday
after. Exception (NYSE Rule 7.2): New Year's Day on a Saturday closes nothing; Dec 31
trades normally.

Early closes (1 pm, e.g. the day after Thanksgiving) are trading days and not listed.
Unscheduled closures (a national day of mourning, say) go in EXTRA_CLOSED by hand.
"""

from __future__ import annotations

from datetime import date, timedelta

# Unscheduled full-day closures the rules can't know about: {date(2027, 1, 9): "Day of mourning"}.
EXTRA_CLOSED: dict[date, str] = {}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday (Mon=0) of the month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    w = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * w) // 451
    month = (h + w - 7 * m + 114) // 31
    day = (h + w - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> dict[date, str]:
    """{date: name} of NYSE full-day closures in `year` (rules plus EXTRA_CLOSED)."""
    out: dict[date, str] = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # Saturday: no closure
        out[_observed(new_year)] = "New Year's Day"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[_easter(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving"
    out[_observed(date(year, 12, 25))] = "Christmas"
    out.update({d: n for d, n in EXTRA_CLOSED.items() if d.year == year})
    return out


def market_holiday(d: date) -> str | None:
    """The holiday's name if the NYSE is closed all day on weekday `d`, else None."""
    return nyse_holidays(d.year).get(d)
