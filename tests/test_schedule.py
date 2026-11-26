"""The publish schedule: the clock's times, weekends and NYSE holidays skipped (app/schedule.py)."""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.market_calendar import market_holiday, nyse_holidays
from app.models import RegimeStatus
from app.schedule import build_schedule, publishes_on, upcoming_runs

# meridian-clock's nyse_holidays() output for these years on Nov 26, 2026. app/market_calendar.py is
# a copy of the clock's file; if this test fails, one of the two changed without the other.
CLOCK_HOLIDAYS = {
    2026: {
        "2026-01-01": "New Year's Day",
        "2026-01-19": "Martin Luther King Jr. Day",
        "2026-02-16": "Washington's Birthday",
        "2026-04-03": "Good Friday",
        "2026-05-25": "Memorial Day",
        "2026-06-19": "Juneteenth",
        "2026-07-03": "Independence Day",
        "2026-09-07": "Labor Day",
        "2026-11-26": "Thanksgiving",
        "2026-12-25": "Christmas",
    },
    2027: {
        "2027-01-01": "New Year's Day",
        "2027-01-18": "Martin Luther King Jr. Day",
        "2027-02-15": "Washington's Birthday",
        "2027-03-26": "Good Friday",
        "2027-05-31": "Memorial Day",
        "2027-06-18": "Juneteenth",
        "2027-07-05": "Independence Day",
        "2027-09-06": "Labor Day",
        "2027-11-25": "Thanksgiving",
        "2027-12-24": "Christmas",
    },
    2028: {  # New Year's Day 2028 is a Saturday: no closure (NYSE Rule 7.2)
        "2028-01-17": "Martin Luther King Jr. Day",
        "2028-02-21": "Washington's Birthday",
        "2028-04-14": "Good Friday",
        "2028-05-29": "Memorial Day",
        "2028-06-19": "Juneteenth",
        "2028-07-04": "Independence Day",
        "2028-09-04": "Labor Day",
        "2028-11-23": "Thanksgiving",
        "2028-12-25": "Christmas",
    },
}


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_calendar_matches_the_clock() -> None:
    for year, expected in CLOCK_HOLIDAYS.items():
        got = {d.isoformat(): name for d, name in nyse_holidays(year).items()}
        assert got == expected, year


def test_early_close_is_a_normal_day() -> None:
    # Day after Thanksgiving closes at 1 pm; the clock runs both jobs.
    assert market_holiday(date(2026, 11, 27)) is None
    assert publishes_on(date(2026, 11, 27))


def test_no_runs_on_weekends_or_holidays() -> None:
    assert not publishes_on(date(2026, 11, 26))  # Thanksgiving
    assert not publishes_on(date(2026, 11, 28))  # Saturday
    assert publishes_on(date(2026, 11, 25))


def test_thanksgiving_noon_next_run_is_friday_morning() -> None:
    """Nov 26, 2026 12:17 ET: the page said 'didn't publish' and 'next update today 4:40 PM'."""
    runs = upcoming_runs(_utc("2026-11-26T17:17:00"), count=3)
    assert [r.isoformat(timespec="minutes") for r in runs] == [
        "2026-11-27T06:00-05:00",
        "2026-11-27T16:40-05:00",
        "2026-11-30T06:00-05:00",
    ]


def test_wednesday_close_run_skips_thanksgiving() -> None:
    # The 16:40 run publishes around 16:45; its next run is Friday 06:00, not Thursday.
    runs = upcoming_runs(_utc("2026-11-25T21:45:00"), count=1)
    assert runs[0].isoformat(timespec="minutes") == "2026-11-27T06:00-05:00"


def test_a_run_before_its_slot_lists_the_slot() -> None:
    # A hand-run at 05:50 still expects the 06:00 run the same morning.
    runs = upcoming_runs(_utc("2026-11-30T10:50:00"), count=1)
    assert runs[0].isoformat(timespec="minutes") == "2026-11-30T06:00-05:00"


def test_daylight_saving_offsets() -> None:
    # Friday before the March 2027 change (EST) and the Monday after (EDT).
    runs = upcoming_runs(_utc("2027-03-12T22:00:00"), count=1)
    assert runs[0].isoformat(timespec="minutes") == "2027-03-15T06:00-04:00"


def test_schedule_block_lists_holidays_in_range() -> None:
    sched = build_schedule(_utc("2026-12-24T22:00:00"))
    assert sched.published_at == "2026-12-24T22:00:00+00:00"
    assert len(sched.next_runs) == 10
    assert sched.next_runs[0] == "2026-12-28T06:00-05:00"
    assert [(c.date, c.name) for c in sched.closed] == [
        ("2026-12-25", "Christmas"),
        ("2027-01-01", "New Year's Day"),
    ]
    assert sched.grace_minutes == 60


def test_closed_today_is_listed() -> None:
    sched = build_schedule(_utc("2026-11-26T17:17:00"))
    assert sched.closed[0].date == "2026-11-26"
    assert sched.closed[0].name == "Thanksgiving"


def test_old_payload_without_schedule_still_loads() -> None:
    status = RegimeStatus(
        phase="Buildout",
        risk_level="CALM",
        liquidity_squeeze=None,
        semi_rsi=None,
        tnx_yield=None,
        actionable_summary="",
        phase_id=1,
        as_of="2026-11-25T21:45:00+00:00",
    )
    assert status.schedule is None
    again = RegimeStatus.model_validate_json(status.model_dump_json())
    assert again.schedule is None
