"""When the next reads are due: meridian-clock's schedule, NYSE holidays skipped.

The page used to work this out itself from a weekday-only copy of the schedule, so on Thanksgiving
(Nov 26, 2026) it said the 06:00 run "didn't publish" and promised a 4:40 PM update, though the
clock skips every job on a market holiday by design. The list is written into regime.json at
publish time and the page only compares it with the clock on the phone.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import MISSED_RUN_GRACE_MINUTES, SCHEDULE_NY, SCHEDULE_RUNS_AHEAD, SCHEDULE_TZ
from app.market_calendar import market_holiday
from app.models import ClosedDay, PublishSchedule

NY = ZoneInfo(SCHEDULE_TZ)


def _times() -> list[time]:
    return sorted(time(int(t[:2]), int(t[3:])) for t in SCHEDULE_NY)


def publishes_on(day: date) -> bool:
    """True when the clock dispatches on `day`: a weekday that is not an NYSE full-day closure."""
    return day.weekday() < 5 and market_holiday(day) is None


def upcoming_runs(after: datetime, count: int = SCHEDULE_RUNS_AHEAD) -> list[datetime]:
    """The next `count` scheduled run times strictly after `after`, as aware New York datetimes."""
    local = after.astimezone(NY)
    out: list[datetime] = []
    day = local.date()
    while len(out) < count:
        if publishes_on(day):
            for t in _times():
                at = datetime.combine(day, t, tzinfo=NY)
                if at > local and len(out) < count:
                    out.append(at)
        day += timedelta(days=1)
    return out


def build_schedule(now: datetime | None = None) -> PublishSchedule:
    """The schedule block for regime.json, counted from the publish time (not the read's as_of: a
    stale fallback republishes an old read, and the next run is still due from now)."""
    now = now or datetime.now(timezone.utc)
    runs = upcoming_runs(now)
    start = now.astimezone(NY).date()
    closed: list[ClosedDay] = []
    day = start
    while day <= runs[-1].date():
        name = market_holiday(day)
        if name is not None and day.weekday() < 5:
            closed.append(ClosedDay(date=day.isoformat(), name=name))
        day += timedelta(days=1)
    return PublishSchedule(
        published_at=now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        next_runs=[r.isoformat(timespec="minutes") for r in runs],
        grace_minutes=MISSED_RUN_GRACE_MINUTES,
        closed=closed,
    )
