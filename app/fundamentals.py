"""The cycle, measured the way it is defined: from company filings (SEC EDGAR XBRL).

Phases (rate cuts are a catalyst, not a phase):
  1 Buildout      hyperscaler capex growing fast and not slowing; chip revenue surging
  2 Digestion     spending still high, but capex or chip-revenue growth has peaked
  3 Monetization  software revenue growth accelerating two quarters running
  0 Between phases

Each company's filings give year-to-date or quarterly figures on its own fiscal calendar.
`quarterize` turns them into single quarters (Q4 = full year minus nine months), each quarter
is assigned to the nearest calendar quarter (Nvidia's quarter ending Jul 26 counts as Q2), and
groups are compared on trailing-twelve-month totals, year over year. A group's quarter is only
used once every company in it has reported, so a reading never mixes complete and partial data.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.config import (
    CHIPS_EX,
    CYCLE,
    CYCLE_COMPANIES,
    CYCLE_GROUP_HOW,
    CYCLE_GROUP_METRIC,
    CYCLE_HISTORY_QUARTERS,
    CYCLE_TAGS,
    FUNDING_GROUP,
    SEC_CONCEPT_URL,
    SEC_CONCURRENCY,
    SEC_DEFAULT_USER_AGENT,
    SEC_REFRESH_HOURS,
    THRESHOLDS,
    VALUATION,
    CycleThresholds,
    ValuationThresholds,
)
from app.data import get_with_retry, redact
from app.models import (
    Catalyst,
    CatalystBar,
    CatalystNote,
    CompanyFiling,
    Cycle,
    CycleCondition,
    CycleMetric,
    CyclePhaseCheck,
    QuarterValue,
)

SecCache = dict[str, Any]  # {"fetched_at": iso, "companies": {ticker: {...}}}
PHASE_NAMES = {1: "Buildout", 2: "Digestion", 3: "Monetization", 0: "Between phases"}
PHASE_SUMMARIES = {
    1: "Hyperscalers are accelerating their spending and chip revenue is surging. Hardware leads; valuations stretch.",
    2: "Spending is still heavy but its growth has peaked. Multiples compress and rotation and sell-offs are normal "
    "until revenue catches up.",
    3: "Software revenue is accelerating: AI spending is turning into enterprise revenue. Earnings, not multiples, "
    "carry the market.",
    0: "The filings don't fit one phase cleanly. Treat it as a transition.",
}
GROUP_LABELS = {
    "hyperscalers": "Hyperscaler capex",
    "chips": "Chip revenue",
    "software": "Software revenue (median company)",
}


# --- Fetching (cached; filings change quarterly) ----------------------------------------------


def _needed() -> list[tuple[str, str, str, str]]:
    """(group, ticker, cik, metric) for every series the cycle needs."""
    out = []
    for group, companies in CYCLE_COMPANIES.items():
        for ticker, cik, _ in companies:
            out.append((group, ticker, cik, CYCLE_GROUP_METRIC[group]))
            if group == "chips":
                out.append((group, ticker, cik, "eps"))
            if group == FUNDING_GROUP:
                out.append((group, ticker, cik, "ocf"))
    return out


def _slim(facts: list[dict[str, Any]], since: str) -> list[dict[str, Any]]:
    keep = ("start", "end", "val", "form", "filed")
    return [{k: f.get(k) for k in keep} for f in facts if str(f.get("end", "")) >= since and f.get("start")]


async def _concept(client: httpx.AsyncClient, cik: str, tag: str, headers: dict[str, str]) -> dict[str, Any] | None:
    try:
        response = await get_with_retry(client, SEC_CONCEPT_URL.format(cik=cik, tag=tag), headers=headers)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None  # the company doesn't report this tag
        raise
    data: dict[str, Any] = response.json()
    return data


async def _fetch_series(
    client: httpx.AsyncClient, ticker: str, cik: str, expected: str, metric: str, headers: dict[str, str], since: str
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for tag in CYCLE_TAGS[metric]:
        data = await _concept(client, cik, tag, headers)
        if data is None:
            continue
        entity = str(data.get("entityName", "")).upper()
        if not entity.startswith(expected):
            return {"error": f"CIK {cik} is {entity!r}, expected {expected!r}"}
        units = data.get("units") or {}
        facts = units.get("USD/shares" if metric == "eps" else "USD") or []
        facts = _slim(facts, since)
        if not facts:
            continue
        latest = max(str(f["end"]) for f in facts)
        if best is None or (latest, len(facts)) > (best["latest"], len(best["facts"])):
            best = {"tag": tag, "facts": facts, "latest": latest, "entity": entity}
    return best or {"error": f"no {metric} facts under {', '.join(CYCLE_TAGS[metric])}"}


async def fetch_filings(client: httpx.AsyncClient | None, cache: SecCache, now: datetime | None = None) -> list[str]:
    """Refresh `cache` in place if it's older than SEC_REFRESH_HOURS. Returns notes."""
    now = now or datetime.now(timezone.utc)
    fetched = cache.get("fetched_at")
    have = cache.get("companies") or {}
    complete = all(metric in (have.get(ticker) or {}) for _, ticker, _, metric in _needed())
    if complete and fetched and now - datetime.fromisoformat(fetched) < timedelta(hours=SEC_REFRESH_HOURS):
        return []  # a cache written before a series was added refetches at once
    headers = {"User-Agent": os.environ.get("SEC_USER_AGENT") or SEC_DEFAULT_USER_AGENT}
    since = (now - timedelta(days=365 * 6)).date().isoformat()
    companies: dict[str, Any] = cache.setdefault("companies", {})
    expected = {t: name for group in CYCLE_COMPANIES.values() for t, _, name in group}
    semaphore = asyncio.Semaphore(SEC_CONCURRENCY)
    notes: list[str] = []
    owns = client is None
    client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def one(ticker: str, cik: str, metric: str) -> None:
        async with semaphore:
            try:
                result = await _fetch_series(client, ticker, cik, expected[ticker], metric, headers, since)
            except Exception as exc:
                result = {"error": redact(f"{type(exc).__name__}: {exc}")}
            await asyncio.sleep(0.15)  # stay well under SEC's 10 requests/second
        slot = companies.setdefault(ticker, {})
        if "error" in result and metric in slot and "facts" in slot[metric]:
            notes.append(f"{ticker} {metric}: refresh failed ({result['error']}); kept the cached filings")
            return
        slot[metric] = result

    try:
        await asyncio.gather(*(one(t, c, m) for _, t, c, m in _needed()))
    finally:
        if owns:
            await client.aclose()
    cache["fetched_at"] = now.isoformat(timespec="seconds")
    errors = [str(m.get("error", "")) for slot in companies.values() for m in slot.values() if isinstance(m, dict)]
    if any("403" in e for e in errors) or any("403" in n for n in notes):
        notes.append(
            "SEC refused some requests (403). SEC asks for a contact address in the request header: set the "
            "SEC_USER_AGENT secret to 'regime-engine <your email>'."
        )
    return notes


# --- Turning filings into calendar quarters -----------------------------------------------------


def quarterize(facts: list[dict[str, Any]]) -> pd.Series:
    """Single-quarter values by period end, from quarterly and year-to-date facts.

    Duplicate (start, end) facts keep the latest filing (restatements win). A quarter is either a
    reported ~3-month fact, or the difference between consecutive year-to-date facts that share a
    start date and end ~3 months apart (Q4 = full year minus nine months). Trailing-twelve-month
    facts (Amazon) share no such chain and are ignored.
    """
    rows: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, str]] = {}
    for f in facts:
        if not str(f.get("form", "")).startswith(("10-Q", "10-K")):
            continue
        try:
            start, end, val = pd.Timestamp(f["start"]), pd.Timestamp(f["end"]), float(f["val"])
        except (KeyError, TypeError, ValueError):
            continue
        filed = str(f.get("filed", ""))
        if (start, end) not in rows or filed > rows[(start, end)][1]:
            rows[(start, end)] = (val, filed)

    quarters: dict[pd.Timestamp, float] = {}
    chains: dict[pd.Timestamp, list[tuple[pd.Timestamp, float, int]]] = defaultdict(list)
    for (start, end), (val, _) in rows.items():
        days = (end - start).days + 1
        if 80 <= days <= 100:
            quarters[end] = val
        chains[start].append((end, val, days))
    for items in chains.values():
        items.sort()
        for (end1, val1, _), (end2, val2, days2) in pairwise(items):
            if days2 > 100 and 80 <= (end2 - end1).days <= 100 and end2 not in quarters:
                quarters[end2] = val2 - val1
    return pd.Series(quarters, dtype=float).sort_index() if quarters else pd.Series(dtype=float)


def _val(series: pd.Series, q: pd.Period) -> float:
    """series[q] for a Period-indexed series (typed: pandas-stubs has no Period overload)."""
    return float(series.get(q))  # type: ignore[arg-type]


def calendar_quarter(end: pd.Timestamp) -> pd.Period:
    this = pd.Period(end, freq="Q")
    previous = this - 1
    to_this = abs((this.end_time.normalize() - end).days)
    to_previous = abs((previous.end_time.normalize() - end).days)
    return this if to_this <= to_previous else previous


def by_calendar_quarter(quarters: pd.Series) -> pd.Series:
    out: dict[pd.Period, float] = {}
    for end, value in quarters.items():
        out[calendar_quarter(pd.Timestamp(str(end)))] = float(value)  # later ends overwrite (sorted input)
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def ttm(cq: pd.Series) -> pd.Series:
    """Trailing-twelve-month sum at each calendar quarter that has four consecutive quarters."""
    out: dict[pd.Period, float] = {}
    for q in cq.index:
        window = [q - i for i in range(4)]
        if all(w in cq.index for w in window):
            out[q] = float(sum(cq[w] for w in window))
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def company_growth(quarters: pd.Series) -> pd.Series:
    """One company's TTM year-over-year growth (%) by calendar quarter."""
    t = ttm(by_calendar_quarter(quarters))
    out = {q: (t[q] / t[q - 4] - 1.0) * 100.0 for q in t.index if (q - 4) in t.index and t[q - 4] > 0}
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def group_growth(series_by_company: dict[str, pd.Series], how: str = "sum") -> pd.Series:
    """Group TTM year-over-year growth (%), only at quarters where every company has both years.

    how="sum": growth of the group's combined dollars. how="median": the median company's growth.
    """
    if not series_by_company:
        return pd.Series(dtype=float)
    ttms = {t: ttm(by_calendar_quarter(s)) for t, s in series_by_company.items()}
    quarters = sorted(set.intersection(*(set(s.index) for s in ttms.values())))
    out: dict[pd.Period, float] = {}
    for q in quarters:
        if not all((q - 4) in s.index and s[q - 4] > 0 for s in ttms.values()):
            continue
        if how == "median":
            out[q] = float(pd.Series([(s[q] / s[q - 4] - 1.0) * 100.0 for s in ttms.values()]).median())
        else:
            now, before = sum(s[q] for s in ttms.values()), sum(s[q - 4] for s in ttms.values())
            out[q] = (now / before - 1.0) * 100.0
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def single_quarter_growth(series_by_company: dict[str, pd.Series], how: str = "sum") -> pd.Series:
    """Group growth of ONE quarter vs the same quarter a year earlier (%), where every company has
    both. Trailing-year growth lags a turn by about two quarters (chips, Q2 2026: 81% trailing,
    124% on the quarter alone); this is shown beside it, not used in the phase rules."""
    if not series_by_company:
        return pd.Series(dtype=float)
    cqs = {t: by_calendar_quarter(s) for t, s in series_by_company.items()}
    quarters = sorted(set.intersection(*(set(s.index) for s in cqs.values())))
    out: dict[pd.Period, float] = {}
    for q in quarters:
        if not all((q - 4) in s.index and s[q - 4] > 0 for s in cqs.values()):
            continue
        if how == "median":
            out[q] = float(pd.Series([(s[q] / s[q - 4] - 1.0) * 100.0 for s in cqs.values()]).median())
        else:
            out[q] = (sum(s[q] for s in cqs.values()) / sum(s[q - 4] for s in cqs.values()) - 1.0) * 100.0
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def funding_ratio(capex: dict[str, pd.Series], ocf: dict[str, pd.Series]) -> pd.Series:
    """Group capex as a % of group operating cash flow, trailing twelve months, at each quarter
    where every company has both. Above 100 the group is spending more than it earns in cash."""
    if not capex or set(capex) != set(ocf):
        return pd.Series(dtype=float)
    c = {t: ttm(by_calendar_quarter(s)) for t, s in capex.items()}
    o = {t: ttm(by_calendar_quarter(s)) for t, s in ocf.items()}
    quarters = sorted(set.intersection(*(set(s.index) for s in (*c.values(), *o.values()))))
    out: dict[pd.Period, float] = {}
    for q in quarters:
        cash = sum(s[q] for s in o.values())
        if cash > 0:
            out[q] = sum(s[q] for s in c.values()) / cash * 100.0
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def median_acceleration(series_by_company: dict[str, pd.Series]) -> pd.Series:
    """The median company's change in growth vs the prior quarter (points), where all can be compared."""
    growth = {t: company_growth(s) for t, s in series_by_company.items()}
    quarters = sorted(set().union(*(set(g.index) for g in growth.values()))) if growth else []
    out = {}
    for q in quarters:
        deltas = [g[q] - g[q - 1] for g in growth.values() if q in g.index and (q - 1) in g.index]
        if deltas and len(deltas) == len(growth):
            out[q] = float(pd.Series(deltas).median())
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def breadth(series_by_company: dict[str, pd.Series]) -> dict[pd.Period, tuple[int, int]]:
    """At each quarter: (companies whose growth rose vs the prior quarter, companies compared)."""
    growth = {t: company_growth(s) for t, s in series_by_company.items()}
    quarters = sorted(set().union(*(set(g.index) for g in growth.values()))) if growth else []
    out = {}
    for q in quarters:
        pairs = [(g[q], g[q - 1]) for g in growth.values() if q in g.index and (q - 1) in g.index]
        if len(pairs) == len(growth):
            out[q] = (sum(1 for now, before in pairs if now > before), len(pairs))
    return out


def qlabel(q: pd.Period | None) -> str | None:
    return None if q is None else f"Q{q.quarter} {q.year}"


# --- Phases --------------------------------------------------------------------------------------


@dataclass
class Readings:
    capex: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    chips: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    software: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    software_accel: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    software_breadth: dict[pd.Period, tuple[int, int]] = field(default_factory=dict)

    def at(self, series: pd.Series, q: pd.Period) -> float | None:
        return _val(series, q) if q in series.index else None

    def peak(self, series: pd.Series, q: pd.Period, window: int) -> float | None:
        """Highest reading over the `window` quarters ending at q (None unless all exist)."""
        values = [self.at(series, q - i) for i in range(window)]
        return None if any(v is None for v in values) else max(v for v in values if v is not None)


def _ge(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a >= b


def _gt(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a > b


def _and(*xs: bool | None) -> bool | None:
    if any(x is False for x in xs):
        return False
    return True if all(x is True for x in xs) else None


def _or(*xs: bool | None) -> bool | None:
    if any(x is True for x in xs):
        return True
    return False if all(x is False for x in xs) else None


def _not(x: bool | None) -> bool | None:
    return None if x is None else not x


def _all(conds: list[CycleCondition]) -> bool | None:
    return _and(*(c.met for c in conds))


def phase_checks(r: Readings, q: pd.Period, t: CycleThresholds = CYCLE) -> list[CyclePhaseCheck]:
    c, h = r.at(r.capex, q), r.at(r.chips, q)
    s = r.at(r.software, q)
    a, ap = r.at(r.software_accel, q), r.at(r.software_accel, q - 1)
    c_peak, h_peak = r.peak(r.capex, q, t.peak_window_quarters), r.peak(r.chips, q, t.peak_window_quarters)
    drop = t.rolled_over_points
    c_rolled = None if c is None or c_peak is None else c <= c_peak - drop
    h_rolled = None if h is None or h_peak is None else h <= h_peak - drop
    up, n = r.software_breadth.get(q, (0, 0))
    broad = None if n == 0 else up * 2 >= n

    def peak_text(now: float | None, top: float | None) -> str:
        return f"{_pct(now)} vs recent peak {_pct(top)} (rolled over if ≥ {drop:.0f} pts below)"

    build = [
        CycleCondition(
            name="Hyperscaler capex growth",
            value=c,
            threshold=f"≥ {t.buildout_capex_growth:.0f}%",
            met=_ge(c, t.buildout_capex_growth),
        ),
        CycleCondition(
            name="Capex growth hasn't rolled over",
            value=None,
            threshold=peak_text(c, c_peak),
            met=_not(c_rolled),
            unit="",
        ),
        CycleCondition(
            name="Chip revenue growth",
            value=h,
            threshold=f"≥ {t.buildout_chip_growth:.0f}%",
            met=_ge(h, t.buildout_chip_growth),
        ),
        CycleCondition(
            name="Chip growth hasn't rolled over",
            value=None,
            threshold=peak_text(h, h_peak),
            met=_not(h_rolled),
            unit="",
        ),
    ]
    digest = [
        CycleCondition(
            name="Hyperscalers still spending heavily",
            value=c,
            threshold=f"capex growth ≥ {t.digestion_capex_floor:.0f}%",
            met=_ge(c, t.digestion_capex_floor),
        ),
        CycleCondition(
            name="Capex or chip growth has rolled over",
            value=None,
            threshold=f"capex {_pct(c)} (peak {_pct(c_peak)}), chips {_pct(h)} (peak {_pct(h_peak)})",
            met=_or(c_rolled, h_rolled),
            unit="",
        ),
    ]
    monet = [
        CycleCondition(
            name="Software revenue growth (median company)",
            value=s,
            threshold=f"≥ {t.monetization_software_growth:.0f}%",
            met=_ge(s, t.monetization_software_growth),
        ),
        CycleCondition(
            name="Median company accelerating two quarters running",
            value=None,
            threshold=f"{_pts(ap)} then {_pts(a)}, total {_total(a, ap)} (needs both > 0 and total ≥ "
            f"{t.monetization_accel_points:.1f})",
            met=_and(
                _gt(a, 0.0), _gt(ap, 0.0), None if a is None or ap is None else a + ap >= t.monetization_accel_points
            ),
            unit="",
        ),
        CycleCondition(
            name="Broad: at least half the companies accelerating",
            value=None,
            threshold=f"{up} of {n} accelerating this quarter" if n else "n/a",
            met=broad,
            unit="",
        ),
    ]
    return [
        CyclePhaseCheck(phase_id=1, name=PHASE_NAMES[1], met=_all(build), conditions=build),
        CyclePhaseCheck(phase_id=2, name=PHASE_NAMES[2], met=_all(digest), conditions=digest),
        CyclePhaseCheck(phase_id=3, name=PHASE_NAMES[3], met=_all(monet), conditions=monet),
    ]


def _pct(x: float | None, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}%"


def _total(a: float | None, b: float | None) -> str:
    return "n/a" if a is None or b is None else f"{a + b:+.3f}"


def _pts(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f} pts"


def phase_of(checks: list[CyclePhaseCheck]) -> int | None:
    """Monetization > Buildout > Digestion > Between. None if a deciding check is unknown.

    Buildout and Digestion are mutually exclusive (rolled over or not); Monetization can overlap
    either, and when it does it is the headline because it is the phase the cycle is waiting for.
    """
    by_id = {c.phase_id: c.met for c in checks}
    for phase_id in (3, 1, 2):
        if by_id[phase_id] is True:
            return phase_id
        if by_id[phase_id] is None:
            return None
    return 0


def phases_met(checks: list[CyclePhaseCheck]) -> list[int]:
    return [c.phase_id for c in checks if c.met is True]


# --- "How we go up": the catalysts ------------------------------------------------------------------
#
# Each catalyst carries three layers written from the numbers on every run (Observation: what the
# data shows; Interpretation: what it suggests; Confirmation required: what has to happen before it
# counts), plus the methodology behind it. Thresholds live in config; the prose only reports them.

BAND_LABELS = {
    "earnings": "Earnings-led",
    "stable": "Stable valuation",
    "expansion": "Multiple expansion",
    "significant": "Significant multiple expansion",
    "unknown": "Not comparable",
}
GRADE_TEXT = {"strong": "Strong", "confirming": "Confirming", "mixed": "Mixed", "weak": "Weak", "unknown": "Unknown"}


@dataclass(frozen=True)
class RateInputs:
    """What the rate-cut catalyst shows beside its status: fed funds and the Treasury path."""

    effr: float | None = None
    effr_date: str | None = None
    points: tuple[tuple[str, float, str], ...] = ()  # (horizon, Treasury par yield %, date), short to long
    gap_2y_prev: float | None = None  # EFFR minus the 2Y, `change_sessions` Treasury days earlier
    prev_date: str | None = None
    change_sessions: int = THRESHOLDS.cuts_change_sessions


def _day(iso: str | pd.Timestamp | None) -> str:
    """'Jun 30, 2026' (no %-d: it breaks on Windows)."""
    if iso is None:
        return "n/a"
    d = pd.Timestamp(str(iso))
    return f"{d:%b} {d.day}, {d.year}"


def _checked(fetched_at: str | None) -> str:
    if not fetched_at:
        return "filings not fetched yet"
    try:
        when = datetime.fromisoformat(fetched_at).astimezone(ZoneInfo("America/New_York"))
    except ValueError:
        return f"filings checked {fetched_at}"
    return f"filings checked {_day(when.date().isoformat())} {when:%H:%M} ET"


def _names(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _signed(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def _sum(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _fiscal_end(quarters: pd.Series, q: pd.Period) -> pd.Timestamp | None:
    """The fiscal quarter end a company's figure for calendar quarter `q` actually comes from."""
    ends = [pd.Timestamp(str(e)) for e in quarters.index if calendar_quarter(pd.Timestamp(str(e))) == q]
    return max(ends) if ends else None


def _close_near(price: pd.Series | None, when: pd.Timestamp, tolerance_days: int) -> tuple[str, float] | None:
    """The last close on or before `when`, if it is within `tolerance_days` (else the price is missing:
    a stale close would compare different periods)."""
    if price is None or price.empty:
        return None
    upto = price[price.index <= when].dropna()
    if upto.empty:
        return None
    day = pd.Timestamp(upto.index[-1])
    if (when - day).days > tolerance_days:
        return None
    return str(day.date()), float(upto.iloc[-1])


def implied_multiple_change(eps_growth_pct: float | None, price_growth_pct: float | None) -> float | None:
    """(1 + price growth) / (1 + EPS growth) - 1, in %, from growth rates in %.

    The change in trailing P/E the two growth rates imply: a proxy, not a quoted P/E, and valid only
    when both cover the same period. None when either is missing or a level went to zero or below
    (EPS turned negative, so there is no P/E to compare).
    """
    if eps_growth_pct is None or price_growth_pct is None:
        return None
    eps_ratio, price_ratio = 1.0 + eps_growth_pct / 100.0, 1.0 + price_growth_pct / 100.0
    if eps_ratio <= 0 or price_ratio <= 0:
        return None
    return (price_ratio / eps_ratio - 1.0) * 100.0


def valuation_band(change_pct: float | None, t: ValuationThresholds = VALUATION) -> str:
    """earnings (multiple contracted) / stable / expansion / significant / unknown."""
    if change_pct is None:
        return "unknown"
    if change_pct <= t.contraction_pct:
        return "earnings"
    if change_pct >= t.significant_pct:
        return "significant"
    if change_pct > t.expansion_pct:
        return "expansion"
    return "stable"


def _grade(led: int, total: int, t: ValuationThresholds) -> str:
    share = led / total
    for step, name in t.grades:
        if share >= step - 1e-9:
            return name
    return t.grades[-1][1]


def valuation_grade(led: int, unknown: int, total: int, t: ValuationThresholds = VALUATION) -> str:
    """The group grade from how many companies have EPS growth above price growth.

    Missing is unknown, never false: the grade is only given when it would be the same whichever way
    the unreadable companies went; otherwise it is "unknown".
    """
    if total <= 0:
        return "unknown"
    low, high = _grade(led, total, t), _grade(led + unknown, total, t)
    return low if low == high else "unknown"


def needed_for(grade: str, total: int, t: ValuationThresholds = VALUATION) -> int:
    """How many companies a grade needs (3 of 4 for "confirming")."""
    step = dict((name, s) for s, name in t.grades)[grade]
    return next(k for k in range(total + 1) if k / total >= step - 1e-9)


def earnings_vs_price(
    eps_by_company: dict[str, pd.Series],
    prices: dict[str, pd.Series],
    q: pd.Period,
    tickers: list[str] | None = None,
    t: ValuationThresholds = VALUATION,
) -> list[dict[str, Any]]:
    """One row per company: trailing-year diluted EPS growth vs share-price growth, year to `q`.

    EPS: the sum of four fiscal quarters mapped to calendar quarters `q-3..q`, vs `q-7..q-4`.
    Price: the last close on or before each calendar quarter end. A company that can't be compared
    stays in the list with `outgrowing` None and the reason, so a gap never shrinks the group.
    """
    out: list[dict[str, Any]] = []
    for ticker in tickers if tickers is not None else list(eps_by_company):
        row: dict[str, Any] = {
            "ticker": ticker,
            "eps_growth": None,
            "price_growth": None,
            "multiple_change": None,
            "outgrowing": None,
            "band": "unknown",
            "low_base": False,
            "reason": None,
        }
        out.append(row)
        eps = eps_by_company.get(ticker)
        if eps is None or eps.empty:
            row["reason"] = "no EPS filings read"
            continue
        cq_ttm = ttm(by_calendar_quarter(eps))
        if q not in cq_ttm.index or (q - 4) not in cq_ttm.index:
            row["reason"] = f"no trailing-year EPS for both {qlabel(q)} and {qlabel(q - 4)}"
            continue
        end_now, end_then = _fiscal_end(eps, q), _fiscal_end(eps, q - 4)
        row["fiscal_end"] = None if end_now is None else str(end_now.date())
        row["fiscal_end_then"] = None if end_then is None else str(end_then.date())
        if end_now is None or end_then is None or abs((end_now - end_then).days - 365) > t.fiscal_year_tolerance_days:
            row["reason"] = (
                f"fiscal quarters ending {_day(end_then)} and {_day(end_now)} aren't a year apart; "
                "the two years can't be compared"
            )
            continue
        eps_now, eps_then = _val(cq_ttm, q), _val(cq_ttm, q - 4)
        row["eps_now"], row["eps_then"] = round(eps_now, 4), round(eps_then, 4)
        price = prices.get(ticker)
        now_close = _close_near(price, q.end_time.normalize(), t.price_date_tolerance_days)
        then_close = _close_near(price, (q - 4).end_time.normalize(), t.price_date_tolerance_days)
        if now_close is None or then_close is None:
            missing = _day((q if now_close is None else q - 4).end_time.normalize())
            row["reason"] = f"no {ticker} close within {t.price_date_tolerance_days} days of {missing}"
            continue
        row["price_date"], row["price_now"] = now_close[0], round(now_close[1], 4)
        row["price_date_then"], row["price_then"] = then_close[0], round(then_close[1], 4)
        if eps_then <= 0:
            row["reason"] = "year-earlier EPS was zero or negative, so growth and the multiple aren't defined"
            continue
        eps_growth = (eps_now / eps_then - 1.0) * 100.0
        price_growth = (now_close[1] / then_close[1] - 1.0) * 100.0
        change = implied_multiple_change(eps_growth, price_growth)
        row.update(
            eps_growth=round(eps_growth, 4),
            price_growth=round(price_growth, 4),
            outgrowing=eps_growth > price_growth,
            multiple_change=None if change is None else round(change, 4),
            band=valuation_band(change, t),
            low_base=eps_now / eps_then >= t.low_base_eps_ratio,
        )
        if change is None:
            row["reason"] = "EPS turned negative, so there is no P/E to compare"
    return out


def _earnings_catalyst(
    rows: list[dict[str, Any]], q: pd.Period | None, fetched_at: str | None, t: ValuationThresholds = VALUATION
) -> Catalyst:
    label = "Semiconductor earnings vs. valuation"
    readable = [r for r in rows if r["outgrowing"] is not None]
    if q is None or not readable:
        reasons = "; ".join(f"{r['ticker']}: {r['reason']}" for r in rows if r.get("reason"))
        return Catalyst(
            key="earnings",
            label=label,
            status="unknown",
            status_text="Unknown",
            detail="Not enough earnings or price history.",
            observation="Not enough earnings or price history." + (f" {reasons}." if reasons else ""),
            rows=rows,
        )
    total = len(rows)
    led = [r for r in rows if r["outgrowing"] is True]
    unknown = [r for r in rows if r["outgrowing"] is None]
    grade = valuation_grade(len(led), len(unknown), total, t)
    ql = qlabel(q)

    def mult(r: dict[str, Any]) -> str:
        return "n/a" if r["multiple_change"] is None else f"{r['multiple_change']:+.1f}%"

    bars = []
    for r in rows:
        if r["eps_growth"] is not None and r["price_growth"] is not None:
            sub = f"EPS {r['eps_growth']:+.0f}% · price {r['price_growth']:+.0f}%"
            if r["reason"]:
                sub += f" · {r['reason']}"
        else:
            sub = str(r["reason"])
        tag = BAND_LABELS[r["band"]] + (" · low base" if r["low_base"] else "")
        tone = r["band"]
        if r["outgrowing"] is False and r["multiple_change"] is None:
            tag, tone = "EPS fell below zero", "significant"
        bars.append(
            CatalystBar(label=r["ticker"], value=r["multiple_change"], text=mult(r), tone=tone, tag=tag, sub=sub)
        )

    detail = f"{len(led)} of {total} over the year to {ql}: " + ", ".join(
        f"{r['ticker']} EPS {r['eps_growth']:+.0f}% vs price {r['price_growth']:+.0f}% (multiple {mult(r)})"
        if r["eps_growth"] is not None
        else f"{r['ticker']} n/a"
        for r in rows
    )

    # Observation: what the data shows.
    measured = [r for r in rows if r["multiple_change"] is not None]
    observation = (
        f"{len(led)} of {total} semiconductor companies grew trailing EPS faster than their share price "
        f"over the year to {ql}."
    )
    if len(measured) >= 2:
        lo = min(measured, key=lambda r: r["multiple_change"])
        hi = max(measured, key=lambda r: r["multiple_change"])
        observation += f" Implied multiples ranged from {mult(lo)} ({lo['ticker']}) to {mult(hi)} ({hi['ticker']})."
    for r in unknown:
        observation += f" {r['ticker']} can't be compared: {r['reason']}."

    # Interpretation: what it suggests. Only states what the bands and the score say.
    expanding = [r for r in measured if r["band"] in ("expansion", "significant")]
    significant = [r for r in measured if r["band"] == "significant"]
    shrinking = [r["ticker"] for r in measured if r["band"] == "earnings"]
    exp_text = _names([f"{r['ticker']} {mult(r)}" for r in expanding])
    if grade == "strong":
        interpretation = (
            "Earnings carried every company's price gain; valuations did not stretch anywhere in the group."
        )
    elif grade == "confirming":
        interpretation = "Earnings explain the gains at most of the group"
        interpretation += f"; valuation expansion is confined to {exp_text}." if expanding else "."
    elif grade == "mixed":
        moderate = _names([f"{r['ticker']} {mult(r)}" for r in expanding if r["band"] == "expansion"])
        if significant:
            sig_text = _names([f"{r['ticker']} {mult(r)}" for r in significant])
            interpretation = (
                "Semiconductor earnings are strong, but valuation expansion remains significant in parts of the "
                f"group ({sig_text})."
            )
            if moderate:
                interpretation += f" {moderate} expanded moderately."
        elif expanding:
            interpretation = (
                f"Semiconductor earnings are strong, but valuation expansion carries part of the gains ({exp_text})."
            )
        else:
            interpretation = "Earnings and prices split the group evenly, with no large valuation expansion."
    elif grade == "weak":
        interpretation = "Price outran earnings at most of the group" + (f" ({exp_text})" if expanding else "")
        interpretation += ": the gains rest more on paying higher multiples than on earnings."
    else:
        interpretation = "The group can't be graded until every company can be compared."
    if shrinking:
        interpretation += (
            f" Where the multiple shrank ({_names(shrinking)}), earnings did the work, or the market doubts they "
            "will last; this gauge can't tell which."
        )
    for r in measured:
        if r["low_base"]:
            interpretation += (
                f" {r['ticker']}'s EPS rose {r['eps_now'] / r['eps_then']:.1f}x from a low base, so its multiple "
                "reading says more about the starting year than about valuation."
            )
    memory = [r["ticker"] for r in rows if r["ticker"] in CHIPS_EX]
    if memory:
        interpretation += f" {_names(memory)} earnings swing with memory prices (cyclical)."

    # Confirmation required: the next grade's bar, and who is closest to moving.
    need = needed_for(t.confirming_grade, total, t)
    if grade in ("confirming", "strong"):
        holding = [r for r in led if r["multiple_change"] is not None]
        confirmation = (
            f"Met: {len(led)} of {total} (needs {need}). It holds while at least {need} keep EPS growth "
            "above price growth."
        )
        if holding:
            edge = max(holding, key=lambda r: r["multiple_change"])
            confirmation += f" Closest to flipping: {edge['ticker']} (multiple {mult(edge)})."
    else:
        confirmation = f"At least {need} of {total} companies with EPS growth above price growth (now {len(led)})."
        behind = [r for r in rows if r["outgrowing"] is False and r["multiple_change"] is not None]
        if behind:
            edge = min(behind, key=lambda r: r["multiple_change"])
            confirmation += (
                f" Closest: {edge['ticker']}, whose price ran {mult(edge)} ahead of its earnings in multiple terms."
            )
        if unknown:
            confirmation += f" {_names([r['ticker'] for r in unknown])} must also become comparable."
    confirmation += f" Next read: once every chip company has reported {qlabel(q + 1)}."

    price_days = sorted({(r["price_date_then"], r["price_date"]) for r in rows if r.get("price_date")})
    used = "; ".join(f"{_day(a)} and {_day(b)}" for a, b in price_days) or "none"
    ends = " · ".join(f"{r['ticker']} {_day(r['fiscal_end'])}" for r in rows if r.get("fiscal_end")) or "n/a"
    methodology = [
        CatalystNote(
            label="Measured",
            value=f"Trailing-12-month EPS vs share price, year to {ql}. Each fiscal quarter counts in the "
            "nearest calendar quarter.",
        ),
        CatalystNote(label="Latest fiscal quarter", value=ends),
        CatalystNote(
            label="EPS",
            value="GAAP diluted (EarningsPerShareDiluted), the sum of four fiscal quarters, from each "
            "company's own 10-Q/10-K at SEC EDGAR. No adjusted EPS is read or mixed in.",
        ),
        CatalystNote(
            label="Price",
            value=f"Alpaca consolidated-tape closes, split-adjusted: the last close on or before each calendar "
            f"quarter end, within {t.price_date_tolerance_days} days. Closes used: {used}.",
        ),
        CatalystNote(
            label="Formula",
            value="Implied multiple change = (1 + price growth) / (1 + EPS growth) - 1. A proxy for the change "
            "in trailing P/E, not a quoted P/E.",
        ),
        CatalystNote(
            label="Bands",
            value=f"≤ {t.contraction_pct:+.0f}% earnings-led · up to {t.expansion_pct:+.0f}% stable · up to "
            f"{t.significant_pct:+.0f}% expansion · {t.significant_pct:+.0f}% or more significant. The score "
            "counts direction only (EPS growth above price growth). Grades: all = strong, "
            f"{need} of {total} = confirming, half = mixed, fewer = weak. Uncalibrated.",
        ),
        CatalystNote(
            label="Limits",
            value="A price can sit up to about five weeks before the fiscal quarter it is compared with. Each "
            "trailing year includes one fiscal Q4 worked out as full year minus nine months (share counts "
            f"differ, so approximate). EPS up {t.low_base_eps_ratio:.0f}x or more is flagged as a low base; "
            "normalized mid-cycle earnings are not estimated.",
        ),
        CatalystNote(label="Updated", value=f"{_checked(fetched_at)}; prices from this run."),
    ]
    return Catalyst(
        key="earnings",
        label=label,
        status=grade,
        status_text=GRADE_TEXT[grade],
        detail=detail + ".",
        headline=f"{len(led)} of {total}",
        headline_note=f"earnings ahead of price, year to {ql}"
        + (f" ({len(unknown)} not comparable)" if unknown else ""),
        bars=bars,
        bars_note="Implied change in trailing P/E. Left of zero: earnings grew faster than the price.",
        observation=observation,
        interpretation=interpretation,
        confirmation=confirmation,
        methodology=methodology,
        rows=rows,
    )


def _software_catalyst(
    r: Readings, q: pd.Period | None, members: dict[str, pd.Series], fetched_at: str | None, t: CycleThresholds = CYCLE
) -> Catalyst:
    label = "Software revenue accelerating"
    s = None if q is None else r.at(r.software, q)
    a = None if q is None else r.at(r.software_accel, q)
    if q is None or s is None or a is None:
        return Catalyst(
            key="software",
            label=label,
            status="unknown",
            detail="Not enough software filings yet.",
            observation="Not enough software filings yet.",
        )
    ap = r.at(r.software_accel, q - 1)
    up, n = r.software_breadth.get(q, (0, 0))
    monet = next(c for c in phase_checks(r, q, t) if c.phase_id == 3)
    # Two different numbers, which disagreed on the first live reads: the median company's
    # growth RATE (15.2% -> 14.2% -> 13.6%) fell while the median CHANGE in each company's
    # growth was positive (+0.64, +0.36), because one large slowdown (Intuit) crossed the
    # middle. "Starting" needs both to point up; otherwise the read is mixed and says why.
    s_prev = r.at(r.software, q - 1)
    level = None if s_prev is None else s - s_prev
    if monet.met is True:
        status, verdict = "yes", "Yes: two quarters running, material and broad"
    elif a > 0 and (level is None or level >= 0):
        status, verdict = "starting", "Starting: the typical company is accelerating, not yet decisively"
    elif a > 0:
        status, verdict = "mixed", "Mixed: most companies are edging up, but the group's median growth rate fell"
    else:
        status, verdict = "not yet", "Not yet: the typical company is slowing"
    was = "" if s_prev is None else f" (was {_pct(s_prev, 1)})"
    ql = qlabel(q)

    rows: list[dict[str, Any]] = []
    bars = []
    for ticker, series in members.items():
        g = company_growth(series)
        now = None if q not in g.index else _val(g, q)
        prev = None if (q - 1) not in g.index else _val(g, q - 1)
        change = None if now is None or prev is None else now - prev
        end = _fiscal_end(series, q)
        rows.append(
            {
                "ticker": ticker,
                "growth": None if now is None else round(now, 4),
                "growth_prev": None if prev is None else round(prev, 4),
                "change": None if change is None else round(change, 4),
                "fiscal_end": None if end is None else str(end.date()),
            }
        )
        bars.append(
            CatalystBar(
                label=ticker,
                value=change,
                text="n/a" if change is None else f"{change:+.{2 if abs(change) < 0.1 else 1}f} pts",
                tone="unknown" if change is None else ("up" if change > 0 else "down"),
                tag=None if change is None else ("Accelerating" if change > 0 else "Slowing"),
                sub=None if now is None else f"growth {now:.1f}%" + ("" if prev is None else f" (was {prev:.1f}%)"),
            )
        )

    observation = (
        f"Group: the median company's revenue growth is {_pct(s, 1)}{was}"
        + ("" if level is None else f", {level:+.1f} pts")
        + f". Companies: the typical company's own growth changed {_pts(ap)} then {_pts(a)}, and {up} of {n} "
        f"are accelerating ({ql})."
    )
    interpretation = {
        "yes": "Acceleration is broad and has lasted two quarters: AI spending is showing up as software revenue.",
        "starting": "The typical company is accelerating and the group's growth rate isn't falling, but it isn't "
        "decisive yet.",
        "mixed": "Company-level acceleration hasn't lifted the group yet: the median growth rate fell while the "
        "typical company's own growth rose. That split happens when the companies speeding up sit below "
        "the middle, or one big slowdown moves the median.",
        "not yet": "The typical company is slowing: no sign yet of AI spending turning into software revenue.",
    }[status]

    def mark(met: bool | None) -> str:
        return "met" if met is True else ("not met" if met is False else "unknown")

    conds = monet.conditions
    confirmation = (
        f"Monetization needs all three: median growth ≥ {t.monetization_software_growth:.0f}% (now {_pct(s, 1)}, "
        f"{mark(conds[0].met)}); the typical company's growth up in each of the last two quarters, totalling ≥ "
        f"{t.monetization_accel_points:.1f} pt (now {_signed(ap)} then {_signed(a)}, total {_signed(_sum(a, ap))}, "
        f"{mark(conds[1].met)}); "
        f"at least half accelerating (now {up} of {n}, {mark(conds[2].met)})."
    )
    if status in ("starting", "mixed"):
        confirmation += " For this card to read better than mixed, the group's median growth rate must not fall."
    ends = " · ".join(f"{x['ticker']} {_day(x['fiscal_end'])}" for x in rows if x["fiscal_end"]) or "n/a"
    methodology = [
        CatalystNote(
            label="Measured",
            value=f"Revenue, trailing 12 months vs a year earlier, by calendar quarter, to {ql}. Group = the median "
            "company, not summed dollars. Acceleration = a company's growth minus its own growth a quarter earlier.",
        ),
        CatalystNote(label="Latest fiscal quarter", value=ends),
        CatalystNote(
            label="Source",
            value="Each company's own 10-Q/10-K revenue at SEC EDGAR (XBRL). Oracle counts as a capex spender, "
            "not software.",
        ),
        CatalystNote(label="Updated", value=_checked(fetched_at) + "."),
    ]
    return Catalyst(
        key="software",
        label=label,
        status=status,
        detail=f"{verdict}. Median growth rate {_pct(s, 1)}{was}; the typical company's own growth "
        f"changed {_pts(ap)} then {_pts(a)}; {up} of {n} companies accelerating ({ql}).",
        headline=_pct(s, 1),
        headline_note=f"median company's revenue growth, trailing year to {ql}{was} · {up} of {n} accelerating",
        bars=bars,
        bars_note="Each company's own change in growth vs the prior quarter. The headline is the group.",
        observation=observation,
        interpretation=interpretation,
        confirmation=confirmation,
        methodology=methodology,
        rows=rows,
    )


def _rates_catalyst(effr_minus_bill: float | None, effr_minus_2y: float | None, rates: RateInputs | None) -> Catalyst:
    label = "Monetary easing expectations"
    rates = rates or RateInputs()
    effr_text = "fed funds" if rates.effr is None else f"fed funds (EFFR {rates.effr:.2f}%, {_day(rates.effr_date)})"
    bars = [
        CatalystBar(
            label=h,
            value=y - rates.effr,
            text=f"{y - rates.effr:+.2f}",
            tone="neutral",
            sub=f"{y:.2f}% on {_day(d)}",
        )
        for h, y, d in rates.points
        if rates.effr is not None
    ]
    methodology = [
        CatalystNote(
            label="Source",
            value="U.S. Treasury daily par yield curve (3M, 6M, 1Y, 2Y) and the NY Fed's effective fed funds "
            "rate (EFFR).",
        ),
        CatalystNote(
            label="Measured",
            value="Each yield minus EFFR, latest daily values. Treasury posts in the evening, so the close run "
            "carries the prior day's curve.",
        ),
        CatalystNote(
            label="Status rule",
            value=f"Cuts priced when the 2Y is more than {THRESHOLDS.cuts_2y_spread_pct:.2f} pts below EFFR (hikes "
            f"when above); the 3M bill at ±{THRESHOLDS.cuts_priced_spread_pct:.2f} if the 2Y is missing. Uncalibrated.",
        ),
        CatalystNote(
            label="Not read",
            value="Fed funds futures and OIS: there is no official, free record to verify them against.",
        ),
        CatalystNote(
            label="Limits",
            value="A yield is a proxy for the expected policy path: it carries a term premium and bill-supply "
            "effects. The 3M is cross-checked; the 6M, 1Y and 2Y are official but not cross-checked. EFFR "
            "posts the next morning and moves only at FOMC decisions.",
        ),
    ]
    common: dict[str, Any] = {
        "key": "cuts",
        "label": label,
        "bars": bars,
        "bars_note": f"Treasury yield minus {effr_text}. Below zero: markets expect lower policy rates over "
        "that horizon."
        if bars
        else None,
        "methodology": methodology,
        "rows": [
            {"horizon": h, "yield": y, "date": d, "gap": None if rates.effr is None else round(y - rates.effr, 4)}
            for h, y, d in rates.points
        ],
    }
    policy = "This is market pricing, not policy: an actual cut shows up only as EFFR falling" + (
        "." if rates.effr is None else f" from {rates.effr:.2f}%."
    )
    if effr_minus_2y is not None:
        # The 2Y averages the expected policy rate over two years; the 3-month bill only spans the
        # next meeting or two and moves with bill supply. Used whenever Treasury's 2Y is in.
        limit = THRESHOLDS.cuts_2y_spread_pct
        gap = -effr_minus_2y
        if gap < -limit:
            status, text, detail = "yes", "Cuts priced", f"Yes: the 2-year yield is {abs(gap):.2f} pts below fed funds."
            meaning = (
                f"Markets price easing: the 2-year yield implies a policy rate averaging roughly {abs(gap):.2f} pts "
                "below today's over two years, term premium included."
            )
        elif gap > limit:
            status, text = "not yet", "Hikes priced"
            detail = f"Not yet: the 2-year yield is {gap:+.2f} pts vs fed funds, so hikes are priced."
            meaning = "Markets price tighter policy: the 2-year yield sits above fed funds."
        else:
            status, text = "not yet", "No clear move"
            detail = f"Not yet: the 2-year yield is {gap:+.2f} pts vs fed funds, no clear move priced."
            meaning = "No clear move priced over two years: the 2-year yield sits near fed funds."
        observation = f"The 2-year Treasury yields {gap:+.2f} pts vs {effr_text}."
        shorter = [f"{b.label} {b.text}" for b in bars if b.label != "2Y"]
        if shorter:
            observation += f" Shorter horizons: {', '.join(shorter)}."
        if rates.gap_2y_prev is not None:
            prev = -rates.gap_2y_prev
            observation += (
                f" {rates.change_sessions} Treasury days earlier ({_day(rates.prev_date)}) the 2-year gap was "
                f"{prev:+.2f}."
            )
            shift = gap - prev
            if abs(shift) >= 0.05:
                meaning += (
                    f" Over the last {rates.change_sessions} Treasury days markets priced "
                    f"{'more' if shift < 0 else 'less'} easing (2-year gap {shift:+.2f} pts)."
                )
        if status == "yes":
            confirmation = (
                f"Market side met: the 2Y is more than {limit:.2f} pts below fed funds. Policy side: an FOMC cut, "
                "seen as EFFR falling."
            )
        else:
            confirmation = (
                f"Market side: the 2-year yield more than {limit:.2f} pts below fed funds (now {gap:+.2f}). "
                "Policy side: an FOMC cut, seen as EFFR falling."
            )
        return Catalyst(
            status=status,
            status_text=text,
            detail=detail,
            headline=f"{gap:+.2f} pts",
            headline_note="2-year Treasury yield vs fed funds",
            observation=observation,
            interpretation=f"{meaning} {policy}",
            confirmation=confirmation,
            **common,
        )
    if effr_minus_bill is None:
        return Catalyst(
            status="unknown",
            status_text="Unknown",
            detail="Rates data unavailable.",
            observation="Rates data unavailable.",
            **common,
        )
    priced = effr_minus_bill > THRESHOLDS.cuts_priced_spread_pct
    gap = -effr_minus_bill
    hikes = gap > THRESHOLDS.cuts_priced_spread_pct
    detail = (
        f"Yes: the 3-month bill is {abs(gap):.2f} pts below fed funds."
        if priced
        else f"Not yet: the 3-month bill is {gap:+.2f} pts vs fed funds{', so hikes are priced' if hikes else ''}."
    )
    return Catalyst(
        status="yes" if priced else "not yet",
        status_text="Cuts priced" if priced else ("Hikes priced" if hikes else "No clear move"),
        detail=detail,
        headline=f"{gap:+.2f} pts",
        headline_note="3-month bill vs fed funds (the 2-year yield is missing)",
        observation=f"The 3-month bill yields {gap:+.2f} pts vs {effr_text}. The 2-year yield is missing this run.",
        interpretation="The 3-month bill only spans the next meeting or two and moves with bill supply. " + policy,
        confirmation=f"Market side: the 3-month bill more than {THRESHOLDS.cuts_priced_spread_pct:.2f} pts below "
        "fed funds; the 2-year read returns when Treasury's 2Y is in. Policy side: an FOMC cut, seen as EFFR falling.",
        **common,
    )


def catalysts(
    r: Readings,
    q: pd.Period | None,
    evp: list[dict[str, Any]],
    effr_minus_bill: float | None,
    effr_minus_2y: float | None = None,
    *,
    software: dict[str, pd.Series] | None = None,
    rates: RateInputs | None = None,
    fetched_at: str | None = None,
) -> list[Catalyst]:
    return [
        _software_catalyst(r, q, software or {}, fetched_at),
        _earnings_catalyst(evp, q, fetched_at),
        _rates_catalyst(effr_minus_bill, effr_minus_2y, rates),
    ]


# --- Assembling the cycle ---------------------------------------------------------------------------


def build_cycle(
    cache: SecCache | None,
    prices: dict[str, pd.Series],
    effr_minus_bill: float | None,
    effr_minus_2y: float | None = None,
    rates: RateInputs | None = None,
) -> Cycle:
    companies = (cache or {}).get("companies") or {}
    filings: list[CompanyFiling] = []
    groups: dict[str, dict[str, pd.Series]] = {}
    eps: dict[str, pd.Series] = {}
    ocf: dict[str, pd.Series] = {}
    notes: list[str] = [str(n) for n in (cache or {}).get("notes") or []]
    for group, members in CYCLE_COMPANIES.items():
        metric = CYCLE_GROUP_METRIC[group]
        groups[group] = {}
        for ticker, _, _ in members:
            slot = (companies.get(ticker) or {}).get(metric) or {}
            if "facts" not in slot:
                filings.append(
                    CompanyFiling(ticker=ticker, group=group, ok=False, error=slot.get("error", "not fetched yet"))
                )
                continue
            series = quarterize(slot["facts"])
            latest_filed = max((str(f.get("filed") or "") for f in slot["facts"]), default="") or None
            filings.append(
                CompanyFiling(
                    ticker=ticker,
                    group=group,
                    ok=not series.empty,
                    tag=slot.get("tag"),
                    latest_period_end=None if series.empty else str(series.index[-1].date()),
                    filed=latest_filed,
                    error=None if not series.empty else "no usable quarters",
                )
            )
            if not series.empty:
                groups[group][ticker] = series
            if group == "chips":
                eps_slot = (companies.get(ticker) or {}).get("eps") or {}
                if "facts" in eps_slot:
                    eps[ticker] = quarterize(eps_slot["facts"])
            if group == FUNDING_GROUP:
                ocf_slot = (companies.get(ticker) or {}).get("ocf") or {}
                if "facts" in ocf_slot:
                    ocf_series = quarterize(ocf_slot["facts"])
                    if not ocf_series.empty:
                        ocf[ticker] = ocf_series

    complete = {g: len(groups[g]) == len(CYCLE_COMPANIES[g]) for g in groups}
    for g, ok in complete.items():
        if not ok:
            missing = [f.ticker for f in filings if f.group == g and not f.ok]
            notes.append(f"{GROUP_LABELS[g]}: missing {', '.join(missing)}; that group can't be read")

    def growth(group: str) -> pd.Series:
        if not complete[group]:
            return pd.Series(dtype=float)
        return group_growth(groups[group], CYCLE_GROUP_HOW[group])

    r = Readings(
        capex=growth("hyperscalers"),
        chips=growth("chips"),
        software=growth("software"),
        software_accel=median_acceleration(groups["software"]) if complete["software"] else pd.Series(dtype=float),
        software_breadth=breadth(groups["software"]) if complete["software"] else {},
    )
    common = sorted(set(r.capex.index) & set(r.chips.index) & set(r.software.index))
    latest = common[-1] if common else None

    history: list[dict[str, Any]] = []
    for q in common[-CYCLE_HISTORY_QUARTERS:]:
        qc = phase_checks(r, q)
        pid = phase_of(qc)
        if pid is not None:
            history.append({"quarter": qlabel(q), "phase_id": pid, "phases_met": phases_met(qc)})
    checks = phase_checks(r, latest) if latest is not None else []
    phase_id = phase_of(checks) if checks else None

    since, run = None, 0
    if phase_id is not None:
        for row in reversed(history):
            if row["phase_id"] != phase_id:
                break
            since, run = row["quarter"], run + 1
    chip_tickers = [t for t, _, _ in CYCLE_COMPANIES["chips"]]
    evp = earnings_vs_price(eps, prices, latest, chip_tickers) if latest is not None else []

    def single(group: str, members: dict[str, pd.Series]) -> pd.Series:
        return single_quarter_growth(members, CYCLE_GROUP_HOW[group])

    def make_metric(
        key: str, label: str, members: list[str], series: pd.Series, one: pd.Series, kind: str
    ) -> CycleMetric:
        tail = series.iloc[-CYCLE_HISTORY_QUARTERS:]
        newest = None if series.empty else pd.Period(str(series.index[-1]), freq="Q")
        return CycleMetric(
            key=key,
            label=label,
            members=members,
            kind=kind,
            latest=None if latest is None or latest not in series.index else round(_val(series, latest), 2),
            previous=None if latest is None or (latest - 1) not in series.index else round(_val(series, latest - 1), 2),
            through=qlabel(newest),
            newest=None if newest is None else round(_val(series, newest), 2),
            single_quarter=None if newest is None or newest not in one.index else round(_val(one, newest), 2),
            history=[
                QuarterValue(quarter=qlabel(pd.Period(str(q), freq="Q")) or "", value=round(float(v), 2))
                for q, v in tail.items()
            ],
        )

    empty = pd.Series(dtype=float)
    metrics = []
    for group, series in (("hyperscalers", r.capex), ("chips", r.chips), ("software", r.software)):
        one = single(group, groups[group]) if complete[group] else empty
        metrics.append(
            make_metric(group, GROUP_LABELS[group], [t for t, _, _ in CYCLE_COMPANIES[group]], series, one, "growth")
        )
        if group == "chips" and complete[group]:
            rest = {t: v for t, v in groups[group].items() if t not in CHIPS_EX}
            if rest and len(rest) < len(groups[group]):
                metrics.append(
                    make_metric(
                        "chips_ex",
                        f"Chip revenue without {', '.join(CHIPS_EX)}",
                        list(rest),
                        group_growth(rest, "sum"),
                        single_quarter_growth(rest, "sum"),
                        "growth",
                    )
                )
    if complete[FUNDING_GROUP]:
        if set(ocf) == set(groups[FUNDING_GROUP]):
            metrics.append(
                make_metric(
                    "funding",
                    "Capex as a share of operating cash flow",
                    list(ocf),
                    funding_ratio(groups[FUNDING_GROUP], ocf),
                    empty,
                    "level",
                )
            )
        elif any("ocf" in (companies.get(t) or {}) for t in groups[FUNDING_GROUP]):
            lacking = sorted(set(groups[FUNDING_GROUP]) - set(ocf))
            notes.append(f"Funding gauge: operating cash flow not read yet for {', '.join(lacking)}")

    # Which filings the phase is waiting for: a group's own series can run a quarter ahead of the
    # quarter every group has (Nov 2026: capex in through Q3, the phase on Q2 until NVDA and AVGO).
    waiting: list[str] = []
    if latest is not None:
        nxt = latest + 1
        for group in ("hyperscalers", "chips", "software"):
            for ticker, series in groups[group].items():
                if nxt not in by_calendar_quarter(series).index:
                    waiting.append(ticker)
        ahead = [
            GROUP_LABELS[g]
            for g, sr in (("hyperscalers", r.capex), ("chips", r.chips), ("software", r.software))
            if nxt in sr.index
        ]
        if ahead and waiting:
            notes.append(
                f"{', '.join(ahead)} already reported through {qlabel(nxt)}. The phase stays on {qlabel(latest)} "
                f"until {', '.join(waiting)} report."
            )

    met = phases_met(checks)
    pid = phase_id if phase_id is not None else 0
    summary = PHASE_SUMMARIES[pid] if phase_id is not None else "Not enough filings to read the cycle."
    if phase_id == 3 and 1 in met:
        summary += " Hardware spending is still accelerating too: monetization is arriving during the buildout."
    elif phase_id == 3 and 2 in met:
        summary += " Hardware growth has rolled over: digestion and monetization are overlapping."
    return Cycle(
        phase_id=pid,
        phase=PHASE_NAMES[pid] if phase_id is not None else "Unknown",
        known=phase_id is not None,
        phases_met=met,
        since=since,
        confirmed=run >= CYCLE.confirm_quarters,
        quarters_in_phase=run,
        through=qlabel(latest),
        summary=summary,
        checks=checks,
        metrics=metrics,
        history=history,
        catalysts=catalysts(
            r,
            latest,
            evp,
            effr_minus_bill,
            effr_minus_2y,
            software=groups["software"] if complete["software"] else {},
            rates=rates,
            fetched_at=(cache or {}).get("fetched_at"),
        ),
        waiting_on=waiting,
        filings=filings,
        fetched_at=(cache or {}).get("fetched_at"),
        notes=notes,
    )
