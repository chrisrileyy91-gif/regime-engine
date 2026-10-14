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

import httpx
import pandas as pd

from app.config import (
    CYCLE,
    CYCLE_COMPANIES,
    CYCLE_GROUP_HOW,
    CYCLE_GROUP_METRIC,
    CYCLE_HISTORY_QUARTERS,
    CYCLE_TAGS,
    SEC_CONCEPT_URL,
    SEC_CONCURRENCY,
    SEC_DEFAULT_USER_AGENT,
    SEC_REFRESH_HOURS,
    THRESHOLDS,
    CycleThresholds,
)
from app.data import get_with_retry, redact
from app.models import Catalyst, CompanyFiling, Cycle, CycleCondition, CycleMetric, CyclePhaseCheck, QuarterValue

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
    if fetched and now - datetime.fromisoformat(fetched) < timedelta(hours=SEC_REFRESH_HOURS):
        return []
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


# --- Earnings vs price, catalysts ------------------------------------------------------------------


def earnings_vs_price(
    eps_by_company: dict[str, pd.Series], prices: dict[str, pd.Series], q: pd.Period
) -> list[dict[str, Any]]:
    out = []
    for ticker, eps in eps_by_company.items():
        cq_ttm = ttm(by_calendar_quarter(eps))
        price = prices.get(ticker)
        if q not in cq_ttm.index or (q - 4) not in cq_ttm.index or price is None or price.empty:
            continue
        before_eps = _val(cq_ttm, q - 4)
        now_px = price[price.index <= q.end_time.normalize()]
        then_px = price[price.index <= (q - 4).end_time.normalize()]
        if before_eps <= 0 or now_px.empty or then_px.empty:
            continue
        eps_growth = (_val(cq_ttm, q) / before_eps - 1.0) * 100.0
        price_growth = (float(now_px.iloc[-1]) / float(then_px.iloc[-1]) - 1.0) * 100.0
        out.append(
            {
                "ticker": ticker,
                "eps_growth": eps_growth,
                "price_growth": price_growth,
                "outgrowing": eps_growth > price_growth,
            }
        )
    return out


def catalysts(
    r: Readings, q: pd.Period | None, evp: list[dict[str, Any]], effr_minus_bill: float | None
) -> list[Catalyst]:
    out = []
    s = None if q is None else r.at(r.software, q)
    a = None if q is None else r.at(r.software_accel, q)
    ap = None if q is None else r.at(r.software_accel, q - 1)
    if q is None or s is None or a is None:
        out.append(
            Catalyst(
                key="software",
                label="Software revenue accelerating",
                status="unknown",
                detail="Not enough software filings yet.",
            )
        )
    else:
        up, n = r.software_breadth.get(q, (0, 0))
        monet = next(c for c in phase_checks(r, q) if c.phase_id == 3)
        if monet.met is True:
            status, verdict = "yes", "Yes: two quarters running, material and broad"
        elif a > 0:
            status, verdict = "starting", "Starting: the typical company is accelerating, not yet decisively"
        else:
            status, verdict = "not yet", "Not yet: the typical company is slowing"
        out.append(
            Catalyst(
                key="software",
                label="Software revenue accelerating",
                status=status,
                detail=f"{verdict}. Median company growth {_pct(s, 1)}, change {_pts(ap)} then {_pts(a)}; "
                f"{up} of {n} companies accelerating ({qlabel(q)}).",
            )
        )
    if not evp:
        out.append(
            Catalyst(
                key="earnings",
                label="Chip earnings outgrowing chip prices",
                status="unknown",
                detail="Not enough earnings or price history.",
            )
        )
    else:
        winners = [e for e in evp if e["outgrowing"]]
        status = "yes" if len(winners) * 2 > len(evp) else "not yet"
        parts = ", ".join(f"{e['ticker']} EPS {e['eps_growth']:+.0f}% vs price {e['price_growth']:+.0f}%" for e in evp)
        out.append(
            Catalyst(
                key="earnings",
                label="Chip earnings outgrowing chip prices",
                status=status,
                detail=f"{len(winners)} of {len(evp)} over the year to {qlabel(q)}: {parts}.",
            )
        )
    if effr_minus_bill is None:
        out.append(Catalyst(key="cuts", label="Rate cuts priced", status="unknown", detail="Rates data unavailable."))
    else:
        priced = effr_minus_bill > THRESHOLDS.cuts_priced_spread_pct
        gap = -effr_minus_bill
        detail = (
            f"Yes: the 3-month bill is {abs(gap):.2f} pts below fed funds."
            if priced
            else f"Not yet: the 3-month bill is {gap:+.2f} pts vs fed funds"
            f"{', so hikes are priced' if gap > THRESHOLDS.cuts_priced_spread_pct else ''}."
        )
        out.append(Catalyst(key="cuts", label="Rate cuts priced", status="yes" if priced else "not yet", detail=detail))
    return out


# --- Assembling the cycle ---------------------------------------------------------------------------


def build_cycle(cache: SecCache | None, prices: dict[str, pd.Series], effr_minus_bill: float | None) -> Cycle:
    companies = (cache or {}).get("companies") or {}
    filings: list[CompanyFiling] = []
    groups: dict[str, dict[str, pd.Series]] = {}
    eps: dict[str, pd.Series] = {}
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
    evp = earnings_vs_price(eps, prices, latest) if latest is not None else []

    metrics = []
    for group, series in (("hyperscalers", r.capex), ("chips", r.chips), ("software", r.software)):
        tail = series.iloc[-CYCLE_HISTORY_QUARTERS:]
        metrics.append(
            CycleMetric(
                key=group,
                label=GROUP_LABELS[group],
                members=[t for t, _, _ in CYCLE_COMPANIES[group]],
                latest=None if latest is None or latest not in series.index else round(_val(series, latest), 2),
                previous=None
                if latest is None or (latest - 1) not in series.index
                else round(_val(series, latest - 1), 2),
                history=[
                    QuarterValue(quarter=qlabel(pd.Period(str(q), freq="Q")) or "", value=round(float(v), 2))
                    for q, v in tail.items()
                ],
            )
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
        catalysts=catalysts(r, latest, evp, effr_minus_bill),
        filings=filings,
        fetched_at=(cache or {}).get("fetched_at"),
        notes=notes,
    )
