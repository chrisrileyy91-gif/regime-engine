"""The fast layer: today's stress, the scapegoat tally, and the overnight line.

The cycle (fundamentals.py) moves quarterly. This module answers the daily questions:
  * How much pressure is the market under right now?  (stress: Calm / Elevated / Stressed)
  * When semis sell off, what is actually moving?     (tally of recent big down days)
  * What happened overnight?                           (Korea, dollar)

Stress only counts things that moved, never slow levels: a 10Y sitting at a high percentile
for months is backdrop, and must not keep the page in "high risk" for months.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from app.config import STRESSED_VIX, TALLY_DAYS, TALLY_DROP_PCT, THRESHOLDS, Thresholds
from app.drivers import day_label
from app.features import MarketFeatures
from app.models import Driver, OvernightItem, Stress, Tally, TallyDay, Today
from app.verify import Verified

STRESS_LABELS = {"CALM": "Calm", "ELEVATED": "Elevated", "STRESSED": "Stressed", "UNKNOWN": "Unknown"}
TALLY_LABELS = {
    "rates": "yields jumped (10Y or 30Y, at the close or intraday)",
    "fear": "VIX spiked",
    "market": "the broad market fell (equal-weight S&P 500 down 1%+)",
    "rotation": "no macro gauge moved, and software rose: money moved from chips to software",
    "chips": "no macro gauge moved, and software fell too",
    "unknown": "couldn't tell (a gauge was missing)",
}
TALLY_ORDER = ("rates", "fear", "market", "rotation", "chips", "unknown")
TALLY_SHORT = {
    "rates": "yields jumped",
    "fear": "VIX spiked",
    "market": "broad market fell",
    "rotation": "rotation into software",
    "chips": "chips alone",
    "unknown": "couldn't tell",
}


def stress(f: MarketFeatures, drivers: list[Driver]) -> Stress:
    active = [d for d in drivers if d.state == "ACTIVE"]
    unknown = [d for d in drivers if d.state == "UNKNOWN"]
    reasons = [d.name for d in active]
    if f.vix is not None and f.vix > STRESSED_VIX:
        level = "STRESSED"
        reasons.insert(0, f"VIX above {STRESSED_VIX:.0f}")
    elif len(active) >= 2:
        level = "STRESSED"
    elif len(active) == 1:
        level = "ELEVATED"
    elif len(unknown) == len(drivers):
        level = "UNKNOWN"
    else:
        level = "CALM"
    if unknown and level != "UNKNOWN":
        reasons.append(f"couldn't check {', '.join(d.name.lower() for d in unknown)}")
    return Stress(level=level, label=STRESS_LABELS[level], reasons=reasons)


def _pct_changes(series: pd.Series | None) -> pd.Series:
    if series is None or len(series) < 2:
        return pd.Series(dtype=float)
    return (series.astype(float).pct_change() * 100.0).dropna()


def _bps_changes(series: pd.Series | None) -> pd.Series:
    if series is None or len(series) < 2:
        return pd.Series(dtype=float)
    return (series.astype(float).diff() * 100.0).dropna()


def _zscores(changes: pd.Series, window: int) -> pd.Series:
    """Each change vs the `window` changes before it (no look-ahead)."""
    baseline = changes.shift(1).rolling(window, min_periods=window)
    return ((changes - baseline.mean()) / baseline.std(ddof=1)).dropna()


def _at(d: dict[Any, Any], ts: Any) -> float | None:
    v = d.get(ts)
    return None if v is None or pd.isna(v) else float(v)


def tally(
    verified: dict[str, Verified],
    extras: dict[str, pd.Series] | None = None,
    t: Thresholds = THRESHOLDS,
) -> Tally:
    """Tag each recent semis drop by what else moved that day, checked in order:

    1. rates   10Y or 30Y up >= 10 bps at the close, or at its intraday high vs the prior close,
               or an unusual 10Y close move (z >= 1.5). Intraday matters: on CPI day (Oct 14) the
               10Y rose 11 bps by mid-session and closed down 3.
    2. fear    VIX up >= 10% at the close.
    3. market  equal-weight S&P 500 (RSP) down >= 1%.
    4. rotation / chips   none of the above. Software up = rotation; software down = chips.

    The last two are what is LEFT when no gauge moved. They do not prove there was no macro news:
    oil, the dollar, policy headlines and company news are not measured. A day is "unknown" unless
    every gauge that could have claimed it was available, and the newest day stays provisional
    until all of its official closes have posted.
    """
    extras = extras or {}
    soxx = _pct_changes(verified["soxx"].series if "soxx" in verified else None)
    if soxx.empty:
        return Tally(summary="Semis prices unavailable, so the tally can't be built.")
    y10_close = verified["y10"].series if "y10" in verified else None
    y10 = _bps_changes(y10_close)
    y10_z = _zscores(y10, t.tnx_delta_z_window) if not y10.empty else pd.Series(dtype=float)
    y10_spike = _spike_bps(y10_close, extras.get("y10_high"))
    y30 = _bps_changes(extras.get("y30"))
    y30_spike = _spike_bps(extras.get("y30"), extras.get("y30_high"))
    vix = _pct_changes(verified["vix"].series if "vix" in verified else None)
    igv = _pct_changes(verified["igv"].series if "igv" in verified else None)
    mkt = _pct_changes(extras.get("market"))
    maps = {
        k: dict(v.items())
        for k, v in (
            ("y10", y10),
            ("z", y10_z),
            ("y10_spike", y10_spike),
            ("y30", y30),
            ("y30_spike", y30_spike),
            ("vix", vix),
            ("igv", igv),
            ("mkt", mkt),
        )
    }

    drops = soxx[soxx <= TALLY_DROP_PCT].iloc[-TALLY_DAYS:]
    newest = soxx.index[-1]
    days = []
    for ts, move in drops.items():
        v = {k: _at(m, ts) for k, m in maps.items()}
        rate_tests = [
            None if v["y10"] is None else v["y10"] >= t.tnx_1d_spike_bps,
            None if v["z"] is None else v["z"] >= t.tnx_delta_z,
            None if v["y10_spike"] is None else v["y10_spike"] >= t.tnx_1d_spike_bps,
            None if v["y30"] is None else v["y30"] >= t.tnx_1d_spike_bps,
            None if v["y30_spike"] is None else v["y30_spike"] >= t.tnx_1d_spike_bps,
        ]
        rates = True if any(x is True for x in rate_tests) else (False if all(x is False for x in rate_tests) else None)
        fear = None if v["vix"] is None else v["vix"] >= t.vix_jump_pct
        broad = None if v["mkt"] is None else v["mkt"] <= t.market_drop_pct
        if rates:
            driver = "rates"
        elif rates is False and fear:
            driver = "fear"
        elif rates is False and fear is False and broad:
            driver = "market"
        elif rates is False and fear is False and broad is False and v["igv"] is not None:
            driver = "rotation" if v["igv"] > 0 else "chips"
        else:
            driver = "unknown"
        missing = [
            name
            for name, key in (
                ("10Y close", "y10"),
                ("10Y intraday high", "y10_spike"),
                ("30Y", "y30"),
                ("30Y intraday high", "y30_spike"),
                ("VIX close", "vix"),
                ("equal-weight S&P", "mkt"),
                ("software", "igv"),
            )
            if v[key] is None
        ]
        # A row is final once every gauge for its date is in; the newest row may still be waiting.
        provisional = bool(missing) and ts == newest
        parts = [
            _fmt_bps("10Y", v["y10"], v["y10_spike"]),
            _fmt_bps("30Y", v["y30"], v["y30_spike"]),
            f"VIX {v['vix']:+.0f}%" if v["vix"] is not None else "VIX n/a",
            f"equal-weight S&P {v['mkt']:+.1f}%" if v["mkt"] is not None else "equal-weight S&P n/a",
            f"software {v['igv']:+.1f}%" if v["igv"] is not None else "software n/a",
        ]
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        days.append(
            TallyDay(
                date=str(pd.Timestamp(str(ts)).date()),
                soxx_pct=round(float(move), 2),
                driver=driver,
                provisional=provisional,
                tnx_bps=None if v["y10"] is None else round(v["y10"], 1),
                tnx_spike_bps=None if v["y10_spike"] is None else round(v["y10_spike"], 1),
                y30_bps=None if v["y30"] is None else round(v["y30"], 1),
                vix_pct=None if v["vix"] is None else round(v["vix"], 1),
                market_pct=None if v["mkt"] is None else round(v["mkt"], 2),
                igv_pct=None if v["igv"] is None else round(v["igv"], 2),
                detail="; ".join(parts),
            )
        )
    days.reverse()  # newest first
    final = [d for d in days if not d.provisional]
    counts = Counter(d.driver for d in final)
    ordered = {k: counts[k] for k in TALLY_ORDER if counts.get(k)}
    return Tally(
        days=days,
        counts=ordered,
        window_sessions=int((soxx.index >= drops.index[0]).sum()) if len(drops) else 0,
        summary=_tally_summary(days, final, counts),
    )


def _spike_bps(close: pd.Series | None, high: pd.Series | None) -> pd.Series:
    """Intraday high minus the prior session's close, in bps."""
    if close is None or high is None or len(close) < 2:
        return pd.Series(dtype=float)
    prior = close.astype(float).shift(1)
    joined = pd.concat([high.astype(float), prior], axis=1, join="inner").dropna()
    return (joined.iloc[:, 0] - joined.iloc[:, 1]) * 100.0


def _fmt_bps(name: str, close: float | None, spike: float | None) -> str:
    text = f"{name} {close:+.0f} bps" if close is not None else f"{name} n/a"
    if spike is not None and spike >= 5 and (close is None or spike - close >= 5):
        text += f" (+{spike:.0f} at its high)"
    return text


def _tally_summary(days: list[TallyDay], final: list[TallyDay], counts: Counter[str]) -> str:
    if not days:
        return f"No semis drop of {abs(TALLY_DROP_PCT):.1f}% or more in the available history."
    n = len(final)
    since = day_label(days[-1].date)
    bits = [f"{TALLY_SHORT[k]} {counts[k]}" for k in TALLY_ORDER if counts.get(k)]
    text = f"Last {n} semis drops of {abs(TALLY_DROP_PCT):.1f}% or more (since {since}): {', '.join(bits)}."
    pending = [d for d in days if d.provisional]
    if pending:
        text += f" {day_label(pending[0].date)} is provisional until its official closes post, and isn't counted."
    residual = counts.get("rotation", 0) + counts.get("chips", 0)
    if n and residual * 2 > n:
        text += (
            f" On {residual} of {n}, none of the macro gauges moved and the drop was concentrated in chips. "
            "That is all the gauges can say: they measure moves, not news, oil or policy headlines, so check "
            "the day list before calling any single day a pure chip story."
        )
    elif n and counts.get("rates", 0) * 2 > n:
        text += " Most came with a yield jump: here, 'valuation' mostly means the discount rate."
    return text


def overnight(verified: dict[str, Verified]) -> list[OvernightItem]:
    out = []
    for key, label in (("kospi", "KOSPI"), ("samsung", "Samsung"), ("dxy", "Dollar index")):
        v = verified.get(key)
        if v is None or v.series is None or v.series.empty:
            out.append(OvernightItem(label=label, value=None, change_pct=None, date=None))
            continue
        s = v.series.dropna()
        change = None if len(s) < 2 else (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
        out.append(
            OvernightItem(
                label=label,
                value=round(float(s.iloc[-1]), 3),
                change_pct=None if change is None else round(change, 2),
                date=str(s.index[-1].date()),
                source=v.status.source,
            )
        )
    return out


def build_today(f: MarketFeatures, drivers: list[Driver], verified: dict[str, Verified]) -> Today:
    return Today(us_session=f.dates.get("soxx"), stress=stress(f, drivers), overnight=overnight(verified))
