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

import pandas as pd

from app.config import STRESSED_VIX, TALLY_DAYS, TALLY_DROP_PCT, THRESHOLDS, Thresholds
from app.drivers import day_label
from app.features import MarketFeatures
from app.models import Driver, OvernightItem, Stress, Tally, TallyDay, Today
from app.verify import Verified

STRESS_LABELS = {"CALM": "Calm", "ELEVATED": "Elevated", "STRESSED": "Stressed", "UNKNOWN": "Unknown"}
TALLY_LABELS = {
    "rates": "a yield jump",
    "fear": "a market-wide volatility spike",
    "chips": "neither (chip-specific selling)",
    "unknown": "unknown (rates or VIX data missing)",
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


def tally(verified: dict[str, Verified], t: Thresholds = THRESHOLDS) -> Tally:
    soxx = _pct_changes(verified["soxx"].series if "soxx" in verified else None)
    if soxx.empty:
        return Tally(summary="Semis prices unavailable, so the tally can't be built.")
    tnx = _bps_changes(verified["y10"].series if "y10" in verified else None)
    tnx_z = _zscores(tnx, t.tnx_delta_z_window) if not tnx.empty else pd.Series(dtype=float)
    vix = _pct_changes(verified["vix"].series if "vix" in verified else None)
    igv = _pct_changes(verified["igv"].series if "igv" in verified else None)

    drops = soxx[soxx <= TALLY_DROP_PCT].iloc[-TALLY_DAYS:]
    tnx_d, tnx_zd, vix_d, igv_d = (dict(x.items()) for x in (tnx, tnx_z, vix, igv))
    days = []
    for ts, move in drops.items():
        bps, z, vchg, ichg = (None if d.get(ts) is None else float(d[ts]) for d in (tnx_d, tnx_zd, vix_d, igv_d))
        rates = None if bps is None else (bps >= t.tnx_1d_spike_bps or (z is not None and z >= t.tnx_delta_z))
        fear = None if vchg is None else vchg >= t.vix_jump_pct
        if rates:
            driver = "rates"
        elif fear:
            driver = "fear"
        elif rates is False and fear is False:
            driver = "chips"
        else:
            driver = "unknown"
        parts = [
            f"10Y {bps:+.0f} bps" if bps is not None else "10Y n/a",
            f"VIX {vchg:+.0f}%" if vchg is not None else "VIX n/a",
        ]
        if ichg is not None:
            parts.append(f"software {ichg:+.1f}%")
        days.append(
            TallyDay(
                date=str(pd.Timestamp(str(ts)).date()),
                soxx_pct=round(float(move), 2),
                driver=driver,
                tnx_bps=None if bps is None else round(bps, 1),
                vix_pct=None if vchg is None else round(vchg, 1),
                igv_pct=None if ichg is None else round(ichg, 2),
                detail=", ".join(parts),
            )
        )
    days.reverse()  # newest first
    counts = Counter(d.driver for d in days)
    ordered = {k: counts.get(k, 0) for k in ("rates", "fear", "chips", "unknown") if counts.get(k, 0) or k != "unknown"}
    if not days:
        summary = f"No semis drop of {abs(TALLY_DROP_PCT):.1f}% or more in the available history."
    else:
        since = day_label(days[-1].date)
        bits = [f"{n} with {TALLY_LABELS[k]}" for k, n in ordered.items() if n]
        summary = (
            f"Of the last {len(days)} semis drops of {abs(TALLY_DROP_PCT):.1f}% or more (since {since}): "
            f"{'; '.join(bits)}."
        )
        chips = counts.get("chips", 0)
        if chips * 2 > len(days):
            summary += (
                " Most sell-offs had no macro trigger: the selling was about the chip trade itself "
                "(positioning, valuation, company news). That is what the 'valuations questioned' "
                "headlines are describing."
            )
        elif counts.get("rates", 0) * 2 > len(days):
            summary += " Most sell-offs came with a yield jump: 'valuation' here mostly means the discount rate."
    sessions = int((soxx.index >= drops.index[0]).sum()) if len(drops) else 0
    return Tally(days=days, counts=dict(ordered), window_sessions=sessions, summary=summary)


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
