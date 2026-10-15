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

from app.config import NEWS_MACRO, STRESSED_VIX, TALLY_DAYS, TALLY_DROP_PCT, THRESHOLDS, Thresholds
from app.drivers import day_label
from app.features import MarketFeatures
from app.models import Driver, OvernightItem, Stress, Tally, TallyDay, Today
from app.verify import Verified

STRESS_LABELS = {"CALM": "Calm", "ELEVATED": "Elevated", "STRESSED": "Stressed", "UNKNOWN": "Unknown"}
TALLY_LABELS = {
    "rates": "yields jumped (2Y, 10Y or 30Y; close or intraday)",
    "oil": "oil moved sharply (USO 4%+ or 2 sigma, either way)",
    "dollar": "the dollar jumped (UUP, 2 sigma)",
    "fear": "VIX spiked",
    "market": "the broad market fell (equal-weight S&P 500 down 1%+)",
    "rotation": "no macro gauge moved, and software rose: money moved from chips to software",
    "chips": "no macro gauge moved, and software fell too",
    "unknown": "couldn't tell (a gauge was missing)",
}
TALLY_ORDER = ("rates", "oil", "dollar", "fear", "market", "rotation", "chips", "unknown")
MACRO_DRIVERS = ("rates", "oil", "dollar", "fear", "market")
RESIDUAL = ("rotation", "chips")
TALLY_SHORT = {
    "rates": "yields jumped",
    "oil": "oil shock",
    "dollar": "dollar jump",
    "fear": "VIX spiked",
    "market": "broad market fell",
    "rotation": "rotation into software",
    "chips": "chips alone",
    "unknown": "couldn't tell",
}
NEWS_SHORT = {
    "company": "company news",
    "ai_demand": "AI-demand worries",
    "policy": "policy",
    "analyst": "analyst calls",
    "macro_data": "economic data",
    "fed": "the Fed",
    "geopolitics": "geopolitics/oil",
    "none_clear": "nothing clear",
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
    news: dict[str, dict[str, Any]] | None = None,
    t: Thresholds = THRESHOLDS,
) -> Tally:
    """Tag each recent semis drop by what else moved that day, checked in order:

    1. rates   10Y or 30Y up >= 10 bps at the close or at its intraday high vs the prior close, or an
               unusual 10Y close move (z >= 1.5). Intraday matters: on CPI day (Oct 14) the 10Y rose
               11 bps by mid-session and closed down 3.
    2. oil     USO move >= 2 sigma either way vs its prior 60 sessions.
    3. dollar  UUP up >= 2 sigma.
    4. fear    VIX up >= 10% at the close.
    5. market  equal-weight S&P 500 (RSP) down >= 1%.
    6. rotation / chips   none of the above (software up = rotation, down = chips alone).

    A gauge can only claim a day if every gauge above it is known; the residual needs all of them.
    Every gauge that moved is listed in `moved`. Headlines (news.py) annotate each day and flag
    disagreement with the gauges; they never change the gauge label. The newest day stays
    provisional (not counted) until all of its official closes have posted.
    """
    extras = extras or {}
    news = news or {}
    soxx = _pct_changes(verified["soxx"].series if "soxx" in verified else None)
    if soxx.empty:
        return Tally(summary="Semis prices unavailable, so the tally can't be built.")
    y10_close = verified["y10"].series if "y10" in verified else None
    y10 = _bps_changes(y10_close)
    oil = _pct_changes(extras.get("oil"))
    usd = _pct_changes(extras.get("dollar"))
    series = {
        "y10": y10,
        "z": _zscores(y10, t.tnx_delta_z_window) if not y10.empty else pd.Series(dtype=float),
        "y10_spike": _spike_bps(y10_close, extras.get("y10_high")),
        "y30": _bps_changes(extras.get("y30")),
        "y30_spike": _spike_bps(extras.get("y30"), extras.get("y30_high")),
        "y2": _bps_changes(extras.get("y2")),
        "vix": _pct_changes(verified["vix"].series if "vix" in verified else None),
        "igv": _pct_changes(verified["igv"].series if "igv" in verified else None),
        "mkt": _pct_changes(extras.get("market")),
        "oil": oil,
        "oil_z": _zscores(oil, t.tally_z_window) if not oil.empty else pd.Series(dtype=float),
        "usd": usd,
        "usd_z": _zscores(usd, t.tally_z_window) if not usd.empty else pd.Series(dtype=float),
    }
    maps = {k: dict(v.items()) for k, v in series.items()}

    drops = soxx[soxx <= TALLY_DROP_PCT].iloc[-TALLY_DAYS:]
    newest = soxx.index[-1]
    y2_series = extras.get("y2")
    y2_last = y2_series.index[-1] if y2_series is not None and not y2_series.empty else None
    days = []
    for ts, move in drops.items():
        v = {k: _at(m, ts) for k, m in maps.items()}
        # Treasury publishes no curve on bond-market holidays (Columbus Day, Veterans Day) while
        # stocks trade. A date before Treasury's latest row with no row of its own had no 2Y move.
        y2_closed = v["y2"] is None and y2_last is not None and ts < y2_last
        rates = _any_true(
            None if v["y10"] is None else v["y10"] >= t.tnx_1d_spike_bps,
            None if v["z"] is None else v["z"] >= t.tnx_delta_z,
            None if v["y10_spike"] is None else v["y10_spike"] >= t.tnx_1d_spike_bps,
            None if v["y30"] is None else v["y30"] >= t.tnx_1d_spike_bps,
            None if v["y30_spike"] is None else v["y30_spike"] >= t.tnx_1d_spike_bps,
            False if y2_closed else (None if v["y2"] is None else v["y2"] >= t.y2_jump_bps),
        )
        gauges = {
            "rates": rates,
            "oil": _any_true(
                None if v["oil_z"] is None else abs(v["oil_z"]) >= t.oil_z,
                None if v["oil"] is None else abs(v["oil"]) >= t.oil_move_pct,
            ),
            "dollar": None if v["usd_z"] is None else v["usd_z"] >= t.dollar_z,
            "fear": None if v["vix"] is None else v["vix"] >= t.vix_jump_pct,
            "market": None if v["mkt"] is None else v["mkt"] <= t.market_drop_pct,
        }
        driver = "unknown"
        for key in MACRO_DRIVERS:
            if gauges[key] is True:
                driver = key
                break
            if gauges[key] is None:
                break
        else:
            if v["igv"] is not None:
                driver = "rotation" if v["igv"] > 0 else "chips"
        missing = [
            name
            for name, key in (
                ("10Y close", "y10"),
                ("10Y intraday high", "y10_spike"),
                ("30Y", "y30"),
                ("30Y intraday high", "y30_spike"),
                ("2Y", "y2"),
                ("oil", "oil_z"),
                ("dollar", "usd_z"),
                ("VIX close", "vix"),
                ("equal-weight S&P", "mkt"),
                ("software", "igv"),
            )
            if v[key] is None and not (key == "y2" and y2_closed)
        ]
        parts = [
            "2Y closed" if y2_closed else (f"2Y {v['y2']:+.0f} bps" if v["y2"] is not None else "2Y n/a"),
            _fmt_bps("10Y", v["y10"], v["y10_spike"]),
            _fmt_bps("30Y", v["y30"], v["y30_spike"]),
            f"oil {v['oil']:+.1f}%" if v["oil"] is not None else "oil n/a",
            f"dollar {v['usd']:+.2f}%" if v["usd"] is not None else "dollar n/a",
            f"VIX {v['vix']:+.0f}%" if v["vix"] is not None else "VIX n/a",
            f"equal-weight S&P {v['mkt']:+.1f}%" if v["mkt"] is not None else "equal-weight S&P n/a",
            f"software {v['igv']:+.1f}%" if v["igv"] is not None else "software n/a",
        ]
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        date = str(pd.Timestamp(str(ts)).date())
        entry = news.get(date)
        label = entry if entry and "category" in entry else None  # failed reads carry no label
        days.append(
            TallyDay(
                date=date,
                soxx_pct=round(float(move), 2),
                driver=driver,
                moved=[k for k in MACRO_DRIVERS if gauges[k] is True],
                provisional=bool(missing) and ts == newest,
                tnx_bps=_r(v["y10"], 1),
                tnx_spike_bps=_r(v["y10_spike"], 1),
                y30_bps=_r(v["y30"], 1),
                y2_bps=_r(v["y2"], 1),
                oil_pct=_r(v["oil"], 2),
                dollar_pct=_r(v["usd"], 2),
                vix_pct=_r(v["vix"], 1),
                market_pct=_r(v["mkt"], 2),
                igv_pct=_r(v["igv"], 2),
                detail="; ".join(parts),
                news=label,
                news_flag=_news_flag(driver, label),
            )
        )
    days.reverse()  # newest first
    final = [d for d in days if not d.provisional]
    counts = Counter(d.driver for d in final)
    return Tally(
        days=days,
        counts={k: counts[k] for k in TALLY_ORDER if counts.get(k)},
        news_counts=_news_counts([d for d in final if d.driver in RESIDUAL]),
        window_sessions=int((soxx.index >= drops.index[0]).sum()) if len(drops) else 0,
        summary=_tally_summary(days, final, counts),
    )


def _any_true(*xs: bool | None) -> bool | None:
    if any(x is True for x in xs):
        return True
    return False if all(x is False for x in xs) else None


def _r(x: float | None, digits: int) -> float | None:
    return None if x is None else round(x, digits)


def _news_flag(driver: str, label: dict[str, Any] | None) -> str | None:
    """Say so when the headlines and the gauges disagree. Never pick a side silently."""
    if not label:
        return None
    category = str(label.get("category", ""))
    if driver in RESIDUAL and category in NEWS_MACRO:
        return f"Headlines point to {NEWS_SHORT[category]}, but no macro gauge moved"
    if driver in MACRO_DRIVERS and category in ("company", "ai_demand", "analyst"):
        return f"A macro gauge moved, but the headlines point to {NEWS_SHORT[category]}"
    return None


def _news_counts(days: list[TallyDay]) -> dict[str, int]:
    counts = Counter(str(d.news.get("category")) for d in days if d.news)
    counts["unlabelled"] = len([d for d in days if not d.news])
    return {k: counts[k] for k in (*NEWS_SHORT, "unlabelled") if counts.get(k)}


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
    residual = [d for d in final if d.driver in RESIDUAL]
    if residual:
        labelled = [d for d in residual if d.news]
        news = Counter(str(d.news.get("category")) for d in labelled if d.news)
        text += f" On {len(residual)} of {n}, no macro gauge moved"
        if labelled:
            parts = [f"{NEWS_SHORT[k]} {news[k]}" for k in NEWS_SHORT if news.get(k)]
            text += f"; the headlines on those days: {', '.join(parts)}"
            if len(labelled) < len(residual):
                text += f" ({len(residual) - len(labelled)} not yet read)"
        text += "."
        flagged = [d for d in final if d.news_flag]
        if flagged:
            text += f" {len(flagged)} day(s) where headlines and gauges disagree are flagged in the day list."
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
