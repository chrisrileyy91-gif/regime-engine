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

import numpy as np
import pandas as pd

from app.config import (
    AI_LEVERAGE_TICKERS,
    NEWS_MACRO,
    STRESSED_VIX,
    TALLY_DAYS,
    TALLY_DROP_PCT,
    THRESHOLDS,
    Thresholds,
)
from app.drivers import bond_market_closed, day_label, yields_counted
from app.features import MarketFeatures
from app.indicators import excess_returns
from app.models import Driver, OvernightItem, Stress, Tally, TallyDay, Today
from app.verify import Verified

STRESS_LABELS = {"CALM": "Calm", "ELEVATED": "Elevated", "STRESSED": "Stressed", "UNKNOWN": "Unknown"}
TALLY_LABELS = {
    "oil": "oil jumped (USO up 4%+ or 2 sigma)",
    "dollar": "the dollar jumped (UUP, 2 sigma)",
    "rates": "the broad market and yields explain at least half the drop, mostly yields",
    "market": "the broad market and yields explain at least half the drop, mostly the broad market",
    "rotation": "mostly chip-specific, and software rose: money moved from chips to software",
    "chips": "mostly chip-specific, and software fell too",
    "unknown": "couldn't tell (a gauge was missing)",
}
TALLY_ORDER = ("oil", "dollar", "rates", "market", "rotation", "chips", "unknown")
# What can claim a day. Oil and the dollar are outside shocks the fit doesn't see, so a jump claims
# the day. Yields and the broad market have to EARN it: together they must explain half the drop at
# SOXX's own recent sensitivity. To Nov 11, 2026 a 10 bps yield jump claimed the day outright; on
# two of its four days the fit explained under a quarter of the drop, and over those 120 sessions
# yields and chips rose together. VIX is not here (it rises BECAUSE chips fell: Nov 5, SOXX -12%,
# VIX +70%), nor is falling oil (Nov 4: oil -8% with the equal-weight S&P +1.9%). A yield jump and
# credit are still listed in `moved`.
SHOCKS = ("oil", "dollar")
LABEL_GAUGES = ("10Y close", "oil", "dollar", "equal-weight S&P", "software", "enough history for the explained share")
MACRO_DRIVERS = ("oil", "dollar", "rates", "market")
RESIDUAL = ("rotation", "chips")
TALLY_SHORT = {
    "rates": "yields",
    "oil": "oil jumped",
    "dollar": "dollar jump",
    "market": "broad market",
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
    """Counted drivers only (the funding card is on trial). A yield jump counts only when chips fell
    with it: yields and chips have been rising together, so a jump on a flat day isn't stress."""
    core = [d for d in drivers if d.counted]
    yields = yields_counted(f, drivers)
    active = [d for d in core if d.state == "ACTIVE" and (d.key != "liquidity" or yields is True)]
    unknown = [d for d in core if d.state == "UNKNOWN"]
    reasons = [d.name for d in active]
    vix_read = any(d.key == "risk_off" and d.state != "UNKNOWN" for d in drivers)  # same-session VIX only
    if vix_read and f.vix is not None and f.vix > STRESSED_VIX:
        level = "STRESSED"
        reasons.insert(0, f"VIX above {STRESSED_VIX:.0f}")
    elif len(active) >= 2:
        level = "STRESSED"
    elif len(active) == 1:
        level = "ELEVATED"
    elif unknown:
        # Calm is a measurement, not a default: nothing active is only Calm when every driver was
        # actually read for this session (Nov 5 16:25 said Calm with SOXX withheld and VIX a day old).
        level = "UNKNOWN"
    else:
        level = "CALM"
    if any(d.key == "liquidity" and d.state == "ACTIVE" for d in core) and yields is False:
        reasons.append("yields jumped, but chips didn't fall with them (not counted)")
    if unknown:
        reasons.append(f"couldn't check {', '.join(d.name.lower() for d in unknown)}")
    watch = [d.name for d in drivers if not d.counted and d.state == "ACTIVE"]
    if watch:
        reasons.append(f"{', '.join(watch)} is active (shown, not counted)")
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


def decompose(
    soxx: pd.Series, market: pd.Series, y10_bps: pd.Series, days: list[Any], t: Thresholds = THRESHOLDS
) -> dict[Any, tuple[float, float, float]]:
    """For each day: (expected SOXX move, share of the actual move it explains 0..1, the part of
    the expected move that comes from the 10Y).

    Expected = what SOXX usually does when the equal-weight S&P 500 and the 10Y move as they did
    that day: an ordinary least-squares fit of SOXX's daily return on the two, over the
    `decomp_window` sessions BEFORE the day (no look-ahead), intercept left out of the estimate.
    A bond holiday counts as a 10Y change of zero. A day without enough history is left out.
    """
    frame = pd.concat({"soxx": soxx, "mkt": market, "dy": y10_bps}, axis=1, join="inner").dropna()
    out: dict[Any, tuple[float, float, float]] = {}
    for ts in days:
        move, mkt = _at(dict(soxx.items()), ts), _at(dict(market.items()), ts)
        dy = _at(dict(y10_bps.items()), ts)
        if dy is None and bond_market_closed(str(pd.Timestamp(str(ts)).date())):
            dy = 0.0
        if move is None or mkt is None or dy is None or move == 0.0:
            continue
        past = frame[frame.index < ts].iloc[-t.decomp_window :]
        if len(past) < t.decomp_min_sessions:
            continue
        x = np.column_stack([np.ones(len(past)), past["mkt"].to_numpy(float), past["dy"].to_numpy(float)])
        coef, *_ = np.linalg.lstsq(x, past["soxx"].to_numpy(float), rcond=None)
        expected = float(coef[1] * mkt + coef[2] * dy)
        out[ts] = (expected, float(min(1.0, max(0.0, expected / move))), float(coef[2] * dy))
    return out


def sensitivity(
    soxx: pd.Series, market: pd.Series, y10_bps: pd.Series, t: Thresholds = THRESHOLDS
) -> tuple[float, float, float, int] | None:
    """How SOXX has moved with the broad market and the 10Y over the newest `decomp_window` sessions:
    (SOXX % per 1% in the equal-weight S&P 500, SOXX % per +10 bps in the 10Y with the market held
    fixed, share of SOXX's daily variation the two account for, sessions used). The same fit the
    tally uses, on the latest window. None without enough history."""
    frame = pd.concat({"soxx": soxx, "mkt": market, "dy": y10_bps}, axis=1, join="inner").dropna()
    past = frame.iloc[-t.decomp_window :]
    if len(past) < t.decomp_min_sessions:
        return None
    y = past["soxx"].to_numpy(float)
    x = np.column_stack([np.ones(len(past)), past["mkt"].to_numpy(float), past["dy"].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0.0:
        return None
    r2 = 1.0 - float(((y - x @ coef) ** 2).sum()) / total
    return float(coef[1]), float(coef[2]) * 10.0, r2, len(past)


def tally(
    verified: dict[str, Verified],
    extras: dict[str, pd.Series] | None = None,
    news: dict[str, dict[str, Any]] | None = None,
    t: Thresholds = THRESHOLDS,
) -> Tally:
    """Tag each recent semis drop by what explains it, checked in order:

    1. oil     USO UP >= 4% or >= 2 sigma vs its prior 60 sessions. Falling oil is not a cause.
    2. dollar  UUP up >= 2 sigma.
    3. rates / market   the broad market and yields explain at least half the drop (`decompose`);
               "rates" when the 10Y accounts for more of that than the broad market, else "market".
    4. rotation / chips   mostly chip-specific (software up = rotation, down = chips alone).

    A yield jump no longer claims a day by itself: it is listed in `moved` (10Y or 30Y up >= 10 bps
    at the close or the intraday high, a 10Y z >= 1.5, or the 2Y up >= 10 bps), and the label has
    to be earned through the explained share. VIX never claims a day (it rises because chips fell).
    Credit (HYG) and the debt-funded AI names annotate. Headlines (news.py) annotate each day and
    flag disagreement; they never change the label. The newest day stays provisional until all of
    its official closes have posted.
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
    credit = _pct_changes(extras.get("credit"))
    mkt = _pct_changes(extras.get("market"))
    leverage = [m for tk in AI_LEVERAGE_TICKERS if not (m := _pct_changes(extras.get(f"lev_{tk.lower()}"))).empty]
    series = {
        "y10": y10,
        "z": _zscores(y10, t.tnx_delta_z_window) if not y10.empty else pd.Series(dtype=float),
        "y10_spike": _spike_bps(y10_close, extras.get("y10_high")),
        "y30": _bps_changes(extras.get("y30")),
        "y30_spike": _spike_bps(extras.get("y30"), extras.get("y30_high")),
        "y2": _bps_changes(extras.get("y2")),
        "vix": _pct_changes(verified["vix"].series if "vix" in verified else None),
        "igv": _pct_changes(verified["igv"].series if "igv" in verified else None),
        "mkt": mkt,
        "oil": oil,
        "oil_z": _zscores(oil, t.tally_z_window) if not oil.empty else pd.Series(dtype=float),
        "usd": usd,
        "usd_z": _zscores(usd, t.tally_z_window) if not usd.empty else pd.Series(dtype=float),
        "credit": credit,
        "credit_z": _zscores(credit, t.tally_z_window) if not credit.empty else pd.Series(dtype=float),
        "lev": pd.concat(leverage, axis=1).mean(axis=1) if leverage else pd.Series(dtype=float),
    }
    series["lev_x"] = (
        excess_returns(series["lev"], soxx, t.decomp_window, t.decomp_min_sessions)
        if leverage
        else pd.Series(dtype=float)
    )
    maps = {k: dict(v.items()) for k, v in series.items()}

    drops = soxx[soxx <= TALLY_DROP_PCT].iloc[-TALLY_DAYS:]
    explained = decompose(soxx, mkt, y10, list(drops.index), t) if not mkt.empty and not y10.empty else {}
    newest = soxx.index[-1]
    y2_series = extras.get("y2")
    y2_last = y2_series.index[-1] if y2_series is not None and not y2_series.empty else None
    days = []
    for ts, move in drops.items():
        v = {k: _at(m, ts) for k, m in maps.items()}
        expected, share, rate_part = explained.get(ts, (None, None, None))
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
            "oil": _any_true(
                None if v["oil_z"] is None else v["oil_z"] >= t.oil_z,
                None if v["oil"] is None else v["oil"] >= t.oil_move_pct,
            ),
            "dollar": None if v["usd_z"] is None else v["usd_z"] >= t.dollar_z,
        }
        credit_moved = _any_true(
            None if v["credit_z"] is None else v["credit_z"] <= t.credit_z,
            None if v["credit"] is None else v["credit"] <= t.credit_move_pct,
        )
        driver = "unknown"
        for key in SHOCKS:
            if gauges[key] is True:
                driver = key
                break
            if gauges[key] is None:
                break
        else:
            if share is not None and expected is not None and rate_part is not None:
                if share >= t.macro_share:
                    # Both parts are signed like the drop when they explain it; the more negative wins.
                    driver = "rates" if rate_part < expected - rate_part else "market"
                elif v["igv"] is not None:
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
                ("equal-weight S&P", "mkt"),
                ("software", "igv"),
            )
            if v[key] is None and not (key == "y2" and y2_closed)
        ]
        if share is None and "equal-weight S&P" not in missing and "10Y close" not in missing:
            missing.append("enough history for the explained share")
        parts = [
            (
                f"expected {expected:+.1f}% from the broad market and yields"
                if share is not None and expected is not None
                else "explained share n/a"
            ),
            f"equal-weight S&P {v['mkt']:+.1f}%" if v["mkt"] is not None else "equal-weight S&P n/a",
            "2Y closed" if y2_closed else (f"2Y {v['y2']:+.0f} bps" if v["y2"] is not None else "2Y n/a"),
            _fmt_bps("10Y", v["y10"], v["y10_spike"]),
            _fmt_bps("30Y", v["y30"], v["y30_spike"]),
            f"oil {v['oil']:+.1f}%" if v["oil"] is not None else "oil n/a",
            f"dollar {v['usd']:+.2f}%" if v["usd"] is not None else "dollar n/a",
            f"software {v['igv']:+.1f}%" if v["igv"] is not None else "software n/a",
            f"VIX {v['vix']:+.0f}%" if v["vix"] is not None else "VIX n/a",
        ]
        if v["credit"] is not None:
            parts.append(f"high-yield bonds {v['credit']:+.2f}%")
        if v["lev"] is not None:
            beyond = "" if v["lev_x"] is None else f" ({v['lev_x']:+.1f} beyond their usual move with chips)"
            parts.append(f"debt-funded AI names {v['lev']:+.1f}%{beyond}")
        if missing:
            parts.append(f"missing: {', '.join(missing)}")
        date = str(pd.Timestamp(str(ts)).date())
        entry = news.get(date)
        label = entry if entry and "category" in entry else None  # failed reads carry no label
        moved = [k for k in SHOCKS if gauges[k] is True]
        if rates is True:
            moved.insert(0, "rates")
        if credit_moved is True:
            moved.append("credit")
        days.append(
            TallyDay(
                date=date,
                soxx_pct=round(float(move), 2),
                driver=driver,
                moved=moved,
                # Only a gauge that can change the label holds the newest day back; the evening 2Y
                # and the intraday highs just annotate.
                provisional=ts == newest and any(m in LABEL_GAUGES for m in missing),
                macro_share=_r(share, 3),
                expected_pct=_r(expected, 2),
                tnx_bps=_r(v["y10"], 1),
                tnx_spike_bps=_r(v["y10_spike"], 1),
                y30_bps=_r(v["y30"], 1),
                y2_bps=_r(v["y2"], 1),
                oil_pct=_r(v["oil"], 2),
                dollar_pct=_r(v["usd"], 2),
                vix_pct=_r(v["vix"], 1),
                market_pct=_r(v["mkt"], 2),
                credit_pct=_r(v["credit"], 2),
                ai_leverage_pct=_r(v["lev"], 2),
                ai_excess_pct=_r(v["lev_x"], 2),
                igv_pct=_r(v["igv"], 2),
                detail="; ".join(parts),
                news=label,
                news_flag=_news_flag(driver, label),
            )
        )
    days.reverse()  # newest first
    fit = sensitivity(soxx, mkt, y10, t) if not mkt.empty and not y10.empty else None
    final = [d for d in days if not d.provisional]
    counts = Counter(d.driver for d in final)
    shares = [d.macro_share for d in final if d.macro_share is not None]
    return Tally(
        days=days,
        counts={k: counts[k] for k in TALLY_ORDER if counts.get(k)},
        news_counts=_news_counts([d for d in final if d.driver in RESIDUAL]),
        window_sessions=int((soxx.index >= drops.index[0]).sum()) if len(drops) else 0,
        avg_macro_share=None if not shares else round(sum(shares) / len(shares), 3),
        market_beta=None if fit is None else round(fit[0], 2),
        rate_beta_10bps=None if fit is None else round(fit[1], 2),
        fit_r2=None if fit is None else round(fit[2], 3),
        fit_sessions=None if fit is None else fit[3],
        summary=_tally_summary(days, final, counts, shares) + _sensitivity_text(fit),
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
        return f"Headlines point to {NEWS_SHORT[category]}, but the gauges read it as chip-specific"
    if driver in MACRO_DRIVERS and category in ("company", "ai_demand", "analyst"):
        return f"The gauges read it as macro, but the headlines point to {NEWS_SHORT[category]}"
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


def _sensitivity_text(fit: tuple[float, float, float, int] | None) -> str:
    if fit is None:
        return ""
    market, rate, r2, n = fit
    return (
        f" Over the last {n} sessions, a 10 bps rise in the 10Y has gone with SOXX {rate:+.1f}% (broad market held "
        f"fixed) and a 1% move in the equal-weight S&P with SOXX {market:+.1f}%; together they account for "
        f"{r2 * 100:.0f}% of SOXX's day-to-day variation."
    )


def _tally_summary(days: list[TallyDay], final: list[TallyDay], counts: Counter[str], shares: list[float]) -> str:
    if not days:
        return f"No semis drop of {abs(TALLY_DROP_PCT):.1f}% or more in the available history."
    n = len(final)
    since = day_label(days[-1].date)
    bits = [f"{TALLY_SHORT[k]} {counts[k]}" for k in TALLY_ORDER if counts.get(k)]
    text = f"Last {n} semis drops of {abs(TALLY_DROP_PCT):.1f}% or more (since {since}): {', '.join(bits)}."
    if shares:
        text += (
            f" On average the broad market and yields explain {sum(shares) / len(shares) * 100:.0f}% of the drop; "
            "the rest is specific to chips."
        )
    pending = [d for d in days if d.provisional]
    if pending:
        text += f" {day_label(pending[0].date)} is provisional until its official closes post, and isn't counted."
    residual = [d for d in final if d.driver in RESIDUAL]
    if residual:
        labelled = [d for d in residual if d.news]
        news = Counter(str(d.news.get("category")) for d in labelled if d.news)
        text += f" {len(residual)} of {n} were mostly chip-specific"
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


def overnight(verified: dict[str, Verified], extras: dict[str, pd.Series] | None = None) -> list[OvernightItem]:
    """Korea's session and the dollar. The dollar is UUP on the consolidated tape (official, same
    day); the dollar index it replaced was a web-search fill on every run without an EODHD key."""
    out = []
    for key, label in (("kospi", "KOSPI"), ("samsung", "Samsung")):
        v = verified.get(key)
        if v is None or v.series is None or v.series.empty:
            out.append(OvernightItem(label=label, value=None, change_pct=None, date=None))
            continue
        out.append(_overnight_item(label, v.series, v.status.source))
    dollar = (extras or {}).get("dollar")
    if dollar is None or dollar.empty:
        out.append(OvernightItem(label="Dollar (UUP)", value=None, change_pct=None, date=None))
    else:
        out.append(_overnight_item("Dollar (UUP)", dollar, "consolidated tape (Alpaca)"))
    return out


def _overnight_item(label: str, series: pd.Series, source: str | None) -> OvernightItem:
    s = series.dropna()
    change = None if len(s) < 2 else (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
    return OvernightItem(
        label=label,
        value=round(float(s.iloc[-1]), 3),
        change_pct=None if change is None else round(change, 2),
        date=str(s.index[-1].date()),
        source=source,
    )


def build_today(
    f: MarketFeatures, drivers: list[Driver], verified: dict[str, Verified], extras: dict[str, pd.Series] | None = None
) -> Today:
    return Today(us_session=f.dates.get("soxx"), stress=stress(f, drivers), overnight=overnight(verified, extras))
