"""Replay the daily stress label over history and score it against what semis actually did.

    python -m app.replay --sessions 250

The thresholds in config.py are uncalibrated. This is the check that was missing: it rebuilds the
label for every past session two ways and shows how each lines up with SOXX's same-day move.

  old  (to Nov 10, 2026): 10Y jump · VIX above 20 · SOXX RSI below 40
  new                   : VIX up 10%+ · SOXX down 4%+ or an unusual drop (z) · a 10Y jump, counted
                          only when SOXX also fell 1.5%+

It also scores the funding backdrop: the debt-funded AI names' move beyond their usual sensitivity
to SOXX, summed over 20 sessions and ranked against the past year. Sessions are grouped by that
rank and scored on what SOXX did over the NEXT five sessions. Five-session windows overlap, so
neighbouring rows share most of their outcome: read the table as a description, not a test.

Needs the Alpaca keys (SOXX). Reads only verified series; changes nothing and publishes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import STRESSED_VIX, TALLY_DROP_PCT, THRESHOLDS, Thresholds
from app.data import fetch_all
from app.indicators import excess_returns, trailing_rank, wilder_rsi
from app.service import tally_extras
from app.verify import verify_all

LEVELS = ("CALM", "ELEVATED", "STRESSED")


@dataclass(frozen=True)
class Row:
    date: str
    soxx_pct: float
    old: str
    new: str
    funding_rank: float | None = None  # 20-session excess of the debt-funded names, percentile vs the past year
    next5_pct: float | None = None  # SOXX over the following five sessions
    next5_drops: int | None = None  # how many of those five fell 1.5%+


def _level(active: int, vix: float) -> str:
    if vix > STRESSED_VIX or active >= 2:
        return "STRESSED"
    return "ELEVATED" if active == 1 else "CALM"


def replay(
    soxx: pd.Series,
    vix: pd.Series,
    y10: pd.Series,
    sessions: int,
    t: Thresholds = THRESHOLDS,
    basket: list[pd.Series] | None = None,
) -> list[Row]:
    """Both labels for each of the last `sessions` sessions SOXX and VIX share. A session with no
    10Y close (a bond holiday) reads as no yield move, as the live page does."""
    frame = pd.concat({"soxx": soxx.astype(float), "vix": vix.astype(float)}, axis=1, join="inner").dropna()
    ret = frame["soxx"].pct_change() * 100.0
    base = ret.shift(1).rolling(t.soxx_z_window, min_periods=t.soxx_z_window)
    ret_z = (ret - base.mean()) / base.std(ddof=1)
    vix_chg = frame["vix"].pct_change() * 100.0
    rsi = wilder_rsi(frame["soxx"], t.rsi_period)

    dy = y10.astype(float).diff() * 100.0
    dy_base = dy.shift(1).rolling(t.tnx_delta_z_window, min_periods=t.tnx_delta_z_window)
    dy_z = (dy - dy_base.mean()) / dy_base.std(ddof=1)
    liquidity = ((dy > t.tnx_1d_spike_bps) | (dy_z > t.tnx_delta_z)).reindex(frame.index).fillna(False)

    names = [(b.astype(float).pct_change() * 100.0).dropna() for b in (basket or []) if len(b) > 1]
    excess = (
        excess_returns(pd.concat(names, axis=1).mean(axis=1), ret.dropna(), t.decomp_window, t.decomp_min_sessions)
        if names
        else pd.Series(dtype=float)
    )
    rolled = excess.rolling(t.funding_window, min_periods=t.funding_window).sum().dropna()
    rank = trailing_rank(rolled, t.funding_rank_window, t.funding_rank_min) if not rolled.empty else rolled
    ahead = (1 + ret / 100.0).rolling(5).apply(lambda w: float(w.prod()), raw=True).shift(-5)
    drops_ahead = (ret <= TALLY_DROP_PCT).astype(float).rolling(5).sum().shift(-5)

    rows = []
    for ts in frame.index[-sessions:]:
        if pd.isna(ret[ts]) or pd.isna(ret_z[ts]) or pd.isna(rsi[ts]) or pd.isna(vix_chg[ts]):
            continue
        level = float(frame["vix"][ts])
        liq = bool(liquidity[ts])
        old = int(liq) + int(level > t.vix_level) + int(rsi[ts] < t.semi_rsi_digestion)
        sold = ret[ts] < t.soxx_drop_floor_pct or ret_z[ts] < t.soxx_drop_z
        counted = liq and ret[ts] <= t.yields_count_soxx_pct  # a yield jump counts only when chips fell
        new = int(counted) + int(vix_chg[ts] > t.vix_jump_pct) + int(sold)
        pct = rank.get(ts)
        fwd, falls = ahead[ts], drops_ahead[ts]
        rows.append(
            Row(
                str(ts.date()),
                round(float(ret[ts]), 2),
                _level(old, level),
                _level(new, level),
                None if pct is None or pd.isna(pct) else round(float(pct), 1),
                None if pd.isna(fwd) else round((float(fwd) - 1) * 100.0, 2),
                None if pd.isna(falls) else int(falls),
            )
        )
    return rows


def summarize(rows: list[Row], scheme: str) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for level in LEVELS:
        moves = [r.soxx_pct for r in rows if getattr(r, scheme) == level]
        if not moves:
            out.append({"label": level, "days": 0, "mean": 0.0, "drops": 0.0, "share": 0.0})
            continue
        drops = sum(1 for m in moves if m <= TALLY_DROP_PCT)
        out.append(
            {
                "label": level,
                "days": len(moves),
                "share": round(len(moves) / len(rows) * 100.0, 1),
                "mean": round(sum(moves) / len(moves), 2),
                "drops": round(drops / len(moves) * 100.0, 1),
            }
        )
    return out


def _mean(values: list[float]) -> str:
    return "n/a" if not values else f"{sum(values) / len(values):+.2f}%"


def funding_lines(rows: list[Row]) -> list[str]:
    """Does the funding backdrop say anything about what comes next? Sessions grouped by how the
    debt-funded names have done against chips over 20 sessions, scored on SOXX's next five."""
    read = [r for r in rows if r.funding_rank is not None and r.next5_pct is not None and r.next5_drops is not None]
    title = "### Funding backdrop (debt-funded AI names vs chips, 20 sessions)"
    if not read:
        return [title, "", "No sessions with the basket and a year of history available."]
    out = [
        f"{title}, {len(read)} sessions read",
        "",
        "| 20-session excess, rank vs past year | Days | Mean SOXX next 5 sessions | Next-5 sessions that fell 1.5%+ |",
        "|---|---|---|---|",
    ]
    for name, lo, hi in (
        ("Bottom fifth (lagging chips)", 0.0, 20.0),
        ("Middle", 20.0, 80.0),
        ("Top fifth", 80.0, 100.1),
    ):
        days = [r for r in read if r.funding_rank is not None and lo <= r.funding_rank < hi]
        drops = [r.next5_drops for r in days if r.next5_drops is not None]
        share = "n/a" if not drops else f"{sum(drops) / (5 * len(drops)) * 100:.0f}%"
        out.append(
            f"| {name} | {len(days)} | {_mean([r.next5_pct for r in days if r.next5_pct is not None])} | {share} |"
        )
    return out


def render(rows: list[Row]) -> str:
    if not rows:
        return "Replay: no sessions with every input. Are the Alpaca keys set?"
    lines = [f"## Stress label replay, {rows[0].date} to {rows[-1].date} ({len(rows)} sessions)", ""]
    for scheme, title in (
        ("old", "Old label (10Y jump, VIX above 20, RSI below 40)"),
        ("new", "New label (VIX up 10%+, SOXX sell-off, 10Y jump only with chips down)"),
    ):
        lines += [
            f"### {title}",
            "",
            "| Label | Days | Share of days | Mean SOXX move | Days SOXX fell 1.5%+ |",
            "|---|---|---|---|---|",
        ]
        for s in summarize(rows, scheme):
            lines.append(f"| {s['label']} | {s['days']} | {s['share']}% | {s['mean']:+}% | {s['drops']}% |")
        lines.append("")
    worst = sorted(rows, key=lambda r: r.soxx_pct)[:15]
    lines += ["### The 15 worst sessions", "", "| Date | SOXX | Old label | New label |", "|---|---|---|---|"]
    lines += [f"| {r.date} | {r.soxx_pct:+.1f}% | {r.old} | {r.new} |" for r in worst]
    lines += ["", *funding_lines(rows)]
    best = sorted(rows, key=lambda r: -r.soxx_pct)[:10]
    lines += ["", "### The 10 best sessions", "", "| Date | SOXX | Old label | New label |", "|---|---|---|---|"]
    lines += [f"| {r.date} | {r.soxx_pct:+.1f}% | {r.old} | {r.new} |" for r in best]
    return "\n".join(lines) + "\n"


Inputs = tuple[pd.Series | None, pd.Series | None, pd.Series | None, list[str], list[pd.Series]]


async def _series() -> Inputs:
    results = await fetch_all()
    extras = tally_extras(results)
    basket = [v for k, v in sorted(extras.items()) if k.startswith("lev_")]
    verified = verify_all(results, datetime.now(timezone.utc).date())
    notes = [f"{k}: {v.status.status} ({'; '.join(v.status.notes)})" for k, v in verified.items() if v.series is None]
    return verified["soxx"].series, verified["vix"].series, verified["y10"].series, notes, basket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=250)
    args = parser.parse_args(argv)
    soxx, vix, y10, notes, basket = asyncio.run(_series())
    if soxx is None or vix is None or y10 is None:
        print("Replay needs verified SOXX, VIX and 10Y series. Withheld this run:")
        for note in notes:
            print(f"  {note}")
        return 1
    text = render(replay(soxx, vix, y10, args.sessions, basket=basket))
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
