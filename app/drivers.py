"""The "why are semis moving?" layer.

Three drivers, each answering one question with official data:
  liquidity  — did yields jump?                    (10Y 1-session change, and how unusual it was)
  risk_off   — is volatility forcing de-risking?   (VIX level)
  semis      — has chip momentum broken?           (SOXX RSI)

Rule learned from the first live read: a one-day move can only be explained by something
that also moved that day. Levels (10Y at 5.2%, where bills sit vs fed funds) go in the
separate `backdrop` sentence, never in the diagnosis. Every value carries its date, and
when the rates data is a session behind the stock data, the diagnosis says so.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.config import THRESHOLDS, Thresholds
from app.features import MarketFeatures
from app.models import Condition, Driver, DriverMetric, InputStatus
from app.regime import _any, _gt, _lt


def _state(metrics: list[DriverMetric]) -> str:
    triggers = [Condition(name=m.label, value=m.value, threshold="", met=m.met) for m in metrics if m.trigger]
    return {True: "ACTIVE", False: "QUIET", None: "UNKNOWN"}[_any(triggers)]


def day_label(iso: str | None) -> str:
    if not iso:
        return "an unknown date"
    d = date.fromisoformat(iso)
    return f"{d:%a %b} {d.day}"  # no %-d: it breaks on Windows


SESSION_INPUTS = ("y10", "vix", "soxx", "igv")  # US-session inputs; the newest official date among them is the session


def session_day(f: MarketFeatures) -> str | None:
    """The US session being read: the newest official close among the US inputs."""
    days = [day for k in SESSION_INPUTS if (day := f.dates.get(k))]
    return max(days) if days else None


def bond_market_closed(iso: str) -> bool:
    """SIFMA's full closes on days stocks trade: Columbus Day (2nd Monday of October) and Veterans
    Day (Nov 11 on a weekday). No Treasury trading means no yield move, not an unknown one."""
    d = date.fromisoformat(iso)
    columbus = d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14
    veterans = d.month == 11 and d.day == 11 and d.weekday() < 5
    return columbus or veterans


def _same_session(met: bool | None, as_of: str | None, session: str | None, *, bond: bool = False) -> bool | None:
    """A trigger is read only on the session's own close. Nov 5 16:25: Cboe hadn't posted the VIX
    settle, so yesterday's 16.6 (not today's 28.2) kept risk-off QUIET and the page said Calm on a
    -12% semis day. A session-behind value is unknown, never quiet."""
    if met is None or as_of is None or session is None or as_of >= session:
        return met
    if bond and bond_market_closed(session) and as_of >= str(date.fromisoformat(session) - timedelta(days=4)):
        return False
    return None


def build_drivers(f: MarketFeatures, inputs: dict[str, InputStatus], t: Thresholds = THRESHOLDS) -> list[Driver]:
    d = f.dates
    prelim = {k: s.preliminary for k, s in inputs.items()}
    session = session_day(f)
    liquidity = [
        DriverMetric(
            label="10Y 1-session change",
            value=f.tnx_chg_1d_bps,
            unit="bps",
            trigger=f"> +{t.tnx_1d_spike_bps:.0f} bps",
            met=_same_session(_gt(f.tnx_chg_1d_bps, t.tnx_1d_spike_bps), d.get("y10"), session, bond=True),
            as_of=d.get("y10"),
        ),
        DriverMetric(
            label="How unusual that change was",
            value=f.tnx_delta_z,
            unit="z",
            trigger=f"> +{t.tnx_delta_z}σ",
            met=_same_session(_gt(f.tnx_delta_z, t.tnx_delta_z), d.get("y10"), session, bond=True),
            as_of=d.get("y10"),
        ),
        DriverMetric(
            label="10Y yield (backdrop)",
            value=f.tnx_pct,
            unit="%",
            trigger=None,
            met=None,
            as_of=d.get("y10"),
            preliminary=prelim.get("y10"),
        ),
    ]
    risk_off = [
        DriverMetric(
            label="VIX",
            value=f.vix,
            unit="pts",
            trigger=f"> {t.vix_level:.0f}",
            met=_same_session(_gt(f.vix, t.vix_level), d.get("vix"), session),
            as_of=d.get("vix"),
            preliminary=prelim.get("vix"),
        ),
        DriverMetric(
            label="VIX 1-session change", value=f.vix_chg_1d_pct, unit="chg", trigger=None, met=None, as_of=d.get("vix")
        ),
    ]
    semis = [
        DriverMetric(
            label="SOXX RSI(14)",
            value=f.soxx_rsi,
            unit="",
            trigger=f"< {t.semi_rsi_digestion:.0f}",
            met=_same_session(_lt(f.soxx_rsi, t.semi_rsi_digestion), d.get("soxx"), session),
            as_of=d.get("soxx"),
        ),
        DriverMetric(
            label="SOXX 1-session move",
            value=f.soxx_chg_1d_pct,
            unit="chg",
            trigger=None,
            met=None,
            as_of=d.get("soxx"),
            preliminary=prelim.get("soxx"),
        ),
    ]
    return [
        Driver(
            key="liquidity",
            name="Liquidity drain",
            asks="Did yields jump enough to pull money out of risk?",
            state=_state(liquidity),
            metrics=liquidity,
        ),
        Driver(
            key="risk_off",
            name="Algorithmic risk-off",
            asks="Is volatility forcing systematic selling?",
            state=_state(risk_off),
            metrics=risk_off,
        ),
        Driver(
            key="semis",
            name="Semi momentum breakdown",
            asks="Has chip momentum itself broken?",
            state=_state(semis),
            metrics=semis,
        ),
    ]


def _move(pct: float) -> str:
    if abs(pct) < 0.1:
        return "was flat"
    return f"{'rose' if pct > 0 else 'fell'} {abs(pct):.1f}%"


def _evidence(driver: Driver) -> str:
    hits = {m.unit: m for m in driver.metrics if m.met is True and m.value is not None}
    if driver.key == "liquidity":
        parts = []
        if "bps" in hits:
            parts.append(f"10Y {hits['bps'].value:+.0f} bps on {day_label(hits['bps'].as_of)}")
        if "z" in hits:
            parts.append(f"a {hits['z'].value:+.1f}σ move")
        return ", ".join(parts)
    if driver.key == "risk_off" and "pts" in hits:
        return f"VIX {hits['pts'].value:.1f}"
    if driver.key == "semis" and "" in hits:
        return f"SOXX RSI {hits[''].value:.0f}"
    return ""


def diagnose(f: MarketFeatures, drivers: list[Driver], t: Thresholds = THRESHOLDS) -> str:
    stock_day = f.dates.get("soxx")
    sentences = []
    if f.soxx_chg_1d_pct is not None:
        lead = f"Semis {_move(f.soxx_chg_1d_pct)} on {day_label(stock_day)}"
        if f.igv_chg_1d_pct is not None and f.dates.get("igv") == stock_day:
            lead += f"; software {_move(f.igv_chg_1d_pct)}"
        sentences.append(lead + ".")

    active = [d for d in drivers if d.state == "ACTIVE"]
    unknown = [d for d in drivers if d.state == "UNKNOWN"]

    # RSI below the line is a 14-session reading, not something that moved today. On an up day it
    # can't explain the move (Oct 21: SOXX +2.1% with RSI 39 was listed as a "breakdown"), so say
    # momentum is still weak and today's gain pushed against it, instead of calling it a driver.
    weak_momentum = None
    if f.soxx_chg_1d_pct is not None and f.soxx_chg_1d_pct >= 0.1:
        weak_momentum = next((d for d in active if d.key == "semis"), None)
        active = [d for d in active if d is not weak_momentum]
    if weak_momentum is not None:
        ev = _evidence(weak_momentum)
        reading = f" ({ev}, below {t.semi_rsi_digestion:.0f})" if ev else ""
        sentences.append(
            f"Chip momentum is still weak{reading}; today's gain was a bounce against it, not a breakdown."
        )

    if active:
        reasons = "; ".join(f"{d.name.lower()} ({ev})" if (ev := _evidence(d)) else d.name.lower() for d in active)
        sentences.append(f"Same-session drivers: {reasons}.")
    elif not unknown and weak_momentum is None:
        sentences.append(
            "None of the three drivers fired, so the move isn't explained by yields, volatility or chip "
            "momentum. Look at company or sector news."
        )
    if unknown:
        sentences.append(
            f"Couldn't check: {', '.join(d.name.lower() for d in unknown)} (no official close for this session)."
        )

    session = session_day(f)
    for key, what in (("y10", "10Y"), ("vix", "VIX"), ("soxx", "SOXX")):
        day = f.dates.get(key)
        if session and day and day < session:
            if key == "y10" and bond_market_closed(session):
                sentences.append(f"The bond market was closed {day_label(session)}, so yields didn't move.")
                continue
            sentences.append(
                f"The official {what} close for {day_label(session)} isn't in yet; "
                f"{day_label(day)}'s isn't used for today's read."
            )
    return " ".join(sentences)


def backdrop(f: MarketFeatures, t: Thresholds = THRESHOLDS) -> str:
    """Slow-moving context. Explains the regime, never a single day's move."""
    parts = []
    if f.tnx_pct is not None:
        text = f"10Y at {f.tnx_pct:.2f}% ({day_label(f.dates.get('y10'))})"
        if f.tnx_pctile is not None:
            text += f", higher than {f.tnx_pctile:.0f}% of the past year"
        parts.append(text + ".")
    if f.effr_minus_3m_pct is not None and f.bill_3m_pct is not None and f.effr_pct is not None:
        gap = -f.effr_minus_3m_pct
        if gap > t.cuts_priced_spread_pct:
            lean = "which points to hikes, not cuts"
        elif gap < -t.cuts_priced_spread_pct:
            lean = "which points to cuts"
        else:
            lean = "so no move is priced"
        parts.append(f"3-month bill at {f.bill_3m_pct:.2f}% vs fed funds {f.effr_pct:.2f}%, {lean}.")
    return " ".join(parts)
