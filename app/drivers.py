"""The "why is tech moving?" layer.

Three drivers, each answering one question with observable data:
  liquidity  — are yields draining liquidity?          (10Y level, 1d change, 1d change z-score)
  risk_off   — is volatility forcing systematic de-risking? (VIX level)
  semis      — has semiconductor momentum broken down?  (SOXX RSI)

The original Gemini framing called the third one "structural margin compression". Margins are
a fundamental that prices cannot show, so this driver is labelled for what it measures:
momentum. Thresholds come from config.py; nothing new is invented here.
"""

from __future__ import annotations

from app.config import THRESHOLDS, Thresholds
from app.features import MarketFeatures
from app.models import Condition, Driver, DriverMetric
from app.regime import _any, _gt, _lt


def _state(metrics: list[DriverMetric]) -> str:
    triggers = [Condition(name=m.label, value=m.value, threshold="", met=m.met) for m in metrics if m.trigger]
    fired = _any(triggers)
    return {True: "ACTIVE", False: "QUIET", None: "UNKNOWN"}[fired]


def build_drivers(f: MarketFeatures, t: Thresholds = THRESHOLDS) -> list[Driver]:
    liquidity = [
        DriverMetric(
            label="10Y yield",
            value=f.tnx_pct,
            unit="%",
            trigger=f"> {t.tnx_level_pct}%",
            met=_gt(f.tnx_pct, t.tnx_level_pct),
        ),
        DriverMetric(
            label="10Y 1-day change",
            value=f.tnx_chg_1d_bps,
            unit="bps",
            trigger=f"> +{t.tnx_1d_spike_bps:.0f} bps",
            met=_gt(f.tnx_chg_1d_bps, t.tnx_1d_spike_bps),
        ),
        DriverMetric(
            label="1-day change vs normal",
            value=f.tnx_delta_z,
            unit="z",
            trigger=f"> +{t.tnx_delta_z}σ",
            met=_gt(f.tnx_delta_z, t.tnx_delta_z),
        ),
    ]
    risk_off = [
        DriverMetric(label="VIX", value=f.vix, unit="pts", trigger=f"> {t.vix_level:.0f}", met=_gt(f.vix, t.vix_level)),
        DriverMetric(label="VIX 1-day change", value=f.vix_chg_1d_pct, unit="chg", trigger=None, met=None),
    ]
    semis = [
        DriverMetric(
            label="SOXX RSI(14)",
            value=f.soxx_rsi,
            unit="",
            trigger=f"< {t.semi_rsi_digestion:.0f}",
            met=_lt(f.soxx_rsi, t.semi_rsi_digestion),
        ),
        DriverMetric(label="SOXX 1-day move", value=f.soxx_chg_1d_pct, unit="chg", trigger=None, met=None),
    ]
    return [
        Driver(
            key="liquidity",
            name="Liquidity drain",
            asks="Are rising yields pulling money out of risk?",
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


def _fmt_move(pct: float) -> str:
    if abs(pct) < 0.1:
        return "flat"
    return f"{'up' if pct > 0 else 'down'} {abs(pct):.1f}%"


def _evidence(d: Driver) -> str:
    """Short, non-repetitive evidence for a driver's triggered metrics."""
    hit = {m.unit: m.value for m in d.metrics if m.met is True and m.value is not None}
    if d.key == "liquidity":
        parts = []
        level = next((m.value for m in d.metrics if m.unit == "%" and m.value is not None), None)
        if level is not None:
            parts.append(f"10Y at {level:.2f}%")
        if "bps" in hit:
            parts.append(f"{hit['bps']:+.0f} bps on the day")
        if "z" in hit:
            parts.append(f"a {hit['z']:+.1f}σ move")
        return ", ".join(parts)
    if d.key == "risk_off":
        return f"VIX {hit['pts']:.1f}" if "pts" in hit else ""
    if d.key == "semis":
        return f"SOXX RSI {hit['']:.0f}" if "" in hit else ""
    return ""


def diagnose(f: MarketFeatures, drivers: list[Driver]) -> str:
    lead = ""
    if f.soxx_chg_1d_pct is not None:
        lead = f"Semis are {_fmt_move(f.soxx_chg_1d_pct)} on the session"
        if f.igv_chg_1d_pct is not None:
            lead += f", software {_fmt_move(f.igv_chg_1d_pct)}"
        lead += ". "

    active = [d for d in drivers if d.state == "ACTIVE"]
    unknown = [d for d in drivers if d.state == "UNKNOWN"]
    if active:
        reasons = "; ".join(f"{d.name.lower()} ({_evidence(d)})" if _evidence(d) else d.name.lower() for d in active)
        body = f"Driving it: {reasons}."
    elif unknown:
        names = ", ".join(d.name.lower() for d in unknown)
        body = f"No active driver among those with data; {names} could not be checked."
    else:
        body = (
            "None of the three drivers is active, so the move is not coming from yields, volatility "
            "or semi momentum. Look at company-specific news."
        )
    return lead + body
