"""Market signals (telemetry) and three-valued logic helpers. Pure functions, no I/O.

The phase call no longer comes from prices: it comes from company filings (fundamentals.py),
because the phases are defined by capex, chip revenue and software revenue and last quarters,
not days. The price-based phase rules this module used to hold flipped with daily moves and
kept the page on "HIGH RISK" for months while the 10Y sat at a high level. What remains here
are the daily signals, shown as context. Missing data is unknown, never false (Kleene logic).
"""

from __future__ import annotations

from app.config import THRESHOLDS, Thresholds
from app.features import MarketFeatures
from app.models import Condition, TelemetrySignal


def _gt(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value > threshold


def _lt(value: float | None, threshold: float) -> bool | None:
    return None if value is None else value < threshold


def _any(conditions: list[Condition]) -> bool | None:
    states = [c.met for c in conditions if not c.unsourced]
    if any(s is True for s in states):
        return True
    if states and all(s is False for s in states):
        return False
    return None


def _all(conditions: list[Condition]) -> bool | None:
    states = [c.met for c in conditions if not c.unsourced]
    if any(s is False for s in states):
        return False
    if states and all(s is True for s in states):
        return True
    return None


def telemetry(f: MarketFeatures, t: Thresholds = THRESHOLDS) -> list[TelemetrySignal]:
    tnx_level = _gt(f.tnx_pctile, t.tnx_pctile_high)
    tnx_spike = _gt(f.tnx_chg_1d_bps, t.tnx_1d_spike_bps)
    tnx_triggered = _any(
        [
            Condition(name="lvl", value=f.tnx_pctile, threshold="", met=tnx_level),
            Condition(name="spike", value=f.tnx_chg_1d_bps, threshold="", met=tnx_spike),
        ]
    )
    ks11_above = None
    if f.ks11 is not None and f.ks11_sma200 is not None:
        ks11_above = f.ks11 > f.ks11_sma200
    return [
        TelemetrySignal(
            indicator="10Y Treasury Yield",
            source="U.S. Treasury",
            value=f.tnx_pct,
            trigger=(
                f"above the {t.tnx_pctile_high:.0f}th percentile of its past year, "
                f"or +{t.tnx_1d_spike_bps:.0f} bps in a session"
            ),
            triggered=tnx_triggered,
            signal="LIQUIDITY DRAIN",
        ),
        TelemetrySignal(
            indicator="Vol Index (VIX)",
            source="Cboe",
            value=f.vix,
            trigger=f"> {t.vix_level}",
            triggered=_gt(f.vix, t.vix_level),
            signal="ALGORITHMIC RISK-OFF",
        ),
        TelemetrySignal(
            indicator="Semi Momentum (SOXX RSI14)",
            source="consolidated tape",
            value=f.soxx_rsi,
            trigger=f"< {t.semi_rsi_digestion}",
            triggered=_lt(f.soxx_rsi, t.semi_rsi_digestion),
            signal="HARDWARE DIGESTION",
        ),
        TelemetrySignal(
            indicator="Semi Momentum (Samsung RSI14)",
            source="Korea Exchange",
            value=f.samsung_rsi,
            trigger=f"< {t.semi_rsi_digestion}",
            triggered=_lt(f.samsung_rsi, t.semi_rsi_digestion),
            signal="HARDWARE DIGESTION",
        ),
        TelemetrySignal(
            indicator="Software vs Hardware (IGV/SOXX)",
            source="consolidated tape",
            value=f.igv_soxx_ratio,
            trigger=f"SMA{t.ratio_sma_window} rising over {t.ratio_slope_lookback} sessions",
            triggered=f.ratio_sma_rising,
            signal="SOFTWARE OUTPERFORMING",
            tone="constructive",
        ),
        TelemetrySignal(
            indicator="Korean Equity Pulse (KOSPI)",
            source="Korea Exchange",
            value=f.ks11,
            trigger=f"close above SMA{t.ks11_sma_window}",
            triggered=ks11_above,
            signal="EMERGING MARKET RE-RATE",
            tone="constructive",
        ),
        TelemetrySignal(
            indicator="US Dollar Index",
            source="ECB rates, ICE formula",
            value=f.dxy,
            trigger=f"> {t.dxy_compression}",
            triggered=_gt(f.dxy, t.dxy_compression),
            signal="FOREIGN REVENUE COMPRESSION",
        ),
    ]
