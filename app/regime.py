"""Regime classification. Pure function of MarketFeatures — no I/O.

Design changes from the original spec:
  * Precedence is explicit: Phase 2 (risk override) > 4 > 1 > 3. The spec's
    if/elif chain sent "squeeze + semi RSI 45-55" to Phase 3 / LOW risk.
  * Phase 3 is no longer the `else` bucket. It needs its own evidence; when no
    rule fires the state is Unclassified, not "fundamentals stabilizing".
  * Kleene three-valued logic: a missing input is unknown, never false. If a
    higher-precedence rule is unknown, a lower phase is reported PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import THRESHOLDS, Thresholds
from app.features import MarketFeatures
from app.models import Condition, DecisionMatrix, RuleEvaluation, TelemetrySignal


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


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: int
    name: str
    risk: str
    summary: str


PHASES: dict[int, PhaseSpec] = {
    1: PhaseSpec(
        1,
        "Phase 1: CapEx & Infrastructure Buildout",
        "MODERATE",
        "Hardware momentum dominant with yields contained. Highly sensitive to a rate spike.",
    ),
    2: PhaseSpec(
        2,
        "Phase 2: Valuation Digestion & Liquidity Squeeze",
        "HIGH",
        "Yield pressure or volatility regime is driving price. Macro dominates single-name narratives.",
    ),
    3: PhaseSpec(
        3,
        "Phase 3: Software Monetization Transition",
        "LOW",
        "Software rotating up versus semis off a trough. Revenue-growth leg not verified by data.",
    ),
    4: PhaseSpec(
        4,
        "Phase 4: Rate Easing",
        "LOW",
        "Market pricing cuts with a soft dollar. Liquidity tailwind for duration and foreign revenue.",
    ),
}
UNCLASSIFIED = PhaseSpec(
    0, "Unclassified / Transition", "UNDETERMINED", "No phase rule is satisfied. Signals are mixed; no regime call."
)

PRECEDENCE = (2, 4, 1, 3)


def evaluate_rules(f: MarketFeatures, t: Thresholds = THRESHOLDS) -> list[RuleEvaluation]:
    p2 = [
        Condition(
            name="10Y vs its past year (percentile)",
            value=f.tnx_pctile,
            threshold=f"> {t.tnx_pctile_high:.0f}th percentile of {t.tnx_pctile_window} sessions",
            met=_gt(f.tnx_pctile, t.tnx_pctile_high),
        ),
        Condition(
            name="10Y 1d change z-score",
            value=f.tnx_delta_z,
            threshold=f"> +{t.tnx_delta_z} ({t.tnx_delta_z_window}d baseline)",
            met=_gt(f.tnx_delta_z, t.tnx_delta_z),
        ),
        Condition(name="VIX", value=f.vix, threshold=f"> {t.vix_level}", met=_gt(f.vix, t.vix_level)),
    ]
    p4 = [
        Condition(
            name="EFFR minus 3M bill (pp) — cuts-priced proxy",
            value=f.effr_minus_3m_pct,
            threshold=f"> {t.cuts_priced_spread_pct}",
            met=_gt(f.effr_minus_3m_pct, t.cuts_priced_spread_pct),
        ),
        Condition(name="DXY", value=f.dxy, threshold=f"< {t.dxy_easing_max}", met=_lt(f.dxy, t.dxy_easing_max)),
    ]
    p1 = [
        Condition(
            name="SOXX RSI(14) Wilder",
            value=f.soxx_rsi,
            threshold=f"> {t.semi_rsi_strong}",
            met=_gt(f.soxx_rsi, t.semi_rsi_strong),
        ),
        Condition(
            name="10Y 5-session change (bps)",
            value=f.tnx_chg_5d_bps,
            threshold=f"< +{t.tnx_5d_slope_max_bps}",
            met=_lt(f.tnx_chg_5d_bps, t.tnx_5d_slope_max_bps),
        ),
    ]
    p3 = [
        Condition(
            name=f"IGV/SOXX SMA{t.ratio_sma_window} rising ({t.ratio_slope_lookback}d)",
            value=f.ratio_sma_rising,
            threshold="true",
            met=f.ratio_sma_rising,
        ),
        Condition(
            name=f"IGV/SOXX below {t.ratio_trough_window}d mean (still in trough)",
            value=f.ratio_below_trough_mean,
            threshold="true",
            met=f.ratio_below_trough_mean,
        ),
        Condition(
            name="Enterprise SaaS ARR growth (%)",
            value=None,
            threshold=f"> {t.saas_arr_growth_pct}",
            met=None,
            unsourced=True,
        ),
    ]
    return [
        RuleEvaluation(phase_id=2, phase=PHASES[2].name, logic="ANY", fired=_any(p2), conditions=p2),
        RuleEvaluation(phase_id=4, phase=PHASES[4].name, logic="ALL", fired=_all(p4), conditions=p4),
        RuleEvaluation(phase_id=1, phase=PHASES[1].name, logic="ALL", fired=_all(p1), conditions=p1),
        RuleEvaluation(phase_id=3, phase=PHASES[3].name, logic="ALL", fired=_all(p3), conditions=p3),
    ]


@dataclass(frozen=True)
class Classification:
    spec: PhaseSpec
    confidence: str
    rules: list[RuleEvaluation]


def classify(f: MarketFeatures, t: Thresholds = THRESHOLDS) -> Classification:
    rules = evaluate_rules(f, t)
    by_id = {r.phase_id: r for r in rules}
    unknown_above = False
    for phase_id in PRECEDENCE:
        rule = by_id[phase_id]
        if rule.fired is True:
            if unknown_above:
                confidence = "PROVISIONAL"
            elif any(c.unsourced for c in rule.conditions):
                confidence = "PARTIAL"
            else:
                confidence = "FULL"
            return Classification(PHASES[phase_id], confidence, rules)
        if rule.fired is None:
            unknown_above = True
    return Classification(UNCLASSIFIED, "PROVISIONAL" if unknown_above else "FULL", rules)


def semi_momentum_label(rsi: float | None, t: Thresholds = THRESHOLDS) -> str:
    if rsi is None:
        return "UNKNOWN"
    if rsi > t.semi_rsi_strong:
        return "STRONG"
    if rsi < t.semi_rsi_digestion:
        return "DIGESTION"
    return "CHOPPY"


def decision_matrix(f: MarketFeatures, c: Classification, t: Thresholds = THRESHOLDS) -> DecisionMatrix:
    squeeze = next(r.fired for r in c.rules if r.phase_id == 2)
    macro = {True: "SQUEEZE", False: "CLEAR", None: "UNKNOWN"}[squeeze]
    state = "UNCLASSIFIED" if c.spec.phase_id == 0 else f"PHASE {c.spec.phase_id}"
    return DecisionMatrix(macro_liquidity=macro, semi_momentum=semi_momentum_label(f.soxx_rsi, t), current_state=state)


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
            signal="PHASE 3 TRANSITION",
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
