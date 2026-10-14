from __future__ import annotations

import pytest

from app.features import MarketFeatures
from app.regime import classify, decision_matrix, semi_momentum_label, telemetry


def calm(**overrides: object) -> MarketFeatures:
    base = MarketFeatures(
        tnx_pct=4.00,
        tnx_pctile=50.0,
        tnx_chg_1d_bps=1.0,
        tnx_chg_5d_bps=2.0,
        tnx_delta_z=0.3,
        vix=15.0,
        soxx_rsi=50.0,
        ratio_sma_rising=False,
        ratio_below_trough_mean=False,
        dxy=100.0,
        bill_3m_pct=4.30,
        effr_pct=4.33,
        effr_minus_3m_pct=0.03,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_spec_dead_zone_bug_squeeze_with_mid_rsi_is_phase_2() -> None:
    """Spec engine: squeeze=True, RSI 50 fell through to Phase 3 / LOW risk."""
    result = classify(calm(vix=24.0, soxx_rsi=50.0))
    assert result.spec.phase_id == 2
    assert result.spec.risk == "HIGH"


@pytest.mark.parametrize(
    ("overrides", "phase_id", "confidence"),
    [
        ({"tnx_pctile": 85.0}, 2, "FULL"),
        ({"tnx_pct": 5.22, "tnx_pctile": 60.0}, 0, "FULL"),  # high level alone no longer fires
        ({"tnx_delta_z": 2.1}, 2, "FULL"),
        ({"vix": 21.0, "soxx_rsi": 62.0}, 2, "FULL"),  # squeeze overrides strong semis
        ({"soxx_rsi": 60.0}, 1, "FULL"),
        ({"soxx_rsi": 60.0, "tnx_chg_5d_bps": 8.0}, 0, "FULL"),  # yields rising: no Phase 1
        ({"effr_minus_3m_pct": 0.40, "dxy": 100.0, "soxx_rsi": 60.0}, 4, "FULL"),  # 4 beats 1
        ({"effr_minus_3m_pct": 0.40, "dxy": 103.0}, 0, "FULL"),  # strong dollar blocks 4
        ({"ratio_sma_rising": True, "ratio_below_trough_mean": True}, 3, "PARTIAL"),
        ({"ratio_sma_rising": True, "ratio_below_trough_mean": False}, 0, "FULL"),
        ({}, 0, "FULL"),  # nothing fires -> Unclassified, never a default Phase 3
    ],
)
def test_phase_table(overrides: dict[str, object], phase_id: int, confidence: str) -> None:
    result = classify(calm(**overrides))
    assert (result.spec.phase_id, result.confidence) == (phase_id, confidence)


def test_missing_vix_with_high_yield_still_fires_squeeze() -> None:
    result = classify(calm(vix=None, tnx_pctile=90.0))
    assert result.spec.phase_id == 2 and result.confidence == "FULL"


def test_missing_vix_makes_lower_phase_provisional() -> None:
    result = classify(calm(vix=None, soxx_rsi=60.0))
    assert result.spec.phase_id == 1 and result.confidence == "PROVISIONAL"
    assert decision_matrix(calm(vix=None), result).macro_liquidity == "UNKNOWN"


def test_all_missing_is_unclassified_provisional() -> None:
    result = classify(MarketFeatures())
    assert result.spec.phase_id == 0 and result.confidence == "PROVISIONAL"
    assert all(r.fired is None for r in result.rules)


def test_arr_leg_is_flagged_unsourced() -> None:
    p3 = next(r for r in classify(calm()).rules if r.phase_id == 3)
    arr = next(c for c in p3.conditions if "ARR" in c.name)
    assert arr.unsourced and arr.met is None


@pytest.mark.parametrize(
    ("rsi", "label"),
    [(None, "UNKNOWN"), (55.01, "STRONG"), (55.0, "CHOPPY"), (40.0, "CHOPPY"), (39.99, "DIGESTION")],
)
def test_semi_momentum_labels(rsi: float | None, label: str) -> None:
    assert semi_momentum_label(rsi) == label


def test_telemetry_yield_spike_triggers_drain_below_level() -> None:
    signals = {s.indicator: s for s in telemetry(calm(tnx_pctile=40.0, tnx_chg_1d_bps=12.0))}
    assert signals["10Y Treasury Yield"].triggered is True


def test_telemetry_unknown_when_input_missing() -> None:
    signals = {s.indicator: s for s in telemetry(MarketFeatures())}
    assert all(s.triggered is None for s in signals.values())
