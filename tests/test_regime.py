from __future__ import annotations

from app.features import MarketFeatures
from app.regime import telemetry


def calm(**overrides: object) -> MarketFeatures:
    base = MarketFeatures(
        tnx_pct=4.00,
        tnx_pctile=50.0,
        tnx_chg_1d_bps=1.0,
        tnx_chg_5d_bps=2.0,
        tnx_delta_z=0.3,
        vix=15.0,
        vix_chg_1d_pct=1.0,
        soxx_chg_1d_pct=0.4,
        soxx_return_z=0.3,
        soxx_rsi=50.0,
        ratio_sma_rising=False,
        bill_3m_pct=4.30,
        effr_pct=4.33,
        effr_minus_3m_pct=0.03,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_telemetry_yield_spike_triggers_drain_below_level() -> None:
    signals = {s.indicator: s for s in telemetry(calm(tnx_pctile=40.0, tnx_chg_1d_bps=12.0))}
    assert signals["10Y Treasury Yield"].triggered is True


def test_telemetry_unknown_when_input_missing() -> None:
    signals = {s.indicator: s for s in telemetry(MarketFeatures())}
    assert all(s.triggered is None for s in signals.values())


def test_telemetry_names_the_source_each_value_came_from() -> None:
    """The table printed "Korea Exchange" for a KOSPI value that came from a web search."""
    sources = {"kospi": "Claude web search (en.sedaily.com)", "y10": "Cboe"}
    signals = {s.indicator: s for s in telemetry(calm(), sources)}
    assert signals["Korean Equity Pulse (KOSPI)"].source == "Claude web search (en.sedaily.com)"
    assert signals["10Y Treasury Yield"].source == "Cboe" and signals["Vol Index (VIX)"].source == "unavailable"


def test_the_fixed_dollar_level_trigger_is_gone() -> None:
    assert not [s for s in telemetry(calm()) if "Dollar" in s.indicator]
