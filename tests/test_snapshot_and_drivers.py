from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import FetchResult
from app.drivers import build_drivers, diagnose
from app.snapshot import HISTORY_LIMIT, resolve_status, write_outputs
from tests.test_regime import calm


def _drivers(**overrides: object) -> dict[str, str]:
    return {d.key: d.state for d in build_drivers(calm(**overrides))}


def test_drivers_quiet_in_calm_market() -> None:
    assert _drivers() == {"liquidity": "QUIET", "risk_off": "QUIET", "semis": "QUIET"}


def test_yield_spike_below_level_activates_liquidity() -> None:
    assert _drivers(tnx_chg_1d_bps=12.0)["liquidity"] == "ACTIVE"


def test_context_metrics_never_trigger() -> None:
    assert _drivers(vix_chg_1d_pct=40.0, soxx_chg_1d_pct=-6.0) == {
        "liquidity": "QUIET",
        "risk_off": "QUIET",
        "semis": "QUIET",
    }


def test_missing_vix_is_unknown_not_quiet() -> None:
    assert _drivers(vix=None)["risk_off"] == "UNKNOWN"


def test_diagnosis_names_active_drivers_with_evidence() -> None:
    f = calm(vix=24.5, soxx_rsi=35.0, soxx_chg_1d_pct=-3.2, igv_chg_1d_pct=-1.0)
    text = diagnose(f, build_drivers(f))
    assert text.startswith("Semis are down 3.2% on the session, software down 1.0%.")
    assert "algorithmic risk-off (VIX 24.5)" in text and "semi momentum breakdown" in text
    assert "liquidity" not in text


def test_diagnosis_points_to_news_when_nothing_active() -> None:
    f = calm(soxx_chg_1d_pct=-2.5)
    assert "company-specific news" in diagnose(f, build_drivers(f))


def test_resolve_status_falls_back_to_previous_snapshot(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    live = resolve_status(calm_market, None, tmp_path)
    write_outputs(live, tmp_path / "state", tmp_path / "site")
    stale = resolve_status(None, "ConnectError: down", tmp_path / "state")
    assert stale.stale and "down" in (stale.stale_reason or "") and stale.phase == live.phase


def test_resolve_status_exits_with_nothing_to_show(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        resolve_status(None, "down", tmp_path)


def test_history_appends_live_skips_stale_and_caps(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    state, site = tmp_path / "state", tmp_path / "site"
    state.mkdir()
    (state / "history.json").write_text(json.dumps([{"as_of": "old"}] * HISTORY_LIMIT))
    live = resolve_status(calm_market, None, state)
    history = write_outputs(live, state, site)
    assert len(history) == HISTORY_LIMIT and history[-1]["as_of"] == live.as_of

    stale = resolve_status(None, "down", state)
    assert write_outputs(stale, state, site)[-1]["as_of"] == live.as_of
    assert (site / "index.html").exists() and json.loads((site / "regime.json").read_text())["stale"]
