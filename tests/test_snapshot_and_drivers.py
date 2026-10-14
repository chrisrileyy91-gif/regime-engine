from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import alert
from app.data import FetchResult
from app.drivers import backdrop, build_drivers, day_label, diagnose
from app.models import InputStatus, Preliminary
from app.snapshot import (
    HISTORY_LIMIT,
    load_krx_cache,
    load_sec_cache,
    resolve_status,
    save_krx_cache,
    save_sec_cache,
    write_outputs,
)
from tests.sec_fixture import sec_cache
from tests.test_regime import calm

DATES = {"y10": "2026-10-13", "vix": "2026-10-13", "soxx": "2026-10-13", "igv": "2026-10-13"}


def features(**overrides: object):  # type: ignore[no-untyped-def]
    f = calm(**overrides)
    f.dates = dict(DATES)
    return f


def states(**overrides: object) -> dict[str, str]:
    return {d.key: d.state for d in build_drivers(features(**overrides), {})}


def test_drivers_quiet_in_calm_market() -> None:
    assert states() == {"liquidity": "QUIET", "risk_off": "QUIET", "semis": "QUIET"}


def test_high_yield_level_alone_does_not_activate_liquidity() -> None:
    """First live read: 10Y 5.22% but falling. A level can't explain a one-day move."""
    assert states(tnx_pct=5.22, tnx_pctile=95.0, tnx_chg_1d_bps=-2.3, tnx_delta_z=-0.8)["liquidity"] == "QUIET"


def test_yield_jump_activates_liquidity() -> None:
    assert states(tnx_chg_1d_bps=12.0)["liquidity"] == "ACTIVE"


def test_context_metrics_never_trigger() -> None:
    assert states(vix_chg_1d_pct=40.0, soxx_chg_1d_pct=-6.0) == {
        "liquidity": "QUIET",
        "risk_off": "QUIET",
        "semis": "QUIET",
    }


def test_missing_vix_is_unknown_not_quiet() -> None:
    assert states(vix=None)["risk_off"] == "UNKNOWN"


def test_day_label_has_no_platform_specific_format() -> None:
    assert day_label("2026-10-05") == "Mon Oct 5"


def test_diagnosis_dates_the_move_and_names_same_session_drivers() -> None:
    f = features(vix=24.5, soxx_rsi=35.0, soxx_chg_1d_pct=-3.2, igv_chg_1d_pct=-1.0)
    text = diagnose(f, build_drivers(f, {}))
    assert text.startswith("Semis fell 3.2% on Tue Oct 13; software fell 1.0%.")
    assert "algorithmic risk-off (VIX 24.5)" in text and "semi momentum breakdown (SOXX RSI 35)" in text
    assert "liquidity" not in text


def test_first_live_read_no_longer_blames_the_level() -> None:
    f = features(tnx_pct=5.22, tnx_pctile=95.0, tnx_chg_1d_bps=-2.3, tnx_delta_z=-0.8, soxx_chg_1d_pct=-2.6)
    text = diagnose(f, build_drivers(f, {}))
    assert "5.22" not in text and "company or sector news" in text
    assert "5.22%" in backdrop(f)


def test_diagnosis_flags_rates_a_session_behind_stocks() -> None:
    f = features(soxx_chg_1d_pct=-2.6)
    f.dates["y10"] = "2026-10-09"
    text = diagnose(f, build_drivers(f, {}))
    assert "The official 10Y close for Tue Oct 13 isn't available yet, so that read uses Fri Oct 9." in text


def test_preliminary_attached_to_metric() -> None:
    prelim = Preliminary(value=5.20, date="2026-10-14", source="Yahoo")
    status = InputStatus(
        key="y10",
        label="10Y",
        status="VERIFIED",
        value=5.22,
        date="2026-10-13",
        source="U.S. Treasury",
        preliminary=prelim,
    )
    drivers = build_drivers(features(), {"y10": status})
    level = next(m for m in drivers[0].metrics if "backdrop" in m.label)
    assert level.preliminary == prelim


@pytest.mark.parametrize(
    ("effr", "two_year", "expected"),
    [(3.88, 4.80, "hikes, not cuts"), (4.33, 3.80, "points to cuts"), (4.33, 4.30, "no move is priced")],
)
def test_backdrop_policy_lean(effr: float, two_year: float, expected: str) -> None:
    f = features(effr_pct=effr, bill_3m_pct=two_year, effr_minus_3m_pct=effr - two_year)
    assert expected in backdrop(f)


def test_resolve_status_falls_back_to_previous_snapshot(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    live = resolve_status(calm_market, None, tmp_path)
    write_outputs(live, tmp_path / "state", tmp_path / "site")
    stale = resolve_status(None, "ConnectError: down", tmp_path / "state")
    assert stale.stale and stale.phase == live.phase
    assert stale.data_quality is not None and stale.data_quality.status == "STALE"


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
    assert history[-1]["data_quality"] == "OK"

    stale = resolve_status(None, "down", state)
    assert write_outputs(stale, state, site)[-1]["as_of"] == live.as_of
    assert (site / "index.html").exists() and json.loads((site / "regime.json").read_text())["stale"]


def test_krx_cache_roundtrip(tmp_path: Path) -> None:
    cache = {"kospi@krx": {"20261013": 6384.24, "20261009": None}}
    save_krx_cache(tmp_path, cache)
    assert load_krx_cache(tmp_path) == cache
    assert load_krx_cache(tmp_path / "missing") == {}


def test_alert_silent_when_clean(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    status = resolve_status(calm_market, None, tmp_path)
    assert alert.build_message(json.loads(status.model_dump_json()), "test") is None


def test_alert_lists_warnings_when_degraded(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    calm_market["soxx@alpaca"] = FetchResult("soxx@alpaca", "SOXX", "alpaca", None, "HTTPStatusError: 403")
    status = resolve_status(calm_market, None, tmp_path)
    message = alert.build_message(json.loads(status.model_dump_json()), "Regime (close)")
    assert message is not None and message.startswith("Regime read DEGRADED (Regime (close))")
    assert "SOXX close unavailable" in message


def test_alert_on_missing_read() -> None:
    assert "FAILED" in (alert.build_message(None, "Regime (morning)") or "")


def test_alert_refuses_non_discord_webhook() -> None:
    with pytest.raises(ValueError):
        alert.post("https://discordapp.com/api/webhooks/x", "hi")


def test_sec_cache_roundtrip(tmp_path: Path) -> None:
    cache = sec_cache()
    save_sec_cache(tmp_path, cache)
    assert load_sec_cache(tmp_path) == json.loads(json.dumps(cache))
    assert load_sec_cache(tmp_path / "missing") == {}


def test_published_read_carries_cycle_and_today(tmp_path: Path, calm_market: dict[str, FetchResult]) -> None:
    status = resolve_status(calm_market, None, tmp_path, sec_cache=sec_cache())
    assert (status.phase, status.phase_id, status.risk_level) == ("Buildout", 1, "CALM")
    assert status.cycle is not None and status.cycle.since == "Q1 2026"
    assert status.today is not None and status.tally is not None
    row = write_outputs(status, tmp_path / "state", tmp_path / "site")[-1]
    assert row["phase"] == "Buildout" and row["cycle_through"] == "Q2 2026" and row["risk"] == "CALM"


def test_old_snapshot_still_loads_as_stale(tmp_path: Path) -> None:
    """regime.json written by the previous version (rules, matrix, confidence) must still parse."""
    state = tmp_path / "state"
    state.mkdir()
    old = {
        "phase": "Phase 2: Valuation Digestion & Liquidity Squeeze",
        "risk_level": "HIGH",
        "liquidity_squeeze": True,
        "semi_rsi": 48.0,
        "tnx_yield": 5.2,
        "actionable_summary": "x",
        "phase_id": 2,
        "confidence": "FULL",
        "as_of": "2026-10-14T10:01:00+00:00",
        "matrix": {"macro_liquidity": "SQUEEZE"},
        "rules": [],
        "telemetry": [],
        "inputs": {},
        "sources": [],
        "notes": [],
    }
    (state / "regime.json").write_text(json.dumps(old))
    stale = resolve_status(None, "down", state)
    assert stale.stale and stale.cycle is None
