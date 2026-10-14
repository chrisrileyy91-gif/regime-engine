from __future__ import annotations

import pandas as pd
import pytest

from app.config import INPUTS
from app.verify import assess, verify_all, verify_input
from tests.conftest import TODAY, failed, flat_then, market, ok, series

TODAY_DATE = TODAY.date()


def check(results: dict, key: str):  # type: ignore[no-untyped-def]
    return verify_input(key, INPUTS[key], results, TODAY_DATE)


def test_everything_agrees(calm_market: dict) -> None:
    verified = verify_all(calm_market, TODAY_DATE)
    statuses = {k: v.status.status for k, v in verified.items()}
    # EFFR has no second copy; the 3M bill's only other source (Cboe ^IRX) is a different basis.
    assert statuses == {**dict.fromkeys(INPUTS, "VERIFIED"), "effr": "OFFICIAL", "t3m": "OFFICIAL"}
    quality = assess(verified)
    assert quality.status == "OK" and not [i for i in quality.issues if i.severity == "warn"]


def test_cboe_10y_confirmed_by_yahoo_not_by_treasury(calm_market: dict) -> None:
    v = check(calm_market, "y10")
    assert v.status.source == "Cboe" and v.status.confirmed_by == ["Yahoo"]
    assert any("consistent with U.S. Treasury" in n for n in v.status.notes)


def test_record_missing_is_withheld_even_if_yahoo_has_it() -> None:
    results = market()
    results["soxx@alpaca"] = failed("soxx@alpaca", "not configured")
    v = check(results, "soxx")
    assert v.status.status == "MISSING" and v.series is None and v.status.value is None
    assert v.status.preliminary is not None  # Yahoo's number is shown as preliminary, never used


def test_methodology_check_cannot_confirm() -> None:
    """Treasury par differs in method from Cboe TNX: it can veto, never confirm."""
    results = market()
    results["y10@yahoo"] = failed("y10@yahoo")
    v = check(results, "y10")
    assert v.status.status == "OFFICIAL" and v.status.confirmed_by == []


def test_one_day_methodology_disagreement_is_noted_when_a_copy_confirms() -> None:
    results = market()
    treasury = results["y10@treasury"].series.copy()  # type: ignore[union-attr]
    treasury.iloc[-2] += 0.30  # one odd day
    results["y10@treasury"] = ok("y10@treasury", treasury)
    v = check(results, "y10")
    assert v.status.status == "VERIFIED"
    assert any("differs, but Yahoo confirms" in n for n in v.status.notes)


def test_systematic_disagreement_withholds_even_when_a_copy_confirms() -> None:
    """First live run: Cboe ^IRX sat 7-10 bp under Treasury every day; Yahoo (a copy) agreed."""
    results = market()
    results["y10@treasury"] = ok("y10@treasury", results["y10@treasury"].series + 0.30)  # type: ignore[operator]
    v = check(results, "y10")
    assert v.status.status == "CONFLICT" and v.series is None
    assert "systematically" in v.status.notes[0]


def test_3m_bill_record_is_treasury_with_cboe_as_veto_only() -> None:
    results = market()
    results["t3m@cboe"] = ok("t3m@cboe", results["t3m@cboe"].series - 0.09)  # type: ignore[operator]
    v = check(results, "t3m")
    assert v.status.status == "OFFICIAL" and v.status.source == "U.S. Treasury"
    assert any("consistent with Cboe" in n for n in v.status.notes)


def test_disagreement_with_nothing_confirming_is_conflict() -> None:
    results = market()
    results["soxx@yahoo"] = ok("soxx@yahoo", results["soxx@yahoo"].series * 1.02)  # type: ignore[operator]
    v = check(results, "soxx")
    assert v.status.status == "CONFLICT" and v.series is None


def test_vendor_record_without_confirmation_is_withheld() -> None:
    results = market()
    results["dxy@yahoo"] = failed("dxy@yahoo")
    v = check(results, "dxy")
    assert v.status.status == "UNCONFIRMED" and v.series is None


def test_ecb_vetoes_gross_dxy_error() -> None:
    results = market()
    results["dxy@yahoo"] = failed("dxy@yahoo")
    results["dxy@eodhd"] = ok("dxy@eodhd", results["dxy@eodhd"].series * 1.05)  # type: ignore[operator]
    assert check(results, "dxy").status.status == "CONFLICT"


def test_korea_eodhd_newer_than_krx_history_matches() -> None:
    results = market()
    krx = results["kospi@krx"].series
    assert krx is not None
    results["kospi@krx"] = ok("kospi@krx", krx.iloc[:-1])  # KRX posts a session late
    v = check(results, "kospi")
    assert v.status.status == "VERIFIED" and v.status.source == "EODHD"
    assert v.status.confirmed_by == ["Yahoo"] and any("hasn't posted" in n for n in v.status.notes)


def test_korea_records_disagree_is_conflict() -> None:
    results = market()
    results["samsung@krx"] = ok("samsung@krx", results["samsung@krx"].series + 500)  # type: ignore[operator]
    assert check(results, "samsung").status.status == "CONFLICT"


def test_bad_print_outside_bounds_is_withheld() -> None:
    results = market(vix=flat_then(15.0, tail=[1500.0]))
    assert check(results, "vix").status.status == "CONFLICT"


def test_check_newer_than_record_becomes_preliminary() -> None:
    record = flat_then(4.0)
    results = market()
    results["y10@cboe"] = ok("y10@cboe", record.iloc[:-1])  # Cboe quote not final yet
    v = check(results, "y10")
    assert v.status.status == "VERIFIED" and v.status.date == str(record.index[-2].date())
    assert v.status.preliminary is not None and v.status.preliminary.date == str(record.index[-1].date())
    issues = assess(verify_all(results, TODAY_DATE)).issues
    assert any("not used until official" in i.message for i in issues)


def test_stale_record_degrades_quality() -> None:
    results = market()
    results["effr@nyfed"] = ok("effr@nyfed", series([3.88] * 30, end=TODAY - pd.Timedelta(days=20)))
    verified = verify_all(results, TODAY_DATE)
    assert verified["effr"].status.stale
    assert assess(verified).status == "DEGRADED"


def test_korea_missing_warns_but_does_not_degrade_the_call() -> None:
    results = market()
    results["kospi@eodhd"] = failed("kospi@eodhd", "not configured (EODHD_API_KEY)")
    results["kospi@krx"] = failed("kospi@krx", "not configured (KRX_API_KEY)")
    quality = assess(verify_all(results, TODAY_DATE))
    assert quality.status == "OK"
    assert any("KOSPI unavailable" in i.message for i in quality.issues)


@pytest.mark.parametrize("key", list(INPUTS))
def test_every_input_has_a_record_source(key: str) -> None:
    assert INPUTS[key].record
