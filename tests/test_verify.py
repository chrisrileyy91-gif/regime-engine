from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.config import INPUTS
from app.data import Split
from app.verify import Verified, assess, verify_all, verify_input
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
    results["kospi@yahoo"] = failed("kospi@yahoo")
    results["kospi@krx"] = failed("kospi@krx")
    v = check(results, "kospi")
    assert v.status.status == "UNCONFIRMED" and v.series is None


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


def test_telemetry_gaps_are_info_not_warnings() -> None:
    """The banner listed KOSPI/Samsung (optional keys unset) as warnings on every run."""
    from app.verify import assess

    results = market()
    for key in [k for k in results if k.startswith(("kospi@", "samsung@"))]:
        results[key] = failed(key, "not configured")
    quality = assess(verify_all(results, TODAY.date()))
    assert quality.status == "OK"
    assert not [i for i in quality.issues if i.severity == "warn"]


# --- Splits (Nov 5, 2026: SOXX 3-for-1, effective after the Nov 4 close) ---------------------
# The 06:00 run withheld SOXX: Alpaca's split-adjusted closes were exactly 1/3 of Yahoo's,
# which hadn't re-based yet. These are the numbers from that run's warning.
SPLIT_DAYS = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"])
ALPACA_ADJUSTED = [171.3, 157.2, 151.2, 155.0, 150.1]
YAHOO_RAW = [513.9, 471.5, 453.6, 465.0, 450.4]
SPLIT_DAY = date(2026, 11, 5)
SOXX_SPLIT = Split("SOXX", pd.Timestamp("2026-11-05"), 3.0)


def split_market(
    record: list[float],
    yahoo: list[float],
    days: pd.DatetimeIndex = SPLIT_DAYS,
    splits: list[Split] | None = None,
    splits_error: str | None = None,
) -> dict:  # type: ignore[type-arg]
    results = market()
    alpaca = ok("soxx@alpaca", pd.Series(record, index=days))
    alpaca.splits = [SOXX_SPLIT] if splits is None else splits
    alpaca.splits_error = splits_error
    results["soxx@alpaca"] = alpaca
    results["soxx@yahoo"] = ok("soxx@yahoo", pd.Series(yahoo, index=days))
    return results


def split_check(results: dict) -> Verified:  # type: ignore[type-arg]
    return verify_input("soxx", INPUTS["soxx"], results, SPLIT_DAY)


def test_check_that_has_not_applied_a_listed_split_is_rebased_and_confirms() -> None:
    v = split_check(split_market(ALPACA_ADJUSTED, YAHOO_RAW))
    assert v.status.status == "VERIFIED" and v.status.confirmed_by == ["Yahoo"]
    assert v.status.value == pytest.approx(150.1) and v.status.date == "2026-11-04"
    assert any("hasn't applied SOXX's 3-for-1 split (ex-date 2026-11-05)" in n for n in v.status.notes)
    assert v.series is not None and list(v.series) == ALPACA_ADJUSTED  # the record is never touched


def test_rebase_only_touches_dates_before_the_ex_date() -> None:
    """16:25 on the ex-date: Yahoo's new bar is post-split, its history still isn't."""
    days = SPLIT_DAYS.append(pd.DatetimeIndex(["2026-11-05"]))
    v = split_check(split_market([*ALPACA_ADJUSTED, 152.0], [*YAHOO_RAW, 152.1], days))
    assert v.status.status == "VERIFIED" and v.status.date == "2026-11-05"


def test_split_listed_but_gap_is_not_the_split_factor_still_withholds() -> None:
    """A listed split never excuses a different disagreement."""
    v = split_check(split_market(ALPACA_ADJUSTED, [x * 3.06 for x in ALPACA_ADJUSTED]))
    assert v.status.status == "CONFLICT" and v.series is None


def test_split_shaped_gap_with_no_listed_split_withholds_and_says_why() -> None:
    v = split_check(split_market(ALPACA_ADJUSTED, YAHOO_RAW, splits=[]))
    assert v.status.status == "CONFLICT" and v.series is None
    assert "constant 3.00x gap looks like a split, but no split is listed for SOXX" in v.status.notes[0]


def test_split_lookup_failure_is_named_in_the_reason() -> None:
    v = split_check(split_market(ALPACA_ADJUSTED, YAHOO_RAW, splits=[], splits_error="HTTP 403"))
    assert v.status.status == "CONFLICT"
    assert "the split lookup failed (HTTP 403)" in v.status.notes[0]


def test_record_that_has_not_applied_a_listed_split_is_withheld() -> None:
    """Yahoo re-based, Alpaca didn't: the record is on the old basis."""
    v = split_check(split_market(YAHOO_RAW, ALPACA_ADJUSTED))
    assert v.status.status == "CONFLICT" and v.series is None
    assert "consolidated tape (Alpaca) hasn't applied SOXX's 3-for-1 split" in v.status.notes[0]


def test_unadjusted_record_across_the_ex_date_is_withheld() -> None:
    """Neither source re-based: they agree, but the record shows the split as a -67% day."""
    days = SPLIT_DAYS.append(pd.DatetimeIndex(["2026-11-05"]))
    v = split_check(split_market([*YAHOO_RAW, 152.0], [*YAHOO_RAW, 152.0], days))
    assert v.status.status == "CONFLICT" and "jump -66%" in v.status.notes[0]


# The 16:25 run on Nov 5 (after the first split fix): Yahoo had re-based Oct 30 but not its
# Nov 2-4 bars, so only 3 of 5 dates were 3x and SOXX was still withheld. Nov 5 is Yahoo's
# preliminary close from that run.
CLOSE_DAYS = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"])
ALPACA_CLOSE_RUN = [*ALPACA_ADJUSTED, 131.9]
YAHOO_HALF_REBASED = [171.3, 157.2, 453.6, 465.0, 450.4, 131.9]


def test_check_rebased_on_only_some_dates_is_rebased_per_date() -> None:
    v = split_check(split_market(ALPACA_CLOSE_RUN, YAHOO_HALF_REBASED, CLOSE_DAYS))
    assert v.status.status == "VERIFIED" and v.status.date == "2026-11-05"
    assert v.status.value == pytest.approx(131.9)
    note = "Yahoo hasn't applied SOXX's 3-for-1 split (ex-date 2026-11-05) to 2026-11-02, 2026-11-03, 2026-11-04"
    assert any(note in n for n in v.status.notes)
    assert v.series is not None and list(v.series) == ALPACA_CLOSE_RUN  # the record is never touched


def test_partial_rebase_never_excuses_a_date_off_by_something_else() -> None:
    yahoo = [171.3, 157.2 * 1.5, 453.6, 465.0, 450.4, 131.9]  # Oct 30 is neither 1x nor 3x
    v = split_check(split_market(ALPACA_CLOSE_RUN, yahoo, CLOSE_DAYS))
    assert v.status.status == "CONFLICT" and v.series is None


def test_record_rebased_on_only_part_of_its_history_is_withheld() -> None:
    """Alpaca half-adjusted: Yahoo's mixed basis would 'agree' date by date, so the record's own
    step across two pre-split closes is what has to catch it."""
    alpaca = [513.9, 471.5, 151.2, 155.0, 150.1, 131.9]
    yahoo = [513.9, 471.5, 453.6, 465.0, 450.4, 131.9]  # agrees with Alpaca on Oct 29-30
    v = split_check(split_market(alpaca, yahoo, CLOSE_DAYS))
    assert v.status.status == "CONFLICT" and v.series is None
    assert "applied SOXX's 3-for-1 split (ex-date 2026-11-05) to only part of its history" in v.status.notes[0]
    assert "between 2026-10-30 and 2026-11-02" in v.status.notes[0]


def test_both_sources_already_rebased_is_plain_verified() -> None:
    v = split_check(split_market(ALPACA_ADJUSTED, ALPACA_ADJUSTED))
    assert v.status.status == "VERIFIED" and not any("split" in n for n in v.status.notes)
