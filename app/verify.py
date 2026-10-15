"""Verified-or-blank: decide, per input, whether a number is trustworthy enough to use.

For each input (config.INPUTS):
  1. The record source with the newest date is the record. None answered: MISSING.
  2. A value outside its plausible range is a bad print: CONFLICT, withheld.
  3. Other record sources (e.g. KRX's copy of a KOSPI close EODHD also has) must match on
     recent common dates, or CONFLICT.
  4. Checks are compared on recent common dates. A confirming check (same instrument, e.g.
     Yahoo ^TNX vs Cboe TNX) that has the record's latest date and agrees confirms it. A
     non-confirming check (different methodology or timing, e.g. Treasury's par curve, the
     ECB fix) can only veto gross errors. Any disagreement withholds the value unless
     something else already confirmed the latest date, in which case it is noted.
  5. VERIFIED = confirmed. OFFICIAL = the record is an official source, not yet confirmed.
     UNCONFIRMED = a vendor record nothing independent has confirmed: withheld.
  6. A check with a newer date than the record becomes `preliminary`: shown, dated and
     labelled, never used in the phase call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.config import (
    CHECK_OVERLAP_SESSIONS,
    INPUTS,
    OFFICIAL_SOURCES,
    RECORD_MATCH_TOLERANCE,
    SANITY_BOUNDS,
    SYSTEMATIC_MIN_DAYS,
    InputSpec,
)
from app.data import FetchResult
from app.models import DataQuality, InputStatus, Preliminary, QualityIssue

SOURCE_NAMES = {
    "treasury": "U.S. Treasury",
    "cboe": "Cboe",
    "nyfed": "NY Fed",
    "alpaca": "consolidated tape (Alpaca)",
    "ecb": "ECB reference rates",
    "krx": "Korea Exchange",
    "eodhd": "EODHD",
    "yahoo": "Yahoo",
}


@dataclass
class Verified:
    status: InputStatus
    series: pd.Series | None  # the record series, or None when withheld


def _ok(results: dict[str, FetchResult], key: str) -> FetchResult | None:
    r = results.get(key)
    return r if r is not None and r.ok else None


def _agrees(record: float, other: float, tolerance: float, kind: str) -> bool:
    if kind == "pct":
        return abs(other - record) / abs(record) * 100.0 <= tolerance
    return abs(other - record) <= tolerance


def _mismatches(primary: pd.Series, other: pd.Series, tolerance: float, kind: str) -> list[str]:
    common = primary.index.intersection(other.index)[-CHECK_OVERLAP_SESSIONS:]
    bad = []
    for day in common:
        p, o = float(primary.loc[day]), float(other.loc[day])
        if not _agrees(p, o, tolerance, kind):
            bad.append(f"{day.date()}: {p:.4g} vs {o:.4g}")
    return bad


def _fmt_day(ts: pd.Timestamp) -> str:
    return str(ts.date())


def verify_input(key: str, spec: InputSpec, results: dict[str, FetchResult], today: date) -> Verified:
    records = [r for k in spec.record if (r := _ok(results, k)) is not None]
    checks = [(r, tol, kind, confirms) for k, tol, kind, confirms in spec.checks if (r := _ok(results, k)) is not None]

    def newest_check() -> Preliminary | None:
        best: Preliminary | None = None
        for r, _, _, confirms in checks:
            assert r.series is not None
            d = _fmt_day(r.series.index[-1])
            if confirms and (best is None or d > best.date):
                best = Preliminary(value=float(r.series.iloc[-1]), date=d, source=SOURCE_NAMES[r.source])
        return best

    def make(
        status: str,
        notes: list[str],
        *,
        value: float | None = None,
        day: str | None = None,
        source: str | None = None,
        confirmed: list[str] | None = None,
        stale: bool = False,
        preliminary: Preliminary | None = None,
        series: pd.Series | None = None,
    ) -> Verified:
        return Verified(
            InputStatus(
                key=key,
                label=spec.label,
                used_in_call=spec.used_in_call,
                status=status,
                value=value,
                date=day,
                source=source,
                confirmed_by=confirmed or [],
                notes=notes,
                stale=stale,
                preliminary=preliminary,
            ),
            series,
        )

    if not records:
        errors = [f"{SOURCE_NAMES.get(results[k].source, k)}: {results[k].error}" for k in spec.record if k in results]
        return make("MISSING", errors or ["no record source answered"], preliminary=newest_check())

    records.sort(key=lambda r: r.series.index[-1] if r.series is not None else pd.Timestamp.min, reverse=True)
    primary = records[0]
    assert primary.series is not None
    series = primary.series
    latest_day, latest = series.index[-1], float(series.iloc[-1])
    day, source = _fmt_day(latest_day), SOURCE_NAMES[primary.source]
    is_official = primary.source in OFFICIAL_SOURCES
    notes: list[str] = []
    confirmed: list[str] = []

    low, high = SANITY_BOUNDS[key]
    if not low <= latest <= high:
        bad_print = f"latest value {latest:.4g} is outside the plausible range {low}–{high}"
        return make("CONFLICT", [bad_print], day=day, source=source)

    for other in records[1:]:
        assert other.series is not None
        other_name = SOURCE_NAMES[other.source]
        bad = _mismatches(series, other.series, RECORD_MATCH_TOLERANCE.get(key, 0.011), "abs")
        if bad:
            return make("CONFLICT", [f"{source} and {other_name} disagree ({'; '.join(bad)})"], day=day, source=source)
        if latest_day in other.series.index:
            confirmed.append(other_name)
        elif len(series.index.intersection(other.series.index)):
            notes.append(f"history matches {other_name}; {other_name} hasn't posted {day} yet")

    disagreements: list[str] = []
    systematic: list[str] = []
    for check, tolerance, kind, confirms in checks:
        assert check.series is not None
        name = SOURCE_NAMES[check.source]
        bad = _mismatches(series, check.series, tolerance, kind)
        compared = len(series.index.intersection(check.series.index)[-CHECK_OVERLAP_SESSIONS:])
        if bad and compared >= SYSTEMATIC_MIN_DAYS and len(bad) >= SYSTEMATIC_MIN_DAYS:
            systematic.append(f"{name} on {len(bad)} of {compared} recent dates ({'; '.join(bad)})")
        elif bad:
            disagreements.append(f"{name} ({'; '.join(bad)})")
        elif confirms and latest_day in check.series.index:
            confirmed.append(name)
        elif not confirms and compared:
            notes.append(f"consistent with {name}")
    if systematic:
        # A disagreement that repeats day after day is a wrong source or a wrong assumption, not
        # noise. A confirming copy of the same number can't overrule it (first live run: Cboe ^IRX
        # sat 7-10 bp under Treasury for five days while Yahoo, a copy of ^IRX, "confirmed" it).
        reason = f"{source} disagrees systematically with " + ", ".join(systematic)
        return make("CONFLICT", [reason], day=day, source=source, preliminary=newest_check())
    if disagreements and not confirmed:
        reason = f"{source} disagrees with " + ", ".join(disagreements)
        return make("CONFLICT", [reason], day=day, source=source, preliminary=newest_check())
    notes += [f"{d} differs, but {', '.join(confirmed)} confirms {day}" for d in disagreements]

    if not confirmed and not is_official:
        why = "no independent source has confirmed it" if checks else "no independent source available"
        return make(
            "UNCONFIRMED", [f"{source} value for {day}: {why}"], day=day, source=source, preliminary=newest_check()
        )

    newest = newest_check()
    preliminary = newest if newest is not None and newest.date > day else None
    stale = (today - latest_day.date()).days > spec.max_age_days
    if stale:
        notes.append(f"latest value is from {day}")
    if not confirmed and not checks and len(records) == 1:
        notes.append("single official source; no independent copy exists")
    return make(
        "VERIFIED" if confirmed else "OFFICIAL",
        notes,
        value=latest,
        day=day,
        source=source,
        confirmed=confirmed,
        stale=stale,
        preliminary=preliminary,
        series=series,
    )


def verify_all(results: dict[str, FetchResult], today: date) -> dict[str, Verified]:
    return {key: verify_input(key, spec, results, today) for key, spec in INPUTS.items()}


def assess(verified: dict[str, Verified]) -> DataQuality:
    issues: list[QualityIssue] = []
    for v in verified.values():
        s = v.status
        if s.status in ("MISSING", "CONFLICT", "UNCONFIRMED"):
            what = "unavailable" if s.status == "MISSING" else "withheld"
            severity = "warn" if s.used_in_call else "info"  # telemetry gaps never alarm
            issues.append(QualityIssue(severity=severity, message=f"{s.label} {what}: {'; '.join(s.notes)}"))
        elif s.stale:
            issues.append(
                QualityIssue(severity="warn" if s.used_in_call else "info", message=f"{s.label}: {'; '.join(s.notes)}")
            )
        elif s.preliminary is not None:
            issues.append(
                QualityIssue(
                    severity="info",
                    message=f"{s.label}: official value is from {s.date}; {s.preliminary.source} already shows "
                    f"{s.preliminary.date} ({s.preliminary.value:.4g}), not used until official",
                )
            )
    call_inputs = [v.status for v in verified.values() if v.status.used_in_call]
    good = [s for s in call_inputs if s.status in ("VERIFIED", "OFFICIAL") and not s.stale]
    verified_count = sum(1 for s in call_inputs if s.status == "VERIFIED")
    status = "OK" if len(good) == len(call_inputs) else "DEGRADED"
    summary = f"{len(good)} of {len(call_inputs)} call inputs usable, {verified_count} independently confirmed."
    return DataQuality(status=status, summary=summary, issues=issues, inputs=[v.status for v in verified.values()])
