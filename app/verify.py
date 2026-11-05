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
  7. Splits (Alpaca records only). Alpaca's bars are split-adjusted as soon as Alpaca processes
     a split; Yahoo can re-base hours later. When Alpaca lists a split for the symbol and every
     compared date before its ex-date differs by exactly that factor, the check is compared on
     the post-split basis (noted). Only that exact pattern is re-based: anything else is still a
     disagreement. A record that hasn't applied a listed split is withheld.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
from app.data import FetchResult, Split
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


def _split_text(symbol: str, factor: float, ex_date: pd.Timestamp) -> str:
    ratio = f"{factor:g}-for-1" if factor > 1 else f"1-for-{1 / factor:g}"
    return f"{symbol}'s {ratio} split (ex-date {_fmt_day(ex_date)})"


def _cumulative_factors(index: pd.Index, splits: list[Split]) -> pd.Series:
    """Per date: the product of every split whose ex-date is after it (1.0 = no split since)."""
    factors = pd.Series(1.0, index=index)
    for split in splits:
        factors[index < split.ex_date] *= split.factor
    return factors


def _within(ratio: float, target: float, tolerance_pct: float) -> bool:
    return abs(ratio / target - 1.0) * 100.0 <= tolerance_pct


@dataclass
class _Rebase:
    series: pd.Series  # the check, on the record's post-split basis when re-based
    note: str | None = None  # the check was re-based
    conflict: str | None = None  # the record itself is on the wrong basis


def _rebase_for_splits(primary: FetchResult, check: FetchResult, tolerance: float) -> _Rebase:
    """Put a check that hasn't applied a listed split on the record's (split-adjusted) basis.

    Re-bases only when EVERY compared date before the ex-date differs by exactly the split
    factor (within the check's own tolerance): a lagging vendor, not a wrong number.
    """
    assert primary.series is not None and check.series is not None
    record, other = primary.series, check.series
    if primary.source != "alpaca" or not primary.splits:
        return _Rebase(other)
    splits = primary.splits
    newest = splits[-1]
    name = SOURCE_NAMES[check.source]
    what = _split_text(primary.symbol, newest.factor, newest.ex_date)

    # The record itself: a raw (unadjusted) record shows the split as a crash on the ex-date.
    pre = record[record.index < newest.ex_date]
    post = record[record.index >= newest.ex_date]
    if len(pre) and len(post):
        jump = float(post.iloc[0]) / float(pre.iloc[-1])
        if abs(math.log(jump * newest.factor)) < abs(math.log(jump)):
            return _Rebase(
                other,
                conflict=f"{SOURCE_NAMES[primary.source]} hasn't applied {what}: its closes jump "
                f"{(jump - 1) * 100:+.0f}% across the ex-date",
            )

    common = record.index.intersection(other.index)[-CHECK_OVERLAP_SESSIONS:]
    factors = _cumulative_factors(common, splits)
    before = [d for d in common if factors[d] != 1.0]
    if not before:
        return _Rebase(other)  # the split is outside the compared window
    ratios = {d: float(other.loc[d]) / float(record.loc[d]) for d in before}
    if all(_within(ratios[d], 1.0, tolerance) for d in before):
        return _Rebase(other)  # both already on the post-split basis
    if all(_within(ratios[d], factors[d], tolerance) for d in before):
        rebased = other / _cumulative_factors(other.index, splits)
        note = f"{name} hasn't applied {what} yet; its earlier closes were compared after re-basing"
        return _Rebase(rebased, note=note)
    if all(_within(ratios[d], 1.0 / factors[d], tolerance) for d in before):
        return _Rebase(other, conflict=f"{SOURCE_NAMES[primary.source]} hasn't applied {what}; {name} has")
    return _Rebase(other)  # some other disagreement: let the normal checks judge it


CLEAN_SPLIT_FACTORS = (2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0)


def _split_like_gap(primary: FetchResult, check: pd.Series, tolerance: float) -> float | None:
    """A constant clean ratio on every compared date (e.g. exactly 3.00x): what a split looks like."""
    assert primary.series is not None
    common = primary.series.index.intersection(check.index)[-CHECK_OVERLAP_SESSIONS:]
    if len(common) < SYSTEMATIC_MIN_DAYS:
        return None
    ratios = [float(check.loc[d]) / float(primary.series.loc[d]) for d in common]
    for factor in CLEAN_SPLIT_FACTORS:
        for target in (factor, 1.0 / factor):
            if all(_within(r, target, tolerance) for r in ratios):
                return target
    return None


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

    rebased_checks = []
    for check, tolerance, kind, confirms in checks:
        if kind == "pct":
            rebase = _rebase_for_splits(primary, check, tolerance)
            if rebase.conflict:
                return make("CONFLICT", [rebase.conflict], day=day, source=source, preliminary=newest_check())
            if rebase.note:
                notes.append(rebase.note)
                check = replace(check, series=rebase.series)
        rebased_checks.append((check, tolerance, kind, confirms))
    checks[:] = rebased_checks

    disagreements: list[str] = []
    systematic: list[str] = []
    for check, tolerance, kind, confirms in checks:
        assert check.series is not None
        name = SOURCE_NAMES[check.source]
        bad = _mismatches(series, check.series, tolerance, kind)
        compared = len(series.index.intersection(check.series.index)[-CHECK_OVERLAP_SESSIONS:])
        if bad and compared >= SYSTEMATIC_MIN_DAYS and len(bad) >= SYSTEMATIC_MIN_DAYS:
            gap = _split_like_gap(primary, check.series, tolerance) if kind == "pct" else None
            hint = ""
            if gap is not None:
                times = f"{gap:.2f}x" if gap > 1 else f"1/{1 / gap:.0f}"
                why = (
                    f"the split lookup failed ({primary.splits_error})"
                    if primary.splits_error
                    else f"no split is listed for {primary.symbol}"
                )
                hint = f"; a constant {times} gap looks like a split, but {why}"
            systematic.append(f"{name} on {len(bad)} of {compared} recent dates ({'; '.join(bad)}){hint}")
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
