"""Publish job: run the engine once and write the static site's data files.

    python -m app.snapshot --state state --site site

Reads the previous run from --state (restored from the "snapshots" release by the workflow),
writes site/regime.json + site/history.json + site/index.html, and refreshes --state.
If the critical feeds are all down, the previous regime.json is republished with
`stale: true`. With no previous run either, it exits 1 and nothing is published.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.claude_fill import FillCache, FillReport, UsageState, fill_gaps
from app.data import FetchResult, KrxCache, fetch_all, redact
from app.fundamentals import SecCache, fetch_filings
from app.models import DataQuality, QualityIssue, RegimeStatus
from app.service import build_status
from app.verify import verify_all

HISTORY_LIMIT = 750  # ~18 months at two runs per weekday
CRITICAL_INPUTS = ("tnx_pct", "vix", "soxx_rsi")
KRX_CACHE_FILE = "krx_cache.json"
SEC_CACHE_FILE = "sec_cache.json"
FILL_CACHE_FILE = "claude_fills.json"
CLAUDE_USAGE_FILE = "claude_usage.json"
PAGE_SOURCE = Path(__file__).resolve().parent.parent / "web" / "index.html"


def history_row(status: RegimeStatus) -> dict[str, Any]:
    return {
        "as_of": status.as_of,
        "phase_id": status.phase_id,
        "phase": status.phase,
        "cycle_through": status.cycle.through if status.cycle else None,
        "risk": status.risk_level,
        "tnx": status.inputs.get("tnx_pct"),
        "vix": status.inputs.get("vix"),
        "soxx_rsi": status.inputs.get("soxx_rsi"),
        "active_drivers": [d.key for d in status.drivers if d.state == "ACTIVE"],
        "data_quality": status.data_quality.status if status.data_quality else None,
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_status(
    results: dict[str, FetchResult] | None,
    error: str | None,
    state: Path,
    fills: FillReport | None = None,
    sec_cache: SecCache | None = None,
) -> RegimeStatus:
    """Live status if any critical input exists, else the previous snapshot marked stale."""
    if results is not None:
        status = build_status(results, fills=fills, sec_cache=sec_cache)
        if any(status.inputs.get(k) is not None for k in CRITICAL_INPUTS):
            return status
        error = "all critical feeds (10Y, VIX, SOXX) unavailable"

    previous = _read_json(state / "regime.json")
    if previous is None:
        raise SystemExit(f"No live data and no previous snapshot to fall back on ({error}).")
    stale = RegimeStatus.model_validate(previous)
    stale.stale = True
    stale.stale_reason = f"Official sources unavailable at publish time ({error}); showing the {stale.as_of} read."
    previous_issues = stale.data_quality.issues if stale.data_quality else []
    stale.data_quality = DataQuality(
        status="STALE",
        summary="This is the previous read; this run could not get official data.",
        issues=[QualityIssue(severity="warn", message=f"This run failed: {error}"), *previous_issues],
        inputs=stale.data_quality.inputs if stale.data_quality else [],
    )
    return stale


def write_outputs(status: RegimeStatus, state: Path, site: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = _read_json(state / "history.json") or []
    if not status.stale:
        history.append(history_row(status))
        history = history[-HISTORY_LIMIT:]

    site.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    payload = status.model_dump_json(indent=1)
    history_json = json.dumps(history, separators=(",", ":"))
    for folder in (site, state):
        (folder / "regime.json").write_text(payload, encoding="utf-8")
        (folder / "history.json").write_text(history_json, encoding="utf-8")
    shutil.copyfile(PAGE_SOURCE, site / "index.html")
    return history


def load_usage(state: Path) -> UsageState:
    data = _read_json(state / CLAUDE_USAGE_FILE)
    return data if isinstance(data, dict) else {}


def save_usage(state: Path, usage: UsageState) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / CLAUDE_USAGE_FILE).write_text(json.dumps(usage, indent=1, sort_keys=True), encoding="utf-8")


async def _fill(
    results: dict[str, FetchResult] | None, usage: UsageState, cache: FillCache | None = None
) -> FillReport | None:
    if results is None:
        return None
    verified = verify_all(results, datetime.now(timezone.utc).date())
    return await fill_gaps(verified, results, usage, cache=cache)


def load_fill_cache(state: Path) -> FillCache:
    data = _read_json(state / FILL_CACHE_FILE)
    return data if isinstance(data, dict) else {}


def save_fill_cache(state: Path, cache: FillCache) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / FILL_CACHE_FILE).write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def load_sec_cache(state: Path) -> SecCache:
    data = _read_json(state / SEC_CACHE_FILE)
    return data if isinstance(data, dict) else {}


def save_sec_cache(state: Path, cache: SecCache) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / SEC_CACHE_FILE).write_text(json.dumps(cache, separators=(",", ":"), sort_keys=True), encoding="utf-8")


async def _filings(cache: SecCache) -> None:
    try:
        cache["notes"] = await fetch_filings(None, cache)
    except Exception as exc:  # keep the cached filings; the cycle still reads from them
        cache["notes"] = [f"SEC refresh failed: {redact(f'{type(exc).__name__}: {exc}')}"]


def load_krx_cache(state: Path) -> KrxCache:
    data = _read_json(state / KRX_CACHE_FILE)
    return data if isinstance(data, dict) else {}


def save_krx_cache(state: Path, cache: KrxCache) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / KRX_CACHE_FILE).write_text(json.dumps(cache, separators=(",", ":"), sort_keys=True), encoding="utf-8")


async def _fetch(krx_cache: KrxCache) -> tuple[dict[str, FetchResult] | None, str | None]:
    try:
        return await fetch_all(krx_cache=krx_cache), None
    except Exception as exc:  # fetch_all isolates feeds; this is a last-resort guard
        return None, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("state"))
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

    sec_cache = load_sec_cache(args.state)
    asyncio.run(_filings(sec_cache))
    save_sec_cache(args.state, sec_cache)
    krx_cache = load_krx_cache(args.state)
    results, error = asyncio.run(_fetch(krx_cache))
    save_krx_cache(args.state, krx_cache)
    usage = load_usage(args.state)
    fill_cache = load_fill_cache(args.state)
    fills = asyncio.run(_fill(results, usage, fill_cache))
    save_usage(args.state, usage)
    save_fill_cache(args.state, fill_cache)
    status = resolve_status(results, error, args.state, fills, sec_cache)
    history = write_outputs(status, args.state, args.site)

    quality = status.data_quality
    cycle = status.cycle
    if cycle is not None:
        state = "confirmed" if cycle.confirmed else "not yet confirmed"
        print(f"cycle={cycle.phase} since {cycle.since} ({state}), filings through {cycle.through}")
        for m in cycle.metrics:
            print(f"  {m.label}: {m.latest} (prev {m.previous})")
        for c in cycle.catalysts:
            print(f"  catalyst {c.key}: {c.status} - {c.detail}")
        for note in cycle.notes:
            print(f"  [cycle] {note}")
    print(f"today={status.risk_level} | tally: {status.tally.summary if status.tally else 'n/a'}")
    print(f"data={quality.status if quality else 'n/a'} | stale={status.stale} | history rows {len(history)}")
    if quality:
        print(f"  {quality.summary}")
        for item in quality.inputs:
            extra = f" confirmed by {', '.join(item.confirmed_by)}" if item.confirmed_by else ""
            print(f"  {item.status:9} {item.label}: {item.value} ({item.date}, {item.source}){extra}")
        if fills is not None and (fills.gaps or fills.notes):
            month = max(usage) if usage else "-"
            spent = usage.get(month, {}).get("usd", 0.0) if usage else 0.0
            print(
                f"  Claude: {len(fills.gaps)} gap(s), {len(fills.fills)} filled, run cost ${fills.cost_usd:.4f}, "
                f"{month} total ${spent:.2f}"
            )
        for issue in quality.issues:
            print(f"  [{issue.severity}] {issue.message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
