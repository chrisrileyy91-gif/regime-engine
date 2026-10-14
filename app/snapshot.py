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
from pathlib import Path
from typing import Any

from app.data import FetchResult, fetch_all
from app.models import RegimeStatus
from app.service import build_status

HISTORY_LIMIT = 750  # ~18 months at two runs per weekday
CRITICAL_INPUTS = ("tnx_pct", "vix", "soxx_rsi")
PAGE_SOURCE = Path(__file__).resolve().parent.parent / "web" / "index.html"


def history_row(status: RegimeStatus) -> dict[str, Any]:
    return {
        "as_of": status.as_of,
        "phase_id": status.phase_id,
        "phase": status.phase,
        "risk": status.risk_level,
        "confidence": status.confidence,
        "tnx": status.inputs.get("tnx_pct"),
        "vix": status.inputs.get("vix"),
        "soxx_rsi": status.inputs.get("soxx_rsi"),
        "active_drivers": [d.key for d in status.drivers if d.state == "ACTIVE"],
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_status(results: dict[str, FetchResult] | None, error: str | None, state: Path) -> RegimeStatus:
    """Live status if any critical input exists, else the previous snapshot marked stale."""
    if results is not None:
        status = build_status(results)
        if any(status.inputs.get(k) is not None for k in CRITICAL_INPUTS):
            return status
        error = "all critical feeds (10Y, VIX, SOXX) unavailable"

    previous = _read_json(state / "regime.json")
    if previous is None:
        raise SystemExit(f"No live data and no previous snapshot to fall back on ({error}).")
    stale = RegimeStatus.model_validate(previous)
    stale.stale = True
    stale.stale_reason = f"Feeds down at publish time ({error}); showing the {stale.as_of} read."
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


async def _fetch() -> tuple[dict[str, FetchResult] | None, str | None]:
    try:
        return await fetch_all(), None
    except Exception as exc:  # fetch_all isolates feeds; this is a last-resort guard
        return None, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("state"))
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

    results, error = asyncio.run(_fetch())
    status = resolve_status(results, error, args.state)
    history = write_outputs(status, args.state, args.site)

    feeds_ok = sum(1 for s in status.sources if s.ok)
    print(f"phase={status.phase_id} {status.phase} | risk={status.risk_level} | confidence={status.confidence}")
    print(f"stale={status.stale} | feeds ok {feeds_ok}/{len(status.sources)} | history rows {len(history)}")
    for source in status.sources:
        if not source.ok:
            print(f"  feed down: {source.key} ({source.symbol}): {source.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
