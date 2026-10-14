"""Orchestration: fetch -> features -> classify -> payload, with stale fallback."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from app.cache import SnapshotStore
from app.data import FetchResult, fetch_all
from app.drivers import build_drivers, diagnose
from app.features import build_features
from app.models import RegimeStatus, SourceStatus
from app.regime import classify, decision_matrix, telemetry

Fetcher = Callable[[], Awaitable[dict[str, FetchResult]]]


class EngineUnavailable(Exception):
    pass


def build_status(results: dict[str, FetchResult]) -> RegimeStatus:
    features = build_features(results)
    result = classify(features)
    squeeze = next(r.fired for r in result.rules if r.phase_id == 2)
    drivers = build_drivers(features)
    notes = list(features.notes)
    if result.confidence == "PARTIAL":
        notes.append("Phase 3 called on the IGV/SOXX ratio alone; the SaaS ARR leg has no data feed.")
    if result.confidence == "PROVISIONAL":
        notes.append("A higher-precedence rule could not be evaluated (missing data); call is provisional.")
    return RegimeStatus(
        phase=result.spec.name,
        risk_level=result.spec.risk,
        liquidity_squeeze=squeeze,
        semi_rsi=None if features.soxx_rsi is None else round(features.soxx_rsi, 2),
        tnx_yield=None if features.tnx_pct is None else round(features.tnx_pct, 3),
        actionable_summary=result.spec.summary,
        phase_id=result.spec.phase_id,
        confidence=result.confidence,
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        matrix=decision_matrix(features, result),
        diagnosis=diagnose(features, drivers),
        drivers=drivers,
        rules=result.rules,
        telemetry=telemetry(features),
        inputs=features.as_dict(),
        sources=[
            SourceStatus(key=r.key, symbol=r.symbol, source=r.source, ok=r.ok, last_date=r.last_date, error=r.error)
            for r in results.values()
        ],
        notes=notes,
    )


def _critical_inputs_missing(status: RegimeStatus) -> bool:
    inputs = status.inputs
    return all(inputs.get(k) is None for k in ("tnx_pct", "vix", "soxx_rsi"))


async def compute_regime(store: SnapshotStore, fetcher: Fetcher = fetch_all) -> RegimeStatus:
    error: str | None = None
    try:
        status = build_status(await fetcher())
        if not _critical_inputs_missing(status):
            await store.save(status.model_dump_json())
            return status
        error = "all critical feeds (10Y, VIX, SOXX) unavailable"
    except Exception as exc:  # never crash the endpoint on upstream failure
        error = f"{type(exc).__name__}: {exc}"

    cached = await store.latest()
    if cached is None:
        raise EngineUnavailable(f"No live data and no cached snapshot ({error}).")
    created_at, payload = cached
    stale = RegimeStatus.model_validate_json(payload)
    stale.stale = True
    stale.stale_reason = f"Serving snapshot from {created_at}: {error}"
    return stale
