"""Orchestration: fetch -> verify -> features -> cycle (filings) + today (prices) -> payload."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone

import pandas as pd

from app.cache import SnapshotStore
from app.claude_fill import FillReport, apply_fills
from app.config import CYCLE_PRICE_TICKERS, MARKET_TICKER
from app.data import FetchResult, fetch_all
from app.drivers import backdrop, build_drivers, diagnose
from app.features import build_features
from app.fundamentals import SecCache, build_cycle
from app.models import Condition, QualityIssue, RegimeStatus, SourceStatus
from app.regime import _any, telemetry
from app.today import build_today, tally
from app.verify import assess, verify_all

Fetcher = Callable[[], Awaitable[dict[str, FetchResult]]]


class EngineUnavailable(Exception):
    pass


def chip_prices(results: dict[str, FetchResult]) -> dict[str, pd.Series]:
    out = {}
    for ticker in CYCLE_PRICE_TICKERS:
        r = results.get(f"{ticker.lower()}@alpaca")
        if r is not None and r.series is not None and not r.series.empty:
            out[ticker] = r.series
    return out


def tally_extras(results: dict[str, FetchResult]) -> dict[str, pd.Series]:
    """Tally-only series: the equal-weight S&P 500 and Cboe's intraday highs and 30Y."""
    keys = {
        "market": f"{MARKET_TICKER.lower()}@alpaca",
        "y10_high": "y10_high@cboe",
        "y30": "y30@cboe",
        "y30_high": "y30_high@cboe",
    }
    out = {}
    for name, key in keys.items():
        r = results.get(key)
        if r is not None and r.series is not None and not r.series.empty:
            out[name] = r.series
    return out


def build_status(
    results: dict[str, FetchResult],
    today: date | None = None,
    fills: FillReport | None = None,
    sec_cache: SecCache | None = None,
) -> RegimeStatus:
    today = today or datetime.now(timezone.utc).date()
    verified = verify_all(results, today)
    if fills is not None:
        verified = apply_fills(verified, fills)
    quality = assess(verified)
    if fills is not None:
        for key in fills.fills:
            quality.issues.append(
                QualityIssue(severity="info", message=f"{verified[key].status.label}: filled by Claude web search")
            )
        quality.issues += [QualityIssue(severity="info", message=n) for n in fills.notes]
    features = build_features(verified)
    drivers = build_drivers(features, {k: v.status for k, v in verified.items()})
    cycle = build_cycle(sec_cache, chip_prices(results), features.effr_minus_3m_pct)
    day = build_today(features, drivers, verified)
    squeeze = _any(
        [
            Condition(name=d.key, value=None, threshold="", met={"ACTIVE": True, "QUIET": False}.get(d.state))
            for d in drivers
            if d.key in ("liquidity", "risk_off")
        ]
    )
    notes = list(cycle.notes)
    if not cycle.known:
        notes.append("The cycle phase needs every company's filings; see the filings list for what is missing.")
    return RegimeStatus(
        phase=cycle.phase,
        risk_level=day.stress.level,
        liquidity_squeeze=squeeze,
        semi_rsi=None if features.soxx_rsi is None else round(features.soxx_rsi, 2),
        tnx_yield=None if features.tnx_pct is None else round(features.tnx_pct, 3),
        actionable_summary=cycle.summary,
        phase_id=cycle.phase_id,
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cycle=cycle,
        today=day,
        tally=tally(verified, tally_extras(results)),
        diagnosis=diagnose(features, drivers),
        backdrop=backdrop(features),
        drivers=drivers,
        data_quality=quality,
        input_dates=dict(features.dates),
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
