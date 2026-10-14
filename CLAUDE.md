# CLAUDE.md — System Contract for the Market Regime Engine

## What this repo is
A rule-based diagnostic engine that classifies the macro-liquidity / semiconductor-cycle regime
and serves it at `GET /api/v1/regime`. It is a **measurement tool, not a trading signal**.
Every threshold is uncalibrated until a backtest says otherwise.

## Commands
- Install: `pip install -e ".[dev]"`
- Serve: `uvicorn app.main:app --reload --port 8000`
- Test: `pytest -v` (coverage: `pytest --cov=app`)
- Types: `python -m mypy app` (strict)
- Lint/format: `ruff check . --fix` then `ruff format .`

## Architecture (one direction of flow, pure core)
`data.py` (httpx I/O) → `features.py` (series → named features) → `regime.py` (pure classifier)
+ `drivers.py` ("why semis are moving") → `service.py` (builds the payload) → two front doors:
- `snapshot.py`: the publish job. Writes `site/regime.json`, `site/history.json`, copies
  `web/index.html`. This is what runs in production.
- `main.py`: FastAPI (`/api/v1/regime`) for local use. Not deployed.
`config.py` is the ONLY place thresholds live. `indicators.py` is pure math. `web/index.html`
is a dumb renderer: all logic lives in Python and arrives in `regime.json`.

## Publishing (GitHub Pages, no server)
- meridian-clock dispatches `regime-open.yml` (10:15 NY) and `regime-close.yml` (16:25 NY) on
  weekdays. Both call `publish.yml`. Two files because the clock fires each file once per day.
- Never add `schedule:` blocks to these workflows; the clock is the only scheduler.
- State lives on the `snapshots` release (regime.json + history.json), never in git. A failed
  download of an existing asset must stop the run.
- If 10Y, VIX and SOXX are all down, the job republishes the previous read with `stale: true`.
  History only grows on live reads (cap 750 rows).

## Hard rules
- Python 3.10+, full type hints, `mypy --strict` clean, `ruff` clean.
- All external I/O via `httpx.AsyncClient`. No yfinance (blocking; mis-aligns US/KR calendars).
- Never crash on a feed failure. A failed feed yields `FetchResult(series=None, error=...)`.
  If 10Y, VIX and SOXX are all missing, serve the last SQLite snapshot with `stale: true`; with
  no snapshot, return 503.
- Missing data is **unknown, never false**. Rules use Kleene three-valued logic.
- Math is written out explicitly. RSI is Wilder-smoothed (TradingView `ta.rsi`), never a plain
  rolling mean.
- Never change a threshold anywhere but `app/config.py`. Never invent a data source; an input
  without a feed is marked `unsourced`.
- Keep the original payload keys (`phase`, `risk_level`, `liquidity_squeeze`, `semi_rsi`,
  `tnx_yield`, `actionable_summary`) — the dashboard depends on them.

## Regime classification (precedence: 2 > 4 > 1 > 3, else Unclassified)
- **Phase 2 — Liquidity Squeeze (ANY):** 10Y > 4.25% · z-score of today's 10Y change vs prior
  60 sessions > +1.5 · VIX > 20. Risk HIGH. Overrides everything.
- **Phase 4 — Rate Easing (ALL):** EFFR − 2Y > 0.10pp (proxy for "futures pricing cuts") ·
  DXY < 102. Risk LOW.
- **Phase 1 — Hardware Buildout (ALL):** SOXX RSI(14) > 55 · 10Y 5-session change < +5 bps.
  Risk MODERATE.
- **Phase 3 — Monetization (ALL):** IGV/SOXX SMA20 rising over 5 sessions · ratio still below
  its 120-day mean · SaaS ARR growth > 15% (**unsourced** — evaluated without it, reported
  `confidence: PARTIAL`). Risk LOW.
- **Unclassified:** nothing fires. Never default to a phase.
- `confidence`: FULL · PARTIAL (unsourced leg skipped) · PROVISIONAL (a higher-precedence rule
  was unknown due to missing data).

## Telemetry (signals, not phase drivers)
10Y > 4.25% or +10 bps/session → LIQUIDITY DRAIN · VIX > 20 → ALGORITHMIC RISK-OFF ·
SOXX or Samsung RSI < 40 → HARDWARE DIGESTION · IGV/SOXX SMA20 rising → PHASE 3 TRANSITION ·
KOSPI close > SMA200 → EM RE-RATE · DXY > 104.5 → FOREIGN REVENUE COMPRESSION.
Semi-momentum label: >55 STRONG, <40 DIGESTION, otherwise CHOPPY.

## Known gaps (do not paper over)
- Thresholds are from the original spec, not fitted. No backtest exists yet.
- SaaS ARR growth has no free feed.
- EFFR−2Y is a proxy for fed funds futures, not the futures curve itself.
- DXY thresholds are ICE DXY scale; FRED's broad dollar index is a different scale — never
  substitute it.
