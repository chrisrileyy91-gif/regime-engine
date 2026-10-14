# CLAUDE.md — System Contract for the Market Regime Engine

## What this repo is
A rule-based diagnostic engine that classifies the macro-liquidity / semiconductor-cycle regime
and publishes it as a phone page (GitHub Pages). It is a **measurement tool, not a trading
signal**. Every threshold is uncalibrated until a backtest says otherwise.

## The data rule: verified or blank, same day
Every number used in the call comes from its source of record **on the day it describes** and
is checked against an independent copy. A number that fails a check is **withheld** (shown as
unavailable, never guessed). See `app/verify.py`. FRED is not used: it only republishes these
records a day later.

| Input | Record (newest wins) | Confirming check | Veto-only check |
|---|---|---|---|
| 10Y | Cboe TNX (history + delayed quote; close final 15:00 ET) | Yahoo ^TNX ±0.02 | Treasury par 10Y ±0.10 |
| 3M bill | U.S. Treasury par 3M (posts in the evening) | — | Cboe IRX → bond-equivalent ±0.15 |
| VIX | Cboe VIX (close final 16:15 ET) | Yahoo ^VIX ±1% | — |
| EFFR | NY Fed (checked against its own target range) | — | — |
| SOXX, IGV | Alpaca consolidated tape (SIP) | Yahoo ±0.3% | — |
| DXY | ICE formula on EODHD live FX | Yahoo DX-Y.NYB ±0.25 | ECB-fix DXY ±1.0 |
| KOSPI, Samsung | EODHD end-of-day (2–3h after close) → KRX Open API (next day) | Yahoo ±0.1% | — |

Statuses: **VERIFIED** (a confirming source has the same date and agrees) · **OFFICIAL**
(official record, not yet confirmed) · **UNCONFIRMED** (vendor record nothing confirmed →
withheld) · **CONFLICT** (disagreement or bad print → withheld) · **MISSING**. A newer value
from a confirming check is shown as `preliminary` and **never** used in the call. Veto-only
checks differ in method or timing: they catch gross errors, they never confirm. A check that
disagrees on 3+ of the last 5 compared dates is **systematic** and withholds the value even if a
confirming copy agrees: Yahoo's ^TNX/^VIX are copies of Cboe's indexes, so they confirm our parse,
not the methodology (first live run: Cboe ^IRX sat 7–10 bp under Treasury all week).

## Last resort: Claude web search (`app/claude_fill.py`)
Runs only when an input's record is MISSING/UNCONFIRMED or a session behind, AND Yahoo has a
value for that exact date, AND that session is over, AND this month's spend is under
`CLAUDE_MONTHLY_BUDGET_USD` (ledger: `claude_usage.json` on the snapshots release). One call per
run covers every gap. CONFLICTs are never sent. Claude is never shown Yahoo's number. A filled
value is used only if it is for the exact date, cites a URL from its own search results, and
agrees with Yahoo within the confirming tolerance; it is then spliced onto the history and
marked VERIFIED with the page named. Model: `claude-haiku-5-5` (falls back to
`claude-sonnet-5-5` on a 400/404), `web_search_20250305`, max 5 searches. Never let Claude be
the only source for a number, and never send it a conflict to break.

## Commands
- Install: `pip install -e ".[dev]"`
- Test: `pytest -q` · Types: `python -m mypy app` (strict) · Lint: `ruff check .` + `ruff format .`
- One live read: `python -m app.snapshot --state state --site site` (needs the env vars below
  for Alpaca and KRX; without them those inputs show as unavailable)
- View: `python -m http.server 8000 -d site`

## Architecture (one direction of flow, pure core)
`data.py` (httpx I/O, parsers) → `verify.py` (verified-or-blank) → `features.py` (named
features) → `regime.py` (pure classifier) + `drivers.py` ("why semis are moving") →
`service.py` (payload) → `snapshot.py` (publish job) / `main.py` (local FastAPI, not deployed).
`alert.py` posts to Pheme. `config.py` is the ONLY place thresholds, sources and tolerances live.
`web/index.html` is a dumb renderer: all logic is in Python and arrives in `regime.json`.

## Publishing (GitHub Pages, no server)
- meridian-clock dispatches `regime-morning.yml` (06:00 NY: finalizes the prior US session,
  picks up the overnight Korea session) and `regime-close.yml` (16:25 NY: 16:00 closing auction,
  16:15 VIX settle, plus the free SIP 15-minute rule). Both call `publish.yml`.
- Never add `schedule:` blocks; the clock is the only scheduler.
- State on the `snapshots` release: `regime.json`, `history.json`, `krx_cache.json`,
  `claude_usage.json`. Never in
  git. A failed download of an existing asset must stop the run.
- Secrets: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ANTHROPIC_API_KEY`, `DISCORD_PHEME_WEBHOOK`;
  optional `EODHD_API_KEY`, `KRX_API_KEY`. EODHD takes its key in the URL, so every error string passes through
  `data.redact()` before it can be published. Never bypass it.
- Pheme alerts when data quality is DEGRADED/STALE or no read was produced. Silent when clean.
- The page detects a missed scheduled run by itself (schedule mirrored in `web/index.html`).

## Hard rules
- Python 3.10+, full type hints, `mypy --strict` clean, `ruff` clean.
- All external I/O via `httpx.AsyncClient` with `get_with_retry`. No yfinance.
- Never crash on a source failure. Never substitute an unofficial value for an official one.
- Missing data is **unknown, never false**. Rules use Kleene three-valued logic.
- A one-day move is explained only by things that moved that session. Levels go in `backdrop`.
- Every displayed value carries its date. Dates are formatted without `%-d` (breaks on Windows).
- RSI is Wilder-smoothed (TradingView `ta.rsi`), never a plain rolling mean.
- Keep the original payload keys (`phase`, `risk_level`, `liquidity_squeeze`, `semi_rsi`,
  `tnx_yield`, `actionable_summary`).

## Regime classification (precedence: 2 > 4 > 1 > 3, else Unclassified)
- **Phase 2 — Liquidity Squeeze (ANY):** 10Y above the 80th percentile of its past 250 sessions
  · z-score of the latest 10Y change vs prior 60 > +1.5 · VIX > 20. Risk HIGH. (The spec's fixed
  "10Y > 4.25%" fired every day once yields moved structurally above it.)
- **Phase 4 — Rate Easing (ALL):** EFFR − 3M bill > 0.10pp (cuts-priced proxy; the 3M bill
  replaced the 2Y because no same-day 2Y source exists, and its horizon is closer to fed funds
  futures) · DXY < 102. Risk LOW.
- **Phase 1 — Hardware Buildout (ALL):** SOXX RSI(14) > 55 · 10Y 5-session change < +5 bps.
- **Phase 3 — Monetization (ALL):** IGV/SOXX SMA20 rising over 5 sessions · ratio below its
  120-day mean · SaaS ARR growth > 15% (**unsourced** → `confidence: PARTIAL`). Risk LOW.
- **Unclassified:** nothing fires. Never default to a phase.

## Known gaps (do not paper over)
- Thresholds are not fitted. No backtest exists yet. Correct data into an untested rule is
  still an untested call.
- SaaS ARR growth has no feed. EFFR − 3M bill is a proxy, not the futures curve. Bills can trade
  away from EFFR on supply alone.
- EFFR is published the next morning and changes only at FOMC decisions. On a decision day the
  16:25 read still carries the pre-decision EFFR.
- DXY is computed from FX quotes (EODHD free plan: 15-minute delay) with ICE's formula and must
  agree with ICE's own value (Yahoo) to be used.
- Korea at 06:00 depends on EODHD posting within ~3.5h of the 02:30 ET close; if it hasn't, the
  prior session is used and the newer Yahoo value shows as preliminary.
- EODHD is optional. Without it, Korea and the dollar index are filled by Claude every run
  (history then comes from Yahoo, labelled unconfirmed) at roughly $0.02–0.05 per run on Haiku.
- A Claude fill is a second independent source, not an official record. If Yahoo is also down,
  the gap stays open.
