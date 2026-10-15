# CLAUDE.md — System Contract for the Market Regime Engine

## What this repo is
A diagnostic engine published as a phone page (GitHub Pages) with two layers on two clocks:
- **The cycle** (`app/fundamentals.py`, quarterly): which phase of the AI capex cycle we are in,
  read from company filings at SEC EDGAR. Buildout → Digestion → Monetization.
- **Today** (`app/today.py`, `app/drivers.py`, daily): stress (Calm / Elevated / Stressed) from
  what moved that session, the scapegoat tally, the overnight line.
It is a **measurement tool, not a trading signal**. Every threshold is uncalibrated until a
backtest says otherwise.

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
| Company filings | SEC EDGAR XBRL companyconcept (each company's own 10-Q/10-K) | — | entity-name check per CIK |
| Chip stock prices | Alpaca SIP quarter-end closes (earnings-vs-price catalyst only) | — | — |
| Tally-only | Cboe TNX/TYX daily high and TYX close; Treasury 2Y; Alpaca SIP RSP, USO (oil), UUP (dollar) | — | — |
| Headlines | Benzinga via Alpaca news API (same keys), labelled by Claude from the list only | cited ids must exist | — |

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
features) → `drivers.py` ("why semis are moving") + `today.py` (stress, tally, overnight) +
`regime.py` (telemetry signals, Kleene helpers) — and separately `fundamentals.py` (SEC fetch,
cached; quarters; phases; catalysts) → `service.py` (payload) → `snapshot.py` (publish job) /
`main.py` (local FastAPI, not deployed).
`alert.py` posts to Pheme. `config.py` is the ONLY place thresholds, sources and tolerances live.
`web/index.html` is a dumb renderer: all logic is in Python and arrives in `regime.json`.

## Publishing (GitHub Pages, no server)
- meridian-clock dispatches `regime-morning.yml` (06:00 NY: finalizes the prior US session,
  picks up the overnight Korea session) and `regime-close.yml` (16:25 NY: 16:00 closing auction,
  16:15 VIX settle, plus the free SIP 15-minute rule). Both call `publish.yml`.
- Never add `schedule:` blocks; the clock is the only scheduler.
- Runners are pinned to `ubuntu-24.04` (ubuntu-latest moves to 26 on Oct 19, 2026). Move the pin
  deliberately, with a test run, not by surprise.
- State on the `snapshots` release: `regime.json`, `history.json`, `krx_cache.json`,
  `claude_usage.json`, `sec_cache.json` (filings, refreshed at most every 20h), `claude_fills.json`,
  `news_cache.json`.
  Never in
  git. A failed download of an existing asset must stop the run.
- Secrets: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ANTHROPIC_API_KEY`, `DISCORD_PHEME_WEBHOOK`,
  `SEC_USER_AGENT` ("regime-engine <contact email>"; SEC refuses anonymous clients); optional
  `EODHD_API_KEY`, `KRX_API_KEY`. Never put a personal email in the repo (it is public). EODHD takes its key in the URL, so every error string passes through
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
  `tnx_yield`, `actionable_summary`). `phase` = the cycle phase; `risk_level` = today's stress.
- Never put a slow-moving level (10Y percentile, DXY level) into the stress label or the cycle.
  That is what kept the old page on HIGH RISK for months.

## The cycle (from filings; precedence 3 > 1 > 2, else Between phases)
Groups (`CYCLE_COMPANIES`): capex = MSFT, GOOGL, AMZN, META, ORCL (summed dollars); chip revenue =
NVDA, AVGO, AMD, MU (summed); software revenue = CRM, NOW, ADBE, INTU, WDAY, PLTR (**median
company**, plus breadth). Oracle is a capex spender: its growth is AI cloud rental, and in the
software group it alone faked an acceleration on the first real read.
- Quarters: `quarterize` keeps the latest filing per (start, end), takes ~3-month facts and
  differences of year-to-date chains (Q4 = FY − 9M), ignores TTM facts (Amazon). Each quarter maps
  to the nearest calendar quarter end. Growth = trailing-12-month total vs a year earlier. A group
  quarter exists only when every company has it.
- **Buildout (ALL):** capex growth ≥ 20% · capex growth not ≥ 5 pts below its 4-quarter peak ·
  chip growth ≥ 25% · chip growth not rolled over (same test).
- **Digestion (ALL):** capex growth ≥ 10% · capex or chip growth rolled over. (Mutually exclusive
  with Buildout.)
- **Monetization (ALL):** median software growth ≥ 10% · median company's growth change > 0 in
  each of the last two quarters, totalling ≥ 1.0 pt · at least half the companies accelerating.
  Can overlap Buildout or Digestion; when it does it is the headline and `phases_met` shows both.
- `since` = first quarter of the current run; `confirmed` after 2 consecutive quarters.
- Catalysts: software (from the Monetization check), chip earnings vs chip prices (TTM EPS vs
  price, year to the latest quarter), rate cuts (3M bill vs EFFR, as before).
- First real read (filings to Q2 2026): Buildout since Q1 2026, confirmed; Q4 2024–Q4 2025 was
  Digestion (chip revenue growth rolled over from 81% to ~50%); software "starting" (+0.64 then
  +0.36 pts, total 0.992 vs 1.0 needed). `tests/fixtures/sec_filings.txt` holds that data.

## Today (daily)
- Drivers (same session only): 10Y +10 bps or z > 1.5 · VIX > 20 · SOXX RSI < 40.
- Stress: 0 active = Calm, 1 = Elevated, 2+ or VIX > 30 = Stressed, all unknown = Unknown.
- Tally: the last 20 sessions with SOXX ≤ −1.5%, each tagged in order:
  1. **rates** — 10Y or 30Y up ≥ 10 bps at the close OR at the intraday high vs the prior close, a
     10Y close z ≥ 1.5, or the Treasury 2Y up ≥ 10 bps (policy repricing: Aug 28 hike odds jumped to
     60%, 2Y +14 bps, 10Y +5). A date before Treasury's latest row with no row is a bond holiday
     (Columbus Day): "2Y closed", not missing. Intraday is required: CPI day Oct 14 the 10Y rose 11.2 bps by its high and
     closed −2.7; a close-to-close test misses it entirely.
  2. **fear** — VIX close ≥ +10%.  3. **market** — equal-weight S&P 500 (RSP) ≤ −1%. Not SPY: chips
     are a big slice of SPY, so a semis rout drags it (Oct 14 SPY −1.1%; Sep 14 SPY −0.5%, RSP +0.1%).
  4. **rotation** (software up) / **chips** (software down) — the residual: no gauge moved.
  A row is "unknown" unless every gauge that could claim it is present. The newest row is
  `provisional` (not counted) while any of its official closes are missing; 06:00 finalizes it.
  The page must never say "no macro trigger": gauges see moves, not news, policy or levels
  (Aug 18 "yields rattle markets" was the 30Y LEVEL at a 2007 high; its move was +1.7 bps).
- Oil (USO |move| ≥ 4% OR |z| ≥ 2 vs prior 60 sessions) and dollar (UUP, z ≥ +2) sit after rates.
  The 4% floor exists because a 60-day z adapts shocks away in a war regime (Sep 1, US strikes on
  Iran: USO +5.5% = 1.65σ). Both fixes came from headline flags, which is what the flags are for. USO
  and UUP are what Crude Logic (frack-around) already uses after Yahoo's CL=F broke; EIA's official
  WTI spot posts in weekly batches, too late for same day. `moved` lists every gauge that moved.
- Headlines (`app/news.py`): for each tally day, Alpaca/Benzinga headlines tagged to NVDA, AVGO,
  AMD, MU, TSM, ASML, INTC, SOXX, SMH, SPY, QQQ from the prior 16:00 ET to that day's 16:00 ET.
  Claude (no web search) must answer through a forced `label_day` tool call (structured input; the
  first version parsed prose and every reply ran out of tokens before its JSON). It picks one
  category and must cite headline ids from the list; any unknown id, unknown category or uncited
  pick is rejected. A rejected day records why (stop_reason, block types, first words) and is
  retried at most once a day, three times, so a broken reply can't be re-billed every run. Finished days are labelled once
  (`news_cache.json`). Headlines never change the gauge label; disagreement sets `news_flag`
  ("headlines point to the Fed, but no macro gauge moved", or the reverse). Links are http(s) only.
- Real-data check (Oct 14, all 20 days recomputed from Cboe/Yahoo): rates 1 (+ Oct 14 provisional),
  fear 2, market 0, rotation 7, chips 9. Spot checks: Jul 28 (China competition, financing) and
  Sep 14 (AI-policy worries, software +5%) were chip stories; Oct 14 was CPI.
- Claude fills are cached (`claude_fills.json`, 30 days) and re-checked against Yahoo and vetoes
  before reuse, so a value found at 09:50 isn't lost when the 16:25 search comes up empty.
- Alpaca daily bars dated today are dropped until 16:20 NY: a manual run at 09:50 once recorded a
  partial session as a VERIFIED close (Yahoo's bar was equally partial).
- DXY, KOSPI and Samsung are telemetry only (`used_in_call=False`): their gaps never degrade the read.

## Known gaps (do not paper over)
- Thresholds are not fitted. No backtest exists yet. Correct data into an untested rule is
  still an untested call.
- EFFR − 3M bill is a proxy, not the futures curve. Bills can trade away from EFFR on supply alone.
- Filings lag: a quarter is only read once the last company in a group reports (~6–8 weeks after
  quarter end), so the cycle runs a quarter behind by design. Q4 values derived as FY − 9M
  include year-end adjustments; Q4 EPS derived this way is approximate (share counts differ).
- Old EPS facts are not split-adjusted unless a later filing restated them; the earnings catalyst
  only uses the latest two years, which are post-split for NVDA and AVGO. Watch for new splits.
- Group membership is a judgment call (who counts as "software"). Changing it changes the read.
- Headlines are one feed (Benzinga) and Claude's label is a reading of them, not a fact. The cited
  headlines are shown so every label can be checked; a wrong label shows up as a mismatch with them.
- USO tracks front-month WTI futures (roll effects), not spot; UUP tracks the dollar-index futures.
  Both are proxies, chosen because they print same-day on the consolidated tape.
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
