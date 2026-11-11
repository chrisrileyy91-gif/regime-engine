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
| KOSPI, Samsung | EODHD end-of-day (2–3h after close) → KRX Open API (next day) | Yahoo ±0.1% | — |
| Company filings | SEC EDGAR XBRL companyconcept (each company's own 10-Q/10-K) | — | entity-name check per CIK |
| Chip stock prices | Alpaca SIP quarter-end closes (earnings-vs-price catalyst only) | — | — |
| Official, not cross-checked | Cboe TNX/TYX daily high and TYX close; Treasury 2Y; Alpaca SIP RSP, USO (oil), UUP (dollar), HYG (credit), CRWV/ORCL/NBIS (debt-funded AI names, own request) | — | — |
| Headlines | Benzinga via Alpaca news API (same keys), labelled by Claude from the list only | cited ids must exist | — |

Statuses: **VERIFIED** (a confirming source has the same date and agrees) · **OFFICIAL**
(official record, not yet confirmed) · **FILLED** (record missing; a web search found a published
value that matches the independent copy; usable, never counted as confirmed) · **UNCONFIRMED** (vendor record nothing confirmed →
withheld) · **CONFLICT** (disagreement or bad print → withheld) · **MISSING**. A newer value
from a confirming check is shown as `preliminary` and **never** used in the call. Veto-only
checks differ in method or timing: they catch gross errors, they never confirm. A check that
disagrees on 3+ of the last 5 compared dates is **systematic** and withholds the value even if a
confirming copy agrees: Yahoo's ^TNX/^VIX are copies of Cboe's indexes, so they confirm our parse,
not the methodology (first live run: Cboe ^IRX sat 7–10 bp under Treasury all week).

Splits: Alpaca bars are split-adjusted (`adjustment=split`) as soon as Alpaca processes a split;
Yahoo can re-base hours later. Each run looks up recent/announced splits (Alpaca corporate-actions,
v1 then v1beta1). When one is listed and EVERY compared date before its ex-date differs by exactly
that factor, Yahoo's earlier closes are re-based and compared (noted on the input). Nothing else is
re-based. A record that hasn't applied a listed split (or jumps by the factor across the ex-date)
is withheld. A constant clean-ratio gap with no listed split is still withheld, and says so.
(Nov 5, 2026: SOXX 3-for-1; the 06:00 run withheld SOXX as "systematic" because Alpaca was exactly
1/3 of Yahoo on 5 of 5 dates, which also blanked the page's "US data" line.)
Yahoo re-bases piecemeal: at 16:25 the same day it had re-based Oct 30 but not its Nov 2–4 bars, so
only 3 of 5 dates were 3x and the all-or-nothing rule withheld SOXX again. Re-basing is per date:
each pre-split compared date must be either equal or off by exactly the factor, and only the lagging
dates are re-based (named in the note). Any other ratio on any date still withholds. Because a mixed
check could then "agree" with a mixed record, the record itself must not step by the split factor
between two pre-split closes (same nearest-to-factor rule as the ex-date jump); if it does, withheld.

## Last resort: Claude web search (`app/claude_fill.py`)
Runs only when an input's record is MISSING/UNCONFIRMED or a session behind, AND Yahoo has a
value for that exact date, AND that session is over, AND this month's spend is under
`CLAUDE_MONTHLY_BUDGET_USD` (ledger: `claude_usage.json` on the snapshots release). One call per
run covers every gap. CONFLICTs are never sent. Claude is never shown Yahoo's number. A filled
value is used only if it is for the exact date, cites a URL from its own search results, and
agrees with Yahoo within the confirming tolerance; it is then spliced onto the history and
marked FILLED with the page named (it was VERIFIED until Nov 2026, which overstated it). Model: `claude-haiku-5-5` (falls back to
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
`regime.py` (telemetry signals, Kleene helpers) + `replay.py` (label replay, manual) — and separately `fundamentals.py` (SEC fetch,
cached; quarters; phases; catalysts) → `service.py` (payload) → `snapshot.py` (publish job) /
`main.py` (local FastAPI, not deployed).
`alert.py` posts to Pheme. `config.py` is the ONLY place thresholds, sources and tolerances live.
`web/index.html` is a dumb renderer: all logic is in Python and arrives in `regime.json`.

## Publishing (GitHub Pages, no server)
- meridian-clock dispatches `regime-morning.yml` (06:00 NY: finalizes the prior US session,
  picks up the overnight Korea session) and `regime-close.yml` (16:40 NY: 16:00 closing auction,
  16:15 VIX settle seen through Cboe's 15-minute-delayed feed, plus the free SIP 15-minute rule).
  It ran at 16:25 until Nov 10, 2026, and the same-day VIX was never in (19 of 19 runs). Both call
  `publish.yml`. `replay.yml` is manual only and publishes nothing.
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
- Never put a level (10Y percentile, VIX level, RSI) into the stress label or the cycle. The 10Y
  level kept the first page on HIGH RISK for months; RSI < 40 then made the label "Elevated" on 24
  of 39 reads whatever the day did. A level belongs in `backdrop`.
- VIX never claims a tally day: it rises because chips fell. Falling oil is not a cause either.

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
- Catalysts: software ("starting" needs the median growth RATE not falling as well as a positive
  median CHANGE; when they disagree it reads "mixed" and prints both), chip earnings vs chip prices
  (TTM EPS vs price, year to the latest quarter), rate cuts (Treasury 2Y vs EFFR, ±0.25; the 3M
  bill is the fallback when the 2Y is missing).
- Shown beside the phase, never in its rules: each series through its OWN newest quarter, that
  quarter alone vs a year earlier (a trailing year turns ~2 quarters late: chips Q2 2026 were 81%
  trailing, 124% on the quarter), chip revenue without MU (memory pricing), and the funding gauge:
  hyperscaler capex as a % of operating cash flow, trailing year (`ocf` series, same EDGAR API).
  `waiting_on` names the companies whose next quarter the phase is waiting for. A cache written
  before a series was added refetches at once instead of waiting out the 20 hours.
- First real read (filings to Q2 2026): Buildout since Q1 2026, confirmed; Q4 2024–Q4 2025 was
  Digestion (chip revenue growth rolled over from 81% to ~50%); software "starting" (+0.64 then
  +0.36 pts, total 0.992 vs 1.0 needed). `tests/fixtures/sec_filings.txt` holds that data.

## Today (daily)
- Drivers (same-session MOVES only; the yield driver is named "Yields jumped", key `liquidity`. It
  was "Liquidity drain" until Nov 11, 2026: the page reads a yield, not reserves or repo): 10Y +10 bps or z > 1.5 · VIX up > 10% · SOXX down > 4% or
  its return z < −1.5 vs the prior 60 sessions. The 4% floor exists because the z adapts after a
  rout. VIX above 20 and SOXX RSI below 40 are backdrop metrics on the driver cards, not triggers.
- A yield jump counts toward stress only when SOXX also fell 1.5%+ that session (`yields_counted`).
  The card still shows ACTIVE and the reasons say "not counted". Nov 10, 2026: 10Y +12 bps, SOXX
  −0.1%, page read Elevated; over the 120 sessions to then a 10 bps rise went with SOXX +2.1%.
- Funding card (`key="funding"`, `counted=False`): the CRWV/ORCL/NBIS average move minus what its
  prior-120-session slope on SOXX predicts (`indicators.excess_returns`), as a z vs the prior 60
  (fires < −1.5), or HYG < −0.75%. Shown, logged in `active_drivers`, named in the diagnosis and
  reasons, never in the label and never blocks Calm. It stays on trial until the replay says it adds
  something the semis move doesn't.
- Stress: 1 active = Elevated, 2+ or VIX > 30 = Stressed; 0 active = Calm only if all three counted drivers
  were read, otherwise Unknown. The session is the newest official date among 10Y, VIX, SOXX, IGV; a
  trigger whose input is older than that is unknown (never quiet, never active, no VIX > 30
  override). Nov 5 16:25 said Calm on a −12% semis day: SOXX withheld and the VIX settle not yet
  posted, so yesterday's 16.6 stood in for today's 28.2. Bond holidays (Columbus Day, Veterans Day
  on a weekday) are the exception: a missing 10Y there is "yields didn't move", liquidity QUIET.
- Tally: the last 20 sessions with SOXX ≤ −1.5%, each tagged in order:
  1. **oil** — USO UP ≥ 4% or z ≥ +2 vs its prior 60 sessions. One-sided: Nov 4, 2026, oil −8.1%
     with the equal-weight S&P +1.9% was tagged an "oil shock". The 4% floor exists because a 60-day
     z adapts shocks away in a war regime (Sep 1, US strikes on Iran: USO +5.5% = 1.65σ).
  2. **dollar** — UUP z ≥ +2.
  3. **rates / market** — the broad market and yields explain ≥ half the drop (`today.decompose`):
     an OLS fit of SOXX's daily return on RSP's return and the 10Y change over the 120 sessions
     BEFORE the day (min 60, no look-ahead, intercept left out of the estimate). "rates" when the
     10Y's part is the larger, else "market". Not SPY: chips are a big slice of SPY, so a semis rout
     drags it (Oct 14 SPY −1.1%; Sep 14 SPY −0.5%, RSP +0.1%).
  A yield jump no longer claims a day (it did until Nov 11, 2026; on Oct 14 and Aug 28 the fit
  explained under a quarter of the drop). It is listed in `moved` as "rates" when the 10Y or 30Y is
  up ≥ 10 bps at the close OR the intraday high vs the prior close (CPI day Oct 14: +11.2 bps at
  the high, closed −2.7), the 10Y close z ≥ 1.5, or the Treasury 2Y is up ≥ 10 bps (Aug 28: hike
  odds to 60%, 2Y +14 bps). A date before Treasury's latest row with no row is a bond holiday:
  "2Y closed", not missing.
  4. **rotation** (software up) / **chips** (software down) — mostly chip-specific.
  Every row carries `macro_share` and `expected_pct`, so the label is never the whole story.
  The summary ends with the same fit on the newest window (`today.sensitivity`): SOXX per +10 bps
  in the 10Y, per 1% in RSP, and the R². It is a description of the last 120 sessions, not a forecast.
  VIX is in the detail line only (to Nov 2026 a "fear" tag claimed days: Nov 5, SOXX −12%, VIX
  +70%, was "VIX spiked"; 6 of 8 headline/gauge disagreements were that kind). `moved` lists every
  gauge that moved, including **credit** (HYG ≤ −0.75% or z ≤ −2), which annotates and never
  labels. The debt-funded AI names' average move sits beside it (`ai_leverage_pct`).
  The debt-funded names also carry `ai_excess_pct` (beyond their usual move with chips).
  A row is "unknown" unless every gauge that could claim it is present. The newest row is
  `provisional` (not counted) only while a gauge that can change its label is missing (10Y close,
  oil, dollar, RSP, IGV); the evening 2Y and the intraday highs only annotate.
  The page must never say "no macro trigger": gauges see moves, not news, policy or levels
  (Aug 18 "yields rattle markets" was the 30Y LEVEL at a 2007 high; its move was +1.7 bps).
- USO and UUP are what Crude Logic (frack-around) already uses after Yahoo's CL=F broke; EIA's
  official WTI spot posts in weekly batches, too late for same day. The dollar on the overnight
  line is UUP too; the DXY input (EODHD FX, ECB veto, web-search fill) was removed in Nov 2026.
- Headlines (`app/news.py`): for each tally day, Alpaca/Benzinga headlines tagged to NVDA, AVGO,
  AMD, MU, TSM, ASML, INTC, SOXX, SMH, SPY, QQQ from the prior 16:00 ET to that day's 16:00 ET.
  Claude (no web search) must answer through a forced `label_day` tool call (structured input; the
  first version parsed prose and every reply ran out of tokens before its JSON). It picks one
  category and must cite headline ids from the list; any unknown id, unknown category or uncited
  pick is rejected. A rejected day records why (stop_reason, block types, first words) and is
  retried at most once a day, three times, so a broken reply can't be re-billed every run. Finished days are labelled once
  (`news_cache.json`). Headlines never change the gauge label; disagreement sets `news_flag`
  ("headlines point to the Fed, but no macro gauge moved", or the reverse). Links are http(s) only.
- Real-data check under the OLD first-match rules (Oct 14, all 20 days recomputed from Cboe/Yahoo):
  rates 1 (+ Oct 14 provisional), fear 2, market 0, rotation 7, chips 9. Not yet redone for the
  explained-share rules. Spot checks: Jul 28 (China competition, financing) and
  Sep 14 (AI-policy worries, software +5%) were chip stories; Oct 14 was CPI.
- Claude fills are cached (`claude_fills.json`, 30 days) and re-checked against Yahoo and vetoes
  before reuse, so a value found at 09:50 isn't lost when the close-run search comes up empty.
- Alpaca daily bars dated today are dropped until 16:20 NY: a manual run at 09:50 once recorded a
  partial session as a VERIFIED close (Yahoo's bar was equally partial).
- KOSPI and Samsung are telemetry only (`used_in_call=False`): their gaps never degrade the read.
- The Signals table prints the source each value actually came from this run, never a fixed name.

## Known gaps (do not paper over)
- Thresholds are not fitted. `python -m app.replay` (workflow "Replay stress label") rebuilds the
  old and new stress labels over history and tabulates them against SOXX's same-day move; it is a
  description, not a fit, and the new label includes SOXX's own move by construction. The
  explained-share threshold (0.5), the SOXX floor (−4%), the 2Y band (±0.25) and the credit
  thresholds are judgment calls until someone fits them.
- The explained share is a two-factor linear fit on 120 sessions: beta drifts, RSP holds some
  chip names, and a day when yields spike intraday and reverse reads as unexplained (it is still
  listed as a yield jump). Treat it as a rough split, not a measurement to the point.
- The funding card's basket is three names (one is also a capex spender) and its excess is a
  one-factor fit: a company-specific day (an earnings miss) fires it as readily as funding stress.
  HYG rarely moves 0.75%. Neither threshold is fitted.
- The funding gauge reads capex vs operating cash flow only. Debt issuance is not read: filings
  omit quarters with no issuance, so a gap can't be told from a zero. Leases and off-balance-sheet
  financing (the CoreWeave-style deals the headlines are about) are invisible to it.
- 2Y − EFFR is a proxy, not the futures curve: the 2Y carries a term premium, and Treasury posts
  it in the evening, so the close read uses the prior day's 2Y.
- Filings lag: a quarter is only read once the last company in every group reports. With AVGO's
  October year-end that is ~10–11 weeks after quarter end (Q3 lands mid-December), so the cycle runs a quarter behind by design. Q4 values derived as FY − 9M
  include year-end adjustments; Q4 EPS derived this way is approximate (share counts differ).
- Old EPS facts are not split-adjusted unless a later filing restated them; the earnings catalyst
  only uses the latest two years, which are post-split for NVDA and AVGO. Watch for new splits.
- Group membership is a judgment call (who counts as "software"). Changing it changes the read.
- Headlines are one feed (Benzinga) and Claude's label is a reading of them, not a fact. The cited
  headlines are shown so every label can be checked; a wrong label shows up as a mismatch with them.
- USO tracks front-month WTI futures (roll effects), not spot; UUP tracks the dollar-index futures.
  Both are proxies, chosen because they print same-day on the consolidated tape.
- EFFR is published the next morning and changes only at FOMC decisions. On a decision day the
  close read still carries the pre-decision EFFR.
- Korea at 06:00 depends on EODHD posting within ~3.5h of the 02:30 ET close; if it hasn't, the
  prior session is used and the newer Yahoo value shows as preliminary.
- EODHD and KRX keys are optional and were not set as of Nov 2026. Without them Korea is filled
  by Claude every run and shown as FILLED; its history (Samsung RSI, KOSPI SMA200) then comes from
  Yahoo. A free KRX Open API key (`KRX_API_KEY`) makes both official.
- A Claude fill is a second independent source, not an official record. If Yahoo is also down,
  the gap stays open.
