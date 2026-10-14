# Market Regime Engine

Rule-based macro regime classifier (liquidity squeeze / hardware buildout / software
monetization / rate easing), published as a phone page from official data sources. Built from a Gemini-generated spec,
with the spec's logic errors corrected — see "Changes from the original spec".

## How it runs
meridian-clock dispatches two workflows on weekdays: `regime-morning.yml` at 06:00 NY (prior US
session finalized from official sources, plus the overnight Korea session) and `regime-close.yml`
at 16:25 NY (after the closing auction and VIX settlement). Each runs `python -m app.snapshot`,
which writes `regime.json` + `history.json`, and deploys `web/index.html` with them to GitHub
Pages: https://chrisrileyy91-gif.github.io/regime-engine/

## Run locally (PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -v
python -m app.snapshot --state state --site site      # one live read
python -m http.server 8000 -d site                    # view it at http://localhost:8000
uvicorn app.main:app --reload --port 8000             # optional JSON API at /api/v1/regime
```

## The page
Phase dial and risk, **why semis are moving** (three drivers, each active / quiet / no data, with
the value against its trigger and a one-line diagnosis), every phase rule with each condition's
✓ / ✕ / ?, the telemetry signals, and the last 30 reads.

## Data: verified or blank
Same-day sources of record (Cboe, consolidated tape via Alpaca, NY Fed, EODHD), each checked
against an independent copy (Yahoo) and, where methods differ, a veto-only check (U.S. Treasury,
ECB, Korea Exchange). A number that fails a
check is withheld, never guessed. The source table and statuses are in `CLAUDE.md`; the page
shows every input's status, source, date and what it was checked against.

When a source is missing or a session behind, a last-resort step asks Claude (web search) for
that exact date's value; it is used only if it cites a real search result and matches Yahoo
independently. Spend is capped per month.

Secrets on the repo: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ANTHROPIC_API_KEY`,
`DISCORD_PHEME_WEBHOOK`, and optionally `EODHD_API_KEY`, `KRX_API_KEY`.

## Payload
Original keys (`phase`, `risk_level`, `liquidity_squeeze`, `semi_rsi`, `tnx_yield`,
`actionable_summary`) plus `confidence`, `matrix`, every rule with every condition's value and
pass/fail/unknown (`rules`), the 7-row telemetry table, raw `inputs`, per-feed `sources`, and
`stale` flags.

## Changes from the original spec
| # | Spec | Problem | Now |
|---|------|---------|-----|
| 1 | `if squeeze and rsi<45 → P2; elif rsi>55 → P1; else → P3` | Squeeze with RSI 45–55 fell to **Phase 3 / LOW risk**. Phase 3 was the default bucket, not a finding. | Explicit precedence 2>4>1>3; Unclassified when nothing fires. |
| 2 | Three RSI threshold sets (55/45, <40, 35–48) | Contradictory | One table in `config.py` |
| 3 | Phase 1 code ignored the yield-slope leg | Rule ≠ contract | Both legs enforced |
| 4 | Phase 4 not implemented | — | EFFR − 3M bill proxy + DXY |
| 5 | Rolling-mean RSI on 1 month of data | ~7 valid points; different indicator (30.2 vs Wilder 37.8 on the same test series) | Wilder RSI on 2y history |
| 6 | `async def` around blocking `yf.download` | Blocks event loop; violates its own httpx rule | httpx async, concurrent fetch |
| 7 | One multi-ticker download incl. ^KS11 | US/KR calendars misalign → NaN on `iloc[-1]` | Per-symbol fetch, inner-join only where needed |
| 8 | No error handling | Contradicted "never crash" | Per-feed isolation, SQLite stale snapshot, 503 last |
| 9 | Missing data silently false | Hides blind spots | Three-valued logic + `confidence` |
| 11 | Fixed "10Y > 4.25%" Phase 2 leg | Fired every day once yields sat above it (first live read: 5.22%, falling, VIX 15) | 10Y above the 80th percentile of its past year |
| 12 | Diagnosis blamed the 10Y level for a one-day drop | A level can't explain a single session | Same-session moves only; levels go in `backdrop` |
| 14 | Treasury curve and FRED as the yield source | Treasury hadn't posted Oct 13 by 20:40; FRED lags a day | Cboe's same-day official TNX/IRX/VIX closes; Treasury kept as a veto-only check |
| 15 | Cboe ^IRX as the 3M bill record | Ran 7–10 bp under Treasury every day; Yahoo's "confirmation" was a copy of the same index | Treasury is the record; persistent disagreement now withholds even when a copy agrees |
| 13 | Yahoo as the only source | Unofficial; first read had ETF bars a session late | Official sources first, verified-or-blank, Pheme alerts, missed-run banner |
| 10 | Gemini dashboard: composite scores, "Action: NEUTRAL HEDGE", ARR slider | Undefined formulas; ARR has no feed; showed VIX 21 as "stable" | Dropped. Every number on the page traces to a feed and a rule |
