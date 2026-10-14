# Market Regime Engine

Rule-based macro regime classifier (liquidity squeeze / hardware buildout / software
monetization / rate easing) served as a FastAPI endpoint. Built from a Gemini-generated spec,
with the spec's logic errors corrected — see "Changes from the original spec".

## How it runs
meridian-clock dispatches two workflows on weekdays: `regime-open.yml` at 10:15 NY (intraday
read) and `regime-close.yml` at 16:25 NY (after the close). Each runs `python -m app.snapshot`,
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

## Data
Yahoo chart API (^TNX, ^VIX, SOXX, IGV, ^KS11, DX-Y.NYB, 005930.KS) and FRED CSV (DGS10, DGS2,
DFF). No API keys needed. Each feed is fetched and cleaned independently; any one can fail
without taking down the endpoint.

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
| 4 | Phase 4 not implemented | — | EFFR−2Y proxy + DXY |
| 5 | Rolling-mean RSI on 1 month of data | ~7 valid points; different indicator (30.2 vs Wilder 37.8 on the same test series) | Wilder RSI on 2y history |
| 6 | `async def` around blocking `yf.download` | Blocks event loop; violates its own httpx rule | httpx async, concurrent fetch |
| 7 | One multi-ticker download incl. ^KS11 | US/KR calendars misalign → NaN on `iloc[-1]` | Per-symbol fetch, inner-join only where needed |
| 8 | No error handling | Contradicted "never crash" | Per-feed isolation, SQLite stale snapshot, 503 last |
| 9 | Missing data silently false | Hides blind spots | Three-valued logic + `confidence` |
| 10 | Gemini dashboard: composite scores, "Action: NEUTRAL HEDGE", ARR slider | Undefined formulas; ARR has no feed; showed VIX 21 as "stable" | Dropped. Every number on the page traces to a feed and a rule |
