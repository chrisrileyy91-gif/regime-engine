# Claude Code prompt — stand up regime-engine and schedule it

Paste everything below the line into Claude Code, opened on `C:\Users\chris\Dev\regime-engine`
(after unzipping regime-engine.zip into `C:\Users\chris\Dev`).

---

Stand up the regime-engine repo, publish it to GitHub Pages, and schedule it in meridian-clock.
Shell is PowerShell. GitHub account: chrisrileyy91-gif (gh CLI is authenticated).

## Part A — regime-engine (this folder)

**Step 0 — read before doing anything.** Read these files in full: `CLAUDE.md`, `README.md`,
`pyproject.toml`, `.gitignore`, `.github/workflows/publish.yml`, `regime-open.yml`,
`regime-close.yml`, `app/snapshot.py`.

ALLOWED to change: nothing in this repo unless a step below fails; if one fails, stop and report
the error instead of editing code.
FORBIDDEN: adding a `schedule:` block to any workflow; committing `state/`, `site/`, `data/` or `.venv/`.

1. Create the venv and install: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"`
2. Verify: `pytest -q` (expect all passed), `ruff check .`, `python -m mypy app` (expect no issues).
3. First live read: `python -m app.snapshot --state state --site site`. Report the two summary
   lines it prints and any "feed down" lines. Expect `feeds ok 10/10`; any lower, report which feed.
4. Show me the read: print `phase`, `risk_level`, `confidence`, `diagnosis`, `tnx_yield`, `semi_rsi`
   from `site/regime.json`, and the `notes` array.
5. Create the repo and push (public, for free Pages, same as treasury-flow-radar; no secrets in it):
   `git init -b main; git add .; git commit -m "feat: regime engine with Pages dashboard"`
   `gh repo create chrisrileyy91-gif/regime-engine --public --source . --push`
6. Enable Pages with GitHub Actions as the source:
   `gh api -X POST repos/chrisrileyy91-gif/regime-engine/pages -f build_type=workflow`
7. First publish: `gh workflow run regime-close.yml`, wait a few seconds, then
   `gh run watch (gh run list --workflow regime-close.yml --limit 1 --json databaseId -q '.[0].databaseId')`.
8. Verify: `gh release view snapshots --json assets -q '.assets[].name'` lists `regime.json` and
   `history.json`, and `Invoke-WebRequest https://chrisrileyy91-gif.github.io/regime-engine/regime.json`
   returns 200. Report the page URL.

## Part B — meridian-clock (`C:\Users\chris\Dev\meridian-clock`)

**Step 0 — read in full:** `CLAUDE.md`, `schedule.json`, `clock.py`, `tests/test_clock.py`.
Then `git pull --rebase`.

ALLOWED: `schedule.json`, `CLAUDE.md`.
FORBIDDEN: `clock.py`, `pheme.py`, `railway.json`, `requirements.txt`, everything under `tests/`.

1. In `schedule.json`, add these two entries at the end of the array, matching the existing
   column alignment:
   `{"repo": "regime-engine", "workflow": "regime-open.yml", "at": "10:15", "days": "weekdays"}`
   `{"repo": "regime-engine", "workflow": "regime-close.yml", "at": "16:25", "days": "weekdays"}`
2. In `CLAUDE.md`, the first paragraph lists the repos the clock dispatches into
   ("market-dashboard, meridianos, the-dashboard and treasury-flow-radar"). Add regime-engine to it.
3. Verify:
   - `Select-String -Path schedule.json -Pattern '"regime-engine"'` → exactly 2 matches
   - `python -c "import json;json.load(open('schedule.json'))"` → no error
   - `python -m pytest -q` → all passed
   - `$env:DRY_RUN="1"; python clock.py --now 2026-10-14T16:30; Remove-Item Env:DRY_RUN` → output
     includes "would dispatch" for regime-open.yml and regime-close.yml (it needs GH_CLOCK_TOKEN in
     the env to list runs; if it isn't set locally, skip this check and say so)
4. Commit and push: `git add schedule.json CLAUDE.md; git commit -m "feat: schedule regime-engine publish (10:15, 16:25 NY)"; git pull --rebase; git push`

## Report back
Branch, commit SHAs for both repos, test/lint results, the live read from Part A step 4, the
Pages URL, and anything that failed.

## One manual check for Chris (Claude Code can't do this)
If `GH_CLOCK_TOKEN` (Railway variable on meridian-clock) is a fine-grained token limited to
selected repositories, add `regime-engine` to it with **Actions: Read and write**, or the clock's
dispatch will get a 404/403 and open a "clock: dispatch failed" issue.
