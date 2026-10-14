"""Last-resort gap filler: Claude with web search finds a value the pipeline is missing.

When it runs (all must hold):
  * an input's record is MISSING or UNCONFIRMED, or is a session behind a confirming check
    (e.g. the Korea Exchange posts a day late). CONFLICTs are never sent: a conflict means a
    source broke, and that should be seen, not papered over;
  * a confirming check (Yahoo) has a value for that exact date, so there is something
    independent to compare against. Without one, Claude's number could never be used;
  * that date's session is over (no intraday prices), and this month's spend is under budget.

What makes a filled number usable:
  * Claude returns it for the exact date asked, with a source URL that appeared in its own
    search results (no invented citations);
  * it agrees with Yahoo within the input's confirming tolerance. Claude is never told
    Yahoo's number, so the agreement is independent.
The filled value is spliced onto the input's history and marked VERIFIED, with the source
page named. Anything else is dropped and noted.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.config import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    CLAUDE_FALLBACK_MODEL,
    CLAUDE_MAX_SEARCHES,
    CLAUDE_MODEL,
    CLAUDE_MONTHLY_BUDGET_USD,
    CLAUDE_PRICES,
    CLAUDE_SEARCH_PRICE,
    CLAUDE_TIMEOUT_SECONDS,
    GAP_QUESTIONS,
    INPUTS,
    KR_SESSION_DONE_NY,
    US_SESSION_DONE_NY,
)
from app.data import FetchResult, redact
from app.models import InputStatus
from app.verify import SOURCE_NAMES, Verified, _agrees

NY = ZoneInfo("America/New_York")
UsageState = dict[str, dict[str, float]]  # "YYYY-MM" -> {"usd": ..., "calls": ...}


@dataclass
class Gap:
    key: str
    date: str  # YYYY-MM-DD, the session we need
    question: str
    expected: float  # the confirming check's value; never shown to Claude
    tolerance: float
    kind: str
    check_name: str
    base: pd.Series  # history before `date` to splice onto
    base_note: str
    vetoes: list[tuple[str, float, float, str]] = field(default_factory=list)  # (name, value, tol, kind)


@dataclass
class Fill:
    key: str
    date: str
    value: float
    source_url: str
    source_name: str


@dataclass
class FillReport:
    fills: dict[str, Fill] = field(default_factory=dict)
    gaps: dict[str, Gap] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    model: str | None = None


def session_done(market: str, day: str, now_ny: datetime) -> bool:
    today = now_ny.date().isoformat()
    if day < today:
        return True
    if day > today:
        return False
    if market in ("us", "kr"):
        cutoff = US_SESSION_DONE_NY if market == "us" else KR_SESSION_DONE_NY
        hour, minute = (int(x) for x in cutoff.split(":"))
        return (now_ny.hour, now_ny.minute) >= (hour, minute)
    return True  # "fx": the latest value is what's wanted


def find_gaps(verified: dict[str, Verified], results: dict[str, FetchResult], now_ny: datetime) -> list[Gap]:
    gaps: list[Gap] = []
    for key, v in verified.items():
        if key not in GAP_QUESTIONS or v.status.status == "CONFLICT":
            continue
        spec = INPUTS[key]
        check = next(
            (
                (r, tol, kind)
                for k, tol, kind, confirms in spec.checks
                if confirms and (r := results.get(k)) is not None and r.ok
            ),
            None,
        )
        if check is None:
            continue
        result, tolerance, kind = check
        assert result.series is not None
        st = v.status
        question, market = GAP_QUESTIONS[key]
        # Newest session the check has that is finished (an in-progress Korean session at night,
        # or a US session before the close, is skipped in favor of the last completed one).
        done = [d for d in (str(ts.date()) for ts in reversed(result.series.index)) if session_done(market, d, now_ny)]
        if not done:
            continue
        if st.status in ("MISSING", "UNCONFIRMED"):
            day = done[0]
        elif st.date is not None and done[0] > st.date:
            day = done[0]  # the record is a session behind
        else:
            continue
        ts = pd.Timestamp(day)

        base, base_note = _history(key, results, ts, v.series, result)
        vetoes = [
            (SOURCE_NAMES[r.source], float(r.series.loc[ts]), tol, kind)
            for k, tol, kind, confirms in spec.checks
            if not confirms and (r := results.get(k)) is not None and r.series is not None and ts in r.series.index
        ]
        gaps.append(
            Gap(
                key=key,
                date=day,
                question=question,
                expected=float(result.series.loc[ts]),
                tolerance=tolerance,
                kind=kind,
                check_name=SOURCE_NAMES[result.source],
                base=base,
                base_note=base_note,
                vetoes=vetoes,
            )
        )
    return gaps


def _history(
    key: str, results: dict[str, FetchResult], before: pd.Timestamp, record: pd.Series | None, check: FetchResult
) -> tuple[pd.Series, str]:
    if record is not None:
        return record[record.index < before], "history from the record source"
    for k in INPUTS[key].record:
        r = results.get(k)
        if r is not None and r.ok and r.series is not None:
            return r.series[r.series.index < before], f"history from {SOURCE_NAMES[r.source]}"
    assert check.series is not None
    return check.series[check.series.index < before], f"history from {SOURCE_NAMES[check.source]} (unconfirmed)"


def build_prompt(gaps: list[Gap]) -> str:
    items = "\n".join(f'- id "{g.key}": {g.question}, for the session dated {g.date}' for g in gaps)
    return (
        "You are filling gaps in a market-data pipeline. For each item, search the web for the value on the "
        "EXACT date given. Use authoritative pages: the exchange or index provider, a major wire service "
        "(Reuters, AP, Bloomberg, Yonhap), or a major financial data site. Never estimate, interpolate, or use "
        "a different date. If no page you found states the value for that exact date, return null.\n\n"
        f"Items:\n{items}\n\n"
        "Answer with only a JSON array inside a ```json code block, one object per item:\n"
        '[{"id": "...", "date": "YYYY-MM-DD", "value": number or null, "source_url": "the page that states '
        'it", "source_name": "publisher", "quote": "the exact sentence or table cell stating the number"}]'
    )


def parse_claude_response(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str], int]:
    """Return (answer objects, URLs the search actually returned, number of searches)."""
    texts: list[str] = []
    urls: set[str] = set()
    searches = 0
    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            texts.append(block.get("text", ""))
        elif kind == "server_tool_use" and block.get("name") == "web_search":
            searches += 1
        elif kind == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, list):
                urls.update(str(item.get("url")) for item in content if isinstance(item, dict) and item.get("url"))
    usage_searches = ((payload.get("usage") or {}).get("server_tool_use") or {}).get("web_search_requests")
    if isinstance(usage_searches, int):
        searches = usage_searches
    text = "\n".join(texts)
    match = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    raw = match[-1] if match else text[text.find("[") : text.rfind("]") + 1]
    try:
        answers = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        answers = []
    return (answers if isinstance(answers, list) else []), urls, searches


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def cost_of(payload: dict[str, Any], model: str, searches: int) -> float:
    usage = payload.get("usage") or {}
    tokens_in = int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
    tokens_in += int(usage.get("cache_creation_input_tokens") or 0)
    tokens_out = int(usage.get("output_tokens") or 0)
    price_in, price_out = CLAUDE_PRICES.get(model, CLAUDE_PRICES[CLAUDE_FALLBACK_MODEL])
    if model == "claude-haiku-5-5" and tokens_in > 100_000:
        price_in, price_out = 0.50, 2.50  # Haiku 5.5's long-prompt tier
    return tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6 + searches * CLAUDE_SEARCH_PRICE


async def _post(client: httpx.AsyncClient, api_key: str, model: str, prompt: str) -> httpx.Response:
    body = {
        "model": model,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": CLAUDE_MAX_SEARCHES}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    last: httpx.Response | None = None
    for attempt in range(3):
        last = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=CLAUDE_TIMEOUT_SECONDS)
        if last.status_code not in (429, 500, 502, 503, 504, 529):
            return last
        await asyncio.sleep(2.0 * (attempt + 1))
    assert last is not None
    return last


def month_key(now_ny: datetime) -> str:
    return now_ny.strftime("%Y-%m")


async def fill_gaps(
    verified: dict[str, Verified],
    results: dict[str, FetchResult],
    usage: UsageState,
    now_ny: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> FillReport:
    now_ny = now_ny or datetime.now(NY)
    report = FillReport()
    gaps = find_gaps(verified, results, now_ny)
    if not gaps:
        return report
    report.gaps = {g.key: g for g in gaps}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        report.notes.append(f"Claude gap filler not configured; {len(gaps)} gap(s) left open")
        return report
    month = usage.setdefault(month_key(now_ny), {"usd": 0.0, "calls": 0.0})
    if month["usd"] >= CLAUDE_MONTHLY_BUDGET_USD:
        report.notes.append(f"Claude gap filler skipped: monthly budget ${CLAUDE_MONTHLY_BUDGET_USD:.2f} reached")
        return report

    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        prompt = build_prompt(gaps)
        model = CLAUDE_MODEL
        response = await _post(client, api_key, model, prompt)
        if response.status_code in (400, 404):
            model = CLAUDE_FALLBACK_MODEL
            response = await _post(client, api_key, model, prompt)
        if response.status_code != 200:
            report.notes.append(f"Claude gap filler failed: HTTP {response.status_code} {redact(response.text[:200])}")
            return report
        payload = response.json()
    except Exception as exc:
        report.notes.append(f"Claude gap filler failed: {redact(f'{type(exc).__name__}: {exc}')}")
        return report
    finally:
        if owns:
            await client.aclose()

    answers, urls, searches = parse_claude_response(payload)
    report.model = model
    report.cost_usd = cost_of(payload, model, searches)
    month["usd"] = round(month["usd"] + report.cost_usd, 6)
    month["calls"] = month["calls"] + 1
    returned_hosts = {_host(u) for u in urls}

    by_id = {str(a.get("id")): a for a in answers if isinstance(a, dict)}
    for gap in gaps:
        answer = by_id.get(gap.key)
        label = INPUTS[gap.key].label
        if answer is None or not isinstance(answer.get("value"), (int, float)):
            report.notes.append(f"{label}: Claude found no value for {gap.date}")
            continue
        url = str(answer.get("source_url") or "")
        if str(answer.get("date")) != gap.date:
            report.notes.append(f"{label}: Claude answered for {answer.get('date')}, not {gap.date}; not used")
            continue
        if not url or (url not in urls and _host(url) not in returned_hosts):
            report.notes.append(f"{label}: Claude's source wasn't among its search results; not used")
            continue
        value = float(answer["value"])
        if not _agrees(gap.expected, value, gap.tolerance, gap.kind):
            report.notes.append(
                f"{label}: Claude found {value:.4g} for {gap.date} ({_host(url)}), {gap.check_name} has "
                f"{gap.expected:.4g}; they disagree, not used"
            )
            continue
        vetoed = [(n, v) for n, v, tol, kind in gap.vetoes if not _agrees(v, value, tol, kind)]
        if vetoed:
            name, other = vetoed[0]
            report.notes.append(
                f"{label}: Claude's {value:.4g} for {gap.date} fails the {name} check ({other:.4g}); not used"
            )
            continue
        report.fills[gap.key] = Fill(gap.key, gap.date, value, url, str(answer.get("source_name") or _host(url)))
    return report


def apply_fills(verified: dict[str, Verified], report: FillReport) -> dict[str, Verified]:
    out = dict(verified)
    for key, fill in report.fills.items():
        gap = report.gaps[key]
        old = verified[key].status
        point = pd.Series([fill.value], index=[pd.Timestamp(fill.date)])
        series = pd.concat([gap.base, point]).sort_index().astype(float)
        out[key] = Verified(
            InputStatus(
                key=key,
                label=old.label,
                used_in_call=old.used_in_call,
                status="VERIFIED",
                value=fill.value,
                date=fill.date,
                source=f"Claude web search ({_host(fill.source_url)})",
                confirmed_by=[gap.check_name],
                notes=[f"filled from {fill.source_url}", gap.base_note],
            ),
            series,
        )
    return out
