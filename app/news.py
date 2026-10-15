"""Headlines for each tally day: the part the gauges can't see.

Gauges measure moves (yields, oil, the dollar, VIX, the broad market). They can't see an export
rule, a guidance cut or a CPI print that the market absorbed without moving them. This module:

1. pulls the day's Benzinga headlines from Alpaca's news API for the chipmakers, the semis ETFs
   and the index ETFs, inside the session window (prior close 16:00 ET to that day's 16:00 ET);
2. asks Claude to CLASSIFY them (no web search: it can only use the headlines it is given) and to
   cite the ids of the headlines that support the label;
3. keeps a label only if every cited id exists, stores the cited headlines with their links and
   times, and caches finished days for good (a past session's headlines don't change).

Headlines annotate a day; they never overwrite a gauge. When the headlines point at macro but no
macro gauge moved, the tally flags the disagreement instead of picking a side.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.claude_fill import UsageState, cost_of, month_key
from app.config import (
    ALPACA_NEWS_URL,
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    CLAUDE_FALLBACK_MODEL,
    CLAUDE_MODEL,
    CLAUDE_MONTHLY_BUDGET_USD,
    CLAUDE_TIMEOUT_SECONDS,
    NEWS_CATEGORIES,
    NEWS_MAX_DAYS_PER_RUN,
    NEWS_MAX_PAGES,
    NEWS_SYMBOLS,
    US_SESSION_DONE_NY,
)
from app.data import get_with_retry, redact

NY = ZoneInfo("America/New_York")
NewsCache = dict[str, dict[str, Any]]  # {YYYY-MM-DD: {category, summary, evidence[], ...} | {failed, last_try}}
MAX_ATTEMPTS = 3  # a day whose answer keeps failing is retried once a day, then left unread
LABEL_TOOL: dict[str, Any] = {
    "name": "label_day",
    "description": "Record the one category that best explains the chip sell-off, citing headline ids.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(NEWS_CATEGORIES)},
            "summary": {"type": "string", "description": "What the cited headlines say, 20 words or fewer."},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Ids of the headlines that support the category (empty only for none_clear).",
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["category", "summary", "evidence", "confidence"],
    },
}


def session_window(day: str) -> tuple[datetime, datetime]:
    """Prior weekday's 16:00 ET to this day's 16:00 ET: everything the session could react to."""
    d = pd.Timestamp(day)
    prior = d - pd.offsets.BDay(1)
    start = datetime.combine(prior.date(), time(16, 0), NY)
    end = datetime.combine(d.date(), time(16, 0), NY)
    return start, end


def _safe_url(url: Any) -> str:
    """Only plain web links reach the page (a feed could carry javascript: or data: URLs)."""
    text = str(url or "").strip()
    return text if text.startswith(("https://", "http://")) else ""


def parse_news(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for page in pages:
        for item in page.get("news") or []:
            nid = str(item.get("id", ""))
            headline = str(item.get("headline") or "").strip()
            if not nid or not headline or nid in seen:
                continue
            seen.add(nid)
            out.append(
                {
                    "id": nid,
                    "headline": headline,
                    "summary": str(item.get("summary") or "")[:280],
                    "created_at": str(item.get("created_at") or ""),
                    "url": _safe_url(item.get("url")),
                    "source": str(item.get("source") or "benzinga"),
                    "symbols": [str(x) for x in item.get("symbols") or []],
                }
            )
    out.sort(key=lambda x: str(x["created_at"]))
    return out


async def fetch_headlines(client: httpx.AsyncClient, day: str) -> list[dict[str, Any]]:
    key_id, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key_id or not secret:
        raise RuntimeError("not configured (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
    start, end = session_window(day)
    params: dict[str, str] = {
        "symbols": ",".join(NEWS_SYMBOLS),
        "start": start.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": "50",
        "sort": "asc",
    }
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    pages: list[dict[str, Any]] = []
    for _ in range(NEWS_MAX_PAGES):
        page = (await get_with_retry(client, ALPACA_NEWS_URL, params=params, headers=headers)).json()
        pages.append(page)
        token = page.get("next_page_token")
        if not token:
            break
        params["page_token"] = str(token)
    return parse_news(pages)


def _et(stamp: str) -> str:
    try:
        return pd.Timestamp(stamp).tz_convert(NY).strftime("%a %H:%M ET")
    except (ValueError, TypeError):
        return stamp


def build_prompt(day: str, soxx_pct: float, gauges: str, headlines: list[dict[str, Any]]) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in NEWS_CATEGORIES.items())
    lines = "\n".join(
        f"[{h['id']}] {_et(h['created_at'])} ({', '.join(h['symbols'][:5])}) {h['headline']}"
        + (f" — {h['summary']}" if h["summary"] else "")
        for h in headlines
    )
    return (
        f"On {day} the iShares Semiconductor ETF (SOXX) fell {abs(soxx_pct):.1f}%. Market gauges that day: "
        f"{gauges}.\n\nBelow are ALL the Benzinga headlines tagged to chipmakers, chip ETFs and index ETFs "
        "between the prior close and that day's close. Use ONLY these headlines. Do not use outside "
        "knowledge or guess.\n\n"
        f"Pick the ONE category that best explains why chip stocks fell that day:\n{cats}\n\n"
        "Rules: cite 1-3 headline ids (the numbers in square brackets) that directly support your pick. If no "
        "headline explains the drop, answer none_clear with no ids. Prefer what moved the whole chip group "
        "over a single company. The summary must restate what the cited headlines say, in 20 words or "
        "fewer, with no added facts.\n\n"
        f"Headlines:\n{lines}\n\n"
        "Record your answer with the label_day tool."
    )


def _answer(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The label_day tool call (the forced, structured answer), else JSON in the text."""
    blocks = payload.get("content") or []
    for block in blocks:
        if block.get("type") == "tool_use" and block.get("name") == LABEL_TOOL["name"]:
            data = block.get("input")
            return (data, "") if isinstance(data, dict) else (None, "tool call without an input object")
    text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data, ""
        except json.JSONDecodeError:
            pass
    stop = payload.get("stop_reason")
    kinds = ",".join(str(b.get("type")) for b in blocks) or "none"
    snippet = redact(" ".join(text.split())[:120])
    return None, f"no structured answer; stop_reason={stop}, blocks={kinds}, text starts {snippet!r}"


def parse_label(payload: dict[str, Any], headlines: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Validate Claude's answer against the headlines it was given. Returns (label, problem)."""
    answer, problem = _answer(payload)
    if answer is None:
        return None, problem
    category = str(answer.get("category", ""))
    if category not in NEWS_CATEGORIES:
        return None, f"unknown category {category!r}"
    by_id = {h["id"]: h for h in headlines}
    cited = [str(x) for x in answer.get("evidence") or []]
    if any(c not in by_id for c in cited):
        return None, "cited a headline that wasn't in the list"
    if category != "none_clear" and not cited:
        return None, "picked a driver without citing a headline"
    evidence = [
        {"headline": by_id[c]["headline"], "url": by_id[c]["url"], "time": _et(by_id[c]["created_at"])}
        for c in cited[:3]
    ]
    confidence = str(answer.get("confidence", "low"))
    return {
        "category": category,
        "summary": str(answer.get("summary", ""))[:200],
        "evidence": evidence,
        "confidence": confidence if confidence in ("high", "medium", "low") else "low",
        "headline_count": len(headlines),
    }, ""


async def _classify(client: httpx.AsyncClient, api_key: str, prompt: str) -> tuple[dict[str, Any] | None, str, str]:
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    for model in (CLAUDE_MODEL, CLAUDE_FALLBACK_MODEL):
        body = {
            "model": model,
            "max_tokens": 1024,
            "tools": [LABEL_TOOL],
            "tool_choice": {"type": "tool", "name": LABEL_TOOL["name"]},
            "messages": [{"role": "user", "content": prompt}],
        }
        response = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=CLAUDE_TIMEOUT_SECONDS)
        if response.status_code in (400, 404) and model == CLAUDE_MODEL:
            continue
        if response.status_code != 200:
            return None, model, f"HTTP {response.status_code} {redact(response.text[:160])}"
        data: dict[str, Any] = response.json()
        return data, model, ""
    return None, CLAUDE_FALLBACK_MODEL, "no model accepted the request"


def session_complete(day: str, now_ny: datetime) -> bool:
    return day < now_ny.date().isoformat() or (
        day == now_ny.date().isoformat() and now_ny.strftime("%H:%M") >= US_SESSION_DONE_NY
    )


async def update_news(
    cache: NewsCache,
    days: list[tuple[str, float, str]],
    usage: UsageState,
    now_ny: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Label each (day, soxx_pct, gauges) not yet in the cache. Returns notes. Mutates cache/usage."""
    now_ny = now_ny or datetime.now(NY)
    today = now_ny.date().isoformat()

    def due(day: str) -> bool:
        entry = cache.get(day)
        if entry is None:
            return True
        if "category" in entry:
            return False
        return int(entry.get("failed", 0)) < MAX_ATTEMPTS and entry.get("last_try") != today

    todo = [d for d in days if due(d[0]) and session_complete(d[0], now_ny)][:NEWS_MAX_DAYS_PER_RUN]
    if not todo:
        return []
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [f"Headlines: Claude not configured; {len(todo)} day(s) unlabelled"]
    month = usage.setdefault(month_key(now_ny), {"usd": 0.0, "calls": 0.0})
    notes: list[str] = []
    owns = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        for day, move, gauges in todo:
            if month["usd"] >= CLAUDE_MONTHLY_BUDGET_USD:
                notes.append(f"Headlines: monthly Claude budget reached; {day} left unlabelled")
                break
            try:
                headlines = await fetch_headlines(client, day)
            except Exception as exc:
                notes.append(f"Headlines for {day} unavailable: {redact(f'{type(exc).__name__}: {exc}')}")
                continue
            if not headlines:
                cache[day] = {
                    "category": "none_clear",
                    "summary": "No headlines in the session window.",
                    "evidence": [],
                    "confidence": "low",
                    "headline_count": 0,
                }
                continue
            payload, model, problem = await _classify(client, api_key, build_prompt(day, move, gauges, headlines))
            if payload is None:
                notes.append(f"Headlines for {day}: Claude failed ({problem})")
                continue
            cost = cost_of(payload, model, 0)
            month["usd"] = round(month["usd"] + cost, 6)
            month["calls"] = month["calls"] + 1
            label, problem = parse_label(payload, headlines)
            if label is None:
                failed = int((cache.get(day) or {}).get("failed", 0)) + 1
                cache[day] = {"failed": failed, "last_try": today, "problem": problem}
                notes.append(f"Headlines for {day}: answer rejected ({problem}); attempt {failed} of {MAX_ATTEMPTS}")
                continue
            label["model"] = model
            label["labelled_at"] = now_ny.isoformat(timespec="minutes")
            cache[day] = label
    finally:
        if owns:
            await client.aclose()
    keep_after = (now_ny - timedelta(days=400)).date().isoformat()
    for day in [d for d in cache if d < keep_after]:
        del cache[day]
    return notes
