"""Headlines: fetched from Alpaca (Benzinga), classified by Claude, validated against the list."""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import respx

import app.news as news
from app.config import ALPACA_NEWS_URL, ANTHROPIC_URL
from app.news import build_prompt, parse_label, parse_news, session_window, update_news

NY = news.NY
DAY = "2026-07-28"


def article(nid: int, headline: str, minute: int = 0) -> dict:
    return {
        "id": nid,
        "headline": headline,
        "summary": "",
        "created_at": f"2026-07-28T14:{minute:02d}:00Z",
        "url": f"https://www.benzinga.com/news/{nid}",
        "source": "benzinga",
        "symbols": ["NVDA"],
    }


HEADLINES = parse_news(
    [
        {
            "news": [
                article(1, "SK Hynix plunges as China memory competition fears grow", 5),
                article(2, "Nvidia slides on renewed AI financing worries", 10),
                article(2, "duplicate id is dropped", 11),
            ],
            "next_page_token": None,
        }
    ]
)


def claude(answer: str) -> dict:
    return {"content": [{"type": "text", "text": answer}], "usage": {"input_tokens": 3000, "output_tokens": 80}}


def test_session_window_spans_prior_close_to_close() -> None:
    start, end = session_window("2026-07-27")  # a Monday
    assert (start.strftime("%a %H:%M"), end.strftime("%a %H:%M")) == ("Fri 16:00", "Mon 16:00")


def test_parse_dedupes_and_sorts() -> None:
    assert [h["id"] for h in HEADLINES] == ["1", "2"]


def test_prompt_lists_every_headline_and_forbids_outside_knowledge() -> None:
    prompt = build_prompt(DAY, -4.8, "10Y -4 bps", HEADLINES)
    assert "[1]" in prompt and "[2]" in prompt and "Use ONLY these headlines" in prompt


def test_valid_label_keeps_cited_headlines_with_links() -> None:
    answer = (
        '{"category": "ai_demand", "summary": "China competition and AI financing worries.", '
        '"evidence": ["1", "2"], "confidence": "high"}'
    )
    label, problem = parse_label(claude(answer), HEADLINES)
    assert problem == "" and label is not None
    assert label["category"] == "ai_demand" and label["evidence"][0]["url"].endswith("/1")


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ('{"category": "ai_demand", "summary": "x", "evidence": ["99"]}', "wasn't in the list"),
        ('{"category": "aliens", "summary": "x", "evidence": ["1"]}', "unknown category"),
        ('{"category": "company", "summary": "x", "evidence": []}', "without citing"),
        ("I think it was chips.", "no JSON"),
    ],
)
def test_invalid_labels_are_rejected(answer: str, problem: str) -> None:
    label, why = parse_label(claude(answer), HEADLINES)
    assert label is None and problem in why


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {"APCA_API_KEY_ID": "id", "APCA_API_SECRET_KEY": "secret", "ANTHROPIC_API_KEY": "sk-ant-x"}.items():
        monkeypatch.setenv(k, v)


@respx.mock
async def test_update_labels_new_days_once_and_tracks_spend(keys: None) -> None:
    respx.get(ALPACA_NEWS_URL).mock(
        return_value=httpx.Response(200, json={"news": [article(1, "SK Hynix plunges", 5)], "next_page_token": None})
    )
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json=claude('{"category": "ai_demand", "summary": "Memory competition.", "evidence": ["1"]}')
        )
    )
    cache: dict = {}
    usage: dict = {}
    now = datetime(2026, 7, 29, 6, 0, tzinfo=NY)
    assert await update_news(cache, [(DAY, -4.8, "quiet")], usage, now) == []
    assert cache[DAY]["category"] == "ai_demand" and usage["2026-07"]["calls"] == 1
    await update_news(cache, [(DAY, -4.8, "quiet")], usage, now)
    assert route.call_count == 1  # a finished day is read once


async def test_unfinished_session_is_not_labelled(keys: None) -> None:
    cache: dict = {}
    now = datetime(2026, 7, 28, 10, 0, tzinfo=NY)
    assert await update_news(cache, [(DAY, -4.8, "quiet")], {}, now) == [] and cache == {}


@respx.mock
async def test_rejected_answer_is_not_cached(keys: None) -> None:
    respx.get(ALPACA_NEWS_URL).mock(return_value=httpx.Response(200, json={"news": [article(1, "x", 5)]}))
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude('{"category": "company", "summary": "x", "evidence": ["7"]}'))
    )
    cache: dict = {}
    notes = await update_news(cache, [(DAY, -4.8, "quiet")], {}, datetime(2026, 7, 29, 6, tzinfo=NY))
    assert cache == {} and any("rejected" in n for n in notes)


@respx.mock
async def test_news_error_never_leaks_keys(keys: None) -> None:
    respx.get(ALPACA_NEWS_URL).mock(return_value=httpx.Response(403, text="forbidden for key id"))
    notes = await update_news({}, [(DAY, -4.8, "quiet")], {}, datetime(2026, 7, 29, 6, tzinfo=NY))
    assert notes and all("secret" not in n for n in notes)


def test_only_web_links_reach_the_page() -> None:
    bad = parse_news([{"news": [{**article(9, "x"), "url": "javascript:alert(1)"}]}])
    assert bad[0]["url"] == ""
