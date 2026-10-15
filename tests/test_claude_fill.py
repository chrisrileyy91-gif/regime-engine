from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
import pytest
import respx

import app.claude_fill as cf
from app.claude_fill import FillReport, apply_fills, build_prompt, cost_of, fill_gaps, find_gaps, parse_claude_response
from app.config import ANTHROPIC_URL
from app.features import build_features
from app.verify import verify_all
from tests.conftest import TODAY, failed, market, ok

NY = cf.NY
TODAY_STR = str(TODAY.date())


def korea_missing() -> dict:
    results = market()
    for key in ("kospi@eodhd", "kospi@krx"):
        results[key] = failed(key, "not configured")
    return results


def after_close() -> datetime:
    return datetime(TODAY.year, TODAY.month, TODAY.day, 16, 25, tzinfo=NY)


def claude_payload(
    answers: str, urls: tuple[str, ...] = ("https://www.reuters.com/markets/kospi",), searches: int = 2
) -> dict:
    return {
        "content": [
            {"type": "server_tool_use", "name": "web_search", "input": {"query": "KOSPI close"}},
            {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": u} for u in urls]},
            {"type": "text", "text": f"Here you go.\n```json\n{answers}\n```"},
        ],
        "usage": {"input_tokens": 20000, "output_tokens": 400, "server_tool_use": {"web_search_requests": searches}},
    }


def kospi_answer(value: float, day: str = TODAY_STR, url: str = "https://www.reuters.com/markets/kospi") -> str:
    return f'[{{"id": "kospi", "date": "{day}", "value": {value}, "source_url": "{url}", "source_name": "Reuters"}}]'


def yahoo_kospi(results: dict) -> float:
    return float(results["kospi@yahoo"].series.iloc[-1])


# --- finding gaps ------------------------------------------------------------------------


def test_no_gaps_when_everything_verified(calm_market: dict) -> None:
    assert find_gaps(verify_all(calm_market, TODAY.date()), calm_market, after_close()) == []


def test_missing_korea_is_a_gap_dated_by_yahoo() -> None:
    results = korea_missing()
    gaps = find_gaps(verify_all(results, TODAY.date()), results, after_close())
    assert [(g.key, g.date) for g in gaps] == [("kospi", TODAY_STR)]
    assert gaps[0].base_note.endswith("(unconfirmed)")  # history had to come from Yahoo


def test_lagging_record_is_a_gap_with_record_history() -> None:
    results = market()
    results["kospi@eodhd"] = failed("kospi@eodhd")
    results["kospi@krx"] = ok("kospi@krx", results["kospi@krx"].series.iloc[:-1])  # KRX a session behind
    gaps = find_gaps(verify_all(results, TODAY.date()), results, after_close())
    assert [(g.key, g.date) for g in gaps] == [("kospi", TODAY_STR)]
    assert gaps[0].base_note == "history from the record source"


def test_conflicts_are_never_sent() -> None:
    results = market()
    results["soxx@yahoo"] = ok("soxx@yahoo", results["soxx@yahoo"].series * 1.05)
    verified = verify_all(results, TODAY.date())
    assert verified["soxx"].status.status == "CONFLICT"
    assert find_gaps(verified, results, after_close()) == []


def test_no_gap_without_a_confirming_value() -> None:
    results = korea_missing()
    results["kospi@yahoo"] = failed("kospi@yahoo")
    assert find_gaps(verify_all(results, TODAY.date()), results, after_close()) == []


def test_unfinished_session_falls_back_to_last_completed_one() -> None:
    results = market()
    results["soxx@alpaca"] = failed("soxx@alpaca")
    morning = datetime(TODAY.year, TODAY.month, TODAY.day, 10, 0, tzinfo=NY)
    previous = str(results["soxx@yahoo"].series.index[-2].date())  # today's US session isn't over
    assert [(g.key, g.date) for g in find_gaps(verify_all(results, TODAY.date()), results, morning)] == [
        ("soxx", previous)
    ]
    assert [(g.key, g.date) for g in find_gaps(verify_all(results, TODAY.date()), results, after_close())] == [
        ("soxx", TODAY_STR)
    ]


def test_in_progress_korean_session_falls_back_to_the_last_close() -> None:
    """Live run at 21:12 ET: Yahoo's newest KOSPI bar was the next day's session, still trading."""
    results = korea_missing()
    yahoo = results["kospi@yahoo"].series
    tomorrow = TODAY + pd.Timedelta(days=1)
    results["kospi@yahoo"] = ok("kospi@yahoo", pd.concat([yahoo, pd.Series([yahoo.iloc[-1]], index=[tomorrow])]))
    night = datetime(TODAY.year, TODAY.month, TODAY.day, 21, 12, tzinfo=NY)
    gaps = find_gaps(verify_all(results, TODAY.date()), results, night)
    assert [(g.key, g.date) for g in gaps] == [("kospi", TODAY_STR)]


@respx.mock
async def test_fill_that_fails_a_veto_check_is_dropped(api_key: None) -> None:
    results = market()
    results["dxy@eodhd"] = failed("dxy@eodhd", "not configured")
    yahoo = float(results["dxy@yahoo"].series.iloc[-1])
    results["dxy@ecb"] = ok("dxy@ecb", results["dxy@ecb"].series + 5.0)  # ECB-fix DXY 5 points away
    answer = f'[{{"id": "dxy", "date": "{TODAY_STR}", "value": {yahoo}, "source_url": "https://www.reuters.com/markets/kospi"}}]'
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=claude_payload(answer)))
    report = await run_fill(results)
    assert report.fills == {} and any("fails the ECB reference rates check" in n for n in report.notes)


def test_prompt_never_reveals_the_check_value() -> None:
    results = korea_missing()
    gaps = find_gaps(verify_all(results, TODAY.date()), results, after_close())
    prompt = build_prompt(gaps)
    assert f"{gaps[0].expected:.2f}" not in prompt and TODAY_STR in prompt


# --- parsing and cost ----------------------------------------------------------------------


def test_parse_response_collects_json_urls_and_searches() -> None:
    answers, urls, searches = parse_claude_response(claude_payload(kospi_answer(6384.24)))
    assert answers[0]["value"] == 6384.24 and "https://www.reuters.com/markets/kospi" in urls and searches == 2


def test_parse_response_without_json_is_empty() -> None:
    payload = {"content": [{"type": "text", "text": "I couldn't find it."}]}
    assert parse_claude_response(payload)[0] == []


def test_cost_of_haiku_run() -> None:
    # 20k in at $0.10/M + 400 out at $0.50/M + 2 searches at $0.01 = $0.0222
    assert cost_of(claude_payload("[]"), "claude-haiku-5-5", 2) == pytest.approx(0.0222)


# --- the full loop ------------------------------------------------------------------------


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setattr(cf.asyncio, "sleep", _no_sleep)


async def _no_sleep(_: float) -> None:
    return None


async def run_fill(results: dict, usage: dict | None = None) -> FillReport:
    usage = usage if usage is not None else {}
    return await fill_gaps(verify_all(results, TODAY.date()), results, usage, now_ny=after_close())


@respx.mock
async def test_agreeing_fill_is_used_and_spliced(api_key: None) -> None:
    results = korea_missing()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_payload(kospi_answer(yahoo_kospi(results))))
    )
    usage: dict = {}
    report = await run_fill(results, usage)
    assert list(report.fills) == ["kospi"] and report.cost_usd > 0
    assert usage[after_close().strftime("%Y-%m")]["calls"] == 1

    verified = apply_fills(verify_all(results, TODAY.date()), report)
    status = verified["kospi"].status
    assert status.status == "VERIFIED" and status.confirmed_by == ["Yahoo"] and "reuters.com" in (status.source or "")
    features = build_features(verified)
    assert features.ks11 == pytest.approx(yahoo_kospi(results)) and features.ks11_sma200 is not None


@respx.mock
async def test_disagreeing_fill_is_dropped(api_key: None) -> None:
    results = korea_missing()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_payload(kospi_answer(yahoo_kospi(results) * 1.02)))
    )
    report = await run_fill(results)
    assert report.fills == {} and any("disagree" in n for n in report.notes)


@respx.mock
async def test_invented_citation_is_dropped(api_key: None) -> None:
    results = korea_missing()
    answer = kospi_answer(yahoo_kospi(results), url="https://made-up-site.example/kospi")
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=claude_payload(answer)))
    report = await run_fill(results)
    assert report.fills == {} and any("wasn't among its search results" in n for n in report.notes)


@respx.mock
async def test_wrong_date_is_dropped(api_key: None) -> None:
    results = korea_missing()
    answer = kospi_answer(yahoo_kospi(results), day="2020-01-02")
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=claude_payload(answer)))
    report = await run_fill(results)
    assert report.fills == {} and any("not 20" in n for n in report.notes)


@respx.mock
async def test_falls_back_to_second_model_on_400(api_key: None) -> None:
    results = korea_missing()
    route = respx.post(ANTHROPIC_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "tool not supported"}}),
            httpx.Response(200, json=claude_payload(kospi_answer(yahoo_kospi(results)))),
        ]
    )
    report = await run_fill(results)
    assert route.call_count == 2 and report.model == "claude-sonnet-5-5" and "kospi" in report.fills


@respx.mock
async def test_budget_reached_skips_the_call(api_key: None) -> None:
    results = korea_missing()
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=claude_payload("[]")))
    usage = {after_close().strftime("%Y-%m"): {"usd": 99.0, "calls": 400.0}}
    report = await run_fill(results, usage)
    assert route.call_count == 0 and any("budget" in n for n in report.notes)


async def test_no_key_leaves_gaps_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = await run_fill(korea_missing())
    assert report.fills == {} and any("not configured" in n for n in report.notes)


@respx.mock
async def test_api_error_never_leaks_the_key(api_key: None) -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(401, text="bad key sk-ant-test-key"))
    report = await run_fill(korea_missing())
    assert all("sk-ant-test-key" not in n for n in report.notes)


@respx.mock
async def test_no_call_when_nothing_is_missing(api_key: None, calm_market: dict) -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json=claude_payload("[]")))
    report = await run_fill(calm_market)
    assert route.call_count == 0 and report.gaps == {} and report.cost_usd == 0


@respx.mock
async def test_fill_flows_through_to_the_published_payload(api_key: None) -> None:
    from app.service import build_status

    results = korea_missing()
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_payload(kospi_answer(yahoo_kospi(results))))
    )
    report = await run_fill(results)
    status = build_status(results, today=TODAY.date(), fills=report)
    assert status.data_quality is not None and status.data_quality.status == "OK"
    kospi = next(i for i in status.data_quality.inputs if i.key == "kospi")
    assert kospi.status == "VERIFIED" and kospi.source == "Claude web search (reuters.com)"
    assert any("filled by Claude" in i.message for i in status.data_quality.issues)


# --- the fill cache ------------------------------------------------------------------------------


@respx.mock
async def test_cached_fill_is_reused_without_calling_claude(api_key: None) -> None:
    """16:25 run, Oct 14: the 09:50 run's KOSPI fill was lost and the second search came up empty."""
    results = korea_missing()
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_payload(kospi_answer(yahoo_kospi(results))))
    )
    cache: dict = {}
    usage: dict = {}
    first = await fill_gaps(verify_all(results, TODAY.date()), results, usage, now_ny=after_close(), cache=cache)
    assert "kospi" in first.fills and cache["kospi"][TODAY_STR]["value"] == pytest.approx(yahoo_kospi(results))
    second = await fill_gaps(verify_all(results, TODAY.date()), results, usage, now_ny=after_close(), cache=cache)
    assert route.call_count == 1 and "kospi" in second.fills
    assert any("reused" in n for n in second.notes)


async def test_cached_fill_that_no_longer_agrees_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    results = korea_missing()
    cache = {"kospi": {TODAY_STR: {"value": yahoo_kospi(results) * 1.05, "source_url": "https://x.example/k"}}}
    report = await fill_gaps(verify_all(results, TODAY.date()), results, {}, now_ny=after_close(), cache=cache)
    assert report.fills == {}


def test_cache_keeps_thirty_days() -> None:
    old = (after_close() - pd.Timedelta(days=40)).date().isoformat()
    cache = {"kospi": {old: {"value": 1.0}}}
    cf._remember(cache, {}, after_close())
    assert cache == {}
