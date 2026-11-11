"""The replay: both stress labels rebuilt over history and scored against SOXX's own move."""

from __future__ import annotations

import numpy as np

from app.replay import render, replay, summarize
from tests.conftest import series

N = 200


def _inputs() -> tuple:
    rng = np.random.default_rng(9)
    soxx = list(100.0 * np.cumprod(1 + rng.normal(0, 0.004, N)))
    # A slow slide (RSI sinks below 40 with no single big day), then one -6% day with VIX +40%.
    for i in range(N - 30, N - 10):
        soxx[i:] = [v * 0.994 for v in soxx[i:]]
    soxx[N - 5 :] = [v * 0.94 for v in soxx[N - 5 :]]
    vix = [15.0] * N
    vix[N - 5] = 21.0
    return series(soxx), series(vix), series([4.0] * N)


def test_old_label_tracks_the_slide_and_new_label_tracks_the_day() -> None:
    soxx, vix, y10 = _inputs()
    rows = replay(soxx, vix, y10, 60)
    crash = min(rows, key=lambda r: r.soxx_pct)
    assert crash.soxx_pct < -5 and crash.new == "STRESSED"  # sell-off + VIX spike, the same session
    slide = [r for r in rows if r.date < crash.date][-12:-2]
    assert all(r.old == "ELEVATED" for r in slide)  # RSI below 40 for days on end
    assert sum(r.new != "CALM" for r in slide) <= 2  # ordinary -0.6% days are not stress


def test_summary_and_markdown() -> None:
    rows = replay(*_inputs(), 60)
    new = {s["label"]: s for s in summarize(rows, "new")}
    assert sum(int(s["days"]) for s in new.values()) == len(rows)
    assert float(new["STRESSED"]["mean"]) < float(new["CALM"]["mean"])
    text = render(rows)
    assert "Old label" in text and "New label" in text and "The 15 worst sessions" in text
    assert render([]).startswith("Replay: no sessions")


def test_a_yield_jump_counts_only_when_chips_fell() -> None:
    soxx, vix, y10 = _inputs()
    y10 = y10.copy()
    y10.iloc[-40:] += 0.15  # +15 bps on an ordinary chip day
    row = replay(soxx, vix, y10, 60)[-40 + 60]
    assert abs(row.soxx_pct) < 1.5 and row.new == "CALM" and row.old != "CALM"


def test_funding_card_is_scored_beside_the_label() -> None:
    soxx, vix, y10 = _inputs()
    name = soxx * 1.0
    name.iloc[-20:] *= 0.92  # the debt-funded name drops 8% on a day chips don't
    hyg = series([80.0] * N)
    rows = replay(soxx, vix, y10, 60, basket=[name], credit=hyg)
    hit = [r for r in rows if r.funding]
    assert hit and all(r.funding is not None for r in rows) and hit[0].next_pct is not None
    text = render(rows)
    assert "Funding card (shown, not counted)" in text and "the stress label read Calm" in text
    assert "No sessions with the basket" in render(replay(soxx, vix, y10, 60))
