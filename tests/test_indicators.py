from __future__ import annotations

import math

import pandas as pd
import pytest

from app.indicators import change_bps, latest_change_zscore, sma, wilder_rsi
from tests.conftest import series

# Wilder's worked example as published in StockCharts ChartSchool (RSI article).
WILDER_CLOSES = [
    44.34,
    44.09,
    44.15,
    43.61,
    44.33,
    44.83,
    45.10,
    45.42,
    45.84,
    46.08,
    45.89,
    46.03,
    45.61,
    46.28,
    46.28,
    46.00,
    46.03,
    46.41,
    46.22,
    45.64,
    46.21,
    46.25,
    45.71,
    46.45,
    45.78,
    45.35,
    44.03,
    44.18,
    44.22,
    44.57,
    43.42,
    42.66,
    43.13,
]


def _rma_reference_rsi(closes: list[float], period: int) -> list[float]:
    """Independent path: TradingView ta.rma = EWM(alpha=1/period) seeded with an SMA."""
    delta = pd.Series(closes).diff().iloc[1:].reset_index(drop=True)
    gains, losses = delta.clip(lower=0), (-delta).clip(lower=0)

    def rma(x: pd.Series) -> pd.Series:
        seeded = pd.concat([pd.Series([x.iloc[:period].mean()]), x.iloc[period:]], ignore_index=True)
        return seeded.ewm(alpha=1 / period, adjust=False).mean()

    rs = rma(gains) / rma(losses)
    return list(100 - 100 / (1 + rs))


def test_wilder_rsi_matches_independent_rma_implementation() -> None:
    ours = wilder_rsi(series(WILDER_CLOSES), 14).iloc[14:].tolist()
    reference = _rma_reference_rsi(WILDER_CLOSES, 14)
    assert ours == pytest.approx(reference, abs=1e-9)
    assert ours[0] == pytest.approx(70.464, abs=1e-3)  # unrounded first value


def test_wilder_differs_from_spec_simple_mean_rsi() -> None:
    """The spec's rolling-mean RSI is a different indicator, not a rounding difference."""
    s = series(WILDER_CLOSES)
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    spec_rsi = float((100 - 100 / (1 + gain / loss)).iloc[-1])
    assert abs(spec_rsi - float(wilder_rsi(s, 14).iloc[-1])) > 1.0


def test_wilder_rsi_warmup_is_nan() -> None:
    rsi = wilder_rsi(series(WILDER_CLOSES), 14)
    assert rsi.iloc[:14].isna().all()


def test_wilder_rsi_extremes() -> None:
    assert wilder_rsi(series(range(1, 40)), 14).iloc[-1] == 100.0
    assert wilder_rsi(series(range(40, 1, -1)), 14).iloc[-1] == 0.0
    assert wilder_rsi(series([5.0] * 30), 14).iloc[-1] == 50.0


def test_wilder_rsi_short_series_all_nan() -> None:
    assert wilder_rsi(series([1, 2, 3]), 14).isna().all()


def test_change_bps() -> None:
    s = series([4.00, 4.05, 4.10, 4.12, 4.20, 4.31])
    assert change_bps(s, 1) == pytest.approx(11.0)
    assert change_bps(s, 5) == pytest.approx(31.0)
    assert change_bps(s, 6) is None


def test_zscore_baseline_excludes_today() -> None:
    # 61 alternating ±1bp changes, then a +10bp day.
    levels = [4.0]
    for i in range(61):
        levels.append(levels[-1] + (0.01 if i % 2 == 0 else -0.01))
    levels.append(levels[-1] + 0.10)
    z = latest_change_zscore(series(levels), 60)
    assert z is not None and z > 5


def test_zscore_none_when_flat_or_short() -> None:
    assert latest_change_zscore(series([4.0] * 80), 60) is None
    assert latest_change_zscore(series([4.0, 4.1]), 60) is None


def test_sma_requires_full_window() -> None:
    out = sma(pd.Series([1.0, 2.0, 3.0]), 3)
    assert math.isnan(out.iloc[1]) and out.iloc[2] == 2.0
