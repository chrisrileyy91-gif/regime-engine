"""Pure indicator math. No I/O. Every formula is written out explicitly."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing (matches TradingView's ta.rsi).

    Seed: avg_gain/avg_loss = simple mean of the first `period` changes.
    Then: avg = (prev_avg * (period - 1) + current) / period.
    RSI = 100 - 100 / (1 + avg_gain / avg_loss).

    The original spec used a plain rolling mean on one month of data, which
    leaves ~7 valid points and diverges from the RSI on any charting platform.
    """
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=close.index)

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    out[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(out, index=close.index)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def change_bps(series: pd.Series, sessions: int) -> float | None:
    """Change over `sessions` observations, in basis points (series in percent)."""
    if len(series) <= sessions:
        return None
    return (float(series.iloc[-1]) - float(series.iloc[-1 - sessions])) * 100.0


def change_pct(series: pd.Series, sessions: int) -> float | None:
    """Percent change over `sessions` observations."""
    if len(series) <= sessions:
        return None
    prior = float(series.iloc[-1 - sessions])
    if prior == 0.0:
        return None
    return (float(series.iloc[-1]) / prior - 1.0) * 100.0


def percentile_rank(series: pd.Series, window: int) -> float | None:
    """Percent of the trailing `window` observations (today included) at or below today's value."""
    clean = series.dropna()
    if len(clean) < window:
        return None
    tail = clean.iloc[-window:].to_numpy(dtype=float)
    return float((tail <= tail[-1]).mean() * 100.0)


def latest_change_zscore(series: pd.Series, window: int) -> float | None:
    """z = (today's 1-session change - mean(prior changes)) / std(prior changes).

    The baseline excludes today so a spike cannot dilute its own z-score.
    """
    changes = series.diff().dropna()
    if len(changes) < window + 1:
        return None
    today = float(changes.iloc[-1])
    baseline = changes.iloc[-1 - window : -1]
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=1))
    if std == 0.0 or math.isnan(std):
        return None
    return (today - mean) / std


def last_valid(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])
