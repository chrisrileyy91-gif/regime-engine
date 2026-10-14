from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from app.data import FetchResult

TODAY = pd.Timestamp.today().normalize()


def series(values: Sequence[float], end: str | pd.Timestamp | None = None) -> pd.Series:
    index = pd.bdate_range(end=end if end is not None else TODAY, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=index)


def flat_then(level: float, n: int = 300, tail: Sequence[float] = ()) -> pd.Series:
    """Gently noisy series around `level`, optionally ending with `tail` values."""
    rng = np.random.default_rng(7)
    body = level + rng.normal(0, level * 0.002, n - len(tail))
    return series(list(body) + list(tail))


def trend(start: float, end: float, n: int = 300, noise: float = 0.004) -> pd.Series:
    rng = np.random.default_rng(11)
    base = np.linspace(start, end, n)
    return series(base * (1 + rng.normal(0, noise, n)))


def ok(key: str, s: pd.Series, source: str | None = None) -> FetchResult:
    src = source or key.split("@")[1]
    return FetchResult(key=key, symbol=key.split("@")[0].upper(), source=src, series=s)


def failed(key: str, error: str = "boom") -> FetchResult:
    return FetchResult(key=key, symbol=key.split("@")[0].upper(), source=key.split("@")[1], series=None, error=error)


def market(**series_by_input: pd.Series) -> dict[str, FetchResult]:
    """Every official source and check for each input, all agreeing (override per input)."""
    base = {
        "y10": flat_then(4.0),
        "t3m": flat_then(4.30),
        "vix": flat_then(15.0),
        "effr": flat_then(4.33),
        "soxx": trend(200, 260),
        "igv": trend(90, 95),
        "dxy": flat_then(100.0),
        "kospi": trend(2500, 2700),
        "samsung": trend(60000, 70000),
    }
    base.update(series_by_input)
    sources = {
        "y10": ("cboe", "yahoo", "treasury"),
        "t3m": ("treasury", "cboe"),
        "vix": ("cboe", "yahoo"),
        "effr": ("nyfed",),
        "soxx": ("alpaca", "yahoo"),
        "igv": ("alpaca", "yahoo"),
        "dxy": ("eodhd", "yahoo", "ecb"),
        "kospi": ("eodhd", "krx", "yahoo"),
        "samsung": ("eodhd", "krx", "yahoo"),
    }
    out = {}
    for name, srcs in sources.items():
        for src in srcs:
            key = f"{name}@{src}"
            out[key] = ok(key, base[name].copy())
    return out


@pytest.fixture
def calm_market() -> dict[str, FetchResult]:
    """10Y ~4.0% flat, VIX ~15, SOXX in steady uptrend, no cuts priced, DXY ~100."""
    return market()
