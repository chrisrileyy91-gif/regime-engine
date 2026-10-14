from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from app.data import FetchResult


def series(values: Sequence[float], end: str = "2026-10-12") -> pd.Series:
    index = pd.bdate_range(end=end, periods=len(values))
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


def ok(key: str, s: pd.Series, source: str = "yahoo") -> FetchResult:
    return FetchResult(key=key, symbol=key.upper(), source=source, series=s)


def failed(key: str, source: str = "yahoo") -> FetchResult:
    return FetchResult(key=key, symbol=key.upper(), source=source, series=None, error="boom")


@pytest.fixture
def calm_market() -> dict[str, FetchResult]:
    """10Y ~4.0% and flat, VIX ~15, SOXX in steady uptrend, no cuts priced, DXY ~100."""
    return {
        "tnx": ok("tnx", flat_then(4.0)),
        "vix": ok("vix", flat_then(15.0)),
        "soxx": ok("soxx", trend(200, 260)),
        "igv": ok("igv", trend(90, 95)),
        "ks11": ok("ks11", trend(2500, 2700)),
        "dxy": ok("dxy", flat_then(100.0)),
        "samsung": ok("samsung", trend(60000, 70000)),
        "dgs10": ok("dgs10", flat_then(4.0), "fred"),
        "dgs2": ok("dgs2", flat_then(4.30), "fred"),
        "effr": ok("effr", flat_then(4.33), "fred"),
    }
