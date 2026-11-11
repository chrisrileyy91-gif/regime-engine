"""Turn verified official series into the named features the classifier reads.

Only series that passed verify.py arrive here. A withheld or missing input becomes None,
never a guessed value; the classifier treats None as unknown, not as "condition not met".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from app.config import THRESHOLDS, Thresholds
from app.indicators import (
    change_bps,
    change_pct,
    last_valid,
    latest_change_zscore,
    latest_return_zscore,
    percentile_rank,
    sma,
    wilder_rsi,
)
from app.verify import Verified


@dataclass
class MarketFeatures:
    tnx_pct: float | None = None
    tnx_pctile: float | None = None  # % of the trailing year at or below today's 10Y
    tnx_chg_1d_bps: float | None = None
    tnx_chg_5d_bps: float | None = None
    tnx_delta_z: float | None = None
    vix: float | None = None
    vix_chg_1d_pct: float | None = None
    soxx_chg_1d_pct: float | None = None
    soxx_return_z: float | None = None  # today's return vs the prior 60 sessions' returns
    igv_chg_1d_pct: float | None = None
    soxx_rsi: float | None = None
    samsung_rsi: float | None = None
    igv_soxx_ratio: float | None = None
    ratio_sma: float | None = None
    ratio_sma_rising: bool | None = None
    ks11: float | None = None
    ks11_sma200: float | None = None
    dollar_chg_1d_pct: float | None = None  # UUP, consolidated tape
    y2_pct: float | None = None  # Treasury 2Y (official, posts in the evening)
    effr_minus_2y_pct: float | None = None
    bill_3m_pct: float | None = None
    effr_pct: float | None = None
    effr_minus_3m_pct: float | None = None
    dates: dict[str, str] = field(default_factory=dict)  # input -> date of the official value used

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("dates")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}


def build_features(
    verified: dict[str, Verified], extras: dict[str, pd.Series] | None = None, t: Thresholds = THRESHOLDS
) -> MarketFeatures:
    """`extras` are the official series that aren't cross-checked inputs: the Treasury 2Y ("y2")
    and UUP ("dollar"). They feed the rate-cut catalyst and the backdrop, never the stress label."""
    f = MarketFeatures()
    extras = extras or {}

    def get(key: str) -> pd.Series | None:
        v = verified.get(key)
        if v is None or v.series is None:
            return None
        f.dates[key] = str(v.series.index[-1].date())
        return v.series

    tnx = get("y10")
    if tnx is not None:
        f.tnx_pct = last_valid(tnx)
        f.tnx_pctile = percentile_rank(tnx, t.tnx_pctile_window)
        f.tnx_chg_1d_bps = change_bps(tnx, 1)
        f.tnx_chg_5d_bps = change_bps(tnx, 5)
        f.tnx_delta_z = latest_change_zscore(tnx, t.tnx_delta_z_window)

    vix = get("vix")
    if vix is not None:
        f.vix = last_valid(vix)
        f.vix_chg_1d_pct = change_pct(vix, 1)

    soxx = get("soxx")
    if soxx is not None:
        f.soxx_rsi = last_valid(wilder_rsi(soxx, t.rsi_period))
        f.soxx_chg_1d_pct = change_pct(soxx, 1)
        f.soxx_return_z = latest_return_zscore(soxx, t.soxx_z_window)

    igv = get("igv")
    if igv is not None:
        f.igv_chg_1d_pct = change_pct(igv, 1)
    if igv is not None and soxx is not None:
        aligned = pd.concat([igv, soxx], axis=1, join="inner").dropna()
        ratio = aligned.iloc[:, 0] / aligned.iloc[:, 1]
        ratio_sma = sma(ratio, t.ratio_sma_window).dropna()
        f.igv_soxx_ratio = last_valid(ratio)
        if len(ratio_sma) > t.ratio_slope_lookback:
            f.ratio_sma = float(ratio_sma.iloc[-1])
            f.ratio_sma_rising = bool(ratio_sma.iloc[-1] > ratio_sma.iloc[-1 - t.ratio_slope_lookback])

    samsung = get("samsung")
    if samsung is not None:
        f.samsung_rsi = last_valid(wilder_rsi(samsung, t.rsi_period))

    kospi = get("kospi")
    if kospi is not None:
        f.ks11 = last_valid(kospi)
        f.ks11_sma200 = last_valid(sma(kospi, t.ks11_sma_window))

    dollar = extras.get("dollar")
    if dollar is not None and not dollar.empty:
        f.dates["dollar"] = str(dollar.index[-1].date())
        f.dollar_chg_1d_pct = change_pct(dollar, 1)

    bill, effr = get("t3m"), get("effr")
    if bill is not None:
        f.bill_3m_pct = last_valid(bill)
    if effr is not None:
        f.effr_pct = last_valid(effr)
    if f.bill_3m_pct is not None and f.effr_pct is not None:
        f.effr_minus_3m_pct = f.effr_pct - f.bill_3m_pct
    y2 = extras.get("y2")
    if y2 is not None and not y2.empty:
        f.dates["y2"] = str(y2.index[-1].date())
        f.y2_pct = last_valid(y2)
    if f.y2_pct is not None and f.effr_pct is not None:
        f.effr_minus_2y_pct = f.effr_pct - f.y2_pct
    return f
