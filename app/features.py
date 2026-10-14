"""Turn raw price series into the named features the classifier reads.

A missing input becomes None — never a guessed value. The classifier treats
None as "unknown", not as "condition not met".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from app.config import THRESHOLDS, Thresholds
from app.data import FetchResult
from app.indicators import change_bps, change_pct, last_valid, latest_change_zscore, sma, wilder_rsi

TNX_SCALE_GUARD = 25.0  # a 10Y "yield" above 25% means the feed is quoting x10
TNX_CROSSCHECK_TOLERANCE_PCT = 0.25


@dataclass
class MarketFeatures:
    tnx_pct: float | None = None
    tnx_source: str | None = None
    tnx_chg_1d_bps: float | None = None
    tnx_chg_5d_bps: float | None = None
    tnx_delta_z: float | None = None
    vix: float | None = None
    vix_chg_1d_pct: float | None = None
    soxx_chg_1d_pct: float | None = None
    igv_chg_1d_pct: float | None = None
    soxx_rsi: float | None = None
    samsung_rsi: float | None = None
    igv_soxx_ratio: float | None = None
    ratio_sma: float | None = None
    ratio_sma_rising: bool | None = None
    ratio_below_trough_mean: bool | None = None
    ks11: float | None = None
    ks11_sma200: float | None = None
    dxy: float | None = None
    dgs2_pct: float | None = None
    effr_pct: float | None = None
    effr_minus_2y_pct: float | None = None
    saas_arr_growth_pct: float | None = None  # unsourced by design
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("notes")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.items()}


def _series(results: dict[str, FetchResult], key: str) -> pd.Series | None:
    result = results.get(key)
    if result is None or not result.ok:
        return None
    return result.series


def build_features(results: dict[str, FetchResult], t: Thresholds = THRESHOLDS) -> MarketFeatures:
    f = MarketFeatures()

    # --- 10Y yield: Yahoo ^TNX primary, FRED DGS10 fallback ---------------------
    tnx = _series(results, "tnx")
    dgs10 = _series(results, "dgs10")
    if tnx is not None and float(tnx.median()) > TNX_SCALE_GUARD:
        tnx = tnx / 10.0
        f.notes.append("^TNX quoted at 10x yield; rescaled to percent.")
    if tnx is None and dgs10 is not None:
        tnx = dgs10
        f.tnx_source = "fred:DGS10"
        f.notes.append("^TNX unavailable; using FRED DGS10 (lags ~1 session).")
    elif tnx is not None:
        f.tnx_source = "yahoo:^TNX"

    if tnx is not None:
        f.tnx_pct = last_valid(tnx)
        f.tnx_chg_1d_bps = change_bps(tnx, 1)
        f.tnx_chg_5d_bps = change_bps(tnx, 5)
        f.tnx_delta_z = latest_change_zscore(tnx, t.tnx_delta_z_window)
        if f.tnx_source == "yahoo:^TNX" and dgs10 is not None and f.tnx_pct is not None:
            fred_last = last_valid(dgs10)
            if fred_last is not None and abs(fred_last - f.tnx_pct) > TNX_CROSSCHECK_TOLERANCE_PCT:
                f.notes.append(
                    f"^TNX ({f.tnx_pct:.3f}) and DGS10 ({fred_last:.3f}) disagree by more "
                    f"than {TNX_CROSSCHECK_TOLERANCE_PCT:.2f}pp — check the feed."
                )

    # --- Volatility -----------------------------------------------------------
    vix = _series(results, "vix")
    if vix is not None:
        f.vix = last_valid(vix)
        f.vix_chg_1d_pct = change_pct(vix, 1)

    # --- Semis ----------------------------------------------------------------
    soxx = _series(results, "soxx")
    if soxx is not None:
        f.soxx_rsi = last_valid(wilder_rsi(soxx, t.rsi_period))
        f.soxx_chg_1d_pct = change_pct(soxx, 1)
    samsung = _series(results, "samsung")
    if samsung is not None:
        f.samsung_rsi = last_valid(wilder_rsi(samsung, t.rsi_period))

    # --- Software vs hardware: IGV / SOXX ratio ---------------------------------
    igv = _series(results, "igv")
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
        if len(ratio) >= t.ratio_trough_window and f.igv_soxx_ratio is not None:
            trough_mean = float(ratio.iloc[-t.ratio_trough_window :].mean())
            f.ratio_below_trough_mean = bool(f.igv_soxx_ratio < trough_mean)

    # --- Korea ----------------------------------------------------------------
    ks11 = _series(results, "ks11")
    if ks11 is not None:
        f.ks11 = last_valid(ks11)
        f.ks11_sma200 = last_valid(sma(ks11, t.ks11_sma_window))

    # --- Dollar ----------------------------------------------------------------
    dxy = _series(results, "dxy")
    if dxy is not None:
        f.dxy = last_valid(dxy)

    # --- Policy path proxy: 2Y vs effective fed funds --------------------------
    dgs2 = _series(results, "dgs2")
    effr = _series(results, "effr")
    if dgs2 is not None:
        f.dgs2_pct = last_valid(dgs2)
    if effr is not None:
        f.effr_pct = last_valid(effr)
    if f.dgs2_pct is not None and f.effr_pct is not None:
        f.effr_minus_2y_pct = f.effr_pct - f.dgs2_pct

    return f
