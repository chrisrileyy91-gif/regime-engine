"""Single source of truth for every threshold the engine uses.

The original spec carried three inconsistent RSI threshold sets (CLAUDE.md, the
engine code, and the decision matrix). Every number lives here now; change a
threshold here and nowhere else. Values marked UNCALIBRATED came from the spec
and have not been backtested.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # --- Phase 2: Liquidity Squeeze (any one fires) — UNCALIBRATED -------------
    tnx_level_pct: float = 4.25  # 10Y yield level, percent
    tnx_delta_z: float = 1.5  # z-score of today's 1d yield change vs trailing window
    tnx_delta_z_window: int = 60  # trading days in the z-score baseline
    vix_level: float = 20.0

    # --- Phase 1: Hardware Buildout (both) — UNCALIBRATED ----------------------
    semi_rsi_strong: float = 55.0
    tnx_5d_slope_max_bps: float = 5.0  # "< +5 bps/week" = 5-session change

    # --- Phase 3: Monetization ------------------------------------------------
    ratio_sma_window: int = 20  # IGV/SOXX ratio SMA
    ratio_slope_lookback: int = 5  # SMA rising = SMA(t) > SMA(t-5)
    ratio_trough_window: int = 120  # "troughing" = ratio still below its 120d mean
    saas_arr_growth_pct: float = 15.0  # no data source wired — see CLAUDE.md

    # --- Phase 4: Rate Easing (both) — UNCALIBRATED -----------------------------
    cuts_priced_spread_pct: float = 0.10  # 2Y < EFFR by more than this = cuts priced
    dxy_easing_max: float = 102.0

    # --- Telemetry table (signals, not phase drivers) --------------------------
    tnx_1d_spike_bps: float = 10.0
    semi_rsi_digestion: float = 40.0
    ks11_sma_window: int = 200
    dxy_compression: float = 104.5

    # --- Indicator parameters --------------------------------------------------
    rsi_period: int = 14


THRESHOLDS = Thresholds()

# Data plumbing
HISTORY_RANGE = "2y"  # enough for SMA200 and a fully warmed-up Wilder RSI
CACHE_TTL_SECONDS = 300
HTTP_TIMEOUT_SECONDS = 15.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) regime-engine/0.1"

YAHOO_SYMBOLS: dict[str, str] = {
    "tnx": "^TNX",
    "vix": "^VIX",
    "soxx": "SOXX",
    "igv": "IGV",
    "ks11": "^KS11",
    "dxy": "DX-Y.NYB",
    "samsung": "005930.KS",
}

FRED_SERIES: dict[str, str] = {
    "dgs10": "DGS10",  # 10Y fallback / cross-check (lags ~1 day)
    "dgs2": "DGS2",  # 2Y yield — market's path-of-policy proxy
    "effr": "DFF",  # effective fed funds rate
}
