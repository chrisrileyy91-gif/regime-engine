"""Single source of truth for every threshold, source and tolerance the engine uses.

Values marked UNCALIBRATED came from the original spec (or from us) and have not been
backtested. Change a number here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # --- Phase 2: Liquidity Squeeze (any one fires) — UNCALIBRATED -------------
    # Yield leg is RELATIVE: the spec's fixed "> 4.25%" fired every day once the
    # 10Y moved structurally above it (first live read: 5.22%, falling, VIX 15).
    tnx_pctile_high: float = 80.0  # 10Y higher than this % of the trailing window
    tnx_pctile_window: int = 250  # ~1 year of Treasury business days
    tnx_delta_z: float = 1.5  # z-score of the latest 1d yield change vs trailing window
    tnx_delta_z_window: int = 60
    vix_level: float = 20.0

    # --- Phase 1: Hardware Buildout (both) — UNCALIBRATED ----------------------
    semi_rsi_strong: float = 55.0
    tnx_5d_slope_max_bps: float = 5.0

    # --- Phase 3: Monetization ------------------------------------------------
    ratio_sma_window: int = 20
    ratio_slope_lookback: int = 5
    ratio_trough_window: int = 120
    saas_arr_growth_pct: float = 15.0  # no data source exists — see CLAUDE.md

    # --- Phase 4: Rate Easing (both) — UNCALIBRATED -----------------------------
    cuts_priced_spread_pct: float = 0.10  # 3M bill below EFFR by more than this = cuts priced
    dxy_easing_max: float = 102.0

    # --- Telemetry (signals, not phase drivers) --------------------------------
    tnx_1d_spike_bps: float = 10.0
    semi_rsi_digestion: float = 40.0
    ks11_sma_window: int = 200
    dxy_compression: float = 104.5

    rsi_period: int = 14


THRESHOLDS = Thresholds()

# --- HTTP ---------------------------------------------------------------------------
HTTP_TIMEOUT_SECONDS = 20.0
HTTP_ATTEMPTS = 3
HTTP_BACKOFF_SECONDS = (1.5, 4.0)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) regime-engine/0.3"
HISTORY_YEARS = 2
CACHE_TTL_SECONDS = 300  # local API memo only

# --- Source endpoints ------------------------------------------------------------------
# Cboe: index history (date, close) plus the delayed quote, whose `close` is the official
# session close once `last_trade_time` passes the index's session end.
CBOE_HISTORY_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_{symbol}.json"
CBOE_QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{symbol}.json"
CBOE_VIX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"  # history fallback
TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all"
)
NYFED_EFFR_URL = "https://markets.newyorkfed.org/api/rates/unsecured/effr/last/60.json"
ECB_DAILY_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
ECB_90D_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_SIP_MIN_AGE_MINUTES = 16  # free plan: SIP queries must end >= 15 minutes ago
EODHD_EOD_URL = "https://eodhd.com/api/eod/{symbol}"
EODHD_LIVE_URL = "https://eodhd.com/api/real-time/{symbol}"
EODHD_HISTORY_DAYS = 365  # the free plan serves one year; enough for SMA200 and RSI
KRX_BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
KRX_HISTORY_SESSIONS = 260  # backfilled once, then cached in the snapshots release
KRX_MAX_CALLS_PER_RUN = 120  # per endpoint; backfill completes over a few runs
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
SECRET_ENV_VARS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "KRX_API_KEY", "EODHD_API_KEY", "ANTHROPIC_API_KEY")

# Cboe index scaling and the earliest New York time a quote counts as that day's close.
# Yahoo's ^TNX and ^VIX are copies of these Cboe indexes: they confirm our parse, not the method.
CBOE_INDEXES: dict[str, tuple[str, float, str]] = {
    "y10@cboe": ("TNX", 10.0, "14:55"),  # 10-year yield x10; bond session ends 15:00
    "t3m@cboe": ("IRX", 10.0, "14:55"),  # 13-week bill discount rate x10
    "vix@cboe": ("VIX", 1.0, "16:14"),  # VIX settles 16:15
}

# ICE U.S. Dollar Index formula (published weights).
DXY_CONSTANT = 50.14348112
DXY_WEIGHTS = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119, "USDCAD": 0.091, "USDSEK": 0.042, "USDCHF": 0.036}

# --- Inputs ---------------------------------------------------------------------------
# `record`: sources of the number itself, newest date wins. `checks`: (source key,
# tolerance, "abs"|"pct", confirms). A confirming check that has the record's date and
# agrees makes the input VERIFIED; a non-confirming check (different methodology or
# timing) can only veto gross errors.
OFFICIAL_SOURCES = {"cboe", "treasury", "nyfed", "alpaca", "krx", "ecb"}


@dataclass(frozen=True)
class InputSpec:
    label: str
    record: tuple[str, ...]
    checks: tuple[tuple[str, float, str, bool], ...]
    used_in_call: bool  # False = telemetry only
    max_age_days: int  # record's latest date older than this (calendar days) = stale


INPUTS: dict[str, InputSpec] = {
    "y10": InputSpec(
        "10Y Treasury yield",
        ("y10@cboe",),
        (("y10@yahoo", 0.02, "abs", True), ("y10@treasury", 0.10, "abs", False)),
        True,
        5,
    ),
    # Treasury is the record: Cboe ^IRX ran 7-10 bp under Treasury's 3M on every day of the first
    # live run, so it is a different instrument/basis and only serves as a gross-error veto.
    # Yahoo ^IRX is a copy of Cboe's index, so it was dropped. Treasury posts the curve in the
    # evening: the 16:25 read uses the prior session, the 06:00 read the latest.
    "t3m": InputSpec(
        "3-month T-bill yield",
        ("t3m@treasury",),
        (("t3m@cboe", 0.15, "abs", False),),
        True,
        5,
    ),
    "vix": InputSpec("VIX", ("vix@cboe",), (("vix@yahoo", 1.0, "pct", True),), True, 5),
    "effr": InputSpec("Effective fed funds rate", ("effr@nyfed",), (), True, 7),
    "soxx": InputSpec("SOXX close", ("soxx@alpaca",), (("soxx@yahoo", 0.3, "pct", True),), True, 5),
    "igv": InputSpec("IGV close", ("igv@alpaca",), (("igv@yahoo", 0.3, "pct", True),), True, 5),
    "dxy": InputSpec(
        "Dollar index (DXY)",
        ("dxy@eodhd",),
        (("dxy@yahoo", 0.25, "abs", True), ("dxy@ecb", 1.0, "abs", False)),
        True,
        4,
    ),
    "kospi": InputSpec("KOSPI", ("kospi@eodhd", "kospi@krx"), (("kospi@yahoo", 0.1, "pct", True),), False, 6),
    "samsung": InputSpec(
        "Samsung Electronics", ("samsung@eodhd", "samsung@krx"), (("samsung@yahoo", 0.1, "pct", True),), False, 6
    ),
}
RECORD_MATCH_TOLERANCE = {"kospi": 0.05, "samsung": 1.0}  # EODHD vs KRX copies of the same close
CHECK_OVERLAP_SESSIONS = 5  # recent common dates compared between sources
SYSTEMATIC_MIN_DAYS = 3  # a check that disagrees on this many compared dates withholds, confirmed or not

YAHOO_SYMBOLS: dict[str, str] = {
    "y10@yahoo": "^TNX",
    "vix@yahoo": "^VIX",
    "soxx@yahoo": "SOXX",
    "igv@yahoo": "IGV",
    "dxy@yahoo": "DX-Y.NYB",
    "kospi@yahoo": "^KS11",
    "samsung@yahoo": "005930.KS",
}
EODHD_EOD_SYMBOLS = {"kospi@eodhd": "KS11.INDX", "samsung@eodhd": "005930.KO"}
EODHD_FX = {
    "EURUSD": "EURUSD.FOREX",
    "USDJPY": "USDJPY.FOREX",
    "GBPUSD": "GBPUSD.FOREX",
    "USDCAD": "USDCAD.FOREX",
    "USDSEK": "USDSEK.FOREX",
    "USDCHF": "USDCHF.FOREX",
}

SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "y10": (0.0, 20.0),
    "t3m": (0.0, 20.0),
    "vix": (5.0, 150.0),
    "effr": (0.0, 25.0),
    "dxy": (60.0, 160.0),
    "soxx": (1.0, 100000.0),
    "igv": (1.0, 100000.0),
    "kospi": (100.0, 100000.0),
    "samsung": (100.0, 10000000.0),
}

# --- Publishing schedule (New York time, weekdays). Mirrors meridian-clock. ---------
SCHEDULE_NY = ("06:00", "16:25")
MISSED_RUN_GRACE_MINUTES = 60

# --- Claude web search: last-resort gap filler ------------------------------------------
# Runs only when an input's record is missing, unconfirmed or a session behind AND Yahoo has
# a value for that same date. Claude's number is used only if it cites a page the search
# actually returned and agrees with Yahoo within the input's confirming tolerance: it is the
# second independent source, never the only one. Conflicts are never sent to Claude.
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_MODEL = "claude-haiku-5-5"
CLAUDE_FALLBACK_MODEL = "claude-sonnet-5-5"  # used if the first model rejects the request
CLAUDE_MAX_SEARCHES = 5  # per run
CLAUDE_MONTHLY_BUDGET_USD = 5.00  # hard stop; spend is tracked in claude_usage.json
CLAUDE_TIMEOUT_SECONDS = 120.0
# USD per million tokens (input, output) and per search; platform.claude.com pricing, Oct 2026.
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
CLAUDE_SEARCH_PRICE = 0.01

# What to ask for, per input: description, units, and the market whose session must be over.
GAP_QUESTIONS: dict[str, tuple[str, str]] = {
    "y10": ("U.S. 10-year Treasury note yield at the close", "us"),
    "t3m": ("U.S. 13-week (3-month) Treasury bill yield at the close, bond-equivalent basis", "us"),
    "vix": ("Cboe Volatility Index (VIX) official closing value", "us"),
    "soxx": ("iShares Semiconductor ETF (SOXX) official closing price in USD", "us"),
    "igv": ("iShares Expanded Tech-Software Sector ETF (IGV) official closing price in USD", "us"),
    "dxy": ("ICE U.S. Dollar Index (DXY) latest value", "fx"),
    "kospi": ("KOSPI index official closing value on the Korea Exchange", "kr"),
    "samsung": ("Samsung Electronics (005930) official closing price on the Korea Exchange, in KRW", "kr"),
}
US_SESSION_DONE_NY = "16:20"  # a US close for today is only requested after this time
KR_SESSION_DONE_NY = "03:00"  # KRX closes 15:30 KST = 02:30 EDT / 01:30 EST
