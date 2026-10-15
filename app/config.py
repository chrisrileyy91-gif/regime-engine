"""Single source of truth for every threshold, source and tolerance the engine uses.

Values marked UNCALIBRATED came from the original spec (or from us) and have not been
backtested. Change a number here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """The daily layer. The cycle's thresholds are in CycleThresholds below. All UNCALIBRATED."""

    # --- Today's drivers (each counts toward stress only when it moved that session) ---------
    tnx_1d_spike_bps: float = 10.0  # 10Y up this much in a session
    tnx_delta_z: float = 1.5  # ...or an unusual move: z-score of the change vs the prior window
    tnx_delta_z_window: int = 60
    vix_level: float = 20.0  # volatility high enough to force systematic de-risking
    semi_rsi_digestion: float = 40.0  # SOXX RSI(14) below this = chip momentum broken
    vix_jump_pct: float = 10.0  # tally: VIX up this much on a semis down day = market-wide fear
    market_drop_pct: float = -1.0  # tally: equal-weight S&P 500 down this much the same day = broad selling

    # --- Backdrop and telemetry (shown, never counted as stress) -----------------------------
    tnx_pctile_high: float = 80.0  # 10Y higher than this % of the trailing window
    tnx_pctile_window: int = 250  # ~1 year of Treasury business days
    ratio_sma_window: int = 20  # IGV/SOXX ratio trend
    ratio_slope_lookback: int = 5
    ks11_sma_window: int = 200
    dxy_compression: float = 104.5

    # --- Rate-cut catalyst ---------------------------------------------------------------------
    cuts_priced_spread_pct: float = 0.10  # 3M bill below EFFR by more than this = cuts priced

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
        False,  # background telemetry since the cycle moved to filings
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

# --- The cycle: from company filings (SEC EDGAR XBRL), not prices ---------------------------
# The three phases are defined by fundamentals: capex and chip revenue (Buildout), spending
# growth peaking while multiples compress (Digestion), software revenue accelerating
# (Monetization). Rate cuts are a catalyst, not a phase. All thresholds UNCALIBRATED.
SEC_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
SEC_DEFAULT_USER_AGENT = "regime-engine research (https://github.com/chrisrileyy91-gif/regime-engine)"
SEC_REFRESH_HOURS = 20  # filings change quarterly; refetch at most once per day
SEC_CONCURRENCY = 4  # SEC allows 10 requests/second

# (ticker, CIK, expected entityName prefix). The name check guards against a wrong CIK.
# Oracle sits with the capex spenders: its growth is now AI cloud capacity (renting GPUs), which
# is buildout, not software monetization. In the software group it alone produced a false
# "acceleration" in the first live read (Oracle 6% -> 22% while the median company slowed).
CYCLE_COMPANIES: dict[str, list[tuple[str, str, str]]] = {
    "hyperscalers": [
        ("MSFT", "0000789019", "MICROSOFT"),
        ("GOOGL", "0001652044", "ALPHABET"),
        ("AMZN", "0001018724", "AMAZON"),
        ("META", "0001326801", "META PLATFORMS"),
        ("ORCL", "0001341439", "ORACLE"),
    ],
    "chips": [
        ("NVDA", "0001045810", "NVIDIA"),
        ("AVGO", "0001730168", "BROADCOM"),
        ("AMD", "0000002488", "ADVANCED MICRO DEVICES"),
        ("MU", "0000723125", "MICRON"),
    ],
    "software": [
        ("CRM", "0001108524", "SALESFORCE"),
        ("NOW", "0001373715", "SERVICENOW"),
        ("ADBE", "0000796343", "ADOBE"),
        ("INTU", "0000896878", "INTUIT"),
        ("WDAY", "0001327811", "WORKDAY"),
        ("PLTR", "0001321655", "PALANTIR"),
    ],
}
# Tags tried in order. The one with the most recent period wins; ties go to the longer history
# (Intuit files both revenue tags, but only one goes back before 2023).
CYCLE_TAGS: dict[str, tuple[str, ...]] = {
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "eps": ("EarningsPerShareDiluted",),
}
CYCLE_GROUP_METRIC = {"hyperscalers": "capex", "chips": "revenue", "software": "revenue"}
# Capex and chip revenue are dollar aggregates (sum). Software is the median company, so one
# fast grower can't stand in for the industry; breadth (how many are accelerating) is shown too.
CYCLE_GROUP_HOW = {"hyperscalers": "sum", "chips": "sum", "software": "median"}
CYCLE_HISTORY_QUARTERS = 12
# Chip stock prices for "earnings vs price": quarter-end closes from the Alpaca SIP feed, fetched
# with SOXX/IGV. Not cross-checked (a quarterly comparison, never part of the daily call).
CYCLE_PRICE_TICKERS = tuple(t for t, _, _ in CYCLE_COMPANIES["chips"])
# The broad market, for the tally. Equal-weight (RSP), not SPY: chips are a large slice of the
# cap-weighted index, so a semis rout drags SPY by itself (Oct 14: SOXX -4.5%, SPY -1.1%; Sep 14:
# SPY -0.5% while RSP +0.1%). Alpaca SIP, fetched with SOXX, not cross-checked (classification only).
MARKET_TICKER = "RSP"
# Tally-only Cboe series (official, not cross-checked; classification only). Close-to-close data
# misses a yield spike that reverses: on CPI day Oct 14 the 10Y rose 11 bps intraday and closed -3.
# The 30Y catches long-end shocks the 10Y misses (Aug 18: 30Y at its highest since 2007).
CBOE_TALLY = {"y10": ("TNX", 10.0), "y30": ("TYX", 10.0)}


@dataclass(frozen=True)
class CycleThresholds:
    buildout_capex_growth: float = 20.0  # hyperscaler capex, trailing-12-month YoY %
    buildout_chip_growth: float = 25.0  # chip revenue, TTM YoY %
    digestion_capex_floor: float = 10.0  # still spending heavily
    rolled_over_points: float = 5.0  # growth this far below its recent peak = "has peaked"
    peak_window_quarters: int = 4  # "recent peak" = highest reading over this many quarters
    monetization_software_growth: float = 10.0  # median software company, revenue TTM YoY %
    monetization_accel_points: float = 1.0  # median company's growth up this much over two quarters
    confirm_quarters: int = 2  # a phase is "confirmed" after this many consecutive quarters


CYCLE = CycleThresholds()

# --- Daily layer -------------------------------------------------------------------------------
TALLY_DROP_PCT = -1.5  # a "big down day" for semis
TALLY_DAYS = 20  # the tally covers the most recent this-many big down days
STRESSED_VIX = 30.0  # VIX above this alone makes the day "Stressed"
