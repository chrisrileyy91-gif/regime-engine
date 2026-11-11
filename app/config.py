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
    vix_jump_pct: float = 10.0  # VIX up this much in a session = a volatility spike
    soxx_drop_z: float = -1.5  # SOXX's 1-session return this unusual vs its prior window = a sell-off
    soxx_drop_floor_pct: float = -4.0  # ...or this big regardless: after a rout the z adapts (same
    # reason as oil_move_pct). Nov 6-10, 2026: a 60-day window that holds a -12% day reads -4% as mild.
    soxx_z_window: int = 60
    # A yield jump counts toward the stress label only when chips also fell this much that session.
    # Nov 10, 2026: 10Y +12 bps, SOXX -0.1%, and the page read Elevated; over the 120 sessions to
    # then a 10 bps rise in the 10Y went with SOXX UP 2.1% (broad market held fixed).
    yields_count_soxx_pct: float = -1.5
    # Funding backdrop: the debt-funded AI names' move beyond what their usual sensitivity to SOXX
    # predicts, summed over the last 20 sessions and ranked against the past year of the same. A
    # level, so it lives in the backdrop. Its first shape was a same-day trigger (z < -1.5): on the
    # 250-session replay it fired on 3 of 63 semis drops and mostly on rallies the basket lagged.
    funding_window: int = 20
    funding_rank_window: int = 250
    funding_rank_min: int = 120
    # Tally: share of a semis drop that the broad market and yields explain (rolling regression of
    # SOXX's daily return on the equal-weight S&P 500's return and the 10Y change, prior sessions only).
    decomp_window: int = 120
    decomp_min_sessions: int = 60
    macro_share: float = 0.5  # at least this much explained = "the broad market fell", else chip-specific
    market_drop_pct: float = -1.0  # shown in the day list only: equal-weight S&P 500 down this much
    credit_z: float = -2.0  # tally: HYG down this unusually = credit stress (annotation, never the label)
    credit_move_pct: float = -0.75  # ...or this big regardless
    oil_z: float = 2.0  # tally: USO move this unusual (either way) vs its prior 60 sessions = oil shock
    oil_move_pct: float = 4.0  # ...or this big regardless: in a war regime a 60-day z adapts the shocks
    # away (Sep 1, US strikes on Iran: USO +5.5% was only 1.65 sigma)
    y2_jump_bps: float = 10.0  # tally: 2Y up this much = policy repricing (Aug 28: hike odds to 60%,
    # 2Y +14 bps while the 10Y/30Y barely moved)
    dollar_z: float = 2.0  # tally: UUP up this unusually = dollar squeeze
    tally_z_window: int = 60

    # --- Backdrop and telemetry (shown, never counted as stress) -----------------------------
    # VIX above 20 and SOXX RSI below 40 were stress drivers until Nov 2026. Both are levels: across
    # the first 39 scheduled reads every non-Calm label had RSI < 40 active, so the label read
    # "Elevated" on a -8.2% day and on a +2.1% day alike. They are backdrop now.
    vix_level: float = 20.0  # volatility high enough to force systematic de-risking
    semi_rsi_digestion: float = 40.0  # SOXX RSI(14) below this = chip momentum weak
    tnx_pctile_high: float = 80.0  # 10Y higher than this % of the trailing window
    tnx_pctile_window: int = 250  # ~1 year of Treasury business days
    ratio_sma_window: int = 20  # IGV/SOXX ratio trend
    ratio_slope_lookback: int = 5
    ks11_sma_window: int = 200

    # --- Rate-cut catalyst ---------------------------------------------------------------------
    cuts_priced_spread_pct: float = 0.10  # 3M bill below EFFR by more than this = cuts priced (fallback)
    cuts_2y_spread_pct: float = 0.25  # 2Y below EFFR by more than this = cuts priced over two years

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
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ALPACA_SIP_MIN_AGE_MINUTES = 16  # free plan: SIP queries must end >= 15 minutes ago
# Splits, so a check that hasn't re-based its history yet isn't read as a wrong number (Nov 5,
# 2026: SOXX 3-for-1; Alpaca's split-adjusted bars were exactly 1/3 of Yahoo's on every date).
# v1 is the current path; v1beta1 is the older one, tried if v1 is not found.
ALPACA_CORPORATE_ACTIONS_URLS = (
    "https://data.alpaca.markets/v1/corporate-actions",
    "https://data.alpaca.markets/v1beta1/corporate-actions",
)
SPLIT_LOOKBACK_DAYS = 45  # ex-dates this far back still affect the compared window
SPLIT_LOOKAHEAD_DAYS = 7  # Alpaca can apply an announced split before its ex-date
EODHD_EOD_URL = "https://eodhd.com/api/eod/{symbol}"
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

# --- Inputs ---------------------------------------------------------------------------
# `record`: sources of the number itself, newest date wins. `checks`: (source key,
# tolerance, "abs"|"pct", confirms). A confirming check that has the record's date and
# agrees makes the input VERIFIED; a non-confirming check (different methodology or
# timing) can only veto gross errors.
OFFICIAL_SOURCES = {"cboe", "treasury", "nyfed", "alpaca", "krx"}


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
    # The dollar index (DXY) was an input until Nov 2026. Without an EODHD key it was filled by a
    # web search every run and labelled verified; the dollar is read from UUP on the consolidated
    # tape instead (TALLY_TICKERS), same day, no search.
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
    "kospi@yahoo": "^KS11",
    "samsung@yahoo": "005930.KS",
}
EODHD_EOD_SYMBOLS = {"kospi@eodhd": "KS11.INDX", "samsung@eodhd": "005930.KO"}

SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "y10": (0.0, 20.0),
    "t3m": (0.0, 20.0),
    "vix": (5.0, 150.0),
    "effr": (0.0, 25.0),
    "soxx": (1.0, 100000.0),
    "igv": (1.0, 100000.0),
    "kospi": (100.0, 100000.0),
    "samsung": (100.0, 10000000.0),
}

# --- Publishing schedule (New York time, weekdays). Mirrors meridian-clock. ---------
# 16:40, not 16:25: Cboe's quote feed is 15 minutes delayed and VIX settles at 16:15, so the settle
# isn't visible before ~16:30. All 19 close runs from Oct 14 to Nov 10, 2026 carried the prior VIX.
SCHEDULE_NY = ("06:00", "16:40")
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

# --- Headlines (the part gauges can't see) -------------------------------------------------
# Benzinga via Alpaca's news API (same keys as the bars): timestamped, ticker-tagged, back to 2015.
# Claude only CLASSIFIES the headlines it is given (no web search) and must cite their ids; the
# page links the actual headlines so every label can be checked. Gauges stay the measurement:
# headlines annotate a day and flag disagreement, they never overwrite a gauge.
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
NEWS_SYMBOLS = ("NVDA", "AVGO", "AMD", "MU", "TSM", "ASML", "INTC", "SOXX", "SMH", "SPY", "QQQ")
NEWS_MAX_PAGES = 4  # x50 headlines per day
NEWS_MAX_DAYS_PER_RUN = 25  # first run backfills the tally; later runs add one day at a time
NEWS_CATEGORIES = {
    "company": "company news (earnings, guidance, deals, a single chipmaker)",
    "ai_demand": "AI spending or demand worries (capex, financing, competition, model efficiency)",
    "policy": "government policy (export controls, tariffs, regulation, subsidies)",
    "analyst": "analyst calls (downgrades, price targets)",
    "macro_data": "economic data (CPI, jobs, PMI, GDP)",
    "fed": "the Federal Reserve (decisions, speeches, rate expectations)",
    "geopolitics": "geopolitics or oil (war, sanctions, supply shocks)",
    "none_clear": "no clear driver in the headlines",
}
NEWS_MACRO = ("macro_data", "fed", "geopolitics")  # headline says macro; compare with the gauges

# What to ask for, per input: description, units, and the market whose session must be over.
GAP_QUESTIONS: dict[str, tuple[str, str]] = {
    "y10": ("U.S. 10-year Treasury note yield at the close", "us"),
    "t3m": ("U.S. 13-week (3-month) Treasury bill yield at the close, bond-equivalent basis", "us"),
    "vix": ("Cboe Volatility Index (VIX) official closing value", "us"),
    "soxx": ("iShares Semiconductor ETF (SOXX) official closing price in USD", "us"),
    "igv": ("iShares Expanded Tech-Software Sector ETF (IGV) official closing price in USD", "us"),
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
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",),
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "eps": ("EarningsPerShareDiluted",),
}
CYCLE_GROUP_METRIC = {"hyperscalers": "capex", "chips": "revenue", "software": "revenue"}
# Capex and chip revenue are dollar aggregates (sum). Software is the median company, so one
# fast grower can't stand in for the industry; breadth (how many are accelerating) is shown too.
CYCLE_GROUP_HOW = {"hyperscalers": "sum", "chips": "sum", "software": "median"}
CYCLE_HISTORY_QUARTERS = 12
# Funding: how much of the spenders' operating cash flow their capex consumes (trailing year). The
# headlines on 9 of the 20 semis drops to Nov 5, 2026 were about who pays for the buildout (Meta's
# credit, CoreWeave's debt, debt-funded chip deals) and nothing on the page measured it. Shown, not
# part of the phase rules. Debt issuance is not read: filings omit quarters with no issuance, so a
# gap can't be told from a zero.
FUNDING_GROUP = "hyperscalers"
# Chip revenue is also shown without these: memory prices, not AI orders, drive Micron's swings
# (single-quarter revenue +346% year over year in Q2 2026).
CHIPS_EX = ("MU",)
# Chip stock prices for "earnings vs price": quarter-end closes from the Alpaca SIP feed, fetched
# with SOXX/IGV. Not cross-checked (a quarterly comparison, never part of the daily call).
CYCLE_PRICE_TICKERS = tuple(t for t, _, _ in CYCLE_COMPANIES["chips"])
# The broad market, for the tally. Equal-weight (RSP), not SPY: chips are a large slice of the
# cap-weighted index, so a semis rout drags SPY by itself (Oct 14: SOXX -4.5%, SPY -1.1%; Sep 14:
# SPY -0.5% while RSP +0.1%). Alpaca SIP, fetched with SOXX, not cross-checked (classification only).
MARKET_TICKER = "RSP"
# Oil and the dollar, same-day, from the same consolidated tape (the choice Crude Logic already
# made after Yahoo's CL=F broke): USO tracks front-month WTI futures, UUP the dollar index.
# EIA's official WTI spot posts in weekly batches, too late for a same-day read.
# HYG (high-yield bonds) is the same-day credit gauge.
TALLY_TICKERS = {"market": MARKET_TICKER, "oil": "USO", "dollar": "UUP", "credit": "HYG"}
# Debt-funded AI names. Their average move is shown beside SOXX on each tally day: when they fall
# harder than chips, the selling is about funding. Any that Alpaca doesn't return are left out.
AI_LEVERAGE_TICKERS = ("CRWV", "ORCL", "NBIS")
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
