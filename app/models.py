"""API response contract for /api/v1/regime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Condition(BaseModel):
    name: str
    value: float | bool | None
    threshold: str
    met: bool | None  # None = input unavailable, not "false"
    unsourced: bool = False  # leg defined by the spec but no data feed exists


class TelemetrySignal(BaseModel):
    indicator: str
    source: str
    value: float | None
    trigger: str
    triggered: bool | None
    signal: str
    tone: str = "risk"  # "risk" or "constructive" — how a triggered signal should read


class QualityIssue(BaseModel):
    severity: str  # "warn" (a reader should know) or "info" (handled automatically)
    message: str


class Preliminary(BaseModel):
    value: float
    date: str
    source: str


class InputStatus(BaseModel):
    """Verification result for one input. `value` is None unless the official record is usable."""

    key: str
    label: str
    status: str  # VERIFIED / OFFICIAL / FILLED (web search matched an independent copy) / CONFLICT / MISSING
    value: float | None
    date: str | None
    source: str | None  # official source actually used
    confirmed_by: list[str] = []
    notes: list[str] = []
    stale: bool = False
    used_in_call: bool = True
    preliminary: Preliminary | None = None  # newer unofficial value, never used in the call


class DataQuality(BaseModel):
    status: str  # OK / DEGRADED / STALE
    summary: str
    issues: list[QualityIssue] = []
    inputs: list[InputStatus] = []


class DriverMetric(BaseModel):
    label: str
    value: float | None
    unit: str  # "%", "bps", "z", "pts", "" — display hint for the page
    trigger: str | None  # None = context only, not a trigger
    met: bool | None
    as_of: str | None = None  # date of the official value
    preliminary: Preliminary | None = None


class Driver(BaseModel):
    key: str  # liquidity / risk_off / semis
    name: str
    asks: str  # the plain-English question this driver answers
    state: str  # ACTIVE / QUIET / UNKNOWN
    metrics: list[DriverMetric]


class SourceStatus(BaseModel):
    key: str
    symbol: str
    source: str
    ok: bool
    last_date: str | None
    error: str | None


class QuarterValue(BaseModel):
    quarter: str
    value: float


class CycleCondition(BaseModel):
    name: str
    value: float | None
    threshold: str
    met: bool | None
    unit: str = "%"


class CyclePhaseCheck(BaseModel):
    phase_id: int
    name: str
    met: bool | None
    conditions: list[CycleCondition]


class CycleMetric(BaseModel):
    key: str
    label: str
    members: list[str]
    latest: float | None  # value at the quarter the PHASE reads (every group reported)
    previous: float | None
    kind: str = "growth"  # "growth" = TTM year-over-year %, "level" = a ratio in %
    through: str | None = None  # this series' own newest quarter (can be ahead of the phase)
    newest: float | None = None  # value at `through`
    single_quarter: float | None = None  # that quarter alone vs a year earlier (%), growth series only
    history: list[QuarterValue] = []


class Catalyst(BaseModel):
    key: str
    label: str
    status: str  # yes / starting / mixed / not yet / unknown
    detail: str
    news: dict[str, Any] | None = None  # {category, summary, evidence[{headline, url, time}], confidence}
    news_flag: str | None = None  # headlines and gauges disagree


class CompanyFiling(BaseModel):
    ticker: str
    group: str
    ok: bool
    error: str | None = None
    tag: str | None = None
    latest_period_end: str | None = None
    filed: str | None = None


class Cycle(BaseModel):
    """The slow layer: which phase of the AI capex cycle the filings say we are in."""

    phase_id: int  # 1 Buildout, 2 Digestion, 3 Monetization, 0 between phases
    phase: str
    known: bool
    phases_met: list[int] = []  # every phase whose checklist is fully met (Monetization can overlap)
    since: str | None  # first quarter of the current run of this phase
    confirmed: bool  # held for CYCLE.confirm_quarters quarters
    quarters_in_phase: int = 0
    through: str | None  # latest calendar quarter every company has reported
    summary: str
    checks: list[CyclePhaseCheck] = []
    metrics: list[CycleMetric] = []
    history: list[dict[str, object]] = []
    catalysts: list[Catalyst] = []
    filings: list[CompanyFiling] = []
    fetched_at: str | None = None
    waiting_on: list[str] = []  # companies whose next quarter the phase is waiting for
    notes: list[str] = []


class Stress(BaseModel):
    """The fast layer: how much pressure the market is under today."""

    level: str  # CALM / ELEVATED / STRESSED / UNKNOWN
    label: str
    reasons: list[str] = []


class TallyDay(BaseModel):
    date: str
    soxx_pct: float
    driver: str  # rates/oil/dollar, else market (macro explains half+), else rotation/chips; or unknown
    moved: list[str] = []  # every gauge that moved that day, credit included
    provisional: bool = False  # newest day still waiting for an official close; not counted
    macro_share: float | None = None  # 0..1: share of the drop the broad market and yields explain
    expected_pct: float | None = None  # SOXX move expected from the broad market and the 10Y that day
    credit_pct: float | None = None  # HYG
    ai_leverage_pct: float | None = None  # average move of the debt-funded AI names
    tnx_bps: float | None = None
    tnx_spike_bps: float | None = None  # 10Y intraday high vs the prior close
    y30_bps: float | None = None
    y2_bps: float | None = None  # Treasury 2Y, the policy-expectations gauge
    vix_pct: float | None = None
    market_pct: float | None = None  # equal-weight S&P 500
    oil_pct: float | None = None  # USO
    dollar_pct: float | None = None  # UUP
    igv_pct: float | None = None
    detail: str
    news: dict[str, Any] | None = None  # {category, summary, evidence[{headline, url, time}], confidence}
    news_flag: str | None = None  # headlines and gauges disagree


class Tally(BaseModel):
    """The scapegoat check: what was actually moving on recent semis sell-offs."""

    days: list[TallyDay] = []
    counts: dict[str, int] = {}
    news_counts: dict[str, int] = {}  # headline categories on the no-gauge days
    window_sessions: int = 0
    avg_macro_share: float | None = None  # mean explained share across the counted days
    market_beta: float | None = None  # SOXX % per 1% in the equal-weight S&P 500, newest window
    rate_beta_10bps: float | None = None  # SOXX % per +10 bps in the 10Y, market held fixed
    fit_r2: float | None = None  # share of SOXX's daily variation the two account for
    fit_sessions: int | None = None
    summary: str = ""


class OvernightItem(BaseModel):
    label: str
    value: float | None
    change_pct: float | None
    date: str | None
    source: str | None = None


class Today(BaseModel):
    us_session: str | None  # date of the US close the read uses (the page computes the next update)
    stress: Stress
    overnight: list[OvernightItem] = []


class RegimeStatus(BaseModel):
    # Fields kept from the original spec's payload (dashboard-compatible), with new meanings:
    phase: str  # the CYCLE phase from company filings (quarterly), e.g. "Buildout"
    risk_level: str  # TODAY's stress: CALM / ELEVATED / STRESSED / UNKNOWN (daily)
    liquidity_squeeze: bool | None  # original key, kept: did yields jump or volatility spike this session
    semi_rsi: float | None
    tnx_yield: float | None
    actionable_summary: str
    # Additions
    phase_id: int  # 1 Buildout, 2 Digestion, 3 Monetization, 0 between phases
    as_of: str
    stale: bool = False
    stale_reason: str | None = None
    cycle: Cycle | None = None
    today: Today | None = None
    tally: Tally | None = None
    diagnosis: str = ""
    backdrop: str = ""  # slow-moving context (levels), deliberately separate from the diagnosis
    drivers: list[Driver] = []
    data_quality: DataQuality | None = None
    input_dates: dict[str, str] = {}
    telemetry: list[TelemetrySignal] = []
    inputs: dict[str, object] = {}
    sources: list[SourceStatus] = []
    notes: list[str] = []
