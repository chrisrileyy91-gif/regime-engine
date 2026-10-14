"""API response contract for /api/v1/regime."""

from __future__ import annotations

from pydantic import BaseModel


class Condition(BaseModel):
    name: str
    value: float | bool | None
    threshold: str
    met: bool | None  # None = input unavailable, not "false"
    unsourced: bool = False  # leg defined by the spec but no data feed exists


class RuleEvaluation(BaseModel):
    phase_id: int
    phase: str
    logic: str  # "ANY" or "ALL"
    fired: bool | None  # three-valued: True / False / unknown
    conditions: list[Condition]


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
    status: str  # VERIFIED / OFFICIAL / CONFLICT / MISSING
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


class DecisionMatrix(BaseModel):
    macro_liquidity: str  # SQUEEZE / CLEAR / UNKNOWN
    semi_momentum: str  # STRONG / CHOPPY / DIGESTION / UNKNOWN
    current_state: str


class SourceStatus(BaseModel):
    key: str
    symbol: str
    source: str
    ok: bool
    last_date: str | None
    error: str | None


class RegimeStatus(BaseModel):
    # Fields kept from the original spec's payload (dashboard-compatible)
    phase: str
    risk_level: str
    liquidity_squeeze: bool | None
    semi_rsi: float | None
    tnx_yield: float | None
    actionable_summary: str
    # Additions
    phase_id: int
    confidence: str  # FULL / PARTIAL / PROVISIONAL
    as_of: str
    stale: bool = False
    stale_reason: str | None = None
    matrix: DecisionMatrix
    diagnosis: str = ""
    backdrop: str = ""  # slow-moving context (levels), deliberately separate from the diagnosis
    drivers: list[Driver] = []
    data_quality: DataQuality | None = None
    input_dates: dict[str, str] = {}
    rules: list[RuleEvaluation]
    telemetry: list[TelemetrySignal]
    inputs: dict[str, object]
    sources: list[SourceStatus]
    notes: list[str]
