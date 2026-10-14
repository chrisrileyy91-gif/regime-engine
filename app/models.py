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


class DriverMetric(BaseModel):
    label: str
    value: float | None
    unit: str  # "%", "bps", "z", "pts", "" — display hint for the page
    trigger: str | None  # None = context only, not a trigger
    met: bool | None


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
    drivers: list[Driver] = []
    rules: list[RuleEvaluation]
    telemetry: list[TelemetrySignal]
    inputs: dict[str, object]
    sources: list[SourceStatus]
    notes: list[str]
