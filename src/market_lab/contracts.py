from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceGrade(StrEnum):
    OFFICIAL_PRIMARY = "official_primary"
    LICENSED_PRIMARY_TRANSPORT = "licensed_primary_transport"
    INDEPENDENT_CROSSCHECK = "independent_crosscheck"
    DERIVED = "derived"
    UNVERIFIED = "unverified"


class AssetClass(StrEnum):
    A_SHARE = "a_share"
    GOLD = "gold"
    GOLD_PRODUCT = "gold_product"
    MACRO = "macro"


class ActionState(StrEnum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    REDUCE = "reduce"
    EXIT = "exit"
    WATCH = "watch"
    ABSTAIN = "abstain"


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class RecordEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str = Field(min_length=1)
    record_key: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    evidence_grade: EvidenceGrade
    event_time: datetime
    published_at: datetime
    first_seen_at: datetime
    available_at: datetime
    revision: int = Field(ge=1)
    payload: dict[str, Any]
    raw_sha256: str | None = None
    normalized_sha256: str | None = None

    _validate_event_time = field_validator(
        "event_time", "published_at", "first_seen_at", "available_at"
    )(_aware)


class PriceBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    asset_class: AssetClass
    session_date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    turnover: float = Field(ge=0)
    previous_close: float | None = Field(default=None, gt=0)
    suspended: bool = False
    limit_up: float | None = Field(default=None, gt=0)
    limit_down: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> PriceBar:
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be the maximum OHLC value")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be the minimum OHLC value")
        if self.limit_up is not None and self.limit_down is not None:
            if self.limit_down >= self.limit_up:
                raise ValueError("limit_down must be below limit_up")
        return self


class SecurityMasterRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    exchange: str
    name: str
    board: str
    list_date: date
    delist_date: date | None = None
    status: str = "listed"
    is_st: bool = False
    industry: str | None = None
    industry_version: str | None = None


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    action_type: str
    announcement_at: datetime
    ex_date: date
    cash_per_share: float = Field(default=0.0, ge=0)
    share_ratio: float = Field(default=0.0, ge=0)

    _validate_announcement = field_validator("announcement_at")(_aware)


class MacroObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    series_id: str
    observation_date: date
    value: float | None
    published_at: datetime
    available_at: datetime
    vintage_date: date | None = None

    _validate_macro_time = field_validator("published_at", "available_at")(_aware)


class EventFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: str
    entities: tuple[str, ...]
    statement: str
    published_at: datetime
    first_seen_at: datetime
    available_at: datetime
    source_uris: tuple[str, ...]
    evidence_grade: EvidenceGrade
    confirmed: bool
    numeric_values: dict[str, float] = Field(default_factory=dict)
    revision: int = Field(default=1, ge=1)
    event_key: str | None = None
    document_ids: tuple[str, ...] = ()
    affected_assets: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0, le=1)
    novelty: float = Field(default=0.5, ge=0, le=1)
    decay_half_life_hours: float = Field(default=24.0, gt=0)
    strict_eligible: bool = False
    conflict_source_uris: tuple[str, ...] = ()

    _validate_event_times = field_validator("published_at", "first_seen_at", "available_at")(_aware)


class ForecastDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision_id: str
    asset_id: str
    asset_class: AssetClass
    forecast_cutoff: datetime
    valid_until: datetime
    horizon_days: int = Field(gt=0)
    action: ActionState
    evidence_mode: str
    probability_up: float = Field(ge=0, le=1)
    expected_return: float
    return_interval: tuple[float, float]
    max_adverse_excursion_probability: float = Field(ge=0, le=1)
    buy_zone: tuple[float, float] | None = None
    chase_invalidation: float | None = Field(default=None, gt=0)
    risk_invalidation: float | None = Field(default=None, gt=0)
    reduce_zone: tuple[float, float] | None = None
    target_weight: float | None = Field(default=None, ge=0, le=1)
    reasons: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()

    _validate_forecast_times = field_validator("forecast_cutoff", "valid_until")(_aware)

    @model_validator(mode="after")
    def validate_decision(self) -> ForecastDecision:
        if self.valid_until <= self.forecast_cutoff:
            raise ValueError("valid_until must be after forecast_cutoff")
        if self.return_interval[0] > self.return_interval[1]:
            raise ValueError("return_interval must be ordered")
        for zone_name in ("buy_zone", "reduce_zone"):
            zone = getattr(self, zone_name)
            if zone is not None and (zone[0] <= 0 or zone[0] > zone[1]):
                raise ValueError(f"{zone_name} must be positive and ordered")
        return self


class PositionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    asset_class: AssetClass
    average_cost: float = Field(gt=0)
    quantity: float = Field(gt=0)
    available_cash: float = Field(default=0, ge=0)
    horizon_days: int = Field(gt=0)
    max_loss_fraction: float | None = Field(default=None, gt=0, le=1)
    allow_add: bool = True
    allow_scale: bool = True
    sector_limit: float | None = Field(default=None, gt=0, le=1)
