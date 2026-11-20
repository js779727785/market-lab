from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_lab.abstention import AbstentionPolicy, DataStatus, DriftStatus, PredictionContext
from market_lab.contracts import ActionState, AssetClass, ForecastDecision
from market_lab.provenance import canonical_sha256


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class CalibratedPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    asset_class: AssetClass
    forecast_cutoff: datetime
    horizon_days: int = Field(gt=0)
    probability_up: float = Field(ge=0, le=1)
    expected_return: float
    return_interval: tuple[float, float]
    quantile_returns: dict[float, float]
    max_adverse_excursion_probability: float = Field(ge=0, le=1)
    model_disagreement: float = Field(ge=0)
    supporting_evidence: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()

    _validate_cutoff = field_validator("forecast_cutoff")(_aware)

    @model_validator(mode="after")
    def validate_distribution(self) -> CalibratedPrediction:
        required = (0.05, 0.2, 0.35, 0.6, 0.75, 0.9)
        missing = [quantile for quantile in required if quantile not in self.quantile_returns]
        if missing:
            raise ValueError(f"missing required return quantiles: {missing}")
        values = [self.quantile_returns[quantile] for quantile in required]
        if values != sorted(values):
            raise ValueError("return quantiles must be monotonically ordered")
        if self.return_interval[0] > self.return_interval[1]:
            raise ValueError("return_interval must be ordered")
        return self


class DecisionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_price: float = Field(gt=0)
    price_tick: float = Field(default=0.01, gt=0)
    transaction_cost: float = Field(default=0.0, ge=0)
    safety_margin: float = Field(default=0.0, ge=0)
    valid_days: int | None = Field(default=None, gt=0)
    evidence_mode: str = "strict"
    data_status: DataStatus = Field(default_factory=DataStatus)
    drift_status: DriftStatus = Field(default_factory=DriftStatus)


class PriceLevelPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    buy_zone: tuple[float, float]
    chase_invalidation: float
    risk_invalidation: float
    reduce_zone: tuple[float, float]


def _round_tick(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 8)


def build_price_levels(
    current_price: float, quantiles: dict[float, float], price_tick: float
) -> PriceLevelPlan:
    def price(quantile: float) -> float:
        return _round_tick(current_price * (1.0 + quantiles[quantile]), price_tick)

    return PriceLevelPlan(
        buy_zone=(price(0.2), price(0.35)),
        chase_invalidation=price(0.6),
        risk_invalidation=price(0.05),
        reduce_zone=(price(0.75), price(0.9)),
    )


class DecisionEngine:
    def __init__(self, abstention_policy: AbstentionPolicy | None = None) -> None:
        self.abstention_policy = abstention_policy or AbstentionPolicy()

    def for_candidate(
        self, prediction: CalibratedPrediction, context: DecisionContext
    ) -> ForecastDecision:
        action = self.abstention_policy.decide(
            PredictionContext(
                probability_up=prediction.probability_up,
                expected_return=prediction.expected_return,
                return_interval=prediction.return_interval,
                transaction_cost=context.transaction_cost,
                safety_margin=context.safety_margin,
                model_disagreement=prediction.model_disagreement,
            ),
            context.data_status,
            context.drift_status,
        )
        levels = build_price_levels(
            context.current_price, prediction.quantile_returns, context.price_tick
        )
        valid_days = context.valid_days or prediction.horizon_days
        identifier = canonical_sha256(
            {
                "asset_id": prediction.asset_id,
                "cutoff": prediction.forecast_cutoff,
                "horizon_days": prediction.horizon_days,
                "distribution": prediction.quantile_returns,
            }
        )
        strict_levels = action is not ActionState.ABSTAIN
        return ForecastDecision(
            decision_id=identifier,
            asset_id=prediction.asset_id,
            asset_class=prediction.asset_class,
            forecast_cutoff=prediction.forecast_cutoff,
            valid_until=prediction.forecast_cutoff + timedelta(days=valid_days),
            horizon_days=prediction.horizon_days,
            action=action,
            evidence_mode=context.evidence_mode,
            probability_up=prediction.probability_up,
            expected_return=prediction.expected_return,
            return_interval=prediction.return_interval,
            max_adverse_excursion_probability=prediction.max_adverse_excursion_probability,
            buy_zone=levels.buy_zone if strict_levels else None,
            chase_invalidation=levels.chase_invalidation if strict_levels else None,
            risk_invalidation=levels.risk_invalidation if strict_levels else None,
            reduce_zone=levels.reduce_zone if strict_levels else None,
            reasons=("calibrated_quantile_levels", *prediction.supporting_evidence),
            counterevidence=prediction.counterevidence,
        )
