from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ActionState


class PredictionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    probability_up: float = Field(ge=0, le=1)
    expected_return: float
    return_interval: tuple[float, float]
    transaction_cost: float = Field(ge=0)
    safety_margin: float = Field(ge=0)
    model_disagreement: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> PredictionContext:
        if self.return_interval[0] > self.return_interval[1]:
            raise ValueError("return_interval must be ordered")
        return self


class DataStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    quality_passed: bool = True
    source_conflict: bool = False
    tradable: bool = True
    liquidity_passed: bool = True
    event_confirmed: bool = True
    regime_coverage: float = Field(default=1.0, ge=0, le=1)


class DriftStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    calibration_passed: bool = True
    drift_passed: bool = True


class AbstentionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: ActionState
    reasons: tuple[str, ...]


class AbstentionPolicy:
    def __init__(
        self,
        *,
        buy_probability: float = 0.65,
        sell_probability: float = 0.35,
        max_model_disagreement: float = 0.2,
        min_regime_coverage: float = 0.1,
    ) -> None:
        if not 0.5 < buy_probability < 1:
            raise ValueError("buy_probability must be between 0.5 and 1")
        if not 0 < sell_probability < 0.5:
            raise ValueError("sell_probability must be between 0 and 0.5")
        self.buy_probability = buy_probability
        self.sell_probability = sell_probability
        self.max_model_disagreement = max_model_disagreement
        self.min_regime_coverage = min_regime_coverage
        self.last_result: AbstentionResult | None = None

    def evaluate(
        self,
        prediction: PredictionContext,
        data_status: DataStatus,
        drift_status: DriftStatus,
    ) -> AbstentionResult:
        reasons: list[str] = []
        if not data_status.quality_passed:
            reasons.append("data_quality_failed")
        if data_status.source_conflict:
            reasons.append("source_conflict")
        if not data_status.tradable:
            reasons.append("untradeable")
        if not data_status.liquidity_passed:
            reasons.append("insufficient_liquidity")
        if not data_status.event_confirmed:
            reasons.append("unconfirmed_event")
        if data_status.regime_coverage < self.min_regime_coverage:
            reasons.append("insufficient_regime_coverage")
        if not drift_status.calibration_passed:
            reasons.append("calibration_failed")
        if not drift_status.drift_passed:
            reasons.append("model_drift")
        if prediction.model_disagreement > self.max_model_disagreement:
            reasons.append("model_disagreement")
        required_edge = prediction.transaction_cost + prediction.safety_margin
        if prediction.expected_return > 0:
            if prediction.return_interval[0] <= required_edge:
                reasons.append("return_interval_lacks_safety_margin")
            if prediction.expected_return <= required_edge:
                reasons.append("expected_return_below_cost")
            if prediction.probability_up < self.buy_probability:
                reasons.append("insufficient_probability_edge")
            proposed = ActionState.BUY
        elif prediction.expected_return < 0:
            if prediction.return_interval[1] >= -required_edge:
                reasons.append("return_interval_lacks_safety_margin")
            if abs(prediction.expected_return) <= required_edge:
                reasons.append("expected_return_below_cost")
            if prediction.probability_up > self.sell_probability:
                reasons.append("insufficient_probability_edge")
            proposed = ActionState.REDUCE
        else:
            reasons.append("zero_expected_edge")
            proposed = ActionState.HOLD
        return AbstentionResult(
            action=ActionState.ABSTAIN if reasons else proposed,
            reasons=tuple(reasons),
        )

    def decide(
        self,
        prediction: PredictionContext,
        data_status: DataStatus,
        drift_status: DriftStatus,
    ) -> ActionState:
        self.last_result = self.evaluate(prediction, data_status, drift_status)
        return self.last_result.action
