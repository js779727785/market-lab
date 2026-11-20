from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from market_lab.contracts import ActionState, ForecastDecision, PositionInput

from .portfolio import PortfolioState
from .sizing import SizingInput, VolatilityRiskSizer


class HoldingPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: ForecastDecision
    current_price: float = Field(gt=0)
    annualized_volatility: float = Field(gt=0)
    sector: str | None = None
    tracking_error: float = Field(default=0.0, ge=0)
    max_tracking_error: float = Field(default=0.05, ge=0)
    asset_weight_limit: float = Field(default=0.2, gt=0, le=1)
    sector_weight_limit: float = Field(default=0.3, gt=0, le=1)
    target_volatility: float = Field(default=0.1, gt=0)
    lot_size: int = Field(default=100, ge=1)


class PositionAdvice(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: ActionState
    target_weight: float | None = Field(default=None, ge=0, le=1)
    target_quantity: int | None = Field(default=None, ge=0)
    max_loss_amount: float | None = Field(default=None, ge=0)
    research_only: bool
    reasons: tuple[str, ...]
    counterevidence: tuple[str, ...]
    binding_constraints: tuple[str, ...] = ()
    buy_zone: tuple[float, float] | None = None
    reduce_zone: tuple[float, float] | None = None
    expires_at: datetime | None = None


class PositionAssistant:
    def __init__(self, sizer: VolatilityRiskSizer | None = None) -> None:
        self.sizer = sizer or VolatilityRiskSizer()

    def evaluate(
        self,
        position: PositionInput,
        prediction: HoldingPrediction,
        portfolio: PortfolioState,
    ) -> PositionAdvice:
        decision = prediction.decision
        if prediction.tracking_error > prediction.max_tracking_error:
            return self._advice(
                decision,
                action=ActionState.ABSTAIN,
                research_only=False,
                reasons=("tracking_error",),
            )
        if (
            decision.risk_invalidation is not None
            and prediction.current_price <= decision.risk_invalidation
        ):
            return self._advice(
                decision,
                action=ActionState.EXIT,
                target_weight=0.0,
                target_quantity=0,
                max_loss_amount=(
                    portfolio.account_value * position.max_loss_fraction
                    if position.max_loss_fraction is not None
                    else None
                ),
                research_only=position.max_loss_fraction is None,
                reasons=("risk_invalidation_breached",),
            )
        if decision.action is ActionState.ABSTAIN:
            return self._advice(
                decision,
                action=ActionState.ABSTAIN,
                research_only=False,
                reasons=("forecast_abstained",),
            )
        if position.max_loss_fraction is None:
            return self._advice(
                decision,
                action=ActionState.HOLD,
                research_only=True,
                reasons=("missing_risk_budget",),
            )
        if decision.risk_invalidation is None:
            return self._advice(
                decision,
                action=ActionState.ABSTAIN,
                research_only=False,
                reasons=("missing_risk_invalidation",),
            )
        current_market_value = position.quantity * prediction.current_price
        current_weight = current_market_value / portfolio.account_value
        sector_limit = position.sector_limit or prediction.sector_weight_limit
        sizing = self.sizer.size(
            SizingInput(
                account_value=portfolio.account_value,
                available_cash=min(position.available_cash, portfolio.cash),
                current_price=prediction.current_price,
                current_weight=current_weight,
                other_sector_weight=portfolio.other_sector_weight(
                    prediction.sector, position.asset_id
                ),
                risk_invalidation=decision.risk_invalidation,
                annualized_volatility=prediction.annualized_volatility,
                max_loss_fraction=position.max_loss_fraction,
                target_volatility=prediction.target_volatility,
                asset_weight_limit=prediction.asset_weight_limit,
                sector_weight_limit=sector_limit,
                lot_size=prediction.lot_size,
            )
        )
        reasons = list(decision.reasons)
        if decision.reduce_zone is not None and prediction.current_price >= decision.reduce_zone[0]:
            action = ActionState.REDUCE
            reasons.append("reduce_zone_reached")
        elif decision.action in {ActionState.REDUCE, ActionState.EXIT}:
            action = decision.action
        elif current_weight > sizing.target_weight * 1.05:
            action = ActionState.REDUCE
            reasons.append("above_target_weight")
        elif current_weight < sizing.target_weight * 0.95 and position.allow_add:
            if (
                decision.chase_invalidation is not None
                and prediction.current_price > decision.chase_invalidation
            ):
                action = ActionState.HOLD
                reasons.append("chase_invalidation_exceeded")
            else:
                action = ActionState.ADD
                reasons.append("below_target_weight")
        else:
            action = ActionState.HOLD
            reasons.append("within_target_band")
        return self._advice(
            decision,
            action=action,
            target_weight=sizing.target_weight,
            target_quantity=sizing.target_quantity,
            max_loss_amount=sizing.max_loss_amount,
            research_only=False,
            reasons=tuple(reasons),
            binding_constraints=sizing.binding_constraints,
        )

    @staticmethod
    def _advice(
        decision: ForecastDecision,
        *,
        action: ActionState,
        research_only: bool,
        reasons: tuple[str, ...],
        target_weight: float | None = None,
        target_quantity: int | None = None,
        max_loss_amount: float | None = None,
        binding_constraints: tuple[str, ...] = (),
    ) -> PositionAdvice:
        return PositionAdvice(
            action=action,
            target_weight=target_weight,
            target_quantity=target_quantity,
            max_loss_amount=max_loss_amount,
            research_only=research_only,
            reasons=reasons,
            counterevidence=decision.counterevidence,
            binding_constraints=binding_constraints,
            buy_zone=decision.buy_zone,
            reduce_zone=decision.reduce_zone,
            expires_at=decision.valid_until,
        )
