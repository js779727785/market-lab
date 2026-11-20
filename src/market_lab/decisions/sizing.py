from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class SizingInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_value: float = Field(gt=0)
    available_cash: float = Field(ge=0)
    current_price: float = Field(gt=0)
    current_weight: float = Field(ge=0, le=1)
    other_sector_weight: float = Field(ge=0, le=1)
    risk_invalidation: float = Field(gt=0)
    annualized_volatility: float = Field(gt=0)
    max_loss_fraction: float = Field(gt=0, le=1)
    target_volatility: float = Field(gt=0)
    asset_weight_limit: float = Field(gt=0, le=1)
    sector_weight_limit: float = Field(gt=0, le=1)
    lot_size: int = Field(default=1, ge=1)


class SizingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_weight: float = Field(ge=0, le=1)
    target_quantity: int = Field(ge=0)
    max_loss_amount: float = Field(ge=0)
    binding_constraints: tuple[str, ...]


class VolatilityRiskSizer:
    def size(self, sizing: SizingInput) -> SizingResult:
        downside_fraction = max(
            (sizing.current_price - sizing.risk_invalidation) / sizing.current_price,
            1e-6,
        )
        caps = {
            "risk_budget": sizing.max_loss_fraction / downside_fraction,
            "volatility_target": sizing.target_volatility / sizing.annualized_volatility,
            "asset_limit": sizing.asset_weight_limit,
            "sector_limit": max(0.0, sizing.sector_weight_limit - sizing.other_sector_weight),
            "cash_limit": min(
                1.0,
                sizing.current_weight + sizing.available_cash / sizing.account_value,
            ),
        }
        target_weight = round(max(0.0, min(caps.values())), 12)
        tolerance = 1e-12
        binding = tuple(
            name for name, value in caps.items() if abs(value - target_weight) <= tolerance
        )
        raw_quantity = target_weight * sizing.account_value / sizing.current_price
        target_quantity = math.floor(raw_quantity / sizing.lot_size) * sizing.lot_size
        realized_weight = target_quantity * sizing.current_price / sizing.account_value
        return SizingResult(
            target_weight=realized_weight,
            target_quantity=target_quantity,
            max_loss_amount=sizing.account_value * sizing.max_loss_fraction,
            binding_constraints=binding,
        )
