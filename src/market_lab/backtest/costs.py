from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TradingRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    commission_rate: float = Field(ge=0)
    minimum_commission: float = Field(ge=0)
    stamp_duty_rate: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    lot_size: int = Field(ge=1)

    def stressed(self, multiplier: float) -> TradingRules:
        if multiplier <= 0:
            raise ValueError("stress multiplier must be positive")
        return self.model_copy(
            update={
                "commission_rate": self.commission_rate * multiplier,
                "stamp_duty_rate": self.stamp_duty_rate * multiplier,
                "slippage_bps": self.slippage_bps * multiplier,
            }
        )


class BacktestExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    one_way_cost_rate: float = Field(default=0.001, ge=0)
    stress_multiplier: float = Field(default=2.0, ge=1)
