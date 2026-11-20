from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .costs import TradingRules


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    side: OrderSide
    quantity: int = Field(gt=0)


class MarketState(BaseModel):
    model_config = ConfigDict(frozen=True)

    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    suspended: bool
    limit_up: float | None = Field(default=None, gt=0)
    limit_down: float | None = Field(default=None, gt=0)


class FillResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    filled: bool
    reason: str
    filled_quantity: int = 0
    price: float | None = None
    notional: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    total_cost: float = 0.0


class ExecutionSimulator:
    def fill(
        self,
        order: Order,
        market: MarketState,
        rules: TradingRules,
    ) -> FillResult:
        if market.suspended or market.volume <= 0:
            return FillResult(filled=False, reason="suspended")
        if (
            order.side == OrderSide.BUY
            and market.limit_up is not None
            and market.low >= market.limit_up
            and market.high <= market.limit_up
        ):
            return FillResult(filled=False, reason="limit_up_locked")
        if (
            order.side == OrderSide.SELL
            and market.limit_down is not None
            and market.high <= market.limit_down
            and market.low >= market.limit_down
        ):
            return FillResult(filled=False, reason="limit_down_locked")
        quantity = order.quantity - order.quantity % rules.lot_size
        if quantity <= 0:
            return FillResult(filled=False, reason="below_lot_size")
        slippage = rules.slippage_bps / 10_000.0
        direction = 1.0 if order.side == OrderSide.BUY else -1.0
        price = round(market.close * (1.0 + direction * slippage), 6)
        notional = price * quantity
        commission = max(rules.minimum_commission, notional * rules.commission_rate)
        tax = notional * rules.stamp_duty_rate if order.side == OrderSide.SELL else 0.0
        return FillResult(
            filled=True,
            reason="filled",
            filled_quantity=quantity,
            price=price,
            notional=notional,
            commission=commission,
            tax=tax,
            total_cost=commission + tax,
        )
