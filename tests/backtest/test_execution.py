from market_lab.backtest.costs import TradingRules
from market_lab.backtest.execution import (
    ExecutionSimulator,
    MarketState,
    Order,
    OrderSide,
)


def rules() -> TradingRules:
    return TradingRules(
        commission_rate=0.0003,
        minimum_commission=5.0,
        stamp_duty_rate=0.0005,
        slippage_bps=10.0,
        lot_size=100,
    )


def market(**updates) -> MarketState:
    payload = {
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.0,
        "volume": 1_000_000,
        "suspended": False,
        "limit_up": 11.0,
        "limit_down": 9.0,
    }
    payload.update(updates)
    return MarketState(**payload)


def test_suspended_and_one_price_limits_do_not_fill() -> None:
    simulator = ExecutionSimulator()

    suspended = simulator.fill(
        Order(side=OrderSide.BUY, quantity=1000),
        market(suspended=True),
        rules(),
    )
    limit_buy = simulator.fill(
        Order(side=OrderSide.BUY, quantity=1000),
        market(open=11.0, high=11.0, low=11.0, close=11.0),
        rules(),
    )
    limit_sell = simulator.fill(
        Order(side=OrderSide.SELL, quantity=1000),
        market(open=9.0, high=9.0, low=9.0, close=9.0),
        rules(),
    )

    assert suspended.filled is False and suspended.reason == "suspended"
    assert limit_buy.filled is False and limit_buy.reason == "limit_up_locked"
    assert limit_sell.filled is False and limit_sell.reason == "limit_down_locked"


def test_fill_applies_lot_size_slippage_commission_and_sell_tax() -> None:
    simulator = ExecutionSimulator()
    buy = simulator.fill(Order(side=OrderSide.BUY, quantity=1050), market(), rules())
    sell = simulator.fill(Order(side=OrderSide.SELL, quantity=1050), market(), rules())

    assert buy.filled_quantity == 1000
    assert buy.price == 10.01
    assert buy.commission == 5.0
    assert buy.tax == 0.0
    assert sell.price == 9.99
    assert sell.tax == 4.995
    assert sell.total_cost > buy.total_cost


def test_double_cost_stress_doubles_variable_cost_parameters() -> None:
    stressed = rules().stressed(2.0)

    assert stressed.commission_rate == 0.0006
    assert stressed.stamp_duty_rate == 0.001
    assert stressed.slippage_bps == 20.0
    assert stressed.minimum_commission == 5.0
