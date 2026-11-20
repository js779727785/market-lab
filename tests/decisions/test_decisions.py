from datetime import UTC, datetime

import pytest

from market_lab.contracts import ActionState, AssetClass, PositionInput
from market_lab.decisions.assistant import HoldingPrediction, PositionAssistant
from market_lab.decisions.levels import CalibratedPrediction, DecisionContext, DecisionEngine
from market_lab.decisions.portfolio import PortfolioPosition, PortfolioState

UTC = UTC


def calibrated(asset_class: AssetClass = AssetClass.A_SHARE) -> CalibratedPrediction:
    return CalibratedPrediction(
        asset_id="600000.SH" if asset_class is AssetClass.A_SHARE else "518880.SH",
        asset_class=asset_class,
        forecast_cutoff=datetime(2026, 11, 20, 8, tzinfo=UTC),
        horizon_days=5,
        probability_up=0.76,
        expected_return=0.04,
        return_interval=(0.01, 0.1),
        quantile_returns={
            0.05: -0.08,
            0.2: -0.03,
            0.35: -0.01,
            0.6: 0.02,
            0.75: 0.06,
            0.9: 0.1,
        },
        max_adverse_excursion_probability=0.18,
        model_disagreement=0.04,
        supporting_evidence=("positive calibrated edge",),
        counterevidence=("macro regime may reverse",),
    )


def test_decision_engine_builds_prices_from_quantiles_and_execution_constraints() -> None:
    decision = DecisionEngine().for_candidate(
        calibrated(),
        DecisionContext(
            current_price=100.0,
            price_tick=0.01,
            transaction_cost=0.004,
            safety_margin=0.004,
            valid_days=5,
        ),
    )

    assert decision.action is ActionState.BUY
    assert decision.buy_zone == pytest.approx((97.0, 99.0))
    assert decision.chase_invalidation == pytest.approx(102.0)
    assert decision.risk_invalidation == pytest.approx(92.0)
    assert decision.reduce_zone == pytest.approx((106.0, 110.0))
    assert decision.valid_until == datetime(2026, 11, 25, 8, tzinfo=UTC)
    assert decision.reasons == ("calibrated_quantile_levels", "positive calibrated edge")
    assert decision.counterevidence == ("macro regime may reverse",)


def test_position_assistant_sizes_by_risk_and_respects_asset_and_sector_limits() -> None:
    decision = DecisionEngine().for_candidate(
        calibrated(),
        DecisionContext(
            current_price=100.0,
            transaction_cost=0.004,
            safety_margin=0.004,
        ),
    )
    portfolio = PortfolioState(
        account_value=1_000_000,
        cash=500_000,
        positions=(
            PortfolioPosition(
                asset_id="000001.SZ",
                market_value=250_000,
                sector="bank",
            ),
        ),
    )
    position = PositionInput(
        asset_id="600000.SH",
        asset_class=AssetClass.A_SHARE,
        average_cost=95.0,
        quantity=100,
        available_cash=500_000,
        horizon_days=20,
        max_loss_fraction=0.02,
        sector_limit=0.3,
    )
    prediction = HoldingPrediction(
        decision=decision,
        current_price=100.0,
        annualized_volatility=0.25,
        sector="bank",
        asset_weight_limit=0.15,
        target_volatility=0.1,
    )

    advice = PositionAssistant().evaluate(position, prediction, portfolio)

    assert advice.action is ActionState.ADD
    assert advice.target_weight == pytest.approx(0.05)
    assert advice.target_quantity == 500
    assert advice.max_loss_amount == pytest.approx(20_000)
    assert "sector_limit" in advice.binding_constraints


def test_missing_risk_budget_returns_research_only_without_target_position() -> None:
    decision = DecisionEngine().for_candidate(
        calibrated(),
        DecisionContext(current_price=100.0, transaction_cost=0.004, safety_margin=0.004),
    )
    position = PositionInput(
        asset_id="600000.SH",
        asset_class=AssetClass.A_SHARE,
        average_cost=95.0,
        quantity=100,
        available_cash=10_000,
        horizon_days=20,
        max_loss_fraction=None,
    )
    prediction = HoldingPrediction(
        decision=decision,
        current_price=100.0,
        annualized_volatility=0.25,
        sector="bank",
    )

    advice = PositionAssistant().evaluate(
        position,
        prediction,
        PortfolioState(account_value=50_000, cash=10_000),
    )

    assert advice.action is ActionState.HOLD
    assert advice.research_only
    assert advice.target_weight is None
    assert advice.target_quantity is None
    assert "missing_risk_budget" in advice.reasons


def test_gold_product_tracking_error_and_price_breach_force_abstain_or_exit() -> None:
    gold_decision = DecisionEngine().for_candidate(
        calibrated(AssetClass.GOLD_PRODUCT),
        DecisionContext(current_price=5.0, transaction_cost=0.002, safety_margin=0.002),
    )
    position = PositionInput(
        asset_id="518880.SH",
        asset_class=AssetClass.GOLD_PRODUCT,
        average_cost=4.8,
        quantity=1_000,
        available_cash=5_000,
        horizon_days=20,
        max_loss_fraction=0.02,
    )
    portfolio = PortfolioState(account_value=20_000, cash=5_000)
    assistant = PositionAssistant()

    tracking_blocked = assistant.evaluate(
        position,
        HoldingPrediction(
            decision=gold_decision,
            current_price=5.0,
            annualized_volatility=0.2,
            tracking_error=0.08,
            max_tracking_error=0.03,
        ),
        portfolio,
    )
    breached = assistant.evaluate(
        position,
        HoldingPrediction(
            decision=gold_decision,
            current_price=4.0,
            annualized_volatility=0.2,
            tracking_error=0.01,
            max_tracking_error=0.03,
        ),
        portfolio,
    )

    assert tracking_blocked.action is ActionState.ABSTAIN
    assert "tracking_error" in tracking_blocked.reasons
    assert breached.action is ActionState.EXIT
    assert "risk_invalidation_breached" in breached.reasons
