from market_lab.abstention import AbstentionPolicy, DataStatus, DriftStatus, PredictionContext
from market_lab.contracts import ActionState
from market_lab.evaluation.selection import (
    ChampionSelector,
    ModelEvaluation,
    PromotionGates,
)


def evaluation(
    key: str,
    samples: tuple[float, ...],
    *,
    stressed_return: float = 0.1,
    calibration_error: float = 0.04,
    regimes: dict[str, float] | None = None,
    shadow_samples: int = 50,
) -> ModelEvaluation:
    return ModelEvaluation(
        model_key=key,
        primary_samples=samples,
        net_return=0.12,
        stressed_net_return=stressed_return,
        calibration_error=calibration_error,
        max_drawdown=-0.08,
        tail_loss=-0.03,
        regime_scores=regimes or {"bull": 0.2, "bear": 0.2},
        shadow_samples=shadow_samples,
        audit_passed=True,
    )


def test_selector_promotes_only_challenger_that_passes_every_gate() -> None:
    champion = evaluation(
        "champion",
        (0.1,) * 20,
        regimes={"bull": 0.1, "bear": 0.1},
    )
    strong = evaluation("strong", (0.2,) * 20)
    weak_bootstrap = evaluation("weak-bootstrap", (0.11,) * 20)
    weak_cost = evaluation("weak-cost", (0.2,) * 20, stressed_return=-0.01)
    weak_calibration = evaluation("weak-calibration", (0.2,) * 20, calibration_error=0.12)
    weak_stability = evaluation(
        "weak-stability",
        (0.2,) * 20,
        regimes={"bull": 0.2, "bear": 0.08},
    )
    weak_shadow = evaluation("weak-shadow", (0.2,) * 20, shadow_samples=5)
    gates = PromotionGates(
        min_bootstrap_lower_bound=0.02,
        max_calibration_degradation=0.02,
        min_regime_win_fraction=0.75,
        min_shadow_samples=30,
        bootstrap_samples=200,
        block_size=5,
    )

    decision = ChampionSelector(seed=20261120).select(
        champion,
        [strong, weak_bootstrap, weak_cost, weak_calibration, weak_stability, weak_shadow],
        gates,
    )

    assert decision.promoted
    assert decision.selected_model_key == "strong"
    assert decision.comparisons["strong"].passed
    assert "bootstrap_lower_bound" in decision.comparisons["weak-bootstrap"].reasons
    assert "double_cost_economic_value" in decision.comparisons["weak-cost"].reasons
    assert "calibration_degradation" in decision.comparisons["weak-calibration"].reasons
    assert "cross_regime_stability" in decision.comparisons["weak-stability"].reasons
    assert "shadow_sample_count" in decision.comparisons["weak-shadow"].reasons
    assert all(0 <= comparison.fdr_q_value <= 1 for comparison in decision.comparisons.values())


def test_abstention_exposes_reasons_and_blocks_unverifiable_actions() -> None:
    policy = AbstentionPolicy()
    prediction = PredictionContext(
        probability_up=0.76,
        expected_return=0.03,
        return_interval=(0.01, 0.05),
        transaction_cost=0.004,
        safety_margin=0.004,
        model_disagreement=0.05,
    )

    action = policy.decide(prediction, DataStatus(), DriftStatus())
    assert action is ActionState.BUY
    assert policy.last_result is not None
    assert policy.last_result.reasons == ()

    blocked = policy.decide(
        prediction,
        DataStatus(quality_passed=False, source_conflict=True),
        DriftStatus(calibration_passed=False),
    )
    assert blocked is ActionState.ABSTAIN
    assert policy.last_result is not None
    assert set(policy.last_result.reasons) == {
        "calibration_failed",
        "data_quality_failed",
        "source_conflict",
    }
