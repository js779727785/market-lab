from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import benjamini_hochberg, block_bootstrap_difference


class ModelEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_key: str = Field(min_length=1)
    primary_samples: tuple[float, ...]
    net_return: float
    stressed_net_return: float
    calibration_error: float = Field(ge=0)
    max_drawdown: float = Field(le=0)
    tail_loss: float = Field(le=0)
    regime_scores: dict[str, float]
    shadow_samples: int = Field(ge=0)
    audit_passed: bool


class PromotionGates(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_bootstrap_lower_bound: float = 0.0
    min_net_return: float = 0.0
    min_stressed_net_return: float = 0.0
    max_calibration_degradation: float = 0.0
    max_drawdown_degradation: float = 0.0
    max_tail_loss_degradation: float = 0.0
    min_regime_win_fraction: float = Field(default=0.6, ge=0, le=1)
    min_shadow_samples: int = Field(default=30, ge=0)
    bootstrap_samples: int = Field(default=1000, gt=0)
    block_size: int = Field(default=20, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    max_fdr_q_value: float = Field(default=0.05, ge=0, le=1)


class ChallengerComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_key: str
    passed: bool
    reasons: tuple[str, ...]
    bootstrap_estimate: float
    bootstrap_lower_bound: float
    bootstrap_upper_bound: float
    p_value: float
    fdr_q_value: float
    regime_win_fraction: float


class SelectionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    incumbent_model_key: str
    selected_model_key: str
    promoted: bool
    comparisons: dict[str, ChallengerComparison]


class ChampionSelector:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def select(
        self,
        champion: ModelEvaluation,
        challengers: Sequence[ModelEvaluation],
        gates: PromotionGates,
    ) -> SelectionDecision:
        if not champion.primary_samples:
            raise ValueError("champion requires primary metric samples")
        bootstrap_results = {
            challenger.model_key: block_bootstrap_difference(
                champion.primary_samples,
                challenger.primary_samples,
                block_size=gates.block_size,
                samples=gates.bootstrap_samples,
                confidence_level=gates.confidence_level,
                seed=self.seed + index,
            )
            for index, challenger in enumerate(challengers)
        }
        q_values = benjamini_hochberg(
            {key: result.p_value for key, result in bootstrap_results.items()}
        )
        comparisons: dict[str, ChallengerComparison] = {}
        passing: list[tuple[float, str]] = []
        for challenger in challengers:
            result = bootstrap_results[challenger.model_key]
            shared_regimes = sorted(set(champion.regime_scores) & set(challenger.regime_scores))
            regime_win_fraction = (
                sum(
                    challenger.regime_scores[regime] > champion.regime_scores[regime]
                    for regime in shared_regimes
                )
                / len(shared_regimes)
                if shared_regimes
                else 0.0
            )
            reasons: list[str] = []
            if not challenger.audit_passed:
                reasons.append("data_or_leakage_audit")
            if result.lower_bound < gates.min_bootstrap_lower_bound:
                reasons.append("bootstrap_lower_bound")
            if q_values[challenger.model_key] > gates.max_fdr_q_value:
                reasons.append("false_discovery_rate")
            if challenger.net_return <= gates.min_net_return:
                reasons.append("base_cost_economic_value")
            if challenger.stressed_net_return <= gates.min_stressed_net_return:
                reasons.append("double_cost_economic_value")
            if (
                challenger.calibration_error
                > champion.calibration_error + gates.max_calibration_degradation
            ):
                reasons.append("calibration_degradation")
            if (
                challenger.max_drawdown < champion.max_drawdown - gates.max_drawdown_degradation
                or challenger.tail_loss < champion.tail_loss - gates.max_tail_loss_degradation
            ):
                reasons.append("tail_risk_degradation")
            if regime_win_fraction < gates.min_regime_win_fraction:
                reasons.append("cross_regime_stability")
            if challenger.shadow_samples < gates.min_shadow_samples:
                reasons.append("shadow_sample_count")
            comparison = ChallengerComparison(
                model_key=challenger.model_key,
                passed=not reasons,
                reasons=tuple(reasons),
                bootstrap_estimate=result.estimate,
                bootstrap_lower_bound=result.lower_bound,
                bootstrap_upper_bound=result.upper_bound,
                p_value=result.p_value,
                fdr_q_value=q_values[challenger.model_key],
                regime_win_fraction=regime_win_fraction,
            )
            comparisons[challenger.model_key] = comparison
            if comparison.passed:
                passing.append((result.estimate, challenger.model_key))
        selected = max(passing)[1] if passing else champion.model_key
        return SelectionDecision(
            incumbent_model_key=champion.model_key,
            selected_model_key=selected,
            promoted=selected != champion.model_key,
            comparisons=comparisons,
        )
