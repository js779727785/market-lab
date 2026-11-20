from datetime import UTC, datetime

import pytest

from market_lab.contracts import AssetClass
from market_lab.research.cases import AuditCaseCandidate, CaseSelector, TriggerRule
from market_lab.research.failures import FailureAnalyzer, FailureCase
from market_lab.research.promotion import ChallengerEvidence, PromotionWorkflow
from market_lab.research.registry import HypothesisRegistry

UTC = UTC


def test_hypothesis_is_registered_with_time_and_target_before_evaluation(tmp_path) -> None:
    registry = HypothesisRegistry(tmp_path / "hypotheses.jsonl")
    registered_at = datetime(2026, 1, 2, 8, tzinfo=UTC)

    hypothesis = registry.register(
        title="real-rate interaction",
        rationale="gold may respond nonlinearly when dollar and real rates diverge",
        target_metric="gold_brier_score",
        expected_direction="decrease",
        registered_at=registered_at,
    )

    assert hypothesis.registered_at == registered_at
    assert hypothesis.target_metric == "gold_brier_score"
    assert hypothesis.evaluated_at is None
    assert registry.history()[0].status == "registered"

    with pytest.raises(ValueError):
        registry.record_evaluation(
            hypothesis.hypothesis_id,
            evaluated_at=registered_at,
            result={"delta": -0.01},
        )


def test_case_selection_is_objective_and_keeps_successes_and_failures() -> None:
    rule = TriggerRule(
        rule_id="large-move",
        metric="absolute_return",
        threshold=0.05,
        registered_at=datetime(2023, 1, 1, tzinfo=UTC),
    )
    candidates = [
        AuditCaseCandidate(
            case_id="success",
            asset_id="AU9999.SGE",
            asset_class=AssetClass.GOLD,
            forecast_cutoff=datetime(2024, 5, 1, tzinfo=UTC),
            rule=rule,
            trigger_value=0.08,
            predicted_return=0.04,
            realized_return=0.06,
            snapshot_hash="a" * 64,
        ),
        AuditCaseCandidate(
            case_id="failure",
            asset_id="600000.SH",
            asset_class=AssetClass.A_SHARE,
            forecast_cutoff=datetime(2025, 8, 1, tzinfo=UTC),
            rule=rule,
            trigger_value=-0.07,
            predicted_return=0.03,
            realized_return=-0.05,
            snapshot_hash="b" * 64,
        ),
        AuditCaseCandidate(
            case_id="outside",
            asset_id="AU9999.SGE",
            asset_class=AssetClass.GOLD,
            forecast_cutoff=datetime(2023, 11, 19, tzinfo=UTC),
            rule=rule,
            trigger_value=0.2,
            predicted_return=0.1,
            realized_return=0.1,
            snapshot_hash="c" * 64,
        ),
        AuditCaseCandidate(
            case_id="below-threshold",
            asset_id="AU9999.SGE",
            asset_class=AssetClass.GOLD,
            forecast_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
            rule=rule,
            trigger_value=0.01,
            predicted_return=0.01,
            realized_return=0.01,
            snapshot_hash="d" * 64,
        ),
    ]

    selected = CaseSelector().select(candidates)

    assert [case.case_id for case in selected] == ["success", "failure"]
    assert {case.outcome for case in selected} == {"success", "failure"}
    assert all(
        datetime(2023, 11, 20, tzinfo=UTC)
        <= case.forecast_cutoff
        <= datetime(2026, 11, 20, 23, 59, 59, tzinfo=UTC)
        for case in selected
    )


def test_failure_analyzer_clusters_without_dropping_case_ids() -> None:
    failures = [
        FailureCase(
            case_id="a",
            asset_class=AssetClass.GOLD,
            regime="usd_shock",
            error_type="wrong_direction",
            event_type="tariff",
            predicted_return=0.03,
            realized_return=-0.04,
        ),
        FailureCase(
            case_id="b",
            asset_class=AssetClass.GOLD,
            regime="usd_shock",
            error_type="wrong_direction",
            event_type="tariff",
            predicted_return=0.02,
            realized_return=-0.03,
        ),
    ]

    clusters = FailureAnalyzer().cluster(failures)

    assert len(clusters) == 1
    assert clusters[0].case_ids == ("a", "b")
    assert clusters[0].count == 2
    assert clusters[0].mean_absolute_error == pytest.approx(0.06)


def test_promotion_review_requires_frozen_shadow_and_never_overwrites_champion(tmp_path) -> None:
    workflow = PromotionWorkflow(
        tmp_path / "promotions.jsonl",
        champion_model_key="linear-v1",
        min_shadow_samples=3,
    )
    start = datetime(2026, 10, 1, tzinfo=UTC)
    end = datetime(2026, 11, 1, tzinfo=UTC)
    weak = ChallengerEvidence(
        challenger_model_key="tree-v2",
        hypothesis_id="hyp-1",
        shadow_started_at=start,
        shadow_ended_at=end,
        frozen_forecast_ids=("f1",),
        gates_passed=True,
        evaluation_hash="e" * 64,
    )
    strong = weak.model_copy(update={"frozen_forecast_ids": ("f1", "f2", "f3")})

    rejected = workflow.review(weak, reviewed_at=end)
    approved = workflow.review(strong, reviewed_at=end)

    assert not rejected.approved
    assert "shadow_sample_count" in rejected.reasons
    assert approved.approved
    assert approved.recommended_champion_model_key == "tree-v2"
    assert workflow.champion_model_key == "linear-v1"
    history = workflow.history()
    assert len(history) == 2
    assert history[1].previous_record_hash == history[0].record_hash
