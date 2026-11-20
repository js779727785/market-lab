from datetime import date

import pytest

from market_lab.evaluation.metrics import EvaluationCase, evaluate_predictions


def cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            session_date=date(2026, 11, 2),
            asset_id="A",
            prediction=0.04,
            actual_return=0.03,
            probability_up=0.8,
            interval=(0.01, 0.05),
            net_return=0.028,
            stressed_net_return=0.026,
            turnover=0.5,
            regime="trend",
        ),
        EvaluationCase(
            session_date=date(2026, 11, 2),
            asset_id="B",
            prediction=-0.03,
            actual_return=-0.02,
            probability_up=0.2,
            interval=(-0.04, 0.0),
            net_return=0.018,
            stressed_net_return=0.016,
            turnover=0.4,
            regime="trend",
        ),
        EvaluationCase(
            session_date=date(2026, 11, 3),
            asset_id="A",
            prediction=0.02,
            actual_return=0.015,
            probability_up=0.75,
            interval=(0.0, 0.03),
            net_return=0.013,
            stressed_net_return=0.011,
            turnover=0.3,
            regime="range",
        ),
        EvaluationCase(
            session_date=date(2026, 11, 3),
            asset_id="B",
            prediction=-0.01,
            actual_return=-0.005,
            probability_up=0.35,
            interval=(-0.02, 0.01),
            net_return=0.003,
            stressed_net_return=0.001,
            turnover=0.2,
            regime="range",
        ),
        EvaluationCase(
            session_date=date(2026, 11, 4),
            asset_id="A",
            prediction=0.01,
            actual_return=0.008,
            probability_up=0.55,
            interval=(-0.01, 0.02),
            net_return=0.006,
            stressed_net_return=0.004,
            turnover=0.1,
            regime="range",
        ),
        EvaluationCase(
            session_date=date(2026, 11, 4),
            asset_id="B",
            prediction=-0.02,
            actual_return=-0.01,
            probability_up=0.25,
            interval=(-0.03, 0.0),
            net_return=0.008,
            stressed_net_return=0.006,
            turnover=0.25,
            regime="trend",
        ),
    ]


def test_evaluation_reports_probability_rank_risk_and_action_coverage() -> None:
    report = evaluate_predictions(cases(), action_threshold=0.7)

    assert report.sample_count == 6
    assert report.rank_ic == pytest.approx(1.0)
    assert report.brier_score < 0.12
    assert report.log_loss < 0.4
    assert 0 <= report.calibration_error <= 1
    assert report.cumulative_net_return > 0
    assert report.stressed_cumulative_net_return > 0
    assert report.max_drawdown <= 0
    assert report.turnover == pytest.approx(1.75)
    assert report.action_accuracy == pytest.approx(1.0)
    assert report.action_coverage == pytest.approx(4 / 6)
    assert report.interval_coverage == pytest.approx(1.0)
    assert report.interval_mean_width > 0
    assert set(report.regime_net_returns) == {"range", "trend"}


def test_accuracy_coverage_curve_never_hides_declining_coverage() -> None:
    report = evaluate_predictions(cases())

    assert report.accuracy_coverage_curve
    coverages = [point.coverage for point in report.accuracy_coverage_curve]
    assert coverages == sorted(coverages, reverse=True)
    assert any(point.coverage < 1.0 for point in report.accuracy_coverage_curve)
