from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import spearmanr


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_date: date
    asset_id: str = Field(min_length=1)
    prediction: float
    actual_return: float
    probability_up: float = Field(ge=0, le=1)
    interval: tuple[float, float]
    net_return: float
    stressed_net_return: float
    turnover: float = Field(default=0.0, ge=0)
    regime: str = "unknown"

    @model_validator(mode="after")
    def validate_interval(self) -> EvaluationCase:
        if self.interval[0] > self.interval[1]:
            raise ValueError("interval must be ordered")
        return self


class AccuracyCoveragePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    accuracy: float | None
    coverage: float
    action_count: int


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_count: int
    rank_ic: float | None
    ic_ir: float | None
    brier_score: float
    log_loss: float
    calibration_error: float
    cumulative_net_return: float
    stressed_cumulative_net_return: float
    annualized_net_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    turnover: float
    action_accuracy: float | None
    action_coverage: float
    accuracy_coverage_curve: tuple[AccuracyCoveragePoint, ...]
    interval_coverage: float
    interval_mean_width: float
    monthly_positive_fraction: float | None
    yearly_positive_fraction: float | None
    regime_net_returns: dict[str, float]


def _rank_ic(cases: Sequence[EvaluationCase]) -> tuple[float | None, float | None]:
    grouped: dict[date, list[EvaluationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.session_date].append(case)
    daily: list[float] = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        predictions = [row.prediction for row in rows]
        actuals = [row.actual_return for row in rows]
        if len(set(predictions)) < 2 or len(set(actuals)) < 2:
            continue
        correlation = float(spearmanr(predictions, actuals).statistic)
        if math.isfinite(correlation):
            daily.append(correlation)
    if not daily:
        predictions = [case.prediction for case in cases]
        actuals = [case.actual_return for case in cases]
        if len(set(predictions)) < 2 or len(set(actuals)) < 2:
            return None, None
        correlation = float(spearmanr(predictions, actuals).statistic)
        return (correlation, None) if math.isfinite(correlation) else (None, None)
    mean_ic = float(np.mean(daily))
    standard_deviation = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    ic_ir = mean_ic / standard_deviation if standard_deviation > 0 else None
    return mean_ic, ic_ir


def _calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        count = int(mask.sum())
        if count:
            error += count / total * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return error


def _compounded(values: Sequence[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity - 1.0


def _daily_returns(cases: Sequence[EvaluationCase], field: str) -> list[float]:
    grouped: dict[date, list[float]] = defaultdict(list)
    for case in cases:
        grouped[case.session_date].append(float(getattr(case, field)))
    return [float(np.mean(grouped[session])) for session in sorted(grouped)]


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    return drawdown


def _period_positive_fraction(cases: Sequence[EvaluationCase], period: str) -> float | None:
    grouped: dict[tuple[int, int] | int, list[float]] = defaultdict(list)
    for case in cases:
        key: tuple[int, int] | int
        if period == "month":
            key = (case.session_date.year, case.session_date.month)
        else:
            key = case.session_date.year
        grouped[key].append(case.net_return)
    if not grouped:
        return None
    positive = sum(_compounded(values) > 0 for values in grouped.values())
    return positive / len(grouped)


def _action_metrics(
    cases: Sequence[EvaluationCase], threshold: float
) -> tuple[float | None, float, int]:
    selected = [
        case
        for case in cases
        if case.probability_up >= threshold or case.probability_up <= 1.0 - threshold
    ]
    if not selected:
        return None, 0.0, 0
    correct = sum(
        (case.probability_up >= threshold and case.actual_return > 0)
        or (case.probability_up <= 1.0 - threshold and case.actual_return < 0)
        for case in selected
    )
    return correct / len(selected), len(selected) / len(cases), len(selected)


def evaluate_predictions(
    cases: Sequence[EvaluationCase], action_threshold: float = 0.6
) -> EvaluationReport:
    if not cases:
        raise ValueError("evaluation requires at least one case")
    if not 0.5 <= action_threshold < 1:
        raise ValueError("action_threshold must be in [0.5, 1)")
    probabilities = np.asarray([case.probability_up for case in cases], dtype=float)
    labels = np.asarray([case.actual_return > 0 for case in cases], dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    rank_ic, ic_ir = _rank_ic(cases)
    daily = _daily_returns(cases, "net_return")
    stressed_daily = _daily_returns(cases, "stressed_net_return")
    standard_deviation = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    sharpe = (
        float(np.mean(daily) / standard_deviation * math.sqrt(252)) if standard_deviation else None
    )
    annualized = (
        (1.0 + _compounded(daily)) ** (252 / len(daily)) - 1.0
        if daily and 1.0 + _compounded(daily) > 0
        else -1.0
    )
    action_accuracy, action_coverage, _ = _action_metrics(cases, action_threshold)
    thresholds = tuple(round(value, 2) for value in np.arange(0.5, 0.96, 0.05))
    curve = tuple(
        AccuracyCoveragePoint(
            threshold=threshold,
            accuracy=accuracy,
            coverage=coverage,
            action_count=count,
        )
        for threshold in thresholds
        for accuracy, coverage, count in [_action_metrics(cases, threshold)]
    )
    regime_values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        regime_values[case.regime].append(case.net_return)
    interval_hits = sum(
        case.interval[0] <= case.actual_return <= case.interval[1] for case in cases
    )
    return EvaluationReport(
        sample_count=len(cases),
        rank_ic=rank_ic,
        ic_ir=ic_ir,
        brier_score=float(np.mean((probabilities - labels) ** 2)),
        log_loss=float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        calibration_error=_calibration_error(probabilities, labels),
        cumulative_net_return=_compounded(daily),
        stressed_cumulative_net_return=_compounded(stressed_daily),
        annualized_net_return=annualized,
        sharpe_ratio=sharpe,
        max_drawdown=_max_drawdown(daily),
        turnover=float(sum(case.turnover for case in cases)),
        action_accuracy=action_accuracy,
        action_coverage=action_coverage,
        accuracy_coverage_curve=curve,
        interval_coverage=interval_hits / len(cases),
        interval_mean_width=float(np.mean([case.interval[1] - case.interval[0] for case in cases])),
        monthly_positive_fraction=_period_positive_fraction(cases, "month"),
        yearly_positive_fraction=_period_positive_fraction(cases, "year"),
        regime_net_returns={
            regime: _compounded(values) for regime, values in sorted(regime_values.items())
        },
    )
