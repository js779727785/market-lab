from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _paired_floats(
    left: Sequence[float], right: Sequence[float], left_name: str, right_name: str
) -> tuple[np.ndarray, np.ndarray]:
    if not left or not right:
        raise ValueError(f"{left_name} and {right_name} must be non-empty")
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same length")
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


@dataclass
class ProbabilityCalibrator:
    _model: IsotonicRegression | None = field(default=None, init=False, repr=False)
    _constant_probability: float | None = field(default=None, init=False, repr=False)
    fitted_count: int = field(default=0, init=False)

    def fit(self, raw_scores: Sequence[float], labels: Sequence[int]) -> Self:
        scores, outcomes = _paired_floats(raw_scores, labels, "raw_scores", "labels")
        if not set(outcomes.tolist()).issubset({0.0, 1.0}):
            raise ValueError("probability calibration labels must be binary")
        self.fitted_count = len(scores)
        if len(np.unique(scores)) < 2 or len(np.unique(outcomes)) < 2:
            self._constant_probability = float(outcomes.mean())
            self._model = None
            return self
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(scores, outcomes)
        self._model = model
        self._constant_probability = None
        return self

    def predict(self, raw_scores: Sequence[float]) -> list[float]:
        if self.fitted_count == 0:
            raise ValueError("probability calibrator has not been fitted")
        if self._constant_probability is not None:
            return [self._constant_probability for _ in raw_scores]
        if self._model is None:
            raise RuntimeError("probability calibrator is in an invalid fitted state")
        return [float(value) for value in self._model.predict(np.asarray(raw_scores, dtype=float))]


@dataclass
class ResidualIntervalCalibrator:
    alpha: float = 0.1
    fitted_count: int = field(default=0, init=False)
    _radius: float | None = field(default=None, init=False, repr=False)

    def fit(self, predictions: Sequence[float], actuals: Sequence[float]) -> Self:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")
        forecast, outcome = _paired_floats(predictions, actuals, "predictions", "actuals")
        residuals = np.abs(outcome - forecast)
        rank = min(len(residuals), math.ceil((len(residuals) + 1) * (1 - self.alpha)))
        self._radius = float(np.partition(residuals, rank - 1)[rank - 1])
        self.fitted_count = len(residuals)
        return self

    def interval(self, predictions: Sequence[float]) -> tuple[list[float], list[float]]:
        if self._radius is None:
            raise ValueError("residual interval calibrator has not been fitted")
        lower = [float(value) - self._radius for value in predictions]
        upper = [float(value) + self._radius for value in predictions]
        return lower, upper
