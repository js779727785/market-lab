from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = tuple(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError(f"missing required model columns: {missing}")


def _target(frame: pl.DataFrame) -> np.ndarray:
    _require_columns(frame, ("target_return",))
    return np.asarray(frame["target_return"].to_list(), dtype=float)


def _matrix(frame: pl.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    _require_columns(frame, features)
    return np.asarray(frame.select(features).to_numpy(), dtype=float)


@dataclass
class RandomBaseline:
    seed: int = 0
    key: str = "random"
    feature_names: tuple[str, ...] = ()

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        del train
        _require_columns(test, ("asset_id", "session_date"))
        predictions: list[float] = []
        for asset_id, session_date in test.select("asset_id", "session_date").iter_rows():
            digest = hashlib.sha256(
                f"{self.seed}|{asset_id}|{session_date.isoformat()}".encode()
            ).digest()
            unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            predictions.append(unit * 2.0 - 1.0)
        return predictions

    def artifact_state(self) -> dict[str, Any]:
        return {"parameters": {"seed": self.seed}, "fitted_state": {}}


@dataclass
class MomentumBaseline:
    feature: str = "return_5"
    key: str = "momentum"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (self.feature,)

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        del train
        _require_columns(test, self.feature_names)
        return [float(value) for value in test[self.feature].to_list()]

    def artifact_state(self) -> dict[str, Any]:
        return {"parameters": {"feature": self.feature}, "fitted_state": {}}


@dataclass
class ReversalBaseline(MomentumBaseline):
    key: str = "reversal"

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        return [-value for value in super().fit_predict(train, test)]


@dataclass
class MarketIndustryBaseline:
    industry_column: str = "industry"
    target_column: str = "target_return"
    key: str = "market_industry"
    _market_mean: float = field(default=0.0, init=False, repr=False)
    _industry_means: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (self.industry_column,)

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        _require_columns(train, (self.industry_column, self.target_column))
        _require_columns(test, (self.industry_column,))
        market_mean = train[self.target_column].mean()
        self._market_mean = 0.0 if market_mean is None else float(cast(int | float, market_mean))
        grouped = train.group_by(self.industry_column).agg(pl.col(self.target_column).mean())
        self._industry_means = {
            str(industry): float(cast(int | float, value))
            for industry, value in grouped.iter_rows()
            if industry is not None and value is not None
        }
        return [
            self._industry_means.get(str(industry), self._market_mean)
            for industry in test[self.industry_column].to_list()
        ]

    def artifact_state(self) -> dict[str, Any]:
        return {
            "parameters": {
                "industry_column": self.industry_column,
                "target_column": self.target_column,
            },
            "fitted_state": {
                "industry_means": dict(sorted(self._industry_means.items())),
                "market_mean": self._market_mean,
            },
        }


@dataclass
class RegularizedLinearModel:
    features: tuple[str, ...]
    alpha: float = 1.0
    key: str = "regularized_linear"
    _pipeline: Pipeline | None = field(default=None, init=False, repr=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.features

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        if not self.features:
            raise ValueError("regularized linear model requires at least one feature")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        pipeline.fit(_matrix(train, self.features), _target(train))
        self._pipeline = pipeline
        return [float(value) for value in pipeline.predict(_matrix(test, self.features))]

    def artifact_state(self) -> dict[str, Any]:
        fitted_state: dict[str, Any] = {}
        if self._pipeline is not None:
            scaler = self._pipeline.named_steps["scale"]
            ridge = self._pipeline.named_steps["ridge"]
            fitted_state = {
                "coefficient": [float(value) for value in ridge.coef_.tolist()],
                "intercept": float(ridge.intercept_),
                "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
                "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
            }
        return {
            "parameters": {"alpha": self.alpha, "features": list(self.features)},
            "fitted_state": fitted_state,
        }
