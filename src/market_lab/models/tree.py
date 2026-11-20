from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import polars as pl

from .baselines import _matrix, _target


@dataclass
class LightGBMCandidate:
    features: tuple[str, ...]
    seed: int = 20261120
    n_estimators: int = 100
    learning_rate: float = 0.03
    max_depth: int = 3
    num_leaves: int = 7
    key: str = "lightgbm_challenger"
    _model: lgb.LGBMRegressor | None = field(default=None, init=False, repr=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.features

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        if not self.features:
            raise ValueError("LightGBM candidate requires at least one feature")
        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=self.seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
        )
        model.fit(_matrix(train, self.features), _target(train), feature_name=list(self.features))
        self._model = model
        return [float(value) for value in model.predict(_matrix(test, self.features))]

    def artifact_state(self) -> dict[str, Any]:
        fitted_state: dict[str, Any] = {}
        if self._model is not None:
            fitted_state = {"booster": self._model.booster_.model_to_string()}
        return {
            "parameters": {
                "features": list(self.features),
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "n_estimators": self.n_estimators,
                "num_leaves": self.num_leaves,
                "seed": self.seed,
            },
            "fitted_state": fitted_state,
        }
