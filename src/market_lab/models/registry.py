from __future__ import annotations

from collections.abc import Callable

import polars as pl

from .base import CandidateModel, ModelArtifact, ModelRun


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], CandidateModel]] = {}

    def register(self, key: str, factory: Callable[[], CandidateModel]) -> None:
        if not key:
            raise ValueError("model key must not be empty")
        if key in self._factories:
            raise ValueError(f"model key already registered: {key}")
        self._factories[key] = factory

    def fit_predict(self, key: str, train: pl.DataFrame, test: pl.DataFrame) -> ModelRun:
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise KeyError(f"unknown model key: {key}") from exc
        model = factory()
        predictions = tuple(float(value) for value in model.fit_predict(train, test))
        if len(predictions) != test.height:
            raise ValueError("model returned a prediction count different from the test rows")
        state = model.artifact_state()
        artifact = ModelArtifact(
            model_key=key,
            model_type=f"{type(model).__module__}.{type(model).__qualname__}",
            feature_names=model.feature_names,
            parameters=state["parameters"],
            fitted_state=state["fitted_state"],
        )
        return ModelRun(predictions=predictions, artifact=artifact)
