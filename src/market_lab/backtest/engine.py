from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict

from .costs import BacktestExecutionConfig
from .splits import PurgedWalkForward, SplitConfig


class BacktestModel(Protocol):
    key: str

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> Sequence[float]: ...


class BacktestCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold_id: int
    asset_id: str
    session_date: date
    prediction: float
    target_return: float
    strategy_return: float
    net_return: float
    stressed_net_return: float
    tradable: bool


class BacktestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_key: str
    cases: tuple[BacktestCase, ...]


class BacktestEngine:
    def __init__(self, split_config: SplitConfig) -> None:
        self.split_config = split_config

    def run(
        self,
        model_spec: BacktestModel,
        dataset: pl.DataFrame,
        execution_config: BacktestExecutionConfig | None = None,
    ) -> BacktestReport:
        config = execution_config or BacktestExecutionConfig()
        ordered = dataset.sort(["session_date", "asset_id"]).with_row_index("_row")
        index = ordered["session_date"].to_list()
        label_end = ordered["label_end"].to_list()
        splitter = PurgedWalkForward(self.split_config)
        cases: list[BacktestCase] = []
        seen: set[tuple[str, date]] = set()
        for fold in splitter.split(index, label_end):
            train = ordered.filter(pl.col("_row").is_in(fold.train_indices)).drop("_row")
            test = ordered.filter(pl.col("_row").is_in(fold.test_indices)).drop("_row")
            predictions = list(model_spec.fit_predict(train, test))
            if len(predictions) != test.height:
                raise ValueError("model returned a prediction count different from the test rows")
            for row, prediction in zip(test.to_dicts(), predictions, strict=True):
                if not bool(row.get("tradable", True)):
                    continue
                key = (str(row["asset_id"]), row["session_date"])
                if key in seen:
                    continue
                seen.add(key)
                target = float(row["target_return"])
                direction = 1.0 if prediction > 0 else -1.0 if prediction < 0 else 0.0
                strategy_return = direction * target
                cost = config.one_way_cost_rate if direction else 0.0
                cases.append(
                    BacktestCase(
                        fold_id=fold.fold_id,
                        asset_id=key[0],
                        session_date=key[1],
                        prediction=float(prediction),
                        target_return=target,
                        strategy_return=strategy_return,
                        net_return=strategy_return - cost,
                        stressed_net_return=strategy_return - cost * config.stress_multiplier,
                        tradable=True,
                    )
                )
        return BacktestReport(
            model_key=model_spec.key,
            cases=tuple(sorted(cases, key=lambda case: (case.session_date, case.asset_id))),
        )
