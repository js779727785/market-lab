from datetime import date, timedelta

import polars as pl

from market_lab.backtest.engine import BacktestEngine
from market_lab.backtest.splits import SplitConfig


class MeanSignalModel:
    key = "mean-signal"

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> list[float]:
        mean_target = train["target_return"].mean()
        return [float(mean_target)] * test.height


def dataset() -> pl.DataFrame:
    start = date(2026, 1, 1)
    dates = [start + timedelta(days=index) for index in range(14)]
    return pl.DataFrame(
        {
            "asset_id": ["A"] * 14,
            "session_date": dates,
            "label_end": [value + timedelta(days=2) for value in dates],
            "feature": [float(index) for index in range(14)],
            "target_return": [0.01 if index % 2 == 0 else -0.005 for index in range(14)],
            "tradable": [True] * 13 + [False],
        }
    )


def test_engine_emits_ordered_immutable_cases_and_skips_untradeable_rows() -> None:
    report = BacktestEngine(
        SplitConfig(min_train_periods=5, test_periods=2, step_periods=2, embargo_periods=0)
    ).run(MeanSignalModel(), dataset())

    assert report.model_key == "mean-signal"
    assert report.cases
    assert [case.session_date for case in report.cases] == sorted(
        case.session_date for case in report.cases
    )
    assert all(case.tradable for case in report.cases)
    assert len({(case.asset_id, case.session_date) for case in report.cases}) == len(report.cases)
