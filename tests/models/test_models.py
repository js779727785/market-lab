import polars as pl

from market_lab.models.baselines import MomentumBaseline, RandomBaseline, RegularizedLinearModel
from market_lab.models.registry import ModelRegistry


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["A", "B", "A", "B", "A", "B"],
            "session_date": [
                "2026-11-02",
                "2026-11-02",
                "2026-11-03",
                "2026-11-03",
                "2026-11-04",
                "2026-11-04",
            ],
            "return_5": [0.02, -0.01, 0.03, -0.02, 0.01, 0.0],
            "volatility_5": [0.01, 0.02, 0.011, 0.022, 0.012, 0.018],
            "target_return": [0.01, -0.005, 0.012, -0.006, 0.008, 0.001],
        }
    ).with_columns(pl.col("session_date").str.to_date())


def test_random_baseline_is_deterministic_by_asset_and_date() -> None:
    model = RandomBaseline(seed=20261120)
    train = frame().head(4)
    test = frame().tail(2)

    assert model.fit_predict(train, test) == model.fit_predict(train, test)


def test_momentum_and_regularized_models_use_declared_features() -> None:
    train = frame().head(4)
    test = frame().tail(2)
    momentum = MomentumBaseline(feature="return_5")
    linear = RegularizedLinearModel(features=("return_5", "volatility_5"), alpha=1.0)

    assert momentum.fit_predict(train, test) == [0.01, 0.0]
    predictions = linear.fit_predict(train, test)
    assert len(predictions) == 2
    assert all(isinstance(value, float) for value in predictions)


def test_registry_serializes_reproducible_model_artifacts() -> None:
    registry = ModelRegistry()
    registry.register("linear", lambda: RegularizedLinearModel(features=("return_5",)))
    train = frame().head(4)
    test = frame().tail(2)

    first = registry.fit_predict("linear", train, test)
    second = registry.fit_predict("linear", train, test)

    assert first.predictions == second.predictions
    assert first.artifact.sha256 == second.artifact.sha256
    assert first.artifact.feature_names == ("return_5",)
