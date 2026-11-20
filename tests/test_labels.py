import polars as pl

from market_lab.labels import build_forward_labels, build_triple_barrier_labels


def price_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["A"] * 6,
            "session_date": [
                "2026-11-02",
                "2026-11-03",
                "2026-11-04",
                "2026-11-05",
                "2026-11-06",
                "2026-11-09",
            ],
            "close": [100.0, 103.0, 101.0, 106.0, 104.0, 110.0],
        }
    ).with_columns(pl.col("session_date").str.to_date())


def test_forward_labels_include_return_and_excursions() -> None:
    labels = build_forward_labels(price_frame(), horizons=(2,))
    first = labels.row(0, named=True)

    assert abs(first["forward_return_2"] - 0.01) < 1e-12
    assert abs(first["max_favorable_excursion_2"] - 0.03) < 1e-12
    assert first["max_adverse_excursion_2"] == 0.0
    assert first["label_end_2"].isoformat() == "2026-11-04"


def test_triple_barrier_stops_at_first_crossing() -> None:
    prices = price_frame().with_columns(pl.lit(0.02).alias("volatility"))
    labels = build_triple_barrier_labels(
        prices,
        volatility_column="volatility",
        horizon=3,
        upper=1.0,
        lower=1.0,
    )

    assert labels.row(0, named=True)["barrier_label"] == "up"
    assert labels.row(2, named=True)["barrier_label"] == "up"
