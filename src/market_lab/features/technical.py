from __future__ import annotations

import polars as pl


def add_technical_features(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.sort(["asset_id", "session_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("asset_id") - 1.0).alias("return_1"),
            (pl.col("close") / pl.col("close").shift(5).over("asset_id") - 1.0).alias("return_5"),
            (
                pl.col("close") / pl.col("close").rolling_mean(window_size=5).over("asset_id") - 1.0
            ).alias("ma5_ratio"),
        )
        .with_columns(
            pl.col("return_1")
            .rolling_std(window_size=5, ddof=1)
            .over("asset_id")
            .alias("volatility_5"),
            (
                (pl.col("volume") - pl.col("volume").rolling_mean(20).over("asset_id"))
                / pl.col("volume").rolling_std(20, ddof=1).over("asset_id")
            ).alias("volume_zscore_20"),
        )
    )
