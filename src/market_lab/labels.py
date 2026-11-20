from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def build_forward_labels(
    prices: pl.DataFrame,
    horizons: Sequence[int] = (1, 5, 20, 60),
) -> pl.DataFrame:
    if prices.is_empty():
        return prices
    result_rows: list[dict[str, object]] = []
    for group in prices.partition_by("asset_id", maintain_order=True):
        ordered = group.sort("session_date")
        rows = ordered.to_dicts()
        for index, row in enumerate(rows):
            output = dict(row)
            current = float(row["close"])
            for horizon in horizons:
                end_index = index + horizon
                if end_index >= len(rows):
                    output[f"forward_return_{horizon}"] = None
                    output[f"max_favorable_excursion_{horizon}"] = None
                    output[f"max_adverse_excursion_{horizon}"] = None
                    output[f"label_end_{horizon}"] = None
                    continue
                future = [
                    float(item["close"]) / current - 1.0 for item in rows[index + 1 : end_index + 1]
                ]
                output[f"forward_return_{horizon}"] = future[-1]
                output[f"max_favorable_excursion_{horizon}"] = max(0.0, max(future))
                output[f"max_adverse_excursion_{horizon}"] = min(0.0, min(future))
                output[f"label_end_{horizon}"] = rows[end_index]["session_date"]
            result_rows.append(output)
    return pl.DataFrame(result_rows)


def build_triple_barrier_labels(
    prices: pl.DataFrame,
    *,
    volatility_column: str,
    horizon: int,
    upper: float,
    lower: float,
) -> pl.DataFrame:
    if horizon <= 0 or upper <= 0 or lower <= 0:
        raise ValueError("horizon and barrier multipliers must be positive")
    result_rows: list[dict[str, object]] = []
    for group in prices.partition_by("asset_id", maintain_order=True):
        rows = group.sort("session_date").to_dicts()
        for index, row in enumerate(rows):
            output = dict(row)
            current = float(row["close"])
            volatility = float(row[volatility_column])
            label = "time"
            barrier_end = None
            for future in rows[index + 1 : index + horizon + 1]:
                value = float(future["close"]) / current - 1.0
                barrier_end = future["session_date"]
                if value >= upper * volatility:
                    label = "up"
                    break
                if value <= -lower * volatility:
                    label = "down"
                    break
            output["barrier_label"] = label
            output["barrier_end"] = barrier_end
            result_rows.append(output)
    return pl.DataFrame(result_rows)
