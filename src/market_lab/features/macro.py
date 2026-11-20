from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..contracts import RecordEnvelope


def latest_macro_values(records: Sequence[RecordEnvelope]) -> dict[str, float | None]:
    latest: dict[str, RecordEnvelope] = {}
    for record in records:
        series_id = str(record.payload.get("series_id", ""))
        if not series_id:
            continue
        existing = latest.get(series_id)
        if existing is None or record.available_at > existing.available_at:
            latest[series_id] = record
    return {
        f"macro_{series_id.lower()}": (
            None if record.payload.get("value") is None else float(record.payload["value"])
        )
        for series_id, record in latest.items()
    }


def attach_macro(prices: pl.DataFrame, records: Sequence[RecordEnvelope]) -> pl.DataFrame:
    values = latest_macro_values(records)
    if not values:
        return prices
    return prices.with_columns(
        *[pl.lit(value, dtype=pl.Float64).alias(name) for name, value in sorted(values.items())]
    )
