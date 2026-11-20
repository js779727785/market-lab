from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ..contracts import RecordEnvelope

FUNDAMENTAL_FIELDS = ("roe", "revenue_growth", "pe_ttm", "pb")


def fundamental_frame(records: Sequence[RecordEnvelope]) -> pl.DataFrame:
    rows = []
    for record in records:
        row = {
            "asset_id": record.payload.get("asset_id"),
            "fundamental_available_at": record.available_at,
        }
        row.update({field: record.payload.get(field) for field in FUNDAMENTAL_FIELDS})
        rows.append(row)
    if not rows:
        return pl.DataFrame(
            schema={
                "asset_id": pl.String,
                "fundamental_available_at": pl.Datetime(time_zone="UTC"),
                **{field: pl.Float64 for field in FUNDAMENTAL_FIELDS},
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("fundamental_available_at").cast(pl.Datetime(time_zone="UTC")),
        *[pl.col(field).cast(pl.Float64, strict=False) for field in FUNDAMENTAL_FIELDS],
    )


def attach_fundamentals(prices: pl.DataFrame, records: Sequence[RecordEnvelope]) -> pl.DataFrame:
    fundamentals = fundamental_frame(records)
    if fundamentals.is_empty():
        result = prices
        for field in FUNDAMENTAL_FIELDS:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))
    else:
        result = prices.sort(["asset_id", "available_at"]).join_asof(
            fundamentals.sort(["asset_id", "fundamental_available_at"]),
            left_on="available_at",
            right_on="fundamental_available_at",
            by="asset_id",
            strategy="backward",
            check_sortedness=False,
        )
    return result.with_columns(
        *[pl.col(field).is_null().alias(f"{field}_missing") for field in FUNDAMENTAL_FIELDS]
    )
