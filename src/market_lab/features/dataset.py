from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from ..contracts import RecordEnvelope
from ..storage import MarketStore
from .fundamental import attach_fundamentals
from .macro import attach_macro
from .technical import add_technical_features

FEATURE_VERSION = "market-features-v1"


class FeatureBuilder:
    def __init__(self, store: MarketStore) -> None:
        self.store = store

    def build(
        self,
        asset_ids: Sequence[str],
        cutoff: datetime,
        horizon: int,
    ) -> pl.DataFrame:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        asset_set = set(asset_ids)
        price_records = [
            record
            for record in self.store.as_of("prices", cutoff)
            if record.payload.get("asset_id") in asset_set
        ]
        prices = self._price_frame(price_records)
        if prices.is_empty():
            return prices
        result = add_technical_features(prices)
        result = attach_fundamentals(
            result,
            [
                record
                for record in self.store.as_of("fundamentals", cutoff)
                if record.payload.get("asset_id") in asset_set
            ],
        )
        result = attach_macro(result, self.store.as_of("macro", cutoff))
        return result.with_columns(
            pl.lit(FEATURE_VERSION).alias("feature_version"),
            pl.lit(horizon).alias("forecast_horizon"),
        ).sort(["asset_id", "session_date"])

    @staticmethod
    def _price_frame(records: Sequence[RecordEnvelope]) -> pl.DataFrame:
        rows = []
        for record in records:
            payload = record.payload
            rows.append(
                {
                    "asset_id": payload["asset_id"],
                    "session_date": payload["session_date"],
                    "available_at": record.available_at,
                    "open": payload["open"],
                    "high": payload["high"],
                    "low": payload["low"],
                    "close": payload["close"],
                    "volume": payload.get("volume", 0.0),
                    "turnover": payload.get("turnover", 0.0),
                }
            )
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).with_columns(
            pl.col("session_date").str.to_date(),
            pl.col("available_at").cast(pl.Datetime(time_zone="UTC")),
            pl.col("open", "high", "low", "close", "volume", "turnover").cast(pl.Float64),
        )
