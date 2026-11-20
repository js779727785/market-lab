from datetime import UTC, datetime

import polars as pl

from market_lab.contracts import EvidenceGrade, RecordEnvelope
from market_lab.features.dataset import FeatureBuilder
from market_lab.features.gold import gold_cny_return_components
from market_lab.storage import MarketStore

UTC = UTC


def price_record(day: int, close: float, *, seen_day: int | None = None) -> RecordEnvelope:
    event = datetime(2026, 11, day, 7, 0, tzinfo=UTC)
    available = datetime(2026, 11, seen_day or day, 8, 0, tzinfo=UTC)
    return RecordEnvelope(
        dataset="prices",
        record_key=f"600000.SH:2026-11-{day:02d}",
        source="fixture",
        source_uri="file:///prices",
        evidence_grade=EvidenceGrade.LICENSED_PRIMARY_TRANSPORT,
        event_time=event,
        published_at=available,
        first_seen_at=available,
        available_at=available,
        revision=1,
        payload={
            "asset_id": "600000.SH",
            "session_date": f"2026-11-{day:02d}",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000 + day,
            "turnover": (1000 + day) * close,
        },
    )


def fundamental_record(*, available_day: int, roe: float) -> RecordEnvelope:
    available = datetime(2026, 11, available_day, 8, 0, tzinfo=UTC)
    return RecordEnvelope(
        dataset="fundamentals",
        record_key=f"600000.SH:2026Q3:{available_day}",
        source="official",
        source_uri="file:///fundamentals",
        evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
        event_time=datetime(2026, 9, 30, tzinfo=UTC),
        published_at=available,
        first_seen_at=available,
        available_at=available,
        revision=1,
        payload={"asset_id": "600000.SH", "roe": roe, "report_period": "2026Q3"},
    )


def test_future_price_change_does_not_change_earlier_features(tmp_path) -> None:
    cutoff = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
    left = MarketStore(tmp_path / "left")
    right = MarketStore(tmp_path / "right")
    history = [price_record(day, 10.0 + day) for day in range(1, 6)]
    left.append_records([*history, price_record(6, 16.0)])
    right.append_records([*history, price_record(6, 1600.0)])

    left_features = FeatureBuilder(left).build(("600000.SH",), cutoff, 5)
    right_features = FeatureBuilder(right).build(("600000.SH",), cutoff, 5)

    assert left_features.to_dicts() == right_features.to_dicts()
    assert left_features["session_date"].max().isoformat() == "2026-11-05"


def test_fundamental_enters_only_after_announcement_availability(tmp_path) -> None:
    store = MarketStore(tmp_path / "store")
    store.append_records([price_record(day, 10.0 + day) for day in range(1, 8)])
    store.append_records([fundamental_record(available_day=6, roe=0.12)])

    before = FeatureBuilder(store).build(("600000.SH",), datetime(2026, 11, 5, 9, tzinfo=UTC), 5)
    after = FeatureBuilder(store).build(("600000.SH",), datetime(2026, 11, 7, 9, tzinfo=UTC), 5)

    assert before["roe"].to_list()[-1] is None
    assert after["roe"].to_list()[-1] == 0.12


def test_gold_cny_return_is_decomposed_into_gold_and_fx_components() -> None:
    result = gold_cny_return_components(
        xau_usd_start=2000.0,
        xau_usd_end=2040.0,
        usd_cny_start=7.0,
        usd_cny_end=7.07,
    )

    assert result["gold_usd_return"] == 0.02
    assert result["fx_return"] == 0.01
    assert abs(result["gold_cny_return"] - 0.0302) < 1e-12


def test_feature_frame_has_version_and_missingness_columns(tmp_path) -> None:
    store = MarketStore(tmp_path / "store")
    store.append_records([price_record(day, 10.0 + day) for day in range(1, 6)])

    features = FeatureBuilder(store).build(("600000.SH",), datetime(2026, 11, 5, 9, tzinfo=UTC), 5)

    assert isinstance(features, pl.DataFrame)
    assert features["feature_version"].unique().to_list() == ["market-features-v1"]
    assert "roe_missing" in features.columns
