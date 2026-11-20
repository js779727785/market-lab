from datetime import UTC, datetime

import pytest

from market_lab.contracts import EvidenceGrade, RecordEnvelope
from market_lab.ingest import IngestionError, IngestionService
from market_lab.storage import MarketStore

UTC = UTC


def item(*, source: str, close: float) -> RecordEnvelope:
    now = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    return RecordEnvelope(
        dataset="prices",
        record_key="600000.SH:2026-11-20",
        source=source,
        source_uri=f"https://example.test/{source}",
        evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
        event_time=now,
        published_at=now,
        first_seen_at=now,
        available_at=now,
        revision=1,
        payload={"open": 10.0, "high": 10.2, "low": 9.9, "close": close},
    )


def test_ingestion_rejects_conflicting_official_records(tmp_path) -> None:
    service = IngestionService(MarketStore(tmp_path / "store"))

    with pytest.raises(IngestionError):
        service.ingest(
            [
                item(source="official-a", close=10.1),
                item(source="official-b", close=10.2),
            ]
        )


def test_ingestion_persists_quality_checked_records(tmp_path) -> None:
    service = IngestionService(MarketStore(tmp_path / "store"))

    result = service.ingest([item(source="official-a", close=10.1)])

    assert result.rows == 1
    assert result.quality.ok is True
    assert len(result.files) == 1
