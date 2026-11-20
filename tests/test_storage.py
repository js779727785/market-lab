from datetime import UTC, datetime, timedelta

from market_lab.contracts import EvidenceGrade, RecordEnvelope
from market_lab.storage import MarketStore

UTC = UTC


def make_record(*, revision: int, available_at: datetime, close: float) -> RecordEnvelope:
    return RecordEnvelope(
        dataset="prices",
        record_key="600000.SH:2026-11-20",
        source="fixture",
        source_uri="file:///prices.csv",
        evidence_grade=EvidenceGrade.LICENSED_PRIMARY_TRANSPORT,
        event_time=datetime(2026, 11, 20, 7, 0, tzinfo=UTC),
        published_at=available_at - timedelta(minutes=2),
        first_seen_at=available_at,
        available_at=available_at,
        revision=revision,
        payload={"asset_id": "600000.SH", "close": close},
    )


def test_as_of_query_hides_later_revision_and_adds_hashes(tmp_path) -> None:
    store = MarketStore(tmp_path / "store")
    first_time = datetime(2026, 11, 20, 7, 5, tzinfo=UTC)
    second_time = datetime(2026, 11, 21, 7, 5, tzinfo=UTC)
    store.append_records(
        [
            make_record(revision=1, available_at=first_time, close=10.0),
            make_record(revision=2, available_at=second_time, close=11.0),
        ]
    )

    early = store.as_of("prices", first_time + timedelta(hours=1))
    late = store.as_of("prices", second_time + timedelta(hours=1))

    assert len(early) == 1
    assert early[0].revision == 1
    assert early[0].payload["close"] == 10.0
    assert early[0].raw_sha256
    assert early[0].normalized_sha256
    assert late[0].revision == 2
    assert late[0].payload["close"] == 11.0


def test_manifest_verification_detects_file_tampering(tmp_path) -> None:
    store = MarketStore(tmp_path / "store")
    now = datetime(2026, 11, 20, 7, 5, tzinfo=UTC)
    store.append_records([make_record(revision=1, available_at=now, close=10.0)])
    manifest = store.dataset_manifest("prices")

    assert manifest.rows == 1
    assert store.verify_dataset("prices").valid is True

    parquet_path = store.root / manifest.files[0].path
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")

    verification = store.verify_dataset("prices")
    assert verification.valid is False
    assert verification.errors[0].code == "file_hash_mismatch"
