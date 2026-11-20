from datetime import UTC, datetime, timedelta

from market_lab.contracts import EvidenceGrade, RecordEnvelope
from market_lab.quality import QualityGate

UTC = UTC


def record(
    *,
    source: str = "official-a",
    grade: EvidenceGrade = EvidenceGrade.OFFICIAL_PRIMARY,
    revision: int = 1,
    payload: dict[str, object] | None = None,
    available_at: datetime | None = None,
) -> RecordEnvelope:
    timestamp = available_at or datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    return RecordEnvelope(
        dataset="prices",
        record_key="600000.SH:2026-11-20",
        source=source,
        source_uri=f"https://example.test/{source}",
        evidence_grade=grade,
        event_time=timestamp - timedelta(hours=1),
        published_at=timestamp - timedelta(minutes=5),
        first_seen_at=timestamp,
        available_at=timestamp,
        revision=revision,
        payload=payload
        or {
            "asset_id": "600000.SH",
            "open": 10.0,
            "high": 10.2,
            "low": 9.9,
            "close": 10.1,
            "volume": 1000,
            "turnover": 10_000,
        },
    )


def test_duplicate_revision_fails_quality_gate() -> None:
    item = record()
    report = QualityGate().run("prices", [item, item])

    assert report.ok is False
    assert "duplicate_revision" in {issue.code for issue in report.issues}


def test_conflicting_official_sources_fail_closed() -> None:
    left = record(source="official-a")
    right = record(source="official-b", payload={**left.payload, "close": 10.2})

    report = QualityGate().run("prices", [left, right])

    assert report.ok is False
    assert "official_source_conflict" in {issue.code for issue in report.issues}


def test_stale_and_unverified_data_are_reported() -> None:
    now = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
    item = record(
        grade=EvidenceGrade.UNVERIFIED,
        available_at=now - timedelta(days=10),
    )

    report = QualityGate(max_staleness=timedelta(days=2)).run("prices", [item], evaluated_at=now)

    assert report.ok is False
    assert report.evidence_coverage[EvidenceGrade.UNVERIFIED] == 1
    assert {issue.code for issue in report.issues} >= {
        "stale_dataset",
        "strict_evidence_missing",
    }
