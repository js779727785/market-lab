from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from .contracts import EvidenceGrade, RecordEnvelope
from .provenance import canonical_sha256


class QualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: str
    message: str
    record_key: str | None = None


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    rows: int
    ok: bool
    issues: tuple[QualityIssue, ...]
    evidence_coverage: dict[EvidenceGrade, int]
    latest_available_at: datetime | None


class QualityGate:
    def __init__(self, max_staleness: timedelta = timedelta(days=3)) -> None:
        self.max_staleness = max_staleness

    def run(
        self,
        dataset: str,
        records: Sequence[RecordEnvelope],
        evaluated_at: datetime | None = None,
    ) -> QualityReport:
        now = evaluated_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        issues: list[QualityIssue] = []
        seen_source_revision: set[tuple[str, str, int]] = set()
        official_by_key: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        for record in records:
            if record.dataset != dataset:
                issues.append(
                    QualityIssue(
                        code="dataset_mismatch",
                        severity="error",
                        message=f"record belongs to {record.dataset}, expected {dataset}",
                        record_key=record.record_key,
                    )
                )
            source_revision = (record.record_key, record.source, record.revision)
            if source_revision in seen_source_revision:
                issues.append(
                    QualityIssue(
                        code="duplicate_revision",
                        severity="error",
                        message="duplicate source/key/revision",
                        record_key=record.record_key,
                    )
                )
            seen_source_revision.add(source_revision)

            if record.evidence_grade == EvidenceGrade.OFFICIAL_PRIMARY:
                official_by_key[record.record_key][record.source].add(
                    canonical_sha256(record.payload)
                )
            if dataset == "prices":
                issues.extend(self._price_issues(record))

        for record_key, by_source in official_by_key.items():
            hashes = {value for values in by_source.values() for value in values}
            if len(by_source) > 1 and len(hashes) > 1:
                issues.append(
                    QualityIssue(
                        code="official_source_conflict",
                        severity="error",
                        message="official sources disagree on normalized payload",
                        record_key=record_key,
                    )
                )

        coverage_counter = Counter(record.evidence_grade for record in records)
        coverage = {grade: coverage_counter.get(grade, 0) for grade in EvidenceGrade}
        strict_rows = (
            coverage[EvidenceGrade.OFFICIAL_PRIMARY]
            + coverage[EvidenceGrade.LICENSED_PRIMARY_TRANSPORT]
        )
        if records and strict_rows == 0:
            issues.append(
                QualityIssue(
                    code="strict_evidence_missing",
                    severity="error",
                    message="dataset has no official or licensed primary evidence",
                )
            )

        latest = max((record.available_at for record in records), default=None)
        if latest is not None and now - latest > self.max_staleness:
            issues.append(
                QualityIssue(
                    code="stale_dataset",
                    severity="error",
                    message=f"latest record is older than {self.max_staleness}",
                )
            )
        if not records:
            issues.append(
                QualityIssue(
                    code="empty_dataset",
                    severity="error",
                    message="dataset has no records",
                )
            )

        return QualityReport(
            dataset=dataset,
            rows=len(records),
            ok=not any(issue.severity == "error" for issue in issues),
            issues=tuple(issues),
            evidence_coverage=coverage,
            latest_available_at=latest,
        )

    @staticmethod
    def _price_issues(record: RecordEnvelope) -> list[QualityIssue]:
        required = {"open", "high", "low", "close"}
        if not required.issubset(record.payload):
            return []
        try:
            open_price = float(record.payload["open"])
            high = float(record.payload["high"])
            low = float(record.payload["low"])
            close = float(record.payload["close"])
        except (TypeError, ValueError):
            return [
                QualityIssue(
                    code="invalid_price_bar",
                    severity="error",
                    message="OHLC values must be numeric",
                    record_key=record.record_key,
                )
            ]
        if (
            min(open_price, high, low, close) <= 0
            or high < max(open_price, close, low)
            or low > min(open_price, close, high)
        ):
            return [
                QualityIssue(
                    code="invalid_price_bar",
                    severity="error",
                    message="OHLC values are non-positive or inconsistent",
                    record_key=record.record_key,
                )
            ]
        return []
