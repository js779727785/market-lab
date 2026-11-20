from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .contracts import RecordEnvelope
from .quality import QualityGate, QualityReport
from .storage import MarketStore


class IngestionError(RuntimeError):
    pass


class IngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    rows: int
    files: tuple[Path, ...]
    quality: QualityReport


class IngestionService:
    def __init__(self, store: MarketStore, quality_gate: QualityGate | None = None) -> None:
        self.store = store
        self.quality_gate = quality_gate or QualityGate()

    def ingest(self, records: Sequence[RecordEnvelope]) -> IngestionResult:
        if not records:
            raise IngestionError("cannot ingest an empty record sequence")
        datasets = {record.dataset for record in records}
        if len(datasets) != 1:
            raise IngestionError("one ingestion batch must contain exactly one dataset")
        dataset = next(iter(datasets))
        evaluated_at = max(record.available_at for record in records)
        quality = self.quality_gate.run(dataset, records, evaluated_at=evaluated_at)
        if not quality.ok:
            codes = ", ".join(issue.code for issue in quality.issues)
            raise IngestionError(f"quality gate failed: {codes}")
        files = self.store.append_records(records)
        return IngestionResult(
            dataset=dataset,
            rows=len(records),
            files=files,
            quality=quality,
        )
