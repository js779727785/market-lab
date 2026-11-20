from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from .contracts import RecordEnvelope
from .provenance import canonical_sha256, file_sha256


class ManifestFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    rows: int


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    rows: int
    files: tuple[ManifestFile, ...]


class VerificationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    path: str | None = None


class DatasetVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    errors: tuple[VerificationError, ...]


class MarketStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def append_records(self, records: Sequence[RecordEnvelope]) -> tuple[Path, ...]:
        if not records:
            return ()
        by_dataset: dict[str, list[RecordEnvelope]] = {}
        for record in records:
            by_dataset.setdefault(record.dataset, []).append(record)

        written: list[Path] = []
        for dataset, items in sorted(by_dataset.items()):
            dataset_dir = self.root / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            rows = [self._storage_row(self._with_hashes(item)) for item in items]
            stamp = max(item.available_at for item in items).strftime("%Y%m%dT%H%M%S%fZ")
            path = dataset_dir / f"part-{stamp}-{uuid.uuid4().hex[:12]}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
            self._refresh_manifest(dataset)
            written.append(path)
        return tuple(written)

    def as_of(self, dataset: str, cutoff: datetime) -> list[RecordEnvelope]:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        paths = self._parquet_paths(dataset)
        if not paths:
            return []
        connection = duckdb.connect()
        try:
            query = """
                SELECT * EXCLUDE (record_rank)
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY record_key
                        ORDER BY available_at DESC, revision DESC, first_seen_at DESC
                    ) AS record_rank
                    FROM read_parquet($files)
                    WHERE available_at <= $cutoff
                )
                WHERE record_rank = 1
                ORDER BY record_key
            """
            result = connection.execute(
                query,
                {"files": [str(path) for path in paths], "cutoff": cutoff},
            )
            columns = [column[0] for column in result.description]
            return [
                self._record_from_row(dict(zip(columns, row, strict=True)))
                for row in result.fetchall()
            ]
        finally:
            connection.close()

    def latest(self, dataset: str) -> list[RecordEnvelope]:
        paths = self._parquet_paths(dataset)
        if not paths:
            return []
        connection = duckdb.connect()
        try:
            row = connection.execute(
                "SELECT max(available_at) FROM read_parquet($files)",
                {"files": [str(path) for path in paths]},
            ).fetchone()
            if row is None or row[0] is None:
                return []
            maximum = row[0]
            if not isinstance(maximum, datetime):
                raise TypeError("maximum available_at must be a datetime")
        finally:
            connection.close()
        return self.as_of(dataset, maximum)

    def dataset_manifest(self, dataset: str) -> DatasetManifest:
        path = self.root / dataset / "manifest.json"
        if not path.exists():
            return DatasetManifest(dataset=dataset, rows=0, files=())
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def verify_dataset(self, dataset: str) -> DatasetVerification:
        manifest = self.dataset_manifest(dataset)
        errors: list[VerificationError] = []
        for item in manifest.files:
            path = self.root / item.path
            if not path.exists():
                errors.append(
                    VerificationError(
                        code="file_missing",
                        message=f"missing dataset file {item.path}",
                        path=item.path,
                    )
                )
                continue
            actual = file_sha256(path)
            if actual != item.sha256:
                errors.append(
                    VerificationError(
                        code="file_hash_mismatch",
                        message=f"hash mismatch for {item.path}",
                        path=item.path,
                    )
                )
        return DatasetVerification(valid=not errors, errors=tuple(errors))

    def _with_hashes(self, record: RecordEnvelope) -> RecordEnvelope:
        raw_hash = record.raw_sha256 or canonical_sha256(
            {
                "source": record.source,
                "source_uri": record.source_uri,
                "revision": record.revision,
                "payload": record.payload,
            }
        )
        normalized_payload = record.model_dump(
            mode="json", exclude={"raw_sha256", "normalized_sha256"}
        )
        normalized_hash = record.normalized_sha256 or canonical_sha256(normalized_payload)
        return record.model_copy(
            update={"raw_sha256": raw_hash, "normalized_sha256": normalized_hash}
        )

    @staticmethod
    def _storage_row(record: RecordEnvelope) -> dict[str, Any]:
        data = record.model_dump(mode="python")
        data["evidence_grade"] = record.evidence_grade.value
        data["payload_json"] = json.dumps(
            record.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        del data["payload"]
        return data

    @staticmethod
    def _record_from_row(row: dict[str, Any]) -> RecordEnvelope:
        row["payload"] = json.loads(row.pop("payload_json"))
        return RecordEnvelope.model_validate(row)

    def _parquet_paths(self, dataset: str) -> list[Path]:
        return sorted((self.root / dataset).glob("part-*.parquet"))

    def _refresh_manifest(self, dataset: str) -> None:
        dataset_dir = self.root / dataset
        files: list[ManifestFile] = []
        rows = 0
        for path in self._parquet_paths(dataset):
            file_rows = pq.ParquetFile(path).metadata.num_rows
            rows += file_rows
            files.append(
                ManifestFile(
                    path=path.relative_to(self.root).as_posix(),
                    sha256=file_sha256(path),
                    rows=file_rows,
                )
            )
        manifest = DatasetManifest(dataset=dataset, rows=rows, files=tuple(files))
        (dataset_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
