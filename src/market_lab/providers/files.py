from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from ..contracts import RecordEnvelope
from .base import FetchRequest, ProviderError


class FileProvider:
    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if request.path is None:
            raise ProviderError("file provider requires request.path")
        for payload in self._read(request.path):
            yield RecordEnvelope.model_validate(payload)

    @staticmethod
    def _read(path: Path) -> Iterable[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("records", [])
            yield from rows
            return
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle)
            return
        raise ProviderError(f"unsupported file type: {path.suffix}")
