from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, time

import httpx

from ..contracts import EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError


class FredProvider:
    endpoint = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("FRED API key is required for vintage-aware ingestion")
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=30)
        self.fetched_at = fetched_at

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if not request.symbols or request.start is None or request.end is None:
            raise ProviderError("FRED requires series symbols, start, and end")
        vintage_date = str(request.params.get("vintage_date") or request.end.isoformat())
        fetched_at = self.fetched_at or datetime.now(UTC)
        for series_id in request.symbols:
            response = await self.client.get(
                self.endpoint,
                params={
                    "api_key": self.api_key,
                    "file_type": "json",
                    "series_id": series_id,
                    "observation_start": request.start.isoformat(),
                    "observation_end": request.end.isoformat(),
                    "realtime_start": vintage_date,
                    "realtime_end": vintage_date,
                },
            )
            response.raise_for_status()
            for observation in response.json().get("observations", []):
                observation_date = datetime.strptime(observation["date"], "%Y-%m-%d").date()
                value_text = observation.get("value")
                value = None if value_text in (None, ".", "") else float(value_text)
                yield RecordEnvelope(
                    dataset="macro",
                    record_key=f"{series_id}:{observation_date.isoformat()}:{vintage_date}",
                    source="fred",
                    source_uri=str(response.request.url),
                    evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
                    event_time=datetime.combine(observation_date, time.min, UTC),
                    published_at=fetched_at,
                    first_seen_at=fetched_at,
                    available_at=fetched_at,
                    revision=1,
                    payload={
                        "series_id": series_id,
                        "observation_date": observation_date.isoformat(),
                        "value": value,
                        "vintage_date": vintage_date,
                        "realtime_start": observation.get("realtime_start"),
                        "realtime_end": observation.get("realtime_end"),
                    },
                )
