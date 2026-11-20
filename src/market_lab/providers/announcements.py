from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from ..contracts import EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError


class AnnouncementProvider:
    """Official-announcement adapter with an injected source-specific parser."""

    def __init__(
        self,
        source: str,
        endpoint: str,
        parser: Callable[[Any], Sequence[dict[str, Any]]],
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.source = source
        self.endpoint = endpoint
        self.parser = parser
        self.client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        response = await self.client.get(self.endpoint, params=request.params)
        response.raise_for_status()
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        fetched_at = datetime.now(UTC)
        for item in self.parser(payload):
            published_at = item.get("published_at")
            if not isinstance(published_at, datetime):
                raise ProviderError("announcement parser must provide published_at datetime")
            yield RecordEnvelope(
                dataset="announcements",
                record_key=str(item["record_key"]),
                source=self.source,
                source_uri=str(item.get("source_uri") or response.request.url),
                evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
                event_time=item.get("event_time") or published_at,
                published_at=published_at,
                first_seen_at=fetched_at,
                available_at=max(published_at, fetched_at),
                revision=int(item.get("revision", 1)),
                payload=dict(item["payload"]),
            )
