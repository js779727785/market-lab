from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from ..contracts import AssetClass, EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError


class BaostockProvider:
    """Adapter around an injected Baostock-compatible query function.

    The provider is optional because Baostock requires its own client login lifecycle.
    Injecting the query function keeps credentials and session state outside the domain layer.
    """

    def __init__(
        self,
        query: Callable[[FetchRequest], Iterable[dict[str, Any]]],
        fetched_at: datetime | None = None,
    ) -> None:
        self.query = query
        self.fetched_at = fetched_at

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if request.dataset != "prices":
            raise ProviderError("Baostock provider currently supports prices only")
        fetched_at = self.fetched_at or datetime.now(UTC)
        shanghai = ZoneInfo("Asia/Shanghai")
        for row in self.query(request):
            session_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
            asset_id = str(row["code"]).replace("sh.", "").replace("sz.", "")
            suffix = ".SH" if str(row["code"]).startswith("sh.") else ".SZ"
            asset_id = f"{asset_id}{suffix}"
            yield RecordEnvelope(
                dataset="prices",
                record_key=f"{asset_id}:{session_date.isoformat()}",
                source="baostock",
                source_uri="http://baostock.com",
                evidence_grade=EvidenceGrade.INDEPENDENT_CROSSCHECK,
                event_time=datetime.combine(session_date, time(15, 0), shanghai),
                published_at=fetched_at,
                first_seen_at=fetched_at,
                available_at=fetched_at,
                revision=1,
                payload={
                    "asset_id": asset_id,
                    "asset_class": AssetClass.A_SHARE.value,
                    "session_date": session_date.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "previous_close": float(row.get("preclose") or 0) or None,
                    "volume": float(row.get("volume") or 0),
                    "turnover": float(row.get("amount") or 0),
                },
            )
