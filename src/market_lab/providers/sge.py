from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from ..contracts import AssetClass, EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError


class SgeGoldProvider:
    endpoint = "https://www.sge.com.cn/sjzx/quotation_daily_new"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        self.client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
        self.fetched_at = fetched_at

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if request.start is None or request.end is None:
            raise ProviderError("SGE prices require start and end dates")
        symbols = request.symbols or ("Au99.99",)
        response = await self.client.get(
            self.endpoint,
            params={
                "start_date": request.start.isoformat(),
                "end_date": request.end.isoformat(),
                "inst_ids": ",".join(symbols),
            },
        )
        response.raise_for_status()
        fetched_at = self.fetched_at or datetime.now(UTC)
        shanghai = ZoneInfo("Asia/Shanghai")
        soup = BeautifulSoup(response.text, "html.parser")
        for row in soup.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
            if len(cells) < 11 or cells[1] not in symbols:
                continue
            try:
                session_date = datetime.strptime(cells[0], "%Y-%m-%d").date()
                open_price, high, low, close = map(float, cells[2:6])
                weighted_average = float(cells[8])
                volume = float(cells[9])
                turnover = float(cells[10])
            except ValueError as exc:
                raise ProviderError(f"invalid SGE row: {cells}") from exc
            asset_id = "AU9999.SGE" if cells[1] == "Au99.99" else f"{cells[1]}.SGE"
            yield RecordEnvelope(
                dataset="prices",
                record_key=f"{asset_id}:{session_date.isoformat()}",
                source="sge",
                source_uri=str(response.request.url),
                evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
                event_time=datetime.combine(session_date, time(15, 30), shanghai),
                published_at=fetched_at,
                first_seen_at=fetched_at,
                available_at=fetched_at,
                revision=1,
                payload={
                    "asset_id": asset_id,
                    "asset_class": AssetClass.GOLD.value,
                    "session_date": session_date.isoformat(),
                    "instrument": cells[1],
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "weighted_average": weighted_average,
                    "volume": volume,
                    "turnover": turnover,
                },
            )
