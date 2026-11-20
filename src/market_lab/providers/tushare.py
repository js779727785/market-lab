from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import httpx

from ..contracts import AssetClass, EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError

SHANGHAI = ZoneInfo("Asia/Shanghai")


class TushareProvider:
    endpoint = "https://api.tushare.pro"

    def __init__(
        self,
        token: str,
        client: httpx.AsyncClient | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        if not token:
            raise ProviderError("Tushare token is required")
        self.token = token
        self.client = client or httpx.AsyncClient(timeout=30)
        self.fetched_at = fetched_at

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if request.dataset != "prices" or request.start is None or request.end is None:
            raise ProviderError("Tushare daily prices require dataset, start, and end")
        params: dict[str, str] = {
            "start_date": request.start.strftime("%Y%m%d"),
            "end_date": request.end.strftime("%Y%m%d"),
        }
        if len(request.symbols) == 1:
            params["ts_code"] = request.symbols[0]
        fields = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
        response = await self.client.post(
            self.endpoint,
            json={
                "api_name": "daily",
                "token": self.token,
                "params": params,
                "fields": fields,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise ProviderError(f"Tushare error {payload.get('code')}: {payload.get('msg')}")
        data = payload.get("data") or {}
        columns = data.get("fields") or []
        fetched_at = self.fetched_at or datetime.now(UTC)
        for item in data.get("items") or []:
            row = dict(zip(columns, item, strict=True))
            session_date = datetime.strptime(row["trade_date"], "%Y%m%d").date()
            event_time = datetime.combine(session_date, time(15, 0), SHANGHAI)
            yield RecordEnvelope(
                dataset="prices",
                record_key=f"{row['ts_code']}:{session_date.isoformat()}",
                source="tushare",
                source_uri=self.endpoint,
                evidence_grade=EvidenceGrade.LICENSED_PRIMARY_TRANSPORT,
                event_time=event_time,
                published_at=fetched_at,
                first_seen_at=fetched_at,
                available_at=fetched_at,
                revision=1,
                payload={
                    "asset_id": row["ts_code"],
                    "asset_class": AssetClass.A_SHARE.value,
                    "session_date": session_date.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "previous_close": float(row["pre_close"]),
                    "volume": float(row["vol"]) * 100.0,
                    "turnover": float(row["amount"]) * 1000.0,
                },
            )
