from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts import AssetClass, EvidenceGrade, RecordEnvelope
from .base import FetchRequest, ProviderError


def _aware_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class LiveAu9999Quote(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str = "AU9999.SGE"
    instrument: str = "Au99.99"
    price: float = Field(gt=0)
    session_date: date
    as_of: datetime
    fetched_at: datetime
    source_uri: str = Field(min_length=1)
    evidence_grade: EvidenceGrade
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    weighted_average: float = Field(gt=0)
    volume: float = Field(ge=0)
    turnover: float = Field(ge=0)
    previous_close: float | None = Field(default=None, gt=0)
    freshness_seconds: float = Field(ge=0)
    is_fresh: bool
    limitations: tuple[str, ...] = ()

    _validate_timestamps = field_validator("as_of", "fetched_at")(_aware_timestamp)


class SgeGoldProvider:
    endpoint = "https://www.sge.com.cn/sjzx/quotation_daily_new"
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Referer": "https://www.sge.com.cn/",
    }

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        self.client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
        self.client.headers.update(self.default_headers)
        self.fetched_at = fetched_at

    async def aclose(self) -> None:
        await self.client.aclose()

    async def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]:
        if request.start is None or request.end is None:
            raise ProviderError("SGE prices require start and end dates")
        symbols = request.symbols or ("Au99.99",)
        try:
            response = await self.client.get(
                self.endpoint,
                params={
                    "start_date": request.start.isoformat(),
                    "end_date": request.end.isoformat(),
                    "inst_ids": ",".join(symbols),
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"SGE request failed: {exc}") from exc
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

    async def fetch_latest(
        self,
        *,
        now: datetime | None = None,
        max_age: timedelta = timedelta(days=3),
    ) -> LiveAu9999Quote:
        reference_time = now or datetime.now(UTC)
        if reference_time.tzinfo is None or reference_time.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        shanghai = ZoneInfo("Asia/Shanghai")
        end = reference_time.astimezone(shanghai).date()
        records = [
            record
            async for record in self.fetch(
                FetchRequest(
                    dataset="prices",
                    start=end - timedelta(days=7),
                    end=end,
                    symbols=("Au99.99",),
                )
            )
        ]
        if not records:
            raise ProviderError("SGE returned no Au99.99 price rows")
        latest = max(records, key=lambda record: record.event_time)
        previous = max(
            (record for record in records if record.event_time < latest.event_time),
            default=None,
            key=lambda record: record.event_time,
        )
        payload = latest.payload
        as_of = latest.event_time
        freshness_seconds = max((reference_time - as_of).total_seconds(), 0.0)
        is_fresh = (
            reference_time >= as_of
            and reference_time - as_of <= max_age
            and latest.evidence_grade
            in {
                EvidenceGrade.OFFICIAL_PRIMARY,
                EvidenceGrade.LICENSED_PRIMARY_TRANSPORT,
            }
        )
        return LiveAu9999Quote(
            price=float(payload["close"]),
            session_date=date.fromisoformat(str(payload["session_date"])),
            as_of=as_of,
            fetched_at=latest.available_at,
            source_uri=latest.source_uri,
            evidence_grade=latest.evidence_grade,
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            weighted_average=float(payload["weighted_average"]),
            volume=float(payload["volume"]),
            turnover=float(payload["turnover"]),
            previous_close=(float(previous.payload["close"]) if previous else None),
            freshness_seconds=freshness_seconds,
            is_fresh=is_fresh,
            limitations=(
                "official_daily_close_not_intraday",
                "live_model_and_news_evidence_not_loaded",
            ),
        )
