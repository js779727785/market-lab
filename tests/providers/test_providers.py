import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from market_lab.contracts import EvidenceGrade
from market_lab.providers.base import FetchRequest, ProviderError
from market_lab.providers.files import FileProvider
from market_lab.providers.fred import FredProvider
from market_lab.providers.sge import SgeGoldProvider
from market_lab.providers.tushare import TushareProvider

UTC = UTC


@pytest.mark.asyncio
async def test_file_provider_reads_jsonl_envelopes(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "prices",
                "record_key": "600000.SH:2026-11-20",
                "source": "licensed-fixture",
                "source_uri": "file:///prices.jsonl",
                "evidence_grade": "licensed_primary_transport",
                "event_time": "2026-11-20T07:00:00+00:00",
                "published_at": "2026-11-20T07:05:00+00:00",
                "first_seen_at": "2026-11-20T07:06:00+00:00",
                "available_at": "2026-11-20T07:06:00+00:00",
                "revision": 1,
                "payload": {"close": 10.1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = [record async for record in FileProvider().fetch(FetchRequest(path=path))]

    assert records[0].record_key == "600000.SH:2026-11-20"
    assert records[0].evidence_grade == EvidenceGrade.LICENSED_PRIMARY_TRANSPORT


@pytest.mark.asyncio
async def test_tushare_provider_maps_daily_response_with_transport_evidence() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": None,
                "data": {
                    "fields": [
                        "ts_code",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "vol",
                        "amount",
                    ],
                    "items": [["600000.SH", "20261120", 10.0, 10.2, 9.9, 10.1, 9.8, 100.0, 1000.0]],
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = TushareProvider(
        token="secret",
        client=client,
        fetched_at=datetime(2026, 11, 20, 8, 30, tzinfo=UTC),
    )
    request = FetchRequest(
        dataset="prices",
        start=date(2026, 11, 20),
        end=date(2026, 11, 20),
        symbols=("600000.SH",),
    )

    records = [record async for record in provider.fetch(request)]
    await client.aclose()

    assert captured["api_name"] == "daily"
    assert records[0].source == "tushare"
    assert records[0].evidence_grade == EvidenceGrade.LICENSED_PRIMARY_TRANSPORT
    assert records[0].payload["turnover"] == 1_000_000.0
    assert records[0].available_at == datetime(2026, 11, 20, 8, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_sge_provider_parses_official_au9999_html() -> None:
    html = """
    <table><tbody><tr>
      <td>2026-11-20</td><td>Au99.99</td><td>900.00</td><td>910.00</td>
      <td>895.00</td><td>905.00</td><td>5.00</td><td>0.56%</td>
      <td>903.20</td><td>1234.5</td><td>1115000000</td><td>-</td>
    </tr></tbody></table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("Mozilla/")
        assert request.headers["referer"] == "https://www.sge.com.cn/"
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SgeGoldProvider(
        client=client,
        fetched_at=datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
    )
    records = [
        record
        async for record in provider.fetch(
            FetchRequest(
                dataset="prices",
                start=date(2026, 11, 20),
                end=date(2026, 11, 20),
                symbols=("Au99.99",),
            )
        )
    ]
    await client.aclose()

    assert records[0].record_key == "AU9999.SGE:2026-11-20"
    assert records[0].evidence_grade == EvidenceGrade.OFFICIAL_PRIMARY
    assert records[0].payload["weighted_average"] == 903.2


@pytest.mark.asyncio
async def test_sge_provider_builds_fresh_latest_au9999_quote_with_previous_close() -> None:
    html = """
    <table><tbody>
    <tr>
      <td>2026-11-19</td><td>Au99.99</td><td>890.00</td><td>901.00</td>
      <td>888.00</td><td>899.00</td><td>5.00</td><td>0.56%</td>
      <td>895.20</td><td>1200.0</td><td>1074000000</td><td>-</td>
    </tr>
    <tr>
      <td>2026-11-20</td><td>Au99.99</td><td>900.00</td><td>910.00</td>
      <td>895.00</td><td>905.00</td><td>5.00</td><td>0.56%</td>
      <td>903.20</td><td>1234.5</td><td>1115000000</td><td>-</td>
    </tr>
    </tbody></table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SgeGoldProvider(
        client=client,
        fetched_at=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
    )

    quote = await provider.fetch_latest(
        now=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
    )

    assert quote.asset_id == "AU9999.SGE"
    assert quote.instrument == "Au99.99"
    assert quote.price == 905.0
    assert quote.previous_close == 899.0
    assert quote.evidence_grade == EvidenceGrade.OFFICIAL_PRIMARY
    assert quote.is_fresh is True
    assert quote.source_uri.startswith(SgeGoldProvider.endpoint)

    stale_quote = await provider.fetch_latest(
        now=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
        max_age=timedelta(hours=1),
    )
    assert stale_quote.is_fresh is False
    await client.aclose()


@pytest.mark.asyncio
async def test_sge_provider_wraps_transport_errors_as_provider_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = SgeGoldProvider(client=client)

    with pytest.raises(ProviderError, match="SGE request failed"):
        await provider.fetch_latest(
            now=datetime(2026, 11, 20, 10, 0, tzinfo=UTC),
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_fred_provider_preserves_vintage_and_release_availability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "observations": [
                    {
                        "realtime_start": "2026-11-20",
                        "realtime_end": "2026-11-20",
                        "date": "2026-11-19",
                        "value": "2.91",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = FredProvider(
        api_key="key",
        client=client,
        fetched_at=datetime(2026, 11, 20, 14, 0, tzinfo=UTC),
    )
    records = [
        record
        async for record in provider.fetch(
            FetchRequest(
                dataset="macro",
                start=date(2026, 11, 19),
                end=date(2026, 11, 19),
                symbols=("DFII10",),
                params={"vintage_date": "2026-11-20"},
            )
        )
    ]
    await client.aclose()

    assert records[0].record_key == "DFII10:2026-11-19:2026-11-20"
    assert records[0].payload["value"] == 2.91
    assert records[0].payload["vintage_date"] == "2026-11-20"
