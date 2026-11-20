import pytest
from fastapi.testclient import TestClient

from market_lab.api import create_app
from market_lab.contracts import EvidenceGrade
from market_lab.providers.sge import LiveAu9999Quote, SgeGoldProvider
from market_lab.service import MarketLabService


def test_api_reports_missing_data_without_fabricating_forecasts(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    status = client.get("/api/status")
    opportunities = client.get("/api/opportunities")

    assert status.status_code == 200
    assert status.json()["evidence_mode"] == "unavailable"
    assert status.json()["live_trading_enabled"] is False
    assert opportunities.status_code == 200
    assert opportunities.json() == {
        "evidence_mode": "unavailable",
        "items": [],
        "reasons": ["no_forecast_bundle"],
    }


def test_demo_api_labels_research_evidence_and_exposes_abstention_reasons(tmp_path) -> None:
    MarketLabService(tmp_path).run_demo()
    client = TestClient(create_app(tmp_path))

    status = client.get("/api/status").json()
    opportunities = client.get("/api/opportunities").json()
    gold = client.get("/api/gold").json()
    ledger = client.get("/api/ledger").json()

    assert status["evidence_mode"] == "demo"
    assert status["strict_forecasts"] == 0
    assert opportunities["evidence_mode"] == "research"
    assert opportunities["items"][0]["asset_class"] == "a_share"
    assert opportunities["items"][0]["evidence_mode"] == "research"
    assert gold["evidence_mode"] == "research"
    assert gold["decision"]["action"] == "abstain"
    assert "return_interval_lacks_safety_margin" in gold["abstention_reasons"]
    assert ledger["verification"]["valid"] is True
    assert ledger["verification"]["entries"] >= 3


def test_position_evaluation_and_operational_pages_are_available(tmp_path) -> None:
    MarketLabService(tmp_path).run_demo()
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/positions/evaluate",
        json={
            "asset_id": "600000.SH",
            "average_cost": 10.2,
            "quantity": 1000,
            "available_cash": 100000,
            "horizon_days": 20,
            "max_loss_fraction": 0.02,
            "allow_add": True,
            "allow_scale": True,
            "sector_limit": 0.3,
        },
    )
    index = client.get("/")

    assert response.status_code == 200
    assert response.json()["asset_id"] == "600000.SH"
    assert response.json()["evidence_mode"] == "research"
    assert response.json()["advice"]["action"] in {"add", "hold", "reduce"}
    assert index.status_code == 200
    assert "A股与黄金决策实验室" in index.text


def test_all_operational_api_surfaces_return_versioned_payloads(tmp_path) -> None:
    MarketLabService(tmp_path).run_demo()
    client = TestClient(create_app(tmp_path))

    for path in (
        "/api/events",
        "/api/backtests",
        "/api/ledger",
        "/api/research",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["schema_version"] == 1


def test_live_position_evaluation_uses_live_quote_without_demo_state(tmp_path, monkeypatch) -> None:
    async def fake_fetch_latest(self, *, now=None):
        return LiveAu9999Quote(
            asset_id="AU9999.SGE",
            instrument="Au99.99",
            price=760.0,
            session_date="2026-11-20",
            as_of="2026-11-20T15:30:00+08:00",
            fetched_at="2026-11-20T10:00:00+00:00",
            source_uri=SgeGoldProvider.endpoint,
            evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
            open=750.0,
            high=803.99,
            low=755.0,
            weighted_average=770.0,
            volume=1000.0,
            turnover=760000.0,
            previous_close=801.93,
            freshness_seconds=3600.0,
            is_fresh=True,
            limitations=(),
        )

    monkeypatch.setattr(SgeGoldProvider, "fetch_latest", fake_fetch_latest)
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/live/positions/evaluate",
        json={
            "asset_id": "AU9999.SGE",
            "average_cost": 1248.0,
            "quantity": 1000.0,
            "available_cash": 0.0,
            "account_value": 760000.0,
            "max_asset_weight": 0.2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_mode"] == "live_risk_only"
    assert payload["quote"]["price"] == 760.0
    assert payload["position"]["drawdown_fraction"] == pytest.approx(-0.3910256)
    assert payload["position"]["asset_weight"] == pytest.approx(1.0)
    assert payload["advice"]["action"] == "reduce"
    assert payload["forecast"]["action"] == "abstain"
    assert payload["forecast"]["prediction_available"] is False
    assert "probability_up" not in payload["forecast"]
    assert payload["evidence"]["historical_model"]["status"] == "unavailable"
    assert payload["evidence"]["news"]["status"] == "unavailable"
    assert "demo" not in response.text


def test_live_position_evaluation_rejects_non_au9999_assets(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/live/positions/evaluate",
        json={
            "asset_id": "600000.SH",
            "average_cost": 10.0,
            "quantity": 1000.0,
        },
    )

    assert response.status_code == 422
    assert "unsupported live asset" in response.json()["detail"]
