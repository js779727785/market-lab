from fastapi.testclient import TestClient

from market_lab.api import create_app
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
