import os
import tomllib

from market_lab.service import MarketLabService


def test_default_configuration_disables_live_trading_and_registers_all_gates() -> None:
    with open("configs/default.toml", "rb") as handle:
        default = tomllib.load(handle)
    with open("configs/model_gates.toml", "rb") as handle:
        gates = tomllib.load(handle)

    assert default["system"]["live_trading_enabled"] is False
    assert default["prediction"]["horizons"] == [5, 20]
    assert default["prediction"]["recent_audit_start"] == "2023-11-20"
    assert default["prediction"]["recent_audit_end"] == "2026-11-20"
    assert default["sources"]["sge_official"] is True
    assert default["risk"]["max_asset_weight"] < 1
    assert default["risk"]["max_sector_weight"] < 1
    assert gates["promotion"]["bootstrap_confidence"] == 0.95
    assert gates["promotion"]["require_positive_double_cost_return"] is True
    assert gates["shadow"]["a_share_min_samples"] > 0
    assert gates["shadow"]["gold_min_samples"] > 0


def test_available_history_report_never_fills_missing_three_year_coverage(tmp_path) -> None:
    service = MarketLabService(tmp_path)
    service.run_demo()

    report = service.run_available_history_audit()

    assert report.evidence_mode == "demo"
    assert report.source_coverage["prices"]["rows"] == 80
    assert report.available_history_start == "2026-10-01"
    assert report.available_history_end == "2026-11-09"
    assert report.recent_three_year_complete is False
    assert report.recent_three_year_case_count == 0
    assert "insufficient_recent_three_year_coverage" in report.limitations
    assert set(report.frozen_forecast_assets) == {"600000.SH", "AU9999.SGE"}
    assert report.ledger_valid is True
    assert service.historical_report_path.exists()


def test_daily_script_is_executable_and_does_not_enable_live_trading() -> None:
    path = "scripts/run_daily.sh"

    assert os.path.isfile(path)
    assert os.access(path, os.X_OK)
    contents = open(path, encoding="utf-8").read()
    assert "market-lab collect" in contents
    assert "market-lab historical-run" in contents
    assert "live" not in contents.casefold()
