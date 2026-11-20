import json
from pathlib import Path

from typer.testing import CliRunner

from market_lab.api import create_app
from market_lab.cli import app
from market_lab.service import MarketLabService


def test_fixture_workflow_runs_ingestion_features_backtest_freeze_and_settlement(tmp_path) -> None:
    service = MarketLabService(tmp_path)

    summary = service.run_demo()

    assert summary.ingested_rows >= 60
    assert summary.feature_rows >= 60
    assert summary.backtest_cases > 0
    assert summary.frozen_forecasts == 2
    assert summary.settled_forecasts == 1
    assert service.ledger.verify().valid
    assert service.status()["evidence_mode"] == "demo"
    assert create_app(tmp_path).title == "Market Lab"


def test_demo_forecasts_freeze_versioned_immutable_snapshots(tmp_path) -> None:
    service = MarketLabService(tmp_path)

    service.run_demo()

    rows = [
        json.loads(line) for line in service.ledger.path.read_text(encoding="utf-8").splitlines()
    ]
    forecasts = [row for row in rows if row["entry_type"] == "forecast"]
    snapshot_dir = (tmp_path / "reports" / "demo_snapshots").resolve()
    assert len(forecasts) == 2
    for row in forecasts:
        snapshots = row["payload"]["snapshots"]
        for kind, suffix in (("model", ".json"), ("config", ".json"), ("code", ".py")):
            reference = snapshots[kind]
            path = Path(reference["path"])
            assert path.parent == snapshot_dir
            assert path.name == f"{kind}-{reference['sha256']}{suffix}"
            assert path.is_file()


def test_cli_exposes_complete_operational_command_chain(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["demo"])
    help_result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert '"evidence_mode": "demo"' in result.output
    for command in (
        "collect",
        "train",
        "backtest",
        "forecast",
        "settle",
        "serve",
        "run-daily",
    ):
        assert command in help_result.output
