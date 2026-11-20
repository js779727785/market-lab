from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from .api import create_app
from .config import Settings
from .ingest import IngestionService
from .providers.base import FetchRequest
from .providers.files import FileProvider
from .service import MarketLabService
from .storage import MarketStore

app = typer.Typer(no_args_is_help=True, help="Auditable A-share and gold market lab")


def _settings() -> Settings:
    return Settings.load(Path.cwd())


def _echo(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command("ingest")
def ingest_file(path: Path) -> None:
    async def collect_records() -> list[Any]:
        return [record async for record in FileProvider().fetch(FetchRequest(path=path))]

    settings = _settings()
    result = IngestionService(MarketStore(settings.data_dir / "store")).ingest(
        asyncio.run(collect_records())
    )
    _echo(result)


@app.command("collect")
def collect() -> None:
    _echo(MarketLabService(Path.cwd()).source_status())


@app.command("train")
def train_demo() -> None:
    service = MarketLabService(Path.cwd())
    summary = service.run_demo()
    _echo(
        {
            "schema_version": 1,
            "evidence_mode": "demo",
            "model": "momentum-baseline-v1",
            "workflow": summary.model_dump(mode="json"),
        }
    )


@app.command("backtest")
def backtest_demo() -> None:
    service = MarketLabService(Path.cwd())
    service.run_demo()
    _echo(service.backtests())


@app.command("forecast")
def forecast_demo() -> None:
    service = MarketLabService(Path.cwd())
    service.run_demo()
    _echo({"opportunities": service.opportunities(), "gold": service.gold()})


@app.command("settle")
def settle(
    forecast_id: str,
    start_price: float,
    end_price: float,
    settled_at: datetime | None = None,
) -> None:
    timestamp = settled_at or datetime.now(UTC)
    _echo(
        MarketLabService(Path.cwd()).settle(
            forecast_id,
            start_price=start_price,
            end_price=end_price,
            settled_at=timestamp,
        )
    )


@app.command("run-daily")
def run_daily(
    demo_mode: bool = typer.Option(False, "--demo"),
    force: bool = False,
) -> None:
    service = MarketLabService(Path.cwd())
    if demo_mode:
        summary = service.run_demo(force=force)
        _echo(
            {
                **summary.model_dump(mode="json"),
                "live_trading_enabled": False,
                "limitations": service.status()["source_limitations"],
            }
        )
        return
    _echo(service.run_available_history_audit())


@app.command("historical-run")
def historical_run() -> None:
    _echo(MarketLabService(Path.cwd()).run_available_history_audit())


@app.command("demo")
def demo(force: bool = False) -> None:
    _echo(MarketLabService(Path.cwd()).run_demo(force=force))


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    import uvicorn

    uvicorn.run(create_app(Path.cwd()), host=host, port=port)


@app.command("verify-data")
def verify_data(dataset: str) -> None:
    settings = _settings()
    result = MarketStore(settings.data_dir / "store").verify_dataset(dataset)
    _echo(result)
    if not result.valid:
        raise typer.Exit(code=1)


@app.command("data-status")
def data_status(dataset: str) -> None:
    settings = _settings()
    manifest = MarketStore(settings.data_dir / "store").dataset_manifest(dataset)
    _echo(manifest)


if __name__ == "__main__":
    app()
