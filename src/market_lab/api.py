from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .providers.base import ProviderError
from .service import MarketLabService


class PositionEvaluationRequest(BaseModel):
    asset_id: str
    average_cost: float = Field(gt=0)
    quantity: float = Field(gt=0)
    available_cash: float = Field(default=0, ge=0)
    horizon_days: int = Field(default=20, gt=0)
    max_loss_fraction: float | None = Field(default=None, gt=0, le=1)
    allow_add: bool = True
    allow_scale: bool = True
    sector_limit: float | None = Field(default=None, gt=0, le=1)


class LivePositionEvaluationRequest(BaseModel):
    asset_id: str = Field(min_length=1)
    average_cost: float = Field(gt=0)
    quantity: float = Field(gt=0)
    available_cash: float = Field(default=0, ge=0)
    other_assets_value: float = Field(default=0, ge=0)
    account_value: float | None = Field(default=None, gt=0)
    max_asset_weight: float = Field(default=0.2, gt=0, le=1)
    horizon_days: int = Field(default=20, gt=0)
    max_loss_fraction: float | None = Field(default=None, gt=0, le=1)


def create_app(root: Path | None = None) -> FastAPI:
    project_root = (root or Path.cwd()).resolve()
    service = MarketLabService(project_root)
    app = FastAPI(title="Market Lab", version="0.1.0")
    app.state.service = service
    web_root = Path(__file__).resolve().parents[2] / "web"
    if web_root.exists():
        app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_root / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return service.status()

    @app.get("/api/opportunities")
    def opportunities() -> dict[str, Any]:
        return service.opportunities()

    @app.get("/api/gold")
    def gold() -> dict[str, Any]:
        return service.gold()

    @app.post("/api/positions/evaluate")
    def evaluate_position(request: PositionEvaluationRequest) -> dict[str, Any]:
        try:
            return service.evaluate_position(request.model_dump())
        except LookupError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown asset: {exc.args[0]}") from exc

    @app.post("/api/live/positions/evaluate")
    async def evaluate_live_position(request: LivePositionEvaluationRequest) -> dict[str, Any]:
        try:
            return await service.evaluate_live_position(request.model_dump())
        except ProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported live asset: {exc.args[0]}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/events")
    def events() -> dict[str, Any]:
        return service.events()

    @app.get("/api/backtests")
    def backtests() -> dict[str, Any]:
        return service.backtests()

    @app.get("/api/ledger")
    def ledger() -> dict[str, Any]:
        return service.ledger_status()

    @app.get("/api/research")
    def research() -> dict[str, Any]:
        return service.research()

    return app


app = create_app()
