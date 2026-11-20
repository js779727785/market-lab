from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARKET_LAB_",
        env_file=".env",
        extra="ignore",
    )

    root: Path
    data_dir: Path
    ledger_dir: Path
    reports_dir: Path
    strict_min_evidence_grade: str = "licensed_primary_transport"
    forecast_horizons: tuple[int, ...] = (5, 20)
    tushare_token: str | None = Field(default=None, repr=False)
    fred_api_key: str | None = Field(default=None, repr=False)
    llm_api_key: str | None = Field(default=None, repr=False)
    llm_base_url: str | None = None
    llm_model: str | None = None

    @classmethod
    def load(cls, root: Path) -> Settings:
        resolved = root.resolve()
        return cls(
            root=resolved,
            data_dir=resolved / "data",
            ledger_dir=resolved / "forecasts",
            reports_dir=resolved / "reports",
        )
