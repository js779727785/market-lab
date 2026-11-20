from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PortfolioPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    market_value: float = Field(ge=0)
    sector: str | None = None


class PortfolioState(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_value: float = Field(gt=0)
    cash: float = Field(ge=0)
    positions: tuple[PortfolioPosition, ...] = ()

    def asset_weight(self, asset_id: str) -> float:
        value = sum(
            position.market_value for position in self.positions if position.asset_id == asset_id
        )
        return value / self.account_value

    def other_sector_weight(self, sector: str | None, excluding_asset_id: str) -> float:
        if sector is None:
            return 0.0
        value = sum(
            position.market_value
            for position in self.positions
            if position.sector == sector and position.asset_id != excluding_asset_id
        )
        return value / self.account_value
