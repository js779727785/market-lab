from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from market_lab.contracts import AssetClass


class FailureCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    asset_class: AssetClass
    regime: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    predicted_return: float
    realized_return: float


class FailureCluster(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_class: AssetClass
    regime: str
    error_type: str
    event_type: str
    case_ids: tuple[str, ...]
    count: int = Field(gt=0)
    mean_absolute_error: float = Field(ge=0)


class FailureAnalyzer:
    def cluster(self, failures: Sequence[FailureCase]) -> tuple[FailureCluster, ...]:
        grouped: dict[tuple[AssetClass, str, str, str], list[FailureCase]] = defaultdict(list)
        for failure in failures:
            grouped[
                (
                    failure.asset_class,
                    failure.regime,
                    failure.error_type,
                    failure.event_type,
                )
            ].append(failure)
        return tuple(
            FailureCluster(
                asset_class=key[0],
                regime=key[1],
                error_type=key[2],
                event_type=key[3],
                case_ids=tuple(sorted(item.case_id for item in items)),
                count=len(items),
                mean_absolute_error=sum(
                    abs(item.predicted_return - item.realized_return) for item in items
                )
                / len(items),
            )
            for key, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0])))
        )
