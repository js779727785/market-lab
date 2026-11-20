from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_lab.contracts import AssetClass


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class TriggerRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    threshold: float = Field(gt=0)
    registered_at: datetime

    _validate_registered = field_validator("registered_at")(_aware)


class AuditCaseCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    asset_class: AssetClass
    forecast_cutoff: datetime
    rule: TriggerRule
    trigger_value: float
    predicted_return: float
    realized_return: float
    snapshot_hash: str = Field(min_length=64, max_length=64)

    _validate_cutoff = field_validator("forecast_cutoff")(_aware)


class SelectedAuditCase(AuditCaseCandidate):
    outcome: Literal["success", "failure"]


class CaseSelector:
    DEFAULT_START = datetime(2023, 11, 20, tzinfo=UTC)
    DEFAULT_END = datetime(2026, 11, 20, 23, 59, 59, tzinfo=UTC)

    def select(
        self,
        candidates: Sequence[AuditCaseCandidate],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[SelectedAuditCase, ...]:
        lower = start or self.DEFAULT_START
        upper = end or self.DEFAULT_END
        _aware(lower)
        _aware(upper)
        if lower > upper:
            raise ValueError("case selection start must not be after end")
        selected: list[SelectedAuditCase] = []
        for candidate in candidates:
            if not lower <= candidate.forecast_cutoff <= upper:
                continue
            if candidate.rule.registered_at >= candidate.forecast_cutoff:
                continue
            if abs(candidate.trigger_value) < candidate.rule.threshold:
                continue
            prediction_sign = (
                1 if candidate.predicted_return > 0 else -1 if candidate.predicted_return < 0 else 0
            )
            realized_sign = (
                1 if candidate.realized_return > 0 else -1 if candidate.realized_return < 0 else 0
            )
            selected.append(
                SelectedAuditCase(
                    **candidate.model_dump(),
                    outcome=(
                        "success"
                        if prediction_sign == realized_sign and prediction_sign
                        else "failure"
                    ),
                )
            )
        return tuple(sorted(selected, key=lambda case: (case.forecast_cutoff, case.case_id)))
