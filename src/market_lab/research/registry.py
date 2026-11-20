from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_lab.provenance import canonical_sha256


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class HypothesisRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    target_metric: str = Field(min_length=1)
    expected_direction: Literal["increase", "decrease"]
    registered_at: datetime
    status: Literal["registered", "evaluated"]
    evaluated_at: datetime | None = None
    result: dict[str, Any] | None = None

    _validate_registered = field_validator("registered_at")(_aware)
    _validate_evaluated = field_validator("evaluated_at")(
        lambda value: _aware(value) if value is not None else value
    )


class AblationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    evaluated_at: datetime
    metric: str
    no_event_value: float
    event_only_value: float
    combined_value: float
    lower_is_better: bool

    _validate_evaluated = field_validator("evaluated_at")(_aware)

    @property
    def combined_increment(self) -> float:
        difference = self.combined_value - self.no_event_value
        return -difference if self.lower_is_better else difference


class HypothesisRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: HypothesisRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def register(
        self,
        *,
        title: str,
        rationale: str,
        target_metric: str,
        expected_direction: Literal["increase", "decrease"],
        registered_at: datetime,
    ) -> HypothesisRecord:
        _aware(registered_at)
        identifier = canonical_sha256(
            {
                "title": title,
                "target_metric": target_metric,
                "registered_at": registered_at,
            }
        )
        record = HypothesisRecord(
            hypothesis_id=identifier,
            title=title,
            rationale=rationale,
            target_metric=target_metric,
            expected_direction=expected_direction,
            registered_at=registered_at,
            status="registered",
        )
        self._append(record)
        return record

    def record_evaluation(
        self,
        hypothesis_id: str,
        *,
        evaluated_at: datetime,
        result: dict[str, Any],
    ) -> HypothesisRecord:
        candidates = [record for record in self.history() if record.hypothesis_id == hypothesis_id]
        if not candidates:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        original = candidates[0]
        if evaluated_at <= original.registered_at:
            raise ValueError("evaluation must occur after hypothesis registration")
        record = original.model_copy(
            update={"status": "evaluated", "evaluated_at": evaluated_at, "result": result}
        )
        self._append(record)
        return record

    def history(self) -> tuple[HypothesisRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            HypothesisRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
