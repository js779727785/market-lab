from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_lab.provenance import canonical_sha256


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class ChallengerEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenger_model_key: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    shadow_started_at: datetime
    shadow_ended_at: datetime
    frozen_forecast_ids: tuple[str, ...]
    gates_passed: bool
    evaluation_hash: str = Field(min_length=64, max_length=64)

    _validate_times = field_validator("shadow_started_at", "shadow_ended_at")(_aware)

    @model_validator(mode="after")
    def validate_shadow_period(self) -> ChallengerEvidence:
        if self.shadow_ended_at <= self.shadow_started_at:
            raise ValueError("shadow period must end after it starts")
        return self


class PromotionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    incumbent_champion_model_key: str
    challenger_model_key: str
    reviewed_at: datetime
    approved: bool
    reasons: tuple[str, ...]
    recommended_champion_model_key: str
    hypothesis_id: str
    evaluation_hash: str
    shadow_sample_count: int = Field(ge=0)
    previous_record_hash: str | None
    record_hash: str

    _validate_reviewed = field_validator("reviewed_at")(_aware)


class PromotionWorkflow:
    def __init__(
        self,
        path: Path,
        *,
        champion_model_key: str,
        min_shadow_samples: int,
    ) -> None:
        if min_shadow_samples < 0:
            raise ValueError("min_shadow_samples must be non-negative")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.champion_model_key = champion_model_key
        self.min_shadow_samples = min_shadow_samples

    def review(self, evidence: ChallengerEvidence, *, reviewed_at: datetime) -> PromotionRecord:
        _aware(reviewed_at)
        reasons: list[str] = []
        if evidence.challenger_model_key == self.champion_model_key:
            reasons.append("challenger_is_incumbent")
        if not evidence.gates_passed:
            reasons.append("promotion_gates")
        if reviewed_at < evidence.shadow_ended_at:
            reasons.append("shadow_period_not_complete")
        if len(evidence.frozen_forecast_ids) < self.min_shadow_samples:
            reasons.append("shadow_sample_count")
        if len(set(evidence.frozen_forecast_ids)) != len(evidence.frozen_forecast_ids):
            reasons.append("duplicate_frozen_forecasts")
        approved = not reasons
        history = self.history()
        previous_hash = history[-1].record_hash if history else None
        provisional = PromotionRecord(
            incumbent_champion_model_key=self.champion_model_key,
            challenger_model_key=evidence.challenger_model_key,
            reviewed_at=reviewed_at,
            approved=approved,
            reasons=tuple(reasons),
            recommended_champion_model_key=(
                evidence.challenger_model_key if approved else self.champion_model_key
            ),
            hypothesis_id=evidence.hypothesis_id,
            evaluation_hash=evidence.evaluation_hash,
            shadow_sample_count=len(evidence.frozen_forecast_ids),
            previous_record_hash=previous_hash,
            record_hash="",
        )
        payload = provisional.model_dump(mode="json", exclude={"record_hash"})
        record = provisional.model_copy(update={"record_hash": canonical_sha256(payload)})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record

    def history(self) -> tuple[PromotionRecord, ...]:
        if not self.path.exists():
            return ()
        records = tuple(
            PromotionRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        previous: str | None = None
        for record in records:
            if record.previous_record_hash != previous:
                raise ValueError("promotion history hash chain is broken")
            payload = record.model_dump(mode="json", exclude={"record_hash"})
            if canonical_sha256(payload) != record.record_hash:
                raise ValueError("promotion history record was modified")
            previous = record.record_hash
        return records
