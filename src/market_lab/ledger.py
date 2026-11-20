from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ForecastDecision
from .provenance import canonical_sha256
from .snapshots import SnapshotBundle


class LedgerIntegrityError(RuntimeError):
    pass


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class SettlementOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    settled_at: datetime
    start_price: float = Field(gt=0)
    end_price: float = Field(gt=0)
    realized_return: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    tradeable: bool
    notes: tuple[str, ...] = ()

    _validate_settled_at = field_validator("settled_at")(_aware)

    @model_validator(mode="after")
    def validate_excursions(self) -> SettlementOutcome:
        if self.max_favorable_excursion < self.max_adverse_excursion:
            raise ValueError("maximum favorable excursion cannot be below adverse excursion")
        return self


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    entry_type: Literal["forecast", "settlement"]
    entry_id: str
    recorded_at: datetime
    previous_hash: str | None
    payload: dict[str, Any]
    entry_hash: str

    _validate_recorded_at = field_validator("recorded_at")(_aware)

    @property
    def forecast_entry_hash(self) -> str | None:
        value = self.payload.get("forecast_entry_hash")
        return str(value) if value is not None else None


class LedgerVerificationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    entry_id: str | None = None


class LedgerVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    entries: int
    errors: tuple[LedgerVerificationError, ...]


class ForecastLedger:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ledger.jsonl"

    def freeze(
        self,
        decision: ForecastDecision,
        snapshots: SnapshotBundle,
    ) -> LedgerEntry:
        payload = {
            "decision": decision.model_dump(mode="json"),
            "snapshots": snapshots.model_dump(mode="json"),
        }
        for entry in self._read_entries():
            if entry.entry_type == "forecast" and entry.entry_id == decision.decision_id:
                if entry.payload != payload:
                    raise LedgerIntegrityError(
                        f"forecast {decision.decision_id} already exists with different payload"
                    )
                return entry
        entry = self._build_entry(
            entry_type="forecast",
            entry_id=decision.decision_id,
            recorded_at=datetime.now(UTC),
            payload=payload,
        )
        self._append(entry)
        return entry

    def settle(self, entry_id: str, outcome: SettlementOutcome) -> LedgerEntry:
        entries = self._read_entries()
        forecast = next(
            (
                entry
                for entry in entries
                if entry.entry_type == "forecast" and entry.entry_id == entry_id
            ),
            None,
        )
        if forecast is None:
            raise LedgerIntegrityError(f"forecast {entry_id} not found")
        if any(
            entry.entry_type == "settlement" and entry.payload.get("forecast_entry_id") == entry_id
            for entry in entries
        ):
            raise LedgerIntegrityError(f"forecast {entry_id} is already settled")
        entry = self._build_entry(
            entry_type="settlement",
            entry_id=f"settlement:{entry_id}",
            recorded_at=outcome.settled_at,
            payload={
                "forecast_entry_id": entry_id,
                "forecast_entry_hash": forecast.entry_hash,
                "outcome": outcome.model_dump(mode="json"),
            },
        )
        self._append(entry)
        return entry

    def verify(self) -> LedgerVerification:
        errors: list[LedgerVerificationError] = []
        entries: list[LedgerEntry] = []
        if not self.path.exists():
            return LedgerVerification(valid=True, entries=0, errors=())
        previous_hash: str | None = None
        forecasts: dict[str, LedgerEntry] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                raw = json.loads(line)
                entry = LedgerEntry.model_validate(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(
                    LedgerVerificationError(
                        code="invalid_entry",
                        message=f"line {line_number}: {exc}",
                    )
                )
                continue
            entries.append(entry)
            expected_hash = canonical_sha256(entry.model_dump(mode="json", exclude={"entry_hash"}))
            if entry.entry_hash != expected_hash:
                errors.append(
                    LedgerVerificationError(
                        code="entry_hash_mismatch",
                        message="entry payload does not match entry hash",
                        entry_id=entry.entry_id,
                    )
                )
            if entry.previous_hash != previous_hash:
                errors.append(
                    LedgerVerificationError(
                        code="previous_hash_mismatch",
                        message="entry does not link to the previous ledger hash",
                        entry_id=entry.entry_id,
                    )
                )
            previous_hash = entry.entry_hash
            if entry.entry_type == "forecast":
                forecasts[entry.entry_id] = entry
                self._verify_snapshot_payload(entry, errors)
            else:
                forecast_id = str(entry.payload.get("forecast_entry_id", ""))
                forecast = forecasts.get(forecast_id)
                if forecast is None:
                    errors.append(
                        LedgerVerificationError(
                            code="forecast_reference_missing",
                            message="settlement references an unknown earlier forecast",
                            entry_id=entry.entry_id,
                        )
                    )
                elif entry.payload.get("forecast_entry_hash") != forecast.entry_hash:
                    errors.append(
                        LedgerVerificationError(
                            code="forecast_hash_mismatch",
                            message="settlement references a different forecast hash",
                            entry_id=entry.entry_id,
                        )
                    )
        return LedgerVerification(
            valid=not errors,
            entries=len(entries),
            errors=tuple(errors),
        )

    def _build_entry(
        self,
        *,
        entry_type: Literal["forecast", "settlement"],
        entry_id: str,
        recorded_at: datetime,
        payload: dict[str, Any],
    ) -> LedgerEntry:
        entries = self._read_entries()
        draft = LedgerEntry(
            schema_version=1,
            entry_type=entry_type,
            entry_id=entry_id,
            recorded_at=recorded_at,
            previous_hash=entries[-1].entry_hash if entries else None,
            payload=payload,
            entry_hash="",
        )
        entry_hash = canonical_sha256(draft.model_dump(mode="json", exclude={"entry_hash"}))
        return draft.model_copy(update={"entry_hash": entry_hash})

    def _append(self, entry: LedgerEntry) -> None:
        payload = json.dumps(
            entry.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LedgerEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(LedgerEntry.model_validate_json(line))
        return entries

    @staticmethod
    def _verify_snapshot_payload(
        entry: LedgerEntry,
        errors: list[LedgerVerificationError],
    ) -> None:
        try:
            snapshots = SnapshotBundle.model_validate(entry.payload["snapshots"])
        except (KeyError, ValueError) as exc:
            errors.append(
                LedgerVerificationError(
                    code="invalid_snapshot_bundle",
                    message=str(exc),
                    entry_id=entry.entry_id,
                )
            )
            return
        for reference in snapshots.references():
            if not reference.valid():
                errors.append(
                    LedgerVerificationError(
                        code="snapshot_hash_mismatch",
                        message=f"snapshot changed or disappeared: {reference.path}",
                        entry_id=entry.entry_id,
                    )
                )
