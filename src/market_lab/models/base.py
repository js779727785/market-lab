from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_lab.provenance import canonical_sha256


class CandidateModel(Protocol):
    key: str
    feature_names: tuple[str, ...]

    def fit_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> Sequence[float]: ...

    def artifact_state(self) -> dict[str, Any]: ...


class ModelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(default=1, ge=1)
    model_key: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    feature_names: tuple[str, ...]
    parameters: dict[str, Any]
    fitted_state: dict[str, Any]
    sha256: str = ""

    @model_validator(mode="after")
    def populate_or_verify_hash(self) -> ModelArtifact:
        payload = self.model_dump(mode="json", exclude={"sha256"})
        expected = canonical_sha256(payload)
        if self.sha256 and self.sha256 != expected:
            raise ValueError("model artifact hash does not match its payload")
        if not self.sha256:
            object.__setattr__(self, "sha256", expected)
        return self


class ModelRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    predictions: tuple[float, ...]
    artifact: ModelArtifact
