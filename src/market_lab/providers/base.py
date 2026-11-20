from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import RecordEnvelope


class FetchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset: str | None = None
    start: date | None = None
    end: date | None = None
    symbols: tuple[str, ...] = ()
    path: Path | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    def fetch(self, request: FetchRequest) -> AsyncIterator[RecordEnvelope]: ...
