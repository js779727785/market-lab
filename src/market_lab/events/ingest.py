from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_lab.contracts import EventFact, EvidenceGrade
from market_lab.provenance import canonical_sha256

from .deduplicate import deduplicate_documents
from .extract import DeterministicFactExtractor


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class EventDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    published_at: datetime
    first_seen_at: datetime
    available_at: datetime
    revision: int = Field(ge=1)
    evidence_grade: EvidenceGrade
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_sha256: str = ""
    normalized_sha256: str = ""

    _validate_times = field_validator("published_at", "first_seen_at", "available_at")(_aware)

    @model_validator(mode="after")
    def populate_hashes(self) -> EventDocument:
        raw = canonical_sha256({"title": self.title, "body": self.body})
        normalized = canonical_sha256(
            {
                "document_id": self.document_id,
                "source": self.source,
                "published_at": self.published_at,
                "revision": self.revision,
                "metadata": self.metadata,
            }
        )
        if self.raw_sha256 and self.raw_sha256 != raw:
            raise ValueError("raw document hash does not match content")
        if self.normalized_sha256 and self.normalized_sha256 != normalized:
            raise ValueError("normalized document hash does not match content")
        if not self.raw_sha256:
            object.__setattr__(self, "raw_sha256", raw)
        if not self.normalized_sha256:
            object.__setattr__(self, "normalized_sha256", normalized)
        return self


class EventPipeline:
    def __init__(self, extractor: DeterministicFactExtractor | None = None) -> None:
        self.extractor = extractor or DeterministicFactExtractor()

    def process(
        self, documents: Sequence[EventDocument], cutoff: datetime
    ) -> tuple[EventFact, ...]:
        _aware(cutoff)
        visible = [document for document in documents if document.available_at <= cutoff]
        facts = [
            fact
            for group in deduplicate_documents(visible)
            for fact in self.extractor.extract(group)
        ]
        by_conflict_key: dict[tuple[str | None, int], list[EventFact]] = defaultdict(list)
        for fact in facts:
            by_conflict_key[(fact.event_key, fact.revision)].append(fact)
        conflicted_ids: dict[str, tuple[str, ...]] = {}
        for candidates in by_conflict_key.values():
            signatures = {
                canonical_sha256(
                    {"statement": candidate.statement, "numeric_values": candidate.numeric_values}
                )
                for candidate in candidates
            }
            if len(candidates) > 1 and len(signatures) > 1:
                sources = tuple(
                    sorted({uri for candidate in candidates for uri in candidate.source_uris})
                )
                for candidate in candidates:
                    conflicted_ids[candidate.event_id] = sources
        resolved = [
            fact.model_copy(
                update={
                    "confirmed": False,
                    "strict_eligible": False,
                    "conflict_source_uris": conflicted_ids[fact.event_id],
                }
            )
            if fact.event_id in conflicted_ids
            else fact
            for fact in facts
        ]
        return tuple(
            sorted(
                resolved,
                key=lambda fact: (fact.available_at, fact.revision, fact.event_id),
            )
        )
