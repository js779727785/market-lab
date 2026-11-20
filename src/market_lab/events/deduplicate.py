from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from market_lab.provenance import canonical_sha256

if TYPE_CHECKING:
    from .ingest import EventDocument


def normalize_text(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


def document_fingerprint(document: EventDocument) -> str:
    return canonical_sha256(
        {
            "title": normalize_text(document.title),
            "body": normalize_text(document.body),
            "event_key": document.metadata.get("event_key"),
            "event_type": document.metadata.get("event_type"),
            "numeric_values": document.metadata.get("numeric_values", {}),
            "revision": document.revision,
        }
    )


def deduplicate_documents(
    documents: Sequence[EventDocument],
) -> tuple[tuple[EventDocument, ...], ...]:
    grouped: dict[str, list[EventDocument]] = defaultdict(list)
    for document in documents:
        grouped[document_fingerprint(document)].append(document)
    return tuple(
        tuple(sorted(group, key=lambda item: (item.first_seen_at, item.source_uri)))
        for _, group in sorted(
            grouped.items(),
            key=lambda item: min(document.first_seen_at for document in item[1]),
        )
    )
