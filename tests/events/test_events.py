from datetime import UTC, datetime, timedelta

from market_lab.contracts import AssetClass, EvidenceGrade
from market_lab.events.causal import CausalReasoner, MarketContext, build_ablation_payload
from market_lab.events.ingest import EventDocument, EventPipeline

UTC = UTC


def document(
    document_id: str,
    *,
    title: str = "央行宣布降息25个基点",
    body: str = "央行宣布将政策利率下调25个基点。",
    source: str = "official-a",
    source_uri: str | None = None,
    first_seen_hour: int = 8,
    revision: int = 1,
    metadata: dict[str, object] | None = None,
) -> EventDocument:
    observed = datetime(2026, 11, 20, first_seen_hour, tzinfo=UTC)
    return EventDocument(
        document_id=document_id,
        source=source,
        source_uri=source_uri or f"https://example.test/{source}/{document_id}/{revision}",
        title=title,
        body=body,
        published_at=observed - timedelta(minutes=5),
        first_seen_at=observed,
        available_at=observed,
        revision=revision,
        evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
        metadata=metadata
        or {
            "event_key": "policy-rate-2026-11-20",
            "event_type": "monetary_easing",
            "entities": ["央行"],
            "numeric_values": {"basis_points": 25.0},
            "affected_assets": ["AU9999.SGE", "A_SHARE"],
            "half_life_hours": 48,
        },
    )


def test_pipeline_respects_first_seen_cutoff_and_preserves_revisions() -> None:
    revision_one = document("rate", first_seen_hour=8, revision=1)
    revision_two = document(
        "rate",
        title="央行修订公告：降息幅度为20个基点",
        body="央行修订此前公告，政策利率下调20个基点。",
        first_seen_hour=10,
        revision=2,
        metadata={
            **revision_one.metadata,
            "numeric_values": {"basis_points": 20.0},
        },
    )
    pipeline = EventPipeline()

    early = pipeline.process([revision_one, revision_two], datetime(2026, 11, 20, 9, tzinfo=UTC))
    late = pipeline.process([revision_one, revision_two], datetime(2026, 11, 20, 11, tzinfo=UTC))

    assert [fact.revision for fact in early] == [1]
    assert [fact.revision for fact in late] == [1, 2]
    assert late[0].first_seen_at == revision_one.first_seen_at
    assert late[1].first_seen_at == revision_two.first_seen_at


def test_pipeline_deduplicates_syndication_and_excludes_opinion() -> None:
    first = document("wire-a", source="official-a")
    syndicated = document(
        "wire-b",
        source="wire-b",
        source_uri="https://wire.test/b",
        first_seen_hour=9,
    )
    opinion = document(
        "opinion",
        title="分析师看多黄金并给出目标价",
        body="我们预计黄金将大涨，目标价900元。",
        source="broker-opinion",
        first_seen_hour=9,
        metadata={"is_opinion": True},
    )

    facts = EventPipeline().process(
        [first, syndicated, opinion], datetime(2026, 11, 20, 10, tzinfo=UTC)
    )

    assert len(facts) == 1
    assert facts[0].source_uris == tuple(sorted((first.source_uri, syndicated.source_uri)))
    assert facts[0].confirmed
    assert facts[0].strict_eligible


def test_conflicting_official_facts_are_preserved_but_fail_strict_eligibility() -> None:
    first = document("rate-a", source="official-a")
    conflict = document(
        "rate-b",
        source="official-b",
        metadata={
            **first.metadata,
            "numeric_values": {"basis_points": 50.0},
        },
    )

    facts = EventPipeline().process([first, conflict], datetime(2026, 11, 20, 10, tzinfo=UTC))

    assert len(facts) == 2
    assert all(not fact.confirmed for fact in facts)
    assert all(not fact.strict_eligible for fact in facts)
    assert all(
        fact.conflict_source_uris == tuple(sorted((first.source_uri, conflict.source_uri)))
        for fact in facts
    )


def test_causal_reasoning_uses_channels_counterevidence_and_time_decay() -> None:
    fact = EventPipeline().process([document("rate")], datetime(2026, 11, 20, 9, tzinfo=UTC))[0]
    reasoner = CausalReasoner()
    context = MarketContext(
        asset_id="AU9999.SGE",
        asset_class=AssetClass.GOLD,
        as_of=datetime(2026, 11, 20, 9, tzinfo=UTC),
        exposures={"real_rates": -1.0, "usd": -0.5, "risk_aversion": 0.3},
    )

    recent = reasoner.infer([fact], context)
    stale = reasoner.infer(
        [fact],
        context.model_copy(update={"as_of": context.as_of + timedelta(hours=96)}),
    )

    assert recent.direction_score > 0
    assert abs(recent.direction_score) > abs(stale.direction_score)
    assert "lower_real_rates" in recent.supporting_channels
    assert "inflation_reacceleration" in recent.opposing_channels
    assert recent.supporting_fact_ids == (fact.event_id,)


def test_ablation_payload_keeps_no_event_event_only_and_combined_views() -> None:
    payload = build_ablation_payload(
        base_features={"momentum": 0.1, "volatility": 0.2},
        event_features={"event_score": 0.3},
    )

    assert payload.no_event == {"momentum": 0.1, "volatility": 0.2}
    assert payload.event_only == {"event_score": 0.3}
    assert payload.combined == {
        "event_score": 0.3,
        "momentum": 0.1,
        "volatility": 0.2,
    }
