from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_lab.contracts import AssetClass, EventFact


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class MarketContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    asset_class: AssetClass
    as_of: datetime
    exposures: dict[str, float] = Field(default_factory=dict)

    _validate_as_of = field_validator("as_of")(_aware)


class EventImpact(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    direction_score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    supporting_channels: tuple[str, ...]
    opposing_channels: tuple[str, ...]
    supporting_fact_ids: tuple[str, ...]
    strict_eligible: bool
    average_decay_weight: float = Field(ge=0, le=1)
    horizon_days: int = Field(gt=0)


class AblationPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    no_event: dict[str, float]
    event_only: dict[str, float]
    combined: dict[str, float]


ChannelSpec = tuple[tuple[tuple[str, float], ...], tuple[str, ...]]


_CHANNELS: dict[str, dict[AssetClass, ChannelSpec]] = {
    "monetary_easing": {
        AssetClass.GOLD: (
            (("lower_real_rates", 0.55), ("weaker_usd", 0.25), ("risk_liquidity", 0.1)),
            ("inflation_reacceleration", "policy_already_priced"),
        ),
        AssetClass.A_SHARE: (
            (("lower_discount_rate", 0.4), ("liquidity_support", 0.35)),
            ("margin_pressure", "policy_already_priced"),
        ),
    },
    "buyback": {
        AssetClass.A_SHARE: (
            (("share_count_reduction", 0.35), ("management_signal", 0.25)),
            ("cash_depletion", "execution_uncertainty"),
        )
    },
    "tariff": {
        AssetClass.A_SHARE: (
            (("domestic_substitution", 0.15),),
            ("export_demand_loss", "input_cost_inflation"),
        ),
        AssetClass.GOLD: (
            (("risk_aversion", 0.3), ("inflation_risk", 0.2)),
            ("stronger_usd",),
        ),
    },
}


class CausalReasoner:
    def infer(self, facts: Sequence[EventFact], market_context: MarketContext) -> EventImpact:
        visible = [fact for fact in facts if fact.available_at <= market_context.as_of]
        score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        supporting: list[str] = []
        opposing: list[str] = []
        fact_ids: list[str] = []
        decay_values: list[float] = []
        for fact in visible:
            channel_spec = _CHANNELS.get(fact.event_type, {}).get(market_context.asset_class)
            if channel_spec is None:
                continue
            positive_channels, negative_channels = channel_spec
            age_hours = max(0.0, (market_context.as_of - fact.available_at).total_seconds() / 3600)
            decay = math.pow(0.5, age_hours / fact.decay_half_life_hours)
            relevance = fact.confidence * fact.novelty * decay
            channel_score = sum(weight for _, weight in positive_channels)
            if market_context.asset_class is AssetClass.GOLD:
                channel_score += max(0.0, -market_context.exposures.get("real_rates", 0.0)) * 0.05
                channel_score += max(0.0, -market_context.exposures.get("usd", 0.0)) * 0.04
            score += channel_score * relevance
            weighted_confidence += fact.confidence * relevance
            total_weight += relevance
            decay_values.append(decay)
            supporting.extend(name for name, _ in positive_channels)
            opposing.extend(negative_channels)
            fact_ids.append(fact.event_id)
        return EventImpact(
            asset_id=market_context.asset_id,
            direction_score=max(-1.0, min(1.0, score)),
            confidence=min(1.0, weighted_confidence / total_weight) if total_weight else 0.0,
            supporting_channels=tuple(dict.fromkeys(supporting)),
            opposing_channels=tuple(dict.fromkeys(opposing)),
            supporting_fact_ids=tuple(dict.fromkeys(fact_ids)),
            strict_eligible=bool(fact_ids) and all(fact.strict_eligible for fact in visible),
            average_decay_weight=sum(decay_values) / len(decay_values) if decay_values else 0.0,
            horizon_days=20,
        )


def build_ablation_payload(
    base_features: Mapping[str, float], event_features: Mapping[str, float]
) -> AblationPayload:
    overlap = set(base_features) & set(event_features)
    if overlap:
        raise ValueError(f"base and event feature names overlap: {sorted(overlap)}")
    return AblationPayload(
        no_event=dict(sorted(base_features.items())),
        event_only=dict(sorted(event_features.items())),
        combined=dict(sorted({**base_features, **event_features}.items())),
    )
