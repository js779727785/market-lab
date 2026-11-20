from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from market_lab.config import Settings
from market_lab.contracts import (
    ActionState,
    AssetClass,
    EvidenceGrade,
    ForecastDecision,
    PriceBar,
    RecordEnvelope,
)

UTC = UTC


def test_record_envelope_requires_timezone_aware_timestamps() -> None:
    with pytest.raises(ValidationError):
        RecordEnvelope(
            dataset="prices",
            record_key="600000.SH:2026-11-20",
            source="fixture",
            source_uri="file:///fixture.csv",
            evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
            event_time=datetime(2026, 11, 20, 7, 0),
            published_at=datetime(2026, 11, 20, 7, 1, tzinfo=UTC),
            first_seen_at=datetime(2026, 11, 20, 7, 2, tzinfo=UTC),
            available_at=datetime(2026, 11, 20, 7, 2, tzinfo=UTC),
            revision=1,
            payload={"close": 10.0},
        )


def test_price_bar_rejects_non_positive_and_inconsistent_ohlc() -> None:
    with pytest.raises(ValidationError):
        PriceBar(
            asset_id="600000.SH",
            asset_class=AssetClass.A_SHARE,
            session_date="2026-11-20",
            open=10.0,
            high=9.9,
            low=9.8,
            close=10.1,
            volume=1000,
            turnover=10_000,
        )


def test_forecast_decision_is_immutable() -> None:
    decision = ForecastDecision(
        decision_id="fixture-1",
        asset_id="AU9999.SGE",
        asset_class=AssetClass.GOLD,
        forecast_cutoff=datetime(2026, 11, 20, 8, 0, tzinfo=UTC),
        valid_until=datetime(2026, 11, 27, 8, 0, tzinfo=UTC),
        horizon_days=5,
        action=ActionState.WATCH,
        evidence_mode="research",
        probability_up=0.55,
        expected_return=0.01,
        return_interval=(-0.03, 0.05),
        max_adverse_excursion_probability=0.2,
        reasons=("fixture",),
        counterevidence=("wide interval",),
    )

    with pytest.raises(ValidationError):
        decision.action = ActionState.BUY


def test_evidence_grade_has_no_implicit_strict_value() -> None:
    assert {grade.value for grade in EvidenceGrade} == {
        "official_primary",
        "licensed_primary_transport",
        "independent_crosscheck",
        "derived",
        "unverified",
    }


def test_settings_loads_project_paths_from_root(tmp_path) -> None:
    settings = Settings.load(tmp_path)

    assert settings.root == tmp_path.resolve()
    assert settings.data_dir == tmp_path.resolve() / "data"
    assert settings.ledger_dir == tmp_path.resolve() / "forecasts"
