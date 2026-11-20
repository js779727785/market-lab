import json
from datetime import UTC, datetime

import pytest

from market_lab.contracts import ActionState, AssetClass, ForecastDecision
from market_lab.ledger import (
    ForecastLedger,
    LedgerIntegrityError,
    SettlementOutcome,
)
from market_lab.snapshots import SnapshotBundle

UTC = UTC


def decision() -> ForecastDecision:
    return ForecastDecision(
        decision_id="AU9999.SGE:2026-11-20:5",
        asset_id="AU9999.SGE",
        asset_class=AssetClass.GOLD,
        forecast_cutoff=datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
        valid_until=datetime(2026, 11, 27, 9, 0, tzinfo=UTC),
        horizon_days=5,
        action=ActionState.WATCH,
        evidence_mode="research",
        probability_up=0.57,
        expected_return=0.012,
        return_interval=(-0.025, 0.048),
        max_adverse_excursion_probability=0.24,
        reasons=("positive real-rate impulse weakened",),
        counterevidence=("interval remains wide",),
    )


def snapshots(tmp_path) -> SnapshotBundle:
    data = tmp_path / "prices.parquet"
    model = tmp_path / "model.bin"
    config = tmp_path / "config.toml"
    code = tmp_path / "source.tar"
    data.write_bytes(b"data")
    model.write_bytes(b"model")
    config.write_text("gate = 0.55\n", encoding="utf-8")
    code.write_bytes(b"code")
    return SnapshotBundle.capture(
        data_files=(data,),
        model_file=model,
        config_file=config,
        code_file=code,
    )


def outcome() -> SettlementOutcome:
    return SettlementOutcome(
        settled_at=datetime(2026, 11, 27, 9, 5, tzinfo=UTC),
        start_price=905.0,
        end_price=918.0,
        realized_return=918.0 / 905.0 - 1.0,
        max_favorable_excursion=0.025,
        max_adverse_excursion=-0.011,
        tradeable=True,
    )


def test_freeze_is_idempotent_for_identical_payload(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger")
    bundle = snapshots(tmp_path)

    first = ledger.freeze(decision(), bundle)
    second = ledger.freeze(decision(), bundle)

    assert first.entry_hash == second.entry_hash
    assert len(ledger.path.read_text(encoding="utf-8").splitlines()) == 1
    assert ledger.verify().valid is True


def test_settlement_appends_without_mutating_frozen_entry(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger")
    frozen = ledger.freeze(decision(), snapshots(tmp_path))
    first_line = ledger.path.read_text(encoding="utf-8").splitlines()[0]

    settled = ledger.settle(frozen.entry_id, outcome())

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == first_line
    assert len(lines) == 2
    assert settled.forecast_entry_hash == frozen.entry_hash
    assert ledger.verify().valid is True


def test_verification_detects_payload_tampering_and_broken_chain(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger")
    frozen = ledger.freeze(decision(), snapshots(tmp_path))
    ledger.settle(frozen.entry_id, outcome())
    rows = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    rows[0]["payload"]["decision"]["expected_return"] = 99.0
    rows[1]["previous_hash"] = "broken"
    ledger.path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    verification = ledger.verify()

    assert verification.valid is False
    assert {error.code for error in verification.errors} >= {
        "entry_hash_mismatch",
        "previous_hash_mismatch",
    }


def test_verification_detects_model_snapshot_change(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger")
    bundle = snapshots(tmp_path)
    ledger.freeze(decision(), bundle)
    bundle.model.path.write_bytes(b"changed model")

    verification = ledger.verify()

    assert verification.valid is False
    assert "snapshot_hash_mismatch" in {error.code for error in verification.errors}


def test_duplicate_settlement_is_rejected(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger")
    frozen = ledger.freeze(decision(), snapshots(tmp_path))
    ledger.settle(frozen.entry_id, outcome())

    with pytest.raises(LedgerIntegrityError):
        ledger.settle(frozen.entry_id, outcome())
