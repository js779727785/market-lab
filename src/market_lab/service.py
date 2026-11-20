from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
from pydantic import BaseModel, ConfigDict

from .backtest.engine import BacktestEngine
from .backtest.splits import SplitConfig
from .config import Settings
from .contracts import AssetClass, EvidenceGrade, ForecastDecision, PositionInput, RecordEnvelope
from .decisions.assistant import HoldingPrediction, PositionAssistant
from .decisions.levels import CalibratedPrediction, DecisionContext, DecisionEngine
from .decisions.portfolio import PortfolioState
from .events.causal import CausalReasoner, MarketContext
from .events.ingest import EventDocument, EventPipeline
from .features.dataset import FeatureBuilder
from .ingest import IngestionService
from .labels import build_forward_labels
from .ledger import ForecastLedger, LedgerIntegrityError, SettlementOutcome
from .models.baselines import MomentumBaseline
from .provenance import canonical_sha256
from .quality import QualityGate
from .snapshots import SnapshotBundle
from .storage import MarketStore

DEMO_CUTOFF = datetime(2026, 11, 20, 8, tzinfo=UTC)


class DemoWorkflowSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    evidence_mode: str = "demo"
    ingested_rows: int
    feature_rows: int
    backtest_cases: int
    frozen_forecasts: int
    settled_forecasts: int
    state_path: str


class HistoricalAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str
    evidence_mode: str
    source_coverage: dict[str, dict[str, Any]]
    available_history_start: str | None
    available_history_end: str | None
    longest_valid_backtest_cases: int
    recent_three_year_start: str = "2023-11-20"
    recent_three_year_end: str = "2026-11-20"
    recent_three_year_complete: bool
    recent_three_year_case_count: int
    frozen_forecast_assets: tuple[str, ...]
    ledger_valid: bool
    limitations: tuple[str, ...]
    report_path: str


class MarketLabService:
    def __init__(self, root: Path) -> None:
        self.settings = Settings.load(root)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        self.store = MarketStore(self.settings.data_dir / "store")
        self.ledger = ForecastLedger(self.settings.ledger_dir)
        self.state_path = self.settings.reports_dir / "dashboard_state.json"
        self.historical_report_path = self.settings.reports_dir / "historical_run.json"

    def _state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("dashboard state must be a JSON object")
        return cast(dict[str, Any], value)

    def status(self) -> dict[str, Any]:
        state = self._state()
        if state is None:
            return {
                "schema_version": 1,
                "evidence_mode": "unavailable",
                "generated_at": None,
                "strict_forecasts": 0,
                "research_forecasts": 0,
                "live_trading_enabled": False,
                "source_limitations": [
                    "no_forecast_bundle",
                    "licensed_sources_not_configured",
                ],
            }
        return {
            "schema_version": 1,
            "evidence_mode": state["evidence_mode"],
            "generated_at": state["generated_at"],
            "strict_forecasts": 0,
            "research_forecasts": len(state["opportunities"]) + 1,
            "live_trading_enabled": False,
            "source_limitations": state["source_limitations"],
        }

    def opportunities(self) -> dict[str, Any]:
        state = self._state()
        if state is None:
            return {
                "evidence_mode": "unavailable",
                "items": [],
                "reasons": ["no_forecast_bundle"],
            }
        return {
            "evidence_mode": "research",
            "items": state["opportunities"],
            "reasons": [],
        }

    def gold(self) -> dict[str, Any]:
        state = self._state()
        if state is None:
            return {
                "evidence_mode": "unavailable",
                "decision": None,
                "abstention_reasons": ["no_forecast_bundle"],
            }
        value = state["gold"]
        if not isinstance(value, dict):
            raise ValueError("gold state must be a JSON object")
        return cast(dict[str, Any], value)

    def events(self) -> dict[str, Any]:
        state = self._state()
        return {
            "schema_version": 1,
            "evidence_mode": "research" if state else "unavailable",
            "items": state["events"] if state else [],
        }

    def backtests(self) -> dict[str, Any]:
        state = self._state()
        return {
            "schema_version": 1,
            "evidence_mode": "historical_demo" if state else "unavailable",
            "reports": state["backtests"] if state else [],
        }

    def ledger_status(self) -> dict[str, Any]:
        verification = self.ledger.verify()
        return {
            "schema_version": 1,
            "verification": verification.model_dump(mode="json"),
            "entries": self._ledger_rows(),
        }

    def research(self) -> dict[str, Any]:
        state = self._state()
        return {
            "schema_version": 1,
            "evidence_mode": "research" if state else "unavailable",
            "champion": "momentum-baseline-v1" if state else None,
            "challengers": [],
            "notes": (
                ["demo fixture is not promotion evidence"] if state else ["no research state"]
            ),
        }

    def source_status(self) -> dict[str, Any]:
        datasets = {}
        store_root = self.settings.data_dir / "store"
        if store_root.exists():
            for path in sorted(store_root.iterdir()):
                if path.is_dir():
                    datasets[path.name] = self.store.dataset_manifest(path.name).model_dump(
                        mode="json"
                    )
        return {
            "schema_version": 1,
            "datasets": datasets,
            "licensed_sources": {
                "tushare": bool(self.settings.tushare_token),
                "fred": bool(self.settings.fred_api_key),
            },
            "limitations": [
                key
                for key, available in {
                    "tushare_token_missing": bool(self.settings.tushare_token),
                    "fred_api_key_missing": bool(self.settings.fred_api_key),
                }.items()
                if not available
            ],
        }

    def evaluate_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self._state()
        if state is None:
            raise LookupError("no forecast bundle")
        opportunity = next(
            (
                item
                for item in state["opportunities"]
                if item["decision"]["asset_id"] == payload["asset_id"]
            ),
            None,
        )
        if opportunity is None:
            raise KeyError(payload["asset_id"])
        decision = ForecastDecision.model_validate(opportunity["decision"])
        position = PositionInput(
            asset_id=payload["asset_id"],
            asset_class=decision.asset_class,
            average_cost=payload["average_cost"],
            quantity=payload["quantity"],
            available_cash=payload.get("available_cash", 0.0),
            horizon_days=payload.get("horizon_days", decision.horizon_days),
            max_loss_fraction=payload.get("max_loss_fraction"),
            allow_add=payload.get("allow_add", True),
            allow_scale=payload.get("allow_scale", True),
            sector_limit=payload.get("sector_limit"),
        )
        current_price = float(opportunity["current_price"])
        account_value = max(
            position.average_cost * position.quantity + position.available_cash,
            current_price * position.quantity,
            1.0,
        )
        advice = PositionAssistant().evaluate(
            position,
            HoldingPrediction(
                decision=decision,
                current_price=current_price,
                annualized_volatility=float(opportunity["annualized_volatility"]),
                sector=str(opportunity["sector"]),
            ),
            PortfolioState(account_value=account_value, cash=position.available_cash),
        )
        return {
            "schema_version": 1,
            "asset_id": position.asset_id,
            "evidence_mode": decision.evidence_mode,
            "advice": advice.model_dump(mode="json"),
        }

    def run_demo(self, *, force: bool = False) -> DemoWorkflowSummary:
        existing = self._state()
        if existing is not None and not force:
            return DemoWorkflowSummary.model_validate(existing["workflow"])
        records = self._demo_price_records()
        ingestion = IngestionService(
            self.store,
            QualityGate(max_staleness=timedelta(days=3650)),
        ).ingest(records)
        feature_frame = FeatureBuilder(self.store).build(
            ["600000.SH", "AU9999.SGE"], DEMO_CUTOFF, horizon=5
        )
        labeled = (
            build_forward_labels(feature_frame, horizons=(1,))
            .drop_nulls(["forward_return_1", "label_end_1", "return_1"])
            .with_columns(
                pl.col("forward_return_1").alias("target_return"),
                pl.col("label_end_1").alias("label_end"),
                pl.lit(True).alias("tradable"),
            )
        )
        backtest = BacktestEngine(
            SplitConfig(
                min_train_periods=20,
                test_periods=4,
                step_periods=4,
                embargo_periods=1,
            )
        ).run(MomentumBaseline(feature="return_1"), labeled)
        code_bytes = Path(__file__).read_bytes()
        code_sha256 = hashlib.sha256(code_bytes).hexdigest()
        decisions, decision_metadata = self._demo_decisions(feature_frame, code_sha256=code_sha256)
        bundle = self._demo_snapshots(ingestion.files, code_bytes=code_bytes)
        frozen = [self.ledger.freeze(decision, bundle) for decision in decisions]
        settled = 0
        if not self._has_settlement(decisions[0].decision_id):
            start_price = float(decision_metadata[decisions[0].decision_id]["current_price"])
            end_price = start_price * 1.02
            self.ledger.settle(
                decisions[0].decision_id,
                SettlementOutcome(
                    settled_at=DEMO_CUTOFF + timedelta(days=6),
                    start_price=start_price,
                    end_price=end_price,
                    realized_return=end_price / start_price - 1.0,
                    max_favorable_excursion=0.03,
                    max_adverse_excursion=-0.01,
                    tradeable=True,
                    notes=("synthetic fixture settlement",),
                ),
            )
            settled = 1
        else:
            settled = 1
        event_payload = self._demo_event()
        a_share_decision, gold_decision = decisions
        a_share_meta = decision_metadata[a_share_decision.decision_id]
        gold_meta = decision_metadata[gold_decision.decision_id]
        summary = DemoWorkflowSummary(
            ingested_rows=len(records),
            feature_rows=feature_frame.height,
            backtest_cases=len(backtest.cases),
            frozen_forecasts=len(frozen),
            settled_forecasts=settled,
            state_path=str(self.state_path),
        )
        state = {
            "schema_version": 1,
            "evidence_mode": "demo",
            "generated_at": DEMO_CUTOFF.isoformat(),
            "source_limitations": [
                "synthetic_price_fixture",
                "no_licensed_historical_news",
                "not_live_evidence",
            ],
            "opportunities": [
                {
                    "symbol": "浦发银行演示样本",
                    "asset_class": "a_share",
                    "sector": "bank",
                    "current_price": a_share_meta["current_price"],
                    "annualized_volatility": a_share_meta["annualized_volatility"],
                    "evidence_mode": "research",
                    "abstention_reasons": a_share_meta["abstention_reasons"],
                    "decision": a_share_decision.model_dump(mode="json"),
                }
            ],
            "gold": {
                "evidence_mode": "research",
                "symbol": "Au99.99 演示样本",
                "current_price": gold_meta["current_price"],
                "abstention_reasons": gold_meta["abstention_reasons"],
                "decision": gold_decision.model_dump(mode="json"),
            },
            "events": event_payload,
            "backtests": [
                {
                    "model_key": backtest.model_key,
                    "case_count": len(backtest.cases),
                    "net_returns": [case.net_return for case in backtest.cases],
                    "stressed_net_returns": [case.stressed_net_return for case in backtest.cases],
                    "case_dates": [case.session_date.isoformat() for case in backtest.cases],
                    "evidence_mode": "historical_demo",
                }
            ],
            "workflow": summary.model_dump(mode="json"),
        }
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def run_available_history_audit(self) -> HistoricalAuditReport:
        state = self._state()
        source_status = self.source_status()
        coverage = {name: dict(manifest) for name, manifest in source_status["datasets"].items()}
        price_manifest = coverage.get("prices", {"rows": 0})
        price_records = self.store.latest("prices") if price_manifest.get("rows", 0) else []
        available_dates = sorted(
            {
                str(record.payload["session_date"])
                for record in price_records
                if record.payload.get("session_date")
            }
        )
        history_start = available_dates[0] if available_dates else None
        history_end = available_dates[-1] if available_dates else None
        recent_start = date(2023, 11, 20)
        recent_end = date(2026, 11, 20)
        recent_complete = bool(
            history_start
            and history_end
            and date.fromisoformat(history_start) <= recent_start
            and date.fromisoformat(history_end) >= recent_end
        )
        backtests = state.get("backtests", []) if state else []
        case_dates = [
            date.fromisoformat(value)
            for report in backtests
            for value in report.get("case_dates", [])
        ]
        recent_case_count = (
            sum(recent_start <= value <= recent_end for value in case_dates)
            if recent_complete
            else 0
        )
        ledger_rows = self._ledger_rows()
        frozen_assets = tuple(
            sorted(
                {
                    str(row["payload"]["decision"]["asset_id"])
                    for row in ledger_rows
                    if row.get("entry_type") == "forecast"
                }
            )
        )
        limitations = list(source_status["limitations"])
        if state:
            limitations.extend(state.get("source_limitations", []))
        if not recent_complete:
            limitations.append("insufficient_recent_three_year_coverage")
        limitations = list(dict.fromkeys(limitations))
        verification = self.ledger.verify()
        provisional = HistoricalAuditReport(
            generated_at=datetime.now(UTC).isoformat(),
            evidence_mode=state["evidence_mode"] if state else "unavailable",
            source_coverage=coverage,
            available_history_start=history_start,
            available_history_end=history_end,
            longest_valid_backtest_cases=sum(
                int(report.get("case_count", 0)) for report in backtests
            ),
            recent_three_year_complete=recent_complete,
            recent_three_year_case_count=recent_case_count,
            frozen_forecast_assets=frozen_assets,
            ledger_valid=verification.valid,
            limitations=tuple(limitations),
            report_path=str(self.historical_report_path),
        )
        self.historical_report_path.write_text(
            provisional.model_dump_json(indent=2), encoding="utf-8"
        )
        return provisional

    def _demo_decisions(
        self, feature_frame: pl.DataFrame, *, code_sha256: str
    ) -> tuple[tuple[ForecastDecision, ForecastDecision], dict[str, dict[str, Any]]]:
        latest = {
            str(row["asset_id"]): row
            for row in feature_frame.sort("session_date").group_by("asset_id").tail(1).to_dicts()
        }
        engine = DecisionEngine()
        a_price = float(latest["600000.SH"]["close"])
        a_prediction = CalibratedPrediction(
            asset_id="600000.SH",
            asset_class=AssetClass.A_SHARE,
            forecast_cutoff=DEMO_CUTOFF,
            horizon_days=5,
            probability_up=0.76,
            expected_return=0.04,
            return_interval=(0.015, 0.08),
            quantile_returns={
                0.05: -0.07,
                0.2: -0.025,
                0.35: -0.01,
                0.6: 0.02,
                0.75: 0.055,
                0.9: 0.09,
            },
            max_adverse_excursion_probability=0.18,
            model_disagreement=0.04,
            supporting_evidence=("synthetic momentum baseline",),
            counterevidence=("demo data cannot establish live edge",),
        )
        a_decision = engine.for_candidate(
            a_prediction,
            DecisionContext(
                current_price=a_price,
                transaction_cost=0.004,
                safety_margin=0.004,
                evidence_mode="research",
            ),
        )
        a_decision = self._version_demo_decision(a_decision, code_sha256)
        a_reasons = (
            list(engine.abstention_policy.last_result.reasons)
            if engine.abstention_policy.last_result
            else []
        )
        gold_price = float(latest["AU9999.SGE"]["close"])
        gold_prediction = CalibratedPrediction(
            asset_id="AU9999.SGE",
            asset_class=AssetClass.GOLD,
            forecast_cutoff=DEMO_CUTOFF,
            horizon_days=5,
            probability_up=0.58,
            expected_return=0.01,
            return_interval=(-0.03, 0.04),
            quantile_returns={
                0.05: -0.05,
                0.2: -0.025,
                0.35: -0.01,
                0.6: 0.01,
                0.75: 0.025,
                0.9: 0.05,
            },
            max_adverse_excursion_probability=0.28,
            model_disagreement=0.12,
            supporting_evidence=("synthetic real-rate scenario",),
            counterevidence=("wide calibrated interval",),
        )
        gold_decision = engine.for_candidate(
            gold_prediction,
            DecisionContext(
                current_price=gold_price,
                transaction_cost=0.003,
                safety_margin=0.004,
                evidence_mode="research",
            ),
        )
        gold_decision = self._version_demo_decision(gold_decision, code_sha256)
        gold_reasons = (
            list(engine.abstention_policy.last_result.reasons)
            if engine.abstention_policy.last_result
            else []
        )
        metadata = {
            a_decision.decision_id: {
                "current_price": a_price,
                "annualized_volatility": 0.24,
                "abstention_reasons": a_reasons,
            },
            gold_decision.decision_id: {
                "current_price": gold_price,
                "annualized_volatility": 0.18,
                "abstention_reasons": gold_reasons,
            },
        }
        return (a_decision, gold_decision), metadata

    def _demo_event(self) -> list[dict[str, Any]]:
        document = EventDocument(
            document_id="demo-policy-1",
            source="demo-official",
            source_uri="https://example.invalid/demo-policy-1",
            title="演示政策利率调整",
            body="演示文档：政策利率下调25个基点。",
            published_at=DEMO_CUTOFF - timedelta(hours=2),
            first_seen_at=DEMO_CUTOFF - timedelta(hours=1),
            available_at=DEMO_CUTOFF - timedelta(hours=1),
            revision=1,
            evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
            metadata={
                "event_key": "demo-policy",
                "event_type": "monetary_easing",
                "entities": ["演示央行"],
                "numeric_values": {"basis_points": 25},
                "affected_assets": ["AU9999.SGE", "A_SHARE"],
                "half_life_hours": 48,
            },
        )
        facts = EventPipeline().process([document], DEMO_CUTOFF)
        impact = CausalReasoner().infer(
            facts,
            MarketContext(
                asset_id="AU9999.SGE",
                asset_class=AssetClass.GOLD,
                as_of=DEMO_CUTOFF,
                exposures={"real_rates": -0.5, "usd": -0.2},
            ),
        )
        return [
            {
                "fact": fact.model_dump(mode="json"),
                "impact": impact.model_dump(mode="json"),
                "evidence_mode": "demo",
            }
            for fact in facts
        ]

    def _demo_snapshots(self, data_files: tuple[Path, ...], *, code_bytes: bytes) -> SnapshotBundle:
        snapshot_dir = self.settings.reports_dir / "demo_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        model_bytes = json.dumps(
            {"model": "momentum-baseline-v1", "feature": "return_1"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        config_bytes = json.dumps(
            {"mode": "demo", "live_trading": False, "horizons": [5, 20]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        model = self._write_demo_snapshot(snapshot_dir, "model", ".json", model_bytes)
        config = self._write_demo_snapshot(snapshot_dir, "config", ".json", config_bytes)
        code = self._write_demo_snapshot(snapshot_dir, "code", ".py", code_bytes)
        return SnapshotBundle.capture(
            data_files=data_files,
            model_file=model,
            config_file=config,
            code_file=code,
        )

    @staticmethod
    def _version_demo_decision(decision: ForecastDecision, code_sha256: str) -> ForecastDecision:
        decision_id = canonical_sha256(
            {
                "base_decision_id": decision.decision_id,
                "demo_code_sha256": code_sha256,
            }
        )
        return decision.model_copy(update={"decision_id": decision_id})

    @staticmethod
    def _write_demo_snapshot(snapshot_dir: Path, kind: str, suffix: str, content: bytes) -> Path:
        digest = hashlib.sha256(content).hexdigest()
        path = snapshot_dir / f"{kind}-{digest}{suffix}"
        try:
            with path.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise LedgerIntegrityError(f"snapshot hash collision at {path}") from None
        return path

    def _demo_price_records(self) -> list[RecordEnvelope]:
        records: list[RecordEnvelope] = []
        start = date(2026, 10, 1)
        for index in range(40):
            session = start + timedelta(days=index)
            available_at = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(
                hours=16
            )
            a_close = 10.0 + index * 0.035 + math.sin(index / 3) * 0.06
            gold_close = 880.0 + index * 0.9 + math.cos(index / 4) * 2.0
            for asset_id, asset_class, close, volume in (
                ("600000.SH", AssetClass.A_SHARE, a_close, 1_000_000 + index * 5_000),
                ("AU9999.SGE", AssetClass.GOLD, gold_close, 50_000 + index * 200),
            ):
                records.append(
                    RecordEnvelope(
                        dataset="prices",
                        record_key=f"{asset_id}:{session.isoformat()}",
                        source="demo-fixture",
                        source_uri=f"file://demo/{asset_id}/{session.isoformat()}",
                        evidence_grade=EvidenceGrade.OFFICIAL_PRIMARY,
                        event_time=available_at,
                        published_at=available_at,
                        first_seen_at=available_at,
                        available_at=available_at,
                        revision=1,
                        payload={
                            "asset_id": asset_id,
                            "asset_class": asset_class.value,
                            "session_date": session.isoformat(),
                            "open": close * 0.997,
                            "high": close * 1.008,
                            "low": close * 0.992,
                            "close": close,
                            "volume": volume,
                            "turnover": close * volume,
                        },
                    )
                )
        return records

    def _ledger_rows(self) -> list[dict[str, Any]]:
        if not self.ledger.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.ledger.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _has_settlement(self, forecast_id: str) -> bool:
        return any(
            row.get("entry_type") == "settlement"
            and row.get("payload", {}).get("forecast_entry_id") == forecast_id
            for row in self._ledger_rows()
        )

    def settle(
        self,
        forecast_id: str,
        *,
        start_price: float,
        end_price: float,
        settled_at: datetime,
    ) -> dict[str, Any]:
        realized = end_price / start_price - 1.0
        try:
            entry = self.ledger.settle(
                forecast_id,
                SettlementOutcome(
                    settled_at=settled_at,
                    start_price=start_price,
                    end_price=end_price,
                    realized_return=realized,
                    max_favorable_excursion=max(realized, 0.0),
                    max_adverse_excursion=min(realized, 0.0),
                    tradeable=True,
                ),
            )
        except LedgerIntegrityError as exc:
            raise ValueError(str(exc)) from exc
        return entry.model_dump(mode="json")
