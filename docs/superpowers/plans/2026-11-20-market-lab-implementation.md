# A股与黄金决策实验室 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, auditable A-share and gold forecasting system with point-in-time data, leakage-resistant backtests, event reasoning, portfolio decisions, immutable forecasts, and a local web product.

**Architecture:** A Python package owns data contracts, provider adapters, features, models, backtests, decision rules, forecast ledger, research registry, and FastAPI endpoints. DuckDB queries immutable Parquet snapshots; the frontend consumes only versioned API payloads. Strict outputs require data-quality, model-selection, calibration, execution-cost, and frozen-evidence gates; unsupported data remains explicitly research-only.

**Tech Stack:** Python 3.12, uv, Pydantic 2, DuckDB, PyArrow, Polars, scikit-learn, LightGBM, FastAPI, httpx, pytest, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-11-20-a-share-gold-decision-lab-design.md`

## Global Constraints

- Do not promise profit, guaranteed direction, fixed accuracy, or exact future prices.
- Strict models may read only records with `available_at <= forecast_cutoff`.
- Keep delisted, suspended, ST, historical constituent, corporate-action, and trading-rule states.
- Freeze forecasts before outcomes and append settlements without mutating frozen payloads.
- External news and documents are untrusted data; extracted facts require source references.
- Automatic research may register challengers but may not promote them without the configured gates.
- Missing licensed data downgrades affected outputs to research mode; it must never be hidden.
- Do not commit or create branches unless the user explicitly requests it.

## Review Focus

- Late or revised records: a later revision must remain invisible to an earlier forecast cutoff; Task 2 tests this.
- Duplicate/conflicting sources: strict ingestion must fail closed and preserve conflict evidence; Task 3 tests this.
- Untradeable A-share sessions: suspended, limit-up buys and limit-down sells must not fill; Task 6 tests this.
- High reported accuracy with tiny coverage: model reports must expose action coverage beside accuracy; Task 8 tests this.
- Tampering and replay: changing a frozen forecast, model snapshot, or prior ledger link must fail verification; Task 4 tests this.

---

### Task 1: Project Bootstrap And Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `src/market_lab/__init__.py`
- Create: `src/market_lab/config.py`
- Create: `src/market_lab/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Produces: `Settings.load(root: Path) -> Settings`
- Produces: `EvidenceGrade`, `AssetClass`, `RecordEnvelope`, `PriceBar`, `SecurityMasterRecord`, `CorporateAction`, `MacroObservation`, `EventFact`, `ForecastDecision`, `PositionInput`

- [ ] Write contract tests for UTC-aware timestamps, positive prices, ordered OHLC values, valid evidence grades, and immutable forecast decisions.
- [ ] Run `uv run pytest tests/test_contracts.py -v` and verify the new tests fail.
- [ ] Implement the package metadata, settings loader, and Pydantic contracts with Python 3.12.
- [ ] Run `uv run pytest tests/test_contracts.py -v` and verify all tests pass.

### Task 2: Point-In-Time Store And Provenance

**Files:**
- Create: `src/market_lab/storage.py`
- Create: `src/market_lab/provenance.py`
- Create: `src/market_lab/quality.py`
- Create: `tests/test_storage.py`
- Create: `tests/test_quality.py`

**Interfaces:**
- Consumes: contracts from Task 1.
- Produces: `MarketStore(root: Path)` with `append_records`, `as_of`, `latest`, `dataset_manifest`, and `verify_dataset`.
- Produces: `QualityGate.run(dataset: str, records: Sequence[RecordEnvelope]) -> QualityReport`.

- [ ] Write tests proving as-of queries hide later revisions and preserve both source and normalized hashes.
- [ ] Write tests for duplicate keys, conflicting official records, stale data, invalid price bars, and evidence-grade coverage.
- [ ] Implement immutable Parquet partitions, a DuckDB catalog, canonical JSON hashing, manifests, and fail-closed quality reports.
- [ ] Run `uv run pytest tests/test_storage.py tests/test_quality.py -v` and verify all tests pass.

### Task 3: Provider Adapters And Ingestion CLI

**Files:**
- Create: `src/market_lab/providers/base.py`
- Create: `src/market_lab/providers/files.py`
- Create: `src/market_lab/providers/tushare.py`
- Create: `src/market_lab/providers/baostock.py`
- Create: `src/market_lab/providers/sge.py`
- Create: `src/market_lab/providers/fred.py`
- Create: `src/market_lab/providers/announcements.py`
- Create: `src/market_lab/ingest.py`
- Create: `tests/providers/test_providers.py`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Produces: `Provider.fetch(request: FetchRequest) -> AsyncIterator[RecordEnvelope]`.
- Produces: CLI commands `market-lab ingest`, `market-lab verify-data`, and `market-lab data-status`.

- [ ] Write fixture-based provider tests for field mapping, timestamps, source URIs, hashes, pagination, retries, and source conflicts.
- [ ] Implement file imports first, then optional Tushare/Baostock A-share, SGE gold, FRED/ALFRED macro, and official-announcement adapters.
- [ ] Make credentials optional and downgrade unavailable adapters without fabricating data.
- [ ] Run `uv run pytest tests/providers tests/test_ingest.py -v` and verify all tests pass.

### Task 4: Tamper-Evident Forecast Ledger

**Files:**
- Create: `src/market_lab/ledger.py`
- Create: `src/market_lab/snapshots.py`
- Create: `tests/test_ledger.py`

**Interfaces:**
- Produces: `ForecastLedger.freeze(ForecastDecision, SnapshotBundle) -> LedgerEntry`.
- Produces: `ForecastLedger.settle(entry_id: str, outcome: SettlementOutcome) -> SettlementEntry`.
- Produces: `ForecastLedger.verify() -> LedgerVerification`.

- [ ] Write tests for idempotent freeze, append-only settlement, payload tampering, broken previous-hash links, model-hash changes, and duplicate settlements.
- [ ] Implement canonical payload hashing, source/model/config snapshot hashes, and a chained JSONL ledger.
- [ ] Run `uv run pytest tests/test_ledger.py -v` and verify all tests pass.

### Task 5: Features, Labels, And Point-In-Time Datasets

**Files:**
- Create: `src/market_lab/features/technical.py`
- Create: `src/market_lab/features/fundamental.py`
- Create: `src/market_lab/features/macro.py`
- Create: `src/market_lab/features/gold.py`
- Create: `src/market_lab/features/dataset.py`
- Create: `src/market_lab/labels.py`
- Create: `tests/features/test_features.py`
- Create: `tests/test_labels.py`

**Interfaces:**
- Produces: `FeatureBuilder.build(asset_ids, cutoff, horizon) -> FeatureFrame`.
- Produces: `build_forward_labels(prices, horizons=(1,5,20,60)) -> LabelFrame`.
- Produces: `build_triple_barrier_labels(prices, volatility, horizon, upper, lower) -> LabelFrame`.

- [ ] Write tests proving features use only prior observations, financials enter on announcement availability, and gold conversion separates USD moves from CNY moves.
- [ ] Write tests for forward excess returns, maximum favorable/adverse excursion, and triple-barrier labels.
- [ ] Implement deterministic feature definitions with version identifiers and missingness indicators.
- [ ] Run `uv run pytest tests/features tests/test_labels.py -v` and verify all tests pass.

### Task 6: Walk-Forward Backtest And Execution Simulator

**Files:**
- Create: `src/market_lab/backtest/splits.py`
- Create: `src/market_lab/backtest/execution.py`
- Create: `src/market_lab/backtest/engine.py`
- Create: `src/market_lab/backtest/costs.py`
- Create: `tests/backtest/test_splits.py`
- Create: `tests/backtest/test_execution.py`
- Create: `tests/backtest/test_engine.py`

**Interfaces:**
- Produces: `PurgedWalkForward.split(index, label_end, config) -> Iterator[Fold]`.
- Produces: `ExecutionSimulator.fill(order, market_state, rules) -> FillResult`.
- Produces: `BacktestEngine.run(model_spec, dataset, execution_config) -> BacktestReport`.

- [ ] Write split tests for purge and embargo around overlapping 5/20-day labels.
- [ ] Write execution tests for suspension, limit-up buy, limit-down sell, board-specific rules, fees, taxes, slippage, and double-cost stress.
- [ ] Implement expanding walk-forward evaluation with point-in-time universe membership and immutable case records.
- [ ] Run `uv run pytest tests/backtest -v` and verify all tests pass.

### Task 7: Baselines, Candidate Models, And Calibration

**Files:**
- Create: `src/market_lab/models/base.py`
- Create: `src/market_lab/models/baselines.py`
- Create: `src/market_lab/models/tree.py`
- Create: `src/market_lab/models/calibration.py`
- Create: `src/market_lab/models/registry.py`
- Create: `tests/models/test_models.py`
- Create: `tests/models/test_calibration.py`

**Interfaces:**
- Produces: `ModelRegistry.register`, `ModelRegistry.fit_predict`, and versioned `ModelArtifact`.
- Produces: market/industry, random, momentum, reversal, regularized-linear, and LightGBM candidates.
- Produces: calibrated direction probabilities and quantile return intervals.

- [ ] Write deterministic model and serialization tests using a small fixture dataset.
- [ ] Write probability calibration and interval-coverage tests that fit calibration only inside each training fold.
- [ ] Implement baseline and candidate models behind one interface with fixed seeds and feature manifests.
- [ ] Run `uv run pytest tests/models -v` and verify all tests pass.

### Task 8: Metrics, Model Selection, And Abstention

**Files:**
- Create: `src/market_lab/evaluation/metrics.py`
- Create: `src/market_lab/evaluation/bootstrap.py`
- Create: `src/market_lab/evaluation/selection.py`
- Create: `src/market_lab/abstention.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_selection.py`

**Interfaces:**
- Produces: `evaluate_predictions(cases) -> EvaluationReport`.
- Produces: `ChampionSelector.select(champion, challengers, gates) -> SelectionDecision`.
- Produces: `AbstentionPolicy.decide(prediction, data_status, drift_status) -> ActionState`.

- [ ] Test RankIC, Brier/log loss, calibration error, net returns, drawdown, turnover, action accuracy, and action coverage.
- [ ] Test rejection when bootstrap lower bounds, double-cost returns, calibration, stability, or shadow sample gates fail.
- [ ] Implement block bootstrap comparisons, false-discovery metadata, cross-regime stability, and explicit abstention reasons.
- [ ] Run `uv run pytest tests/evaluation -v` and verify all tests pass.

### Task 9: News Facts And Causal Event Reasoning

**Files:**
- Create: `src/market_lab/events/ingest.py`
- Create: `src/market_lab/events/deduplicate.py`
- Create: `src/market_lab/events/extract.py`
- Create: `src/market_lab/events/causal.py`
- Create: `src/market_lab/events/security.py`
- Create: `tests/events/test_events.py`
- Create: `tests/events/test_security.py`

**Interfaces:**
- Produces: `EventPipeline.process(documents, cutoff) -> Sequence[EventFact]`.
- Produces: `CausalReasoner.infer(facts, market_context) -> EventImpact` with supporting and opposing channels.
- Produces: optional `StructuredLLMExtractor` behind a provider-neutral interface; deterministic extraction remains available.

- [ ] Write tests for first-seen cutoff, revision preservation, syndicated-news deduplication, opinion exclusion, conflicting facts, evidence links, and time decay.
- [ ] Write prompt-injection tests proving source text cannot alter system behavior or tool configuration.
- [ ] Implement deterministic rules for official events and a schema-constrained optional LLM extractor for complex text.
- [ ] Add event-only versus no-event versus combined ablation payloads.
- [ ] Run `uv run pytest tests/events -v` and verify all tests pass.

### Task 10: Decision Engine And Position Assistant

**Files:**
- Create: `src/market_lab/decisions/levels.py`
- Create: `src/market_lab/decisions/sizing.py`
- Create: `src/market_lab/decisions/portfolio.py`
- Create: `src/market_lab/decisions/assistant.py`
- Create: `tests/decisions/test_decisions.py`

**Interfaces:**
- Produces: `DecisionEngine.for_candidate(prediction, context) -> ForecastDecision`.
- Produces: `PositionAssistant.evaluate(PositionInput, prediction, portfolio) -> PositionAdvice`.

- [ ] Test buy zones, chase invalidation, risk invalidation, reduce zones, expiry, volatility sizing, concentration limits, product tracking error, and missing-risk-budget behavior.
- [ ] Implement price levels from calibrated quantiles and execution constraints, never from generated prose.
- [ ] Implement hold/add/reduce/exit/watch/abstain states with supporting and opposing evidence.
- [ ] Run `uv run pytest tests/decisions -v` and verify all tests pass.

### Task 11: Research Flywheel And Frozen Case Audits

**Files:**
- Create: `src/market_lab/research/registry.py`
- Create: `src/market_lab/research/failures.py`
- Create: `src/market_lab/research/cases.py`
- Create: `src/market_lab/research/promotion.py`
- Create: `tests/research/test_research.py`

**Interfaces:**
- Produces: `HypothesisRegistry.register`, `FailureAnalyzer.cluster`, `CaseSelector.select`, and `PromotionWorkflow.review`.

- [ ] Test that hypotheses store registration time and target metric before evaluation.
- [ ] Test objective case selection includes successful and failed 2023-11-20 through 2026-11-20 cases.
- [ ] Test challengers cannot overwrite champions and require a frozen shadow period.
- [ ] Implement failure clusters, event-case reports, ablation reports, and append-only promotion history.
- [ ] Run `uv run pytest tests/research -v` and verify all tests pass.

### Task 12: API, Web Product, And End-To-End Workflow

**Files:**
- Create: `src/market_lab/api.py`
- Create: `src/market_lab/service.py`
- Create: `src/market_lab/cli.py`
- Create: `web/index.html`
- Create: `web/styles.css`
- Create: `web/app.js`
- Create: `tests/test_api.py`
- Create: `tests/test_end_to_end.py`
- Create: `README.md`

**Interfaces:**
- Produces: `/api/status`, `/api/opportunities`, `/api/gold`, `/api/positions/evaluate`, `/api/events`, `/api/backtests`, `/api/ledger`, `/api/research`.
- Produces: CLI `collect`, `train`, `backtest`, `forecast`, `settle`, `serve`, and `run-daily`.

- [ ] Write API tests for strict/research evidence labels, abstention reasons, position evaluation, ledger verification, and missing-data behavior.
- [ ] Write an end-to-end fixture workflow from ingestion through feature build, backtest, decision, freeze, settlement, and dashboard response.
- [ ] Implement the FastAPI service, responsive operational UI, and README runbook.
- [ ] Run `uv run pytest -v`, `uv run market-lab demo`, and a local server smoke test.
- [ ] Inspect desktop and mobile screenshots for overlap, truncation, empty charts, and misleading evidence labels.

### Task 13: Historical Run And Final Verification

**Files:**
- Create: `configs/default.toml`
- Create: `configs/model_gates.toml`
- Create: `reports/.gitkeep`
- Create: `scripts/run_daily.sh`

**Interfaces:**
- Consumes all earlier tasks.
- Produces a reproducible data-status report, available-history backtest, three-year audit when source coverage permits, and a frozen next-session forecast.

- [ ] Configure source availability, transaction costs, prediction horizons, risk limits, and promotion gates.
- [ ] Run all data quality checks and record exact source coverage and limitations.
- [ ] Run the longest valid walk-forward backtest plus the fixed recent-three-year audit without filling unavailable history.
- [ ] Freeze one A-share and gold forecast bundle and verify the full ledger chain.
- [ ] Run `uv run pytest -v`, `uv run ruff check .`, and `uv run mypy src`.
- [ ] Document any unavailable licensed sources, insufficient evidence, or models that remain research-only.
