# Market Lab

可审计的 A 股全市场选股、黄金判断和持仓辅助系统。系统默认不做自动实盘交易；数据质量、校准、经济价值、漂移或证据门槛未通过时输出 `ABSTAIN`。

## 当前能力

- Point-in-time Parquet 数据仓与 DuckDB as-of 查询。
- 数据来源、修订、首见时间、哈希清单和冲突门禁。
- A 股与黄金特征、purged walk-forward、交易成本和双倍成本压力。
- 可解释基线、LightGBM 挑战者接口、概率与区间校准。
- RankIC、概率损失、回撤、换手、行动准确率和覆盖率。
- 新闻事实抽取、转载去重、冲突降级、因果通道与提示注入防护。
- 分位数价格区间、风险预算仓位、持仓助手和防篡改预测账本。
- 假设预注册、失败聚类、客观案例审计和冻结影子晋级建议。
- FastAPI、CLI 和响应式本地操作界面。

## 安装

```bash
uv sync
```

## 演示工作流

演示数据是确定性的合成行情，只验证系统链路，不能作为历史或实时投资证据。

```bash
uv run market-lab demo
uv run market-lab serve --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。

## CLI

```bash
uv run market-lab collect
uv run market-lab train
uv run market-lab backtest
uv run market-lab forecast
uv run market-lab settle FORECAST_ID START_PRICE END_PRICE
uv run market-lab historical-run
uv run market-lab run-daily
uv run market-lab run-daily --demo
```

`train`、`backtest` 和 `forecast` 的内置样例明确标注 `demo`。`historical-run` 与默认 `run-daily` 只审计当前可用数据；历史不足时报告 `insufficient_recent_three_year_coverage`，不会静默补齐缺失区间。`run-daily --demo` 才会显式执行合成链路。

## 数据源

- 上金所官方页面适配器：`Au99.99`。
- FRED/ALFRED：宏观与历史 vintage；没有真实 vintage 时不回填可用时间。
- Tushare：可选授权传输，需要 `MARKET_LAB_TUSHARE_TOKEN`。
- Baostock：独立交叉核验接口。
- 官方公告：注入来源专用解析器。

缺少授权、历史新闻覆盖或足够冻结影子样本时，受影响输出保持 `research` 或 `ABSTAIN`。

## 验证

```bash
uv run pytest -v
uv run ruff check .
uv run mypy src
```
