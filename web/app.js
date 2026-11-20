const state = { status: null, opportunities: null, gold: null, events: null, backtests: null, ledger: null, research: null };

const escapeHtml = (value) => String(value ?? "--")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const pct = (value) => value == null ? "--" : `${(Number(value) * 100).toFixed(2)}%`;
const priceRange = (range) => range ? `${Number(range[0]).toFixed(2)} - ${Number(range[1]).toFixed(2)}` : "--";
const actionClass = (action) => `action-${action || "watch"}`;

async function fetchJson(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `${path} ${response.status}`);
  return payload;
}

async function loadData() {
  const paths = {
    status: "/api/status",
    opportunities: "/api/opportunities",
    gold: "/api/gold",
    events: "/api/events",
    backtests: "/api/backtests",
    ledger: "/api/ledger",
    research: "/api/research",
  };
  const entries = await Promise.all(Object.entries(paths).map(async ([key, path]) => [key, await fetchJson(path)]));
  entries.forEach(([key, value]) => { state[key] = value; });
  render();
}

function render() {
  const status = state.status;
  document.getElementById("mode-badge").textContent = status.evidence_mode.toUpperCase();
  document.getElementById("generated-at").textContent = status.generated_at ? `截止 ${new Date(status.generated_at).toLocaleString("zh-CN")}` : "无预测快照";
  document.getElementById("strict-count").textContent = status.strict_forecasts;
  document.getElementById("research-count").textContent = status.research_forecasts;
  document.getElementById("ledger-valid").textContent = state.ledger.verification.valid ? "通过" : "失败";
  const backtest = state.backtests.reports[0];
  document.getElementById("case-count").textContent = backtest?.case_count ?? 0;
  const alert = document.getElementById("global-alert");
  if (status.source_limitations?.length) {
    alert.textContent = `证据限制：${status.source_limitations.join(" / ")}`;
    alert.classList.remove("hidden");
  } else alert.classList.add("hidden");
  renderOpportunities();
  renderGold();
  renderEvents();
  renderBacktests();
  renderLedger();
  renderResearch();
  drawBacktest(backtest?.net_returns || []);
  if (window.lucide) window.lucide.createIcons();
}

function renderOpportunities() {
  const items = state.opportunities.items || [];
  document.getElementById("opportunity-rows").innerHTML = items.length ? items.map((item) => {
    const d = item.decision;
    return `<tr><td><strong>${escapeHtml(d.asset_id)}</strong><br><span>${escapeHtml(item.symbol)}</span></td><td><span class="action-badge ${actionClass(d.action)}">${escapeHtml(d.action)}</span></td><td>${pct(d.probability_up)}</td><td>${pct(d.expected_return)}</td><td>${priceRange(d.buy_zone)}</td><td>${d.risk_invalidation ? Number(d.risk_invalidation).toFixed(2) : "--"}</td><td>${escapeHtml(d.evidence_mode)}</td></tr>`;
  }).join("") : `<tr><td colspan="7">暂无可验证机会</td></tr>`;
  document.getElementById("stocks-detail").innerHTML = items.map((item) => {
    const d = item.decision;
    return `<div class="record"><h3>${escapeHtml(d.asset_id)} · ${escapeHtml(d.action)}</h3><p>支持：${escapeHtml(d.reasons.join(" / "))}</p><p>反证：${escapeHtml(d.counterevidence.join(" / "))}</p><p>区间：${priceRange(d.return_interval)} · 有效期 ${escapeHtml(d.valid_until)}</p></div>`;
  }).join("") || `<div class="empty-state">无数据</div>`;
}

function detailRows(rows) {
  return rows.map(([label, value]) => `<div class="detail-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
}

function renderGold() {
  const gold = state.gold;
  const d = gold.decision;
  const action = d?.action || "unavailable";
  const badge = document.getElementById("gold-action");
  badge.textContent = action;
  badge.className = `action-badge ${actionClass(action)}`;
  document.getElementById("gold-summary").innerHTML = d ? detailRows([
    ["上涨概率", pct(d.probability_up)],
    ["预期收益", pct(d.expected_return)],
    ["预测区间", priceRange(d.return_interval)],
    ["拒绝原因", gold.abstention_reasons.join(" / ") || "无"],
  ]) : `<div class="empty-state">无数据</div>`;
  document.getElementById("gold-detail").innerHTML = d ? `<div class="record"><h3>${escapeHtml(gold.symbol)} · ${escapeHtml(action)}</h3><p>支持：${escapeHtml(d.reasons.join(" / "))}</p><p>反证：${escapeHtml(d.counterevidence.join(" / "))}</p><p>拒绝原因：${escapeHtml(gold.abstention_reasons.join(" / ") || "无")}</p><p class="mono">decision_id: ${escapeHtml(d.decision_id)}</p></div>` : `<div class="empty-state">无数据</div>`;
}

function renderEvents() {
  document.getElementById("event-list").innerHTML = (state.events.items || []).map((item) => `<div class="record"><h3>${escapeHtml(item.fact.event_type)} · ${escapeHtml(item.fact.statement)}</h3><p>来源：${escapeHtml(item.fact.source_uris.join(" / "))}</p><p>支持通道：${escapeHtml(item.impact.supporting_channels.join(" / "))}</p><p>反向通道：${escapeHtml(item.impact.opposing_channels.join(" / "))}</p><p>方向分数：${Number(item.impact.direction_score).toFixed(3)} · 严格可用：${item.fact.strict_eligible ? "是" : "否"}</p></div>`).join("") || `<div class="empty-state">无事件</div>`;
}

function renderBacktests() {
  document.getElementById("backtest-detail").innerHTML = (state.backtests.reports || []).map((report) => `<div class="record"><h3>${escapeHtml(report.model_key)}</h3><p>案例数：${report.case_count}</p><p>基础成本累计：${pct(report.net_returns.reduce((equity, value) => equity * (1 + value), 1) - 1)}</p><p>双倍成本累计：${pct(report.stressed_net_returns.reduce((equity, value) => equity * (1 + value), 1) - 1)}</p><p>证据：${escapeHtml(report.evidence_mode)}</p></div>`).join("") || `<div class="empty-state">无回测</div>`;
}

function renderLedger() {
  const verification = state.ledger.verification;
  document.getElementById("ledger-detail").innerHTML = `<div class="record"><h3>校验 ${verification.valid ? "通过" : "失败"}</h3><p>条目：${verification.entries}</p><p>错误：${escapeHtml((verification.errors || []).map((item) => item.code).join(" / ") || "无")}</p></div>${(state.ledger.entries || []).slice(-6).reverse().map((entry) => `<div class="record"><h3>${escapeHtml(entry.entry_type)} · ${escapeHtml(entry.entry_id)}</h3><p class="mono">hash: ${escapeHtml(entry.entry_hash)}</p><p>${escapeHtml(entry.recorded_at)}</p></div>`).join("")}`;
}

function renderResearch() {
  document.getElementById("research-detail").innerHTML = `<div class="record"><h3>冠军：${escapeHtml(state.research.champion || "未建立")}</h3><p>挑战者：${state.research.challengers.length}</p><p>${escapeHtml(state.research.notes.join(" / "))}</p></div>`;
}

function drawBacktest(returns) {
  const canvas = document.getElementById("backtest-chart");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f8f9f7";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!returns.length) return;
  const equity = [1];
  returns.forEach((value) => equity.push(equity.at(-1) * (1 + value)));
  const min = Math.min(...equity);
  const max = Math.max(...equity);
  const range = Math.max(max - min, 0.0001);
  context.strokeStyle = "#18794e";
  context.lineWidth = 3;
  context.beginPath();
  equity.forEach((value, index) => {
    const x = 18 + index / Math.max(equity.length - 1, 1) * (canvas.width - 36);
    const y = 18 + (max - value) / range * (canvas.height - 36);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".view").forEach((panel) => panel.classList.remove("active"));
  button.classList.add("active");
  document.querySelector(`[data-view-panel="${button.dataset.view}"]`).classList.add("active");
  document.getElementById("page-title").textContent = button.textContent.trim();
}));

document.getElementById("refresh-button").addEventListener("click", () => loadData().catch(showError));
document.getElementById("position-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = Object.fromEntries(form.entries());
  ["average_cost", "quantity", "available_cash", "max_loss_fraction", "sector_limit"].forEach((key) => { payload[key] = Number(payload[key]); });
  payload.horizon_days = 20;
  payload.allow_add = true;
  payload.allow_scale = true;
  try {
    const result = await fetchJson("/api/positions/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    document.getElementById("position-result").className = "detail-list";
    document.getElementById("position-result").innerHTML = detailRows([
      ["动作", result.advice.action],
      ["目标仓位", pct(result.advice.target_weight)],
      ["目标数量", result.advice.target_quantity ?? "--"],
      ["最大风险金额", result.advice.max_loss_amount == null ? "--" : Number(result.advice.max_loss_amount).toFixed(2)],
      ["约束", result.advice.binding_constraints.join(" / ") || "无"],
      ["原因", result.advice.reasons.join(" / ")],
    ]);
  } catch (error) { showError(error); }
});

function showError(error) {
  const alert = document.getElementById("global-alert");
  alert.textContent = error.message;
  alert.classList.remove("hidden");
}

loadData().catch(showError);
