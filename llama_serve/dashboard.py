"""A single-file live view of the server, served at `/dashboard`.

Deliberately dependency-free: no build step, no CDN, no framework. It polls
`/metrics.json` and `/metrics/requests` and renders what it gets. The point is
to be able to watch the scheduler behave — queue depth rising, slots filling,
preemptions firing, cache hit rate climbing as prefixes get published — while a
benchmark runs, without leaving the terminal to set up Grafana.

Prometheus remains the real integration surface. This is a debugging window.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-serve</title>
<style>
  :root {
    --bg: #fbfbfa; --fg: #1a1a19; --muted: #6b6b66; --line: #e3e3df;
    --card: #ffffff; --accent: #b5502a; --good: #2f6f4f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17181a; --fg: #e8e8e4; --muted: #97978f; --line: #2c2e31;
      --card: #1e2023; --accent: #e08b5f; --good: #6fbf95;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
    font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
  }
  h1 { font-size: 17px; margin: 0 0 2px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 12.5px; margin-bottom: 20px; }
  .grid {
    display: grid; gap: 12px; margin-bottom: 20px;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px 14px;
  }
  .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase;
       letter-spacing: 0.04em; }
  .v { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 3px; }
  .v small { font-size: 12px; color: var(--muted); font-weight: normal; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em;
       color: var(--muted); margin: 22px 0 8px; font-weight: 600; }
  .scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line);
           white-space: nowrap; font-size: 12.5px; }
  th { color: var(--muted); font-weight: 600; text-align: right;
       position: sticky; top: 0; background: var(--card); }
  th:first-child, td:first-child { text-align: left; }
  tr:last-child td { border-bottom: none; }
  .p { color: var(--accent); }
  .ok { color: var(--good); }
  footer { color: var(--muted); font-size: 12px; margin-top: 22px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
</style>
</head>
<body>
<h1>llama-serve</h1>
<div class="sub" id="sub">connecting…</div>

<div class="grid" id="tiles"></div>

<h2>Latency, over real completed requests</h2>
<div class="scroll"><table id="lat">
  <thead><tr><th>metric</th><th>n</th><th>mean</th><th>p50</th><th>p90</th><th>p99</th><th>max</th></tr></thead>
  <tbody></tbody>
</table></div>

<h2>Recent requests</h2>
<div class="scroll"><table id="reqs">
  <thead><tr><th>id</th><th>prio</th><th>queue s</th><th>ttft s</th><th>total s</th>
    <th>tok/s</th><th>prompt</th><th>cached</th><th>gen</th><th>preempt</th><th>finish</th></tr></thead>
  <tbody></tbody>
</table></div>

<footer>
  Scrape <code>/metrics</code> for Prometheus, <code>/metrics.json</code> for this
  data as JSON, <code>/metrics/requests</code> for raw per-request rows.
  A quantile is blank when there are too few samples to justify it.
</footer>

<script>
const fmt = (v, d = 3) => (v === null || v === undefined) ? "—" : (+v).toFixed(d);
const int = (v) => (v === null || v === undefined) ? "—" : (+v).toLocaleString();

function tile(k, v, sub) {
  return `<div class="card"><div class="k">${k}</div>
          <div class="v">${v}${sub ? ` <small>${sub}</small>` : ""}</div></div>`;
}

async function tick() {
  let m, rq;
  try {
    [m, rq] = await Promise.all([
      fetch("/metrics.json").then(r => r.json()),
      fetch("/metrics/requests?limit=25").then(r => r.json()),
    ]);
  } catch (e) {
    document.getElementById("sub").textContent = "server unreachable";
    return;
  }
  const e = m.engine || {}, kv = e.kv_cache || {}, sc = e.scheduler || {};
  document.getElementById("sub").textContent =
    `${e.engine || "engine"} · ${int(e.max_concurrent_seqs)} slots · policy ${sc.policy || "—"}` +
    ` · window ${fmt(m.window_s, 1)}s`;

  document.getElementById("tiles").innerHTML = [
    tile("in flight", int(m.requests.in_flight)),
    tile("queued", int(e.pending)),
    tile("running", int(e.running), `/ ${int(e.max_concurrent_seqs)}`),
    tile("finished", int(m.requests.finished), m.requests.errored ? `${m.requests.errored} err` : ""),
    tile("throughput", fmt(m.requests.throughput_rps, 2), "req/s"),
    tile("generated", fmt(m.tokens.generated_tps, 1), "tok/s"),
    tile("slot use", fmt(100 * (e.avg_slot_utilization || 0), 1) + "%"),
    tile("step", fmt(e.avg_step_ms, 1), "ms"),
    tile("prefix cache", fmt(100 * (kv.hit_rate || 0), 1) + "%",
         kv.total_blocks ? `${int(kv.used_blocks)}/${int(kv.total_blocks)} blk` : ""),
    tile("preemptions", int(sc.preemptions || 0), `${int(sc.promotions || 0)} promoted`),
  ].join("");

  const lat = m.latency || {};
  document.querySelector("#lat tbody").innerHTML = Object.entries(lat).map(([k, s]) => `
    <tr><td>${k}</td><td>${int(s.n)}</td><td>${fmt(s.mean)}</td><td>${fmt(s.p50)}</td>
    <td>${fmt(s.p90)}</td><td>${fmt(s.p99)}</td><td>${fmt(s.max)}</td></tr>`).join("");

  document.querySelector("#reqs tbody").innerHTML = (rq.rows || []).slice().reverse().map(r => `
    <tr><td>${r.id}</td><td>${r.priority}</td><td>${fmt(r.queue_time_s)}</td>
    <td>${fmt(r.ttft_s)}</td><td>${fmt(r.total_time_s)}</td><td>${fmt(r.output_tps, 1)}</td>
    <td>${int(r.prompt_tokens)}</td>
    <td class="${r.cached_prompt_tokens ? "ok" : ""}">${int(r.cached_prompt_tokens)}</td>
    <td>${int(r.generated_tokens)}</td>
    <td class="${r.preemptions ? "p" : ""}">${int(r.preemptions)}</td>
    <td>${r.finish_reason ?? "—"}</td></tr>`).join("");
}

tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""
