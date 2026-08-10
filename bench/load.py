"""Milestone 7: the load-testing harness.

Sweeps offered load across a range of QPS levels, repeats every level several
times, and writes both a per-request CSV and a per-level summary with mean and
spread. Run it against two servers to compare engines on identical workloads.

**Open loop, Poisson arrivals.** Requests are launched on a schedule fixed
*before* the run starts — inter-arrival times drawn from an exponential
distribution at rate λ — and the harness never waits for a response before
sending the next request. This is the difference between measuring a server and
measuring yourself. A closed-loop harness with N workers cannot offer more load
than the server can absorb: when the server slows down, the client slows down
with it, latency looks flat, and the queue that would have formed in production
never forms. Open loop lets the queue form, which is the entire thing worth
measuring.

**Every level is run `--runs` times.** A single measurement of a latency tail on
a laptop is a coin flip: thermal state, background processes and page cache all
move it. Levels are reported as mean ± sample standard deviation across runs,
and the raw per-request rows for every run are committed, so the spread can be
checked rather than trusted.

**Saturation is reported, not hidden.** Above the server's capacity the queue
grows for the whole run and latency grows with it, so the numbers at those
levels describe a backlog rather than a steady state. Each level therefore
carries `achieved_qps` alongside `offered_qps`, plus counts of requests that
were rejected (HTTP 503, queue full) or still unfinished at the drain deadline.
A level where achieved falls below offered is saturated, and is labelled that
way in the output instead of being quietly averaged in.

    # find what this machine can actually sustain
    python bench/load.py --probe --max-tokens 32

    # the sweep
    python -m llama_serve.server --engine continuous --max-seqs 8
    python bench/load.py --label continuous --qps 2,4,6,8,10 --runs 3

    python -m llama_serve.server --engine simple --max-seqs 1
    python bench/load.py --label simple-baseline --qps 2,4,6,8,10 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import random
import statistics
import time
from pathlib import Path

import httpx

# A prompt with a shared preamble, because that is what real traffic looks like
# and it is what the prefix cache is for. The per-request tail keeps the
# prompts distinct.
PREAMBLE = (
    "You are a support assistant for a storage product. Answer concisely, cite "
    "the manual section, and never speculate about hardware you were not told "
    "about. Use the same format as the examples you were given.\n\n"
)
QUESTIONS = [
    "A disk shows as foreign after reseating it.",
    "The management interface is unreachable over HTTPS.",
    "Rebuild speed is much slower than documented.",
    "Two controllers disagree about the cluster time.",
    "The replication link keeps flapping every few minutes.",
    "A LUN cannot be expanded past four terabytes.",
    "The event log is filling with checksum warnings.",
    "Cache battery health reports unknown after a power event.",
    "A volume went read-only after a controller failover.",
    "Snapshots are consuming more space than configured.",
    "The array rejects a replacement disk as unsupported.",
    "Latency spikes every night at the same hour.",
]


def pct(values: list[float], q: float) -> float | None:
    """Percentile by nearest index on the sorted samples: `round(q * (n - 1))`.

    No interpolation, and deliberately the same convention as
    `llama_serve.metrics._pct`, so a percentile reported by this harness and one
    scraped from `/metrics` mean the same thing and can be put side by side.
    Returns None for an empty sample rather than inventing a tail.
    """
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 5)


def spread(values: list[float]) -> dict:
    """Mean and sample standard deviation, plus the range."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 5),
        "stdev": round(statistics.stdev(vals), 5) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 5),
        "max": round(max(vals), 5),
    }


async def one_request(client, idx: int, max_tokens: int, timeout: float) -> dict:
    """Send one request; return a row describing what happened to it."""
    prompt = PREAMBLE + QUESTIONS[idx % len(QUESTIONS)] + f" (case {idx})"
    t0 = time.perf_counter()
    row = {
        "idx": idx,
        "sent_t": t0,
        "status": None,
        "client_ttft_s": None,
        "wall_s": None,
        "queue_time_s": None,
        "ttft_s": None,
        "total_time_s": None,
        "output_tps": None,
        "prompt_tokens": None,
        "cached_prompt_tokens": None,
        "generated_tokens": None,
        "preemptions": None,
    }
    try:
        r = await client.post(
            "/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,  # pin output length, or this measures EOS behaviour
            },
            timeout=timeout,
        )
        row["status"] = r.status_code
        row["wall_s"] = time.perf_counter() - t0
        if r.status_code == 200:
            b = r.json()
            t = b["timings"]
            row.update(
                {
                    "queue_time_s": t.get("queue_time_s"),
                    "ttft_s": t.get("ttft_s"),
                    "total_time_s": t.get("total_time_s"),
                    "output_tps": t.get("output_tps"),
                    "prompt_tokens": b["usage"]["prompt_tokens"],
                    "cached_prompt_tokens": b["usage"]["cached_prompt_tokens"],
                    "generated_tokens": b["usage"]["completion_tokens"],
                    "preemptions": t.get("preemptions"),
                }
            )
    except Exception as exc:  # timeout, connection reset, ...
        row["status"] = type(exc).__name__
        row["wall_s"] = time.perf_counter() - t0
    return row


async def run_level(client, qps: float, duration: float, max_tokens: int, seed: int,
                    drain_timeout: float) -> tuple[dict, list[dict]]:
    """One measurement run at one offered-QPS level."""
    rng = random.Random(seed)

    # The whole arrival schedule is fixed before anything is sent, so a slow
    # server cannot slow the offered load down.
    schedule, t = [], 0.0
    while t < duration:
        schedule.append(t)
        t += rng.expovariate(qps)

    await client.post("/metrics/reset")
    tasks: list[asyncio.Task] = []
    t0 = time.perf_counter()

    async def fire(idx: int, at: float):
        delay = at - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        return await one_request(client, idx, max_tokens, drain_timeout)

    for i, at in enumerate(schedule):
        tasks.append(asyncio.create_task(fire(i, at)))

    done, pending = await asyncio.wait(tasks, timeout=duration + drain_timeout)
    for p in pending:
        p.cancel()
    rows = [d.result() for d in done]
    rows.sort(key=lambda r: r["idx"])
    elapsed = time.perf_counter() - t0
    # Throughput is completions per second of *observation*, and the observation
    # window is at least the arrival window. A Poisson draw whose last arrival
    # lands early would otherwise report a throughput the server never sustained.
    wall = max(duration, elapsed)

    ok = [r for r in rows if r["status"] == 200]
    rejected = [r for r in rows if r["status"] == 503]
    failed = [r for r in rows if r["status"] not in (200, 503)]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    qt = [r["queue_time_s"] for r in ok if r["queue_time_s"] is not None]
    tps = [r["output_tps"] for r in ok if r["output_tps"] is not None]
    gen = sum(r["generated_tokens"] or 0 for r in ok)
    cached = sum(r["cached_prompt_tokens"] or 0 for r in ok)
    prompt = sum(r["prompt_tokens"] or 0 for r in ok)

    summary = {
        "offered_qps": qps,
        "scheduled": len(schedule),
        "completed": len(ok),
        "rejected_503": len(rejected),
        "failed": len(failed),
        "unfinished_at_deadline": len(pending),
        "wall_s": round(elapsed, 3),
        "window_s": round(wall, 3),
        # Against the schedule actually drawn, not the nominal rate: a Poisson
        # draw over a short window routinely lands 20% either side of lambda,
        # and dividing by the nominal rate would report that as a server effect.
        "offered_qps_actual": round(len(schedule) / duration, 4),
        "achieved_qps": round(len(ok) / wall, 4) if wall else 0.0,
        "generated_tps": round(gen / wall, 3) if wall else 0.0,
        "ttft_mean_s": round(statistics.fmean(ttft), 5) if ttft else None,
        "ttft_p50_s": pct(ttft, 0.50),
        "ttft_p90_s": pct(ttft, 0.90) if len(ttft) >= 10 else None,
        "ttft_p99_s": pct(ttft, 0.99) if len(ttft) >= 100 else None,
        "ttft_max_s": round(max(ttft), 5) if ttft else None,
        "queue_p50_s": pct(qt, 0.50),
        "queue_p99_s": pct(qt, 0.99) if len(qt) >= 100 else None,
        "output_tps_mean": round(statistics.fmean(tps), 3) if tps else None,
        "prompt_cache_hit_rate": round(cached / prompt, 4) if prompt else 0.0,
    }
    return summary, rows


async def drain(client, timeout: float = 60.0) -> None:
    """Wait until the server reports nothing in flight, so runs do not bleed."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            m = (await client.get("/metrics.json")).json()
            if m["requests"]["in_flight"] <= 0:
                return
        except Exception:
            return
        await asyncio.sleep(0.5)


def summarise(results_path: Path, out_path: Path) -> int:
    """One row per label: the highest offered level the server kept up with.

    "Kept up with" is the harness's own saturation rule — mean achieved QPS at
    least 90% of the mean offered rate actually drawn. Above that level the
    queue grows for the whole run, so the latency describes a backlog and the
    throughput is a drain rate, not a capacity.
    """
    data = json.loads(results_path.read_text())
    lines = [
        "# Highest sustained load, by configuration",
        "",
        f"Generated from `{results_path}` by `bench/load.py --summary`.",
        "",
        "Sustained = the highest offered level whose mean achieved QPS stayed "
        "within 10% of the offered rate, across all runs at that level. TTFT "
        "percentiles are pooled over every completed request at that level.",
        "",
        "| configuration | model | out tok | slots | sustained QPS | p50 TTFT | p99 TTFT | offered | runs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for label, run in data.items():
        ok = [lv for lv in run["levels"] if not lv["saturated"]]
        if not ok:
            continue
        best = max(ok, key=lambda x: x["achieved_qps"]["mean"])
        env, harness = run["environment"], run["harness"]
        model = Path(env["model"]).name if env.get("model") else "?"
        rows.append((best["achieved_qps"]["mean"], label, model, harness, env, best))

    for achieved, label, model, harness, env, best in sorted(rows, reverse=True):
        p = best["ttft_pooled"]
        lines.append(
            f"| `{label}` | {model} | {harness['max_tokens']} | {env['max_seqs']} "
            f"| **{achieved:.2f}** | {p['p50']} s | "
            f"{'—' if p['p99'] is None else str(p['p99']) + ' s'} "
            f"| {best['offered_qps']:g} | {best['runs']} |"
        )

    if rows:
        top = max(rows)
        lines += [
            "",
            f"Peak measured: **{top[0]:.2f} QPS** on `{top[1]}` "
            f"({top[2]}, {top[3]['max_tokens']} output tokens, {top[4]['max_seqs']} sequence slots, "
            f"{top[4]['platform']}).",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_path}")
    return 0


def compare(labels: str, results_path: Path, out_path: Path) -> int:
    """Render the comparison table from data already on disk.

    Separate from the measurement path on purpose: the table in the README is
    generated from the committed results file, so it cannot drift from it.
    """
    base_label, cand_label = (x.strip() for x in labels.split(","))
    data = json.loads(results_path.read_text())
    base = {lv["offered_qps"]: lv for lv in data[base_label]["levels"]}
    cand = {lv["offered_qps"]: lv for lv in data[cand_label]["levels"]}

    def reduction(a, b):
        return f"{100 * (a - b) / a:.1f} %" if a and b else "—"

    def cell(v):
        return "—" if v is None else f"{v:.4f}"

    lines = [
        f"# Load sweep: `{cand_label}` vs `{base_label}`",
        "",
        f"Generated from `{results_path}` by `bench/load.py --compare`. ",
        f"Workload: {data[cand_label]['harness']['workload']}, "
        f"{data[cand_label]['harness']['max_tokens']} output tokens, "
        f"open-loop Poisson arrivals, "
        f"{data[cand_label]['harness']['runs_per_level']} runs of "
        f"{data[cand_label]['harness']['duration_s']} s per level.",
        "",
        f"Model: `{data[cand_label]['environment']['model']}` on "
        f"{data[cand_label]['environment']['platform']}.",
        "",
        "TTFT percentiles are pooled over every completed request at a level "
        "(p90 needs 10 samples, p99 needs 100; below that the cell is `—`). "
        "Achieved QPS is mean ± stdev across runs. **SAT** marks a level where "
        "the server could not keep up, so its latencies describe a growing "
        "backlog rather than a steady state.",
        "",
        "| offered QPS | achieved (base) | achieved (cand) | p50 base | p50 cand | p50 lower | p99 base | p99 cand | p99 lower |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for q in sorted(cand):
        b, c = base[q], cand[q]
        bp, cp = b["ttft_pooled"], c["ttft_pooled"]
        sat = ("<br>SAT base" if b["saturated"] else "") + ("<br>SAT cand" if c["saturated"] else "")
        lines.append(
            f"| {q:g}{sat} | {b['achieved_qps']['mean']:.2f} ± {b['achieved_qps']['stdev']:.2f} "
            f"| {c['achieved_qps']['mean']:.2f} ± {c['achieved_qps']['stdev']:.2f} "
            f"| {cell(bp['p50'])} | {cell(cp['p50'])} | {reduction(bp['p50'], cp['p50'])} "
            f"| {cell(bp['p99'])} | {cell(cp['p99'])} | {reduction(bp['p99'], cp['p99'])} |"
        )

    def best(levels):
        ok = [x for x in levels.values() if not x["saturated"]]
        return max(ok, key=lambda x: x["achieved_qps"]["mean"]) if ok else None

    bb, cb = best(base), best(cand)
    if bb and cb:
        lines += [
            "",
            "## Highest sustained load (last level the server kept up with)",
            "",
            "| | achieved QPS | p50 TTFT | p99 TTFT |",
            "|---|---|---|---|",
            f"| `{base_label}` | {bb['achieved_qps']['mean']:.2f} | {cell(bb['ttft_pooled']['p50'])} s "
            f"| {cell(bb['ttft_pooled']['p99'])} s |",
            f"| `{cand_label}` | {cb['achieved_qps']['mean']:.2f} | {cell(cb['ttft_pooled']['p50'])} s "
            f"| {cell(cb['ttft_pooled']['p99'])} s |",
            "",
            f"Capacity ratio: **{cb['achieved_qps']['mean'] / bb['achieved_qps']['mean']:.2f}x**.",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_path}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", help="run name, used as the results key")
    ap.add_argument("--qps", default="2,4,6,8,10", help="comma-separated offered QPS levels")
    ap.add_argument("--runs", type=int, default=3, help="repeats per level (>= 3 for a spread)")
    ap.add_argument("--duration", type=float, default=12.0, help="seconds of arrivals per run")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--drain-timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--probe", action="store_true", help="find sustainable QPS, then exit")
    ap.add_argument(
        "--compare",
        help="two labels already in --out, as 'baseline,candidate'; renders the "
        "comparison table and exits without running anything",
    )
    ap.add_argument("--compare-out", default="results/load_comparison.md")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="render every label's highest sustained level from --out, then exit",
    )
    ap.add_argument("--summary-out", default="results/load_summary.md")
    ap.add_argument("--out", default="results/load_sweep.json")
    ap.add_argument("--csv-dir", default="results/load_raw")
    args = ap.parse_args()

    levels = [float(x) for x in args.qps.split(",") if x.strip()]

    if args.compare:
        return compare(args.compare, Path(args.out), Path(args.compare_out))
    if args.summary:
        return summarise(Path(args.out), Path(args.summary_out))

    async with httpx.AsyncClient(base_url=args.url, timeout=args.drain_timeout) as c:
        health = (await c.get("/health")).json()
        cfg = (await c.get("/config")).json()

        if args.probe:
            # Ramp until achieved QPS stops tracking offered QPS. Cheap, one run
            # per level, only used to choose the sweep range honestly.
            print(f"probing {health['engine']} (max_seqs={health['max_seqs']})")
            prev, probe_rows = None, []
            for qps in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48):
                s, _ = await run_level(c, qps, 8.0, args.max_tokens, args.seed, 30.0)
                await drain(c)
                ratio = s["achieved_qps"] / s["offered_qps_actual"]
                print(
                    f"  offered {qps:>4} (actual {s['offered_qps_actual']:>5.2f}) -> "
                    f"achieved {s['achieved_qps']:>6.2f} ({ratio:5.0%})  "
                    f"completed {s['completed']}/{s['scheduled']}  "
                    f"ttft p50 {s['ttft_p50_s']}  max {s['ttft_max_s']}"
                )
                probe_rows.append(
                    {
                        "offered_qps": qps,
                        "offered_qps_actual": s["offered_qps_actual"],
                        "achieved_qps": s["achieved_qps"],
                        "tracking_ratio": round(ratio, 4),
                        "completed": s["completed"],
                        "ttft_p50_s": s["ttft_p50_s"],
                        "ttft_max_s": s["ttft_max_s"],
                    }
                )
                if ratio < 0.85:
                    print(f"\n  saturates between {prev} and {qps} QPS at {args.max_tokens} tokens")
                    break
                prev = qps

            probe_out = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "engine": health["engine"],
                "max_seqs": health["max_seqs"],
                "max_tokens": args.max_tokens,
                "duration_s": 8.0,
                "runs_per_level": 1,
                "last_sustained_qps": prev,
                "first_saturated_qps": probe_rows[-1]["offered_qps"] if probe_rows else None,
                "levels": probe_rows,
            }
            path = Path("results/load_probe.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(path.read_text()) if path.exists() else {}
            existing[f"{health['engine']}-{args.max_tokens}tok"] = probe_out
            path.write_text(json.dumps(existing, indent=2) + "\n")
            print(f"wrote {path}")
            return 0

        if not args.label:
            ap.error("--label is required unless --probe is given")

        print(
            f"engine={health['engine']} max_seqs={health['max_seqs']} "
            f"levels={levels} runs={args.runs} duration={args.duration}s "
            f"max_tokens={args.max_tokens}"
        )
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{args.label}.csv"
        fields = [
            "label", "offered_qps", "run", "idx", "status", "wall_s", "queue_time_s",
            "ttft_s", "total_time_s", "output_tps", "prompt_tokens",
            "cached_prompt_tokens", "generated_tokens", "preemptions",
        ]
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()

            level_summaries = []
            for qps in levels:
                runs = []
                level_rows: list[dict] = []
                for run in range(args.runs):
                    s, rows = await run_level(
                        c, qps, args.duration, args.max_tokens,
                        args.seed + run, args.drain_timeout,
                    )
                    s["run"] = run
                    runs.append(s)
                    level_rows.extend(r for r in rows if r["status"] == 200)
                    for r in rows:
                        writer.writerow({**r, "label": args.label, "offered_qps": qps, "run": run})
                    fh.flush()
                    print(
                        f"  {qps:>5} QPS run {run}: achieved {s['achieved_qps']:>6.2f} "
                        f"ttft p50 {s['ttft_p50_s']} p99 {s['ttft_p99_s']} "
                        f"completed {s['completed']}/{s['scheduled']}"
                    )
                    await drain(c)

                # Percentiles are computed over every request at this level,
                # pooled across runs: a p99 needs 100 samples, and one 20-second
                # run at a low rate does not have them. The per-run spread below
                # is what shows the variance; the pooled tail is what shows the
                # tail.
                pooled = [r["ttft_s"] for r in level_rows if r["ttft_s"] is not None]
                pooled_q = {
                    "n": len(pooled),
                    "p50": pct(pooled, 0.50),
                    "p90": pct(pooled, 0.90) if len(pooled) >= 10 else None,
                    "p99": pct(pooled, 0.99) if len(pooled) >= 100 else None,
                    "mean": round(statistics.fmean(pooled), 5) if pooled else None,
                    "max": round(max(pooled), 5) if pooled else None,
                }
                achieved = [r["achieved_qps"] for r in runs]
                offered_actual = [r["offered_qps_actual"] for r in runs]
                saturated = statistics.fmean(achieved) < 0.9 * statistics.fmean(offered_actual)
                level_summaries.append(
                    {
                        "offered_qps": qps,
                        "offered_qps_actual": spread(offered_actual),
                        "runs": args.runs,
                        "saturated": saturated,
                        "achieved_qps": spread(achieved),
                        "ttft_pooled": pooled_q,
                        "ttft_mean_s": spread([r["ttft_mean_s"] for r in runs]),
                        "ttft_p50_s": spread([r["ttft_p50_s"] for r in runs]),
                        "ttft_p90_s": spread([r["ttft_p90_s"] for r in runs]),
                        "ttft_p99_s": spread([r["ttft_p99_s"] for r in runs]),
                        "ttft_max_s": spread([r["ttft_max_s"] for r in runs]),
                        "queue_p50_s": spread([r["queue_p50_s"] for r in runs]),
                        "generated_tps": spread([r["generated_tps"] for r in runs]),
                        "prompt_cache_hit_rate": spread([r["prompt_cache_hit_rate"] for r in runs]),
                        "completed_total": sum(r["completed"] for r in runs),
                        "rejected_503_total": sum(r["rejected_503"] for r in runs),
                        "failed_total": sum(r["failed"] for r in runs),
                        "unfinished_total": sum(r["unfinished_at_deadline"] for r in runs),
                        "per_run": runs,
                    }
                )
                flag = "  [SATURATED]" if saturated else ""
                lv = level_summaries[-1]
                print(
                    f"  -> {qps} QPS: achieved {lv['achieved_qps']['mean']}"
                    f" ± {lv['achieved_qps']['stdev']} | pooled n={pooled_q['n']} "
                    f"ttft p50 {pooled_q['p50']} p90 {pooled_q['p90']} "
                    f"p99 {pooled_q['p99']}{flag}\n"
                )

    out = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness": {
            "arrivals": "open loop, Poisson",
            "duration_s": args.duration,
            "runs_per_level": args.runs,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "seed": args.seed,
            "workload": "shared ~40-token preamble + a distinct question per request",
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "model": health["model"],
            "engine": health["engine"],
            "backend": cfg["backend"],
            "max_seqs": health["max_seqs"],
            "n_ctx": health["n_ctx"],
            "n_ctx_per_seq": cfg["n_ctx_per_seq"],
            "block_size": cfg["block_size"],
            "cache_seqs": cfg["cache_seqs"],
            "prefix_cache": cfg["enable_prefix_cache"],
            "preemption": cfg["enable_preemption"],
            "policy": cfg["policy"],
            "model_load_time_s": health["load_time_s"],
        },
        "raw_csv": str(csv_path),
        "levels": level_summaries,
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"wrote {path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
