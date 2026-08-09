"""Measures what preemption buys, on the case continuous batching could not fix.

`bench/late_arrival.py --mode saturated` is the open problem from milestone 3:
every sequence slot occupied by an equally long generation, so there is no free
slot and no early finisher, and iteration-level batching has nothing to exploit.
Measured there, a short request arriving 1 s in waited 9.11 s under continuous
batching and 8.75 s under static — the two engines are the same in that regime.

This experiment runs the same saturated workload and sends the late request as a
*high-priority* one. The scheduler now has an option the batcher did not: take a
slot away from a low-priority generation, run the urgent request, and resume the
victim from where it stopped.

Three things are recorded, because a scheduler that only improved one of them
would not be worth shipping:

  1. the urgent request's TTFT — the number preemption exists to reduce
  2. what it cost the victims — their completion time and how many tokens had
     to be recomputed on resume
  3. whether the victims still produced their full output — a preemption that
     truncates a generation is a bug wearing a feature's clothes

Run twice against the same server binary, once with preemption and once without:

    python -m llama_serve.server --engine continuous --max-seqs 8
    python bench/preemption.py --label preempt-on

    python -m llama_serve.server --engine continuous --max-seqs 8 --no-preemption
    python bench/preemption.py --label preempt-off
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def _run(client, payload) -> dict:
    """Stream a request; return client-observed TTFT, total time, chunk count."""
    t0 = time.perf_counter()
    ttft, n = None, 0
    async with client.stream("POST", "/generate", json={**payload, "stream": True}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            d = json.loads(body)
            if "text" in d:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
            elif d.get("done"):
                timings = d.get("timings", {})
    return {
        "ttft_s": ttft,
        "total_s": time.perf_counter() - t0,
        "chunks": n,
        "timings": timings,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--long-tokens", type=int, default=300)
    ap.add_argument("--short-tokens", type=int, default=16)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3, help="runs to average over")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="results/preemption.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as c:
        health = (await c.get("/health")).json()
        cfg = (await c.get("/config")).json()
        max_seqs = health["max_seqs"]
        print(
            f"engine={health['engine']} max_seqs={max_seqs} "
            f"preemption={cfg['enable_preemption']} starvation_s={cfg['starvation_s']}"
        )

        long_payload = {
            "prompt": "Write a long detailed essay about the history of maritime navigation.",
            "max_tokens": args.long_tokens,
            "temperature": 0.0,
            "ignore_eos": True,  # pin the length, or the experiment measures EOS behaviour
            "priority": 9,  # background work
        }
        urgent_payload = {
            "prompt": "Say hello.",
            "max_tokens": args.short_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "priority": 0,  # interactive
        }

        runs = []
        for i in range(args.repeats):
            await c.post("/metrics/reset")
            t_start = time.perf_counter()
            longs = [asyncio.create_task(_run(c, long_payload)) for _ in range(max_seqs)]
            await asyncio.sleep(args.delay)

            m = (await c.get("/metrics")).json()
            in_flight = m["requests"]["in_flight"]

            urgent = await _run(c, urgent_payload)
            long_results = await asyncio.gather(*longs)
            wall = time.perf_counter() - t_start
            stats = (await c.get("/metrics")).json().get("engine", {})

            short_outputs = [r["chunks"] for r in long_results]
            runs.append(
                {
                    "run": i,
                    "in_flight_at_send": in_flight,
                    "urgent_ttft_s": round(urgent["ttft_s"], 4) if urgent["ttft_s"] else None,
                    "urgent_total_s": round(urgent["total_s"], 4),
                    "long_total_max_s": round(max(r["total_s"] for r in long_results), 4),
                    "long_chunks_min": min(short_outputs),
                    "long_chunks_max": max(short_outputs),
                    "wall_s": round(wall, 4),
                    "preemptions": stats.get("scheduler", {}).get("preemptions"),
                    "resumes": stats.get("resumes"),
                    "resumed_tokens_recomputed": stats.get("resumed_tokens_recomputed"),
                }
            )
            print(
                f"  run {i}: in_flight={in_flight} urgent TTFT={runs[-1]['urgent_ttft_s']}s "
                f"preemptions={runs[-1]['preemptions']} "
                f"longest long req={runs[-1]['long_total_max_s']}s"
            )

    def agg(key):
        vals = [r[key] for r in runs if r[key] is not None]
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {
            "mean": round(mean, 4),
            "stdev": round(var**0.5, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n": len(vals),
        }

    out = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": health["engine"],
        "preemption_enabled": cfg["enable_preemption"],
        "starvation_s": cfg["starvation_s"],
        "max_preemptions": cfg["max_preemptions"],
        "max_seqs": max_seqs,
        "long_tokens": args.long_tokens,
        "short_tokens": args.short_tokens,
        "delay_s": args.delay,
        "repeats": args.repeats,
        "urgent_ttft_s": agg("urgent_ttft_s"),
        "long_total_max_s": agg("long_total_max_s"),
        "wall_s": agg("wall_s"),
        # If preemption ever truncated a victim, this drops below long_tokens.
        "victim_output_tokens_min": min(r["long_chunks_min"] for r in runs),
        "victim_output_tokens_max": max(r["long_chunks_max"] for r in runs),
        "expected_output_tokens": args.long_tokens,
        "runs": runs,
    }

    print(f"\n  urgent TTFT      : {out['urgent_ttft_s']}")
    print(f"  longest long req : {out['long_total_max_s']}")
    print(
        f"  victim output    : {out['victim_output_tokens_min']}-"
        f"{out['victim_output_tokens_max']} tokens (expected {args.long_tokens})"
    )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
