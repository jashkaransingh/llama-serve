"""Measures what the prefix cache actually saves.

Workload: a long shared preamble (system prompt + few-shot examples) followed
by a short per-request question — the shape of essentially every RAG or
agent deployment, where the first ~90% of every prompt is identical.

Requests are sent in sequential waves rather than all at once, because a cache
can only hit on a prefix some earlier request has already published. Firing N
identical-prefix requests simultaneously would have them all miss and would
measure nothing.

Run the server twice, with and without `--no-prefix-cache`, and compare.

    python bench/prefix_cache.py --label cache-on
    python bench/prefix_cache.py --label cache-off
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import httpx

PREAMBLE = (
    "You are a meticulous technical support assistant for an enterprise storage product. "
    "Follow these rules exactly. Always cite the relevant manual section. Never speculate "
    "about hardware you have not been told about. Escalate anything involving data loss. "
    "Here are worked examples of the expected format.\n\n"
    "Example 1. Question: The array reports a degraded volume. Answer: Per manual section "
    "4.2, a degraded volume indicates a failed member disk. Verify the disk LED state, then "
    "replace the failed member and allow the rebuild to complete before further changes.\n\n"
    "Example 2. Question: Throughput dropped after a firmware update. Answer: Per manual "
    "section 9.1, confirm the write cache policy was preserved across the update; firmware "
    "updates reset the policy to write-through on some controller revisions.\n\n"
    "Example 3. Question: A snapshot schedule stopped running. Answer: Per manual section "
    "6.5, verify the schedule owner account has not expired, then confirm free capacity in "
    "the snapshot reserve exceeds the configured threshold.\n\n"
    "Now answer the following question in the same format.\n\n"
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
]


async def one(client, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        "/generate",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
    )
    r.raise_for_status()
    b = r.json()
    return {
        "wall_s": time.perf_counter() - t0,
        "ttft_s": b["timings"]["ttft_s"],
        "prompt_tokens": b["usage"]["prompt_tokens"],
        "cached_prompt_tokens": b["usage"]["cached_prompt_tokens"],
        "text": b["text"],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/prefix_cache.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=300.0) as c:
        health = (await c.get("/health")).json()
        cfg = (await c.get("/config")).json()
        await c.post("/metrics/reset")
        print(f"engine={health['engine']} prefix_cache={cfg['enable_prefix_cache']}")

        rows: list[dict] = []
        for w in range(args.waves):
            picked = [QUESTIONS[(w * args.concurrency + i) % len(QUESTIONS)] for i in range(args.concurrency)]
            res = await asyncio.gather(*(one(c, PREAMBLE + q, args.max_tokens) for q in picked))
            for r, q in zip(res, picked, strict=True):
                r["wave"] = w
                r["question"] = q
            rows.extend(res)
            hit = sum(r["cached_prompt_tokens"] for r in res)
            tot = sum(r["prompt_tokens"] for r in res)
            print(f"  wave {w}: {hit}/{tot} prompt tokens served from cache")

        metrics = (await c.get("/metrics.json")).json()

    # Wave 0 can never hit (nothing published yet); report it separately so the
    # steady-state number is not diluted by the unavoidable cold start.
    warm = [r for r in rows if r["wave"] > 0]
    cold = [r for r in rows if r["wave"] == 0]

    def summarize(rs):
        if not rs:
            return {}
        return {
            "n": len(rs),
            "ttft_mean_s": round(statistics.mean(r["ttft_s"] for r in rs), 5),
            "ttft_median_s": round(statistics.median(r["ttft_s"] for r in rs), 5),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "cached_prompt_tokens": sum(r["cached_prompt_tokens"] for r in rs),
        }

    # Correctness, not just speed: at temperature 0 the answer to a given
    # question must not depend on whether its prefix came out of the cache.
    # Comparing this fingerprint between the cache-on and cache-off runs is what
    # turns "it was faster" into "it was faster and still right".
    by_q: dict[str, str] = {}
    for r in rows:
        by_q.setdefault(r["question"], r["text"])
    fingerprint = hashlib.sha256(
        "\u0000".join(f"{q}=>{by_q[q]}" for q in sorted(by_q)).encode()
    ).hexdigest()[:16]

    out = {
        "label": args.label,
        "output_fingerprint": fingerprint,
        "prefix_cache_enabled": cfg["enable_prefix_cache"],
        "block_size": cfg["block_size"],
        "waves": args.waves,
        "concurrency": args.concurrency,
        "cold_wave": summarize(cold),
        "warm_waves": summarize(warm),
        "all": summarize(rows),
        "kv_cache_stats": metrics.get("engine", {}).get("kv_cache", {}),
        "server_prompt_cache_hit_rate": metrics["tokens"]["prompt_cache_hit_rate"],
    }

    print(f"\n  output fingerprint   : {fingerprint}")
    print(f"  cold wave  TTFT mean : {out['cold_wave'].get('ttft_mean_s')} s")
    print(f"  warm waves TTFT mean : {out['warm_waves'].get('ttft_mean_s')} s")
    print(f"  prompt cache hit rate: {out['server_prompt_cache_hit_rate']}")
    print(f"  kv stats             : {json.dumps(out['kv_cache_stats'])}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
