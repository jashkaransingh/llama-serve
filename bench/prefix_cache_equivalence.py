"""Does serving a prompt from the prefix cache change the answer?

This is the correctness experiment for milestone 4 on the *real* model. The unit
tests prove block-level sharing is exact against the mock backend, where the KV
cache is deterministic by construction. They cannot prove llama.cpp agrees.

**Why this needs its own experiment.** llama.cpp on Metal is not bit-identical
across batch shapes: the same prompt decoded in a batch of 1 and in a batch of 4
can diverge, because floating-point reduction order changes with batch width and
a near-tie at the argmax then falls the other way. `bench/prefix_cache.py`
measured exactly that — two cache-off runs at concurrency 4 produced identical
text, and a cache-off run at concurrency 1 produced different text. So a naive
"cache-on output == cache-off output" comparison over a concurrent workload
tests the GPU's determinism, not the cache.

The controlled version, run here: **one request in flight at a time, greedy
sampling, same prompt, same server.** The only variable is whether the prompt's
prefix was computed or reused.

    pass 1  cold: nothing cached for this prompt   -> text_cold
    pass 2  warm: prefix served from the cache     -> text_warm

If prefix sharing is faithful, `text_cold == text_warm` for every prompt, and
`cached_prompt_tokens` is > 0 on the warm pass — the second check matters, since
a cache that never hits would pass the first one trivially.

    python bench/prefix_cache_equivalence.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def ask(client: httpx.AsyncClient, prompt: str, max_tokens: int) -> dict:
    r = await client.post(
        "/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy: the only remaining variable is the cache
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    )
    r.raise_for_status()
    b = r.json()
    return {
        "text": b["text"],
        "prompt_tokens": b["usage"]["prompt_tokens"],
        "cached": b["usage"]["cached_prompt_tokens"],
        "ttft_s": b["timings"]["ttft_s"],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default="results/prefix_cache_equivalence.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=300.0) as c:
        cfg = (await c.get("/config")).json()
        health = (await c.get("/health")).json()
        if not cfg["enable_prefix_cache"]:
            print("ERROR: server is running with --no-prefix-cache; nothing to compare")
            return 2
        await c.post("/metrics/reset")

        rows = []
        for q in QUESTIONS:
            prompt = PREAMBLE + q
            cold = await ask(c, prompt, args.max_tokens)  # publishes on finish
            warm = await ask(c, prompt, args.max_tokens)  # should hit
            rows.append(
                {
                    "question": q,
                    "prompt_tokens": cold["prompt_tokens"],
                    "cold_cached_tokens": cold["cached"],
                    "warm_cached_tokens": warm["cached"],
                    "identical": cold["text"] == warm["text"],
                    "cold_ttft_s": cold["ttft_s"],
                    "warm_ttft_s": warm["ttft_s"],
                    "cold_text": cold["text"],
                    "warm_text": warm["text"],
                }
            )
            flag = "OK " if rows[-1]["identical"] else "DIFF"
            print(
                f"  [{flag}] {q[:44]:<44} warm reused {warm['cached']}/{cold['prompt_tokens']} tokens"
            )

        metrics = (await c.get("/metrics.json")).json()

    hit = [r for r in rows if r["warm_cached_tokens"] > 0]
    identical = [r for r in rows if r["identical"]]
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": health["engine"],
        "model": health["model"],
        "block_size": cfg["block_size"],
        "n_prompts": len(rows),
        "n_warm_hits": len(hit),
        "n_identical": len(identical),
        "all_identical": len(identical) == len(rows),
        "concurrency": 1,
        "sampling": "greedy (temperature=0, repeat_penalty=1.0)",
        "rows": rows,
        "kv_cache_stats": metrics.get("engine", {}).get("kv_cache", {}),
    }
    print(
        f"\n  {len(identical)}/{len(rows)} prompts identical cold vs warm; "
        f"{len(hit)}/{len(rows)} actually hit the cache"
    )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0 if out["all_identical"] and len(hit) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
