"""Measures the one thing that separates static from continuous batching.

Experiment: saturate the server with `--n-long` requests that each generate a
long completion. Then, after `--delay` seconds — once those are unambiguously
mid-generation — send one short request and measure its time-to-first-token.

  static batching:     the late request cannot join the running batch. Its TTFT
                       is bounded below by the *remaining* time of the longest
                       request already in flight.
  continuous batching: the late request joins at the next decode step. Its TTFT
                       should be close to a single prefill.

Run it against both engines and compare. The absolute numbers are hardware
specific; the ratio is the point.

    python bench/late_arrival.py --label static
    python bench/late_arrival.py --label continuous
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def _stream_ttft(client, payload) -> dict:
    """Send a streaming request; return client-observed TTFT and total time."""
    t0 = time.perf_counter()
    ttft = None
    n = 0
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
    return {"ttft_s": ttft, "total_s": time.perf_counter() - t0, "chunks": n}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True, help="engine name, for the results file")
    ap.add_argument("--n-long", type=int, default=8)
    ap.add_argument("--long-tokens", type=int, default=200)
    ap.add_argument("--short-tokens", type=int, default=16)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="results/late_arrival.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as c:
        health = (await c.get("/health")).json()
        await c.post("/metrics/reset")
        print(f"engine={health['engine']} max_seqs={health['max_seqs']}")

        long_payload = {
            "prompt": "Write a long detailed essay about the history of maritime navigation.",
            "max_tokens": args.long_tokens,
            "temperature": 0.0,
            # Force the exact length: TinyLlama emits EOS after ~30 tokens on
            # this prompt, which would make the "long" requests short and the
            # experiment meaningless.
            "ignore_eos": True,
        }
        short_payload = {
            "prompt": "Say hello.",
            "max_tokens": args.short_tokens,
            "temperature": 0.0,
        }

        t_start = time.perf_counter()
        longs = [asyncio.create_task(_stream_ttft(c, long_payload)) for _ in range(args.n_long)]
        await asyncio.sleep(args.delay)

        # Confirm the long requests really are mid-flight before we measure.
        m = (await c.get("/metrics")).json()
        in_flight = m["requests"]["in_flight"]
        print(f"t={time.perf_counter() - t_start:.2f}s: {in_flight} requests in flight; sending late request")

        late = await _stream_ttft(c, short_payload)
        long_results = await asyncio.gather(*longs)
        wall = time.perf_counter() - t_start
        metrics = (await c.get("/metrics")).json()

    long_ttfts = [r["ttft_s"] for r in long_results if r["ttft_s"] is not None]
    out = {
        "label": args.label,
        "engine": health["engine"],
        "max_seqs": health["max_seqs"],
        "n_long": args.n_long,
        "long_tokens": args.long_tokens,
        "delay_s": args.delay,
        "in_flight_at_send": in_flight,
        "late_request_ttft_s": round(late["ttft_s"], 4) if late["ttft_s"] else None,
        "late_request_total_s": round(late["total_s"], 4),
        "long_ttft_mean_s": round(sum(long_ttfts) / len(long_ttfts), 4) if long_ttfts else None,
        "long_total_max_s": round(max(r["total_s"] for r in long_results), 4),
        "wall_s": round(wall, 4),
        "engine_stats": metrics.get("engine", {}),
    }

    print(f"\n  late-request TTFT : {out['late_request_ttft_s']} s")
    print(f"  longest long req  : {out['long_total_max_s']} s")
    print(f"  wall clock        : {out['wall_s']} s")
    print(f"  slot utilization  : {out['engine_stats'].get('avg_slot_utilization')}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
