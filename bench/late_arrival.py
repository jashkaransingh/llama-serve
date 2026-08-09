"""Measures the one thing that separates static from continuous batching.

Experiment: saturate the server with long-running requests, wait `--delay`
seconds so they are unambiguously mid-generation, then send one short request
and measure its time-to-first-token.

    static batching:     the late request cannot join a batch that has already
                         started. Its TTFT is bounded below by the *remaining*
                         time of the batch it missed.
    continuous batching: the late request is folded into the next decode step,
                         so its TTFT should be close to a bare prefill.

Two modes, because the honest answer differs between them:

  --mode headroom  (default)  n_long = max_seqs - 1, leaving one free sequence
                              slot. This isolates the scheduling difference:
                              capacity exists, and the only question is whether
                              the engine can hand it to a request that arrived
                              late. Continuous can; static cannot.

  --mode saturated            n_long = max_seqs, so every slot is occupied and
                              all long requests are the same length. Continuous
                              batching has nothing to exploit here — there is no
                              free slot and no early finisher — and it should be
                              reported as no better than static. Fixing *this*
                              case needs preemption (milestone 5), not batching.

Run against both engines and compare. Absolute numbers are hardware specific;
the ratio within a mode is the point.

    python bench/late_arrival.py --label static-headroom --mode headroom
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
    ap.add_argument("--label", required=True, help="run name, used as the results key")
    ap.add_argument("--mode", choices=["headroom", "saturated"], default="headroom")
    ap.add_argument("--n-long", type=int, default=None, help="override; defaults from --mode")
    ap.add_argument("--long-tokens", type=int, default=300)
    ap.add_argument("--short-tokens", type=int, default=16)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="results/late_arrival.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as c:
        health = (await c.get("/health")).json()
        max_seqs = health["max_seqs"]
        n_long = args.n_long
        if n_long is None:
            n_long = max_seqs - 1 if args.mode == "headroom" else max_seqs
        await c.post("/metrics/reset")
        print(f"engine={health['engine']} max_seqs={max_seqs} mode={args.mode} n_long={n_long}")

        long_payload = {
            "prompt": "Write a long detailed essay about the history of maritime navigation.",
            "max_tokens": args.long_tokens,
            "temperature": 0.0,
            # Pin the output length. TinyLlama emits EOS after ~30 tokens on
            # this prompt, which would leave the late request nothing to wait
            # for and make the whole experiment vacuous.
            "ignore_eos": True,
        }
        short_payload = {"prompt": "Say hello.", "max_tokens": args.short_tokens, "temperature": 0.0}

        t_start = time.perf_counter()
        longs = [asyncio.create_task(_stream_ttft(c, long_payload)) for _ in range(n_long)]
        await asyncio.sleep(args.delay)

        # Confirm the long requests really are mid-flight before measuring.
        m = (await c.get("/metrics.json")).json()
        in_flight = m["requests"]["in_flight"]
        print(f"t={time.perf_counter() - t_start:.2f}s: {in_flight} in flight; sending late request")

        late = await _stream_ttft(c, short_payload)
        long_results = await asyncio.gather(*longs)
        wall = time.perf_counter() - t_start
        metrics = (await c.get("/metrics.json")).json()

    long_ttfts = [r["ttft_s"] for r in long_results if r["ttft_s"] is not None]
    out = {
        "label": args.label,
        "engine": health["engine"],
        "mode": args.mode,
        "max_seqs": max_seqs,
        "n_long": n_long,
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
