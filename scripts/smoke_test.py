"""Smoke test: starts nothing, assumes a server is already listening.

Usage: python scripts/smoke_test.py [--url http://127.0.0.1:8000] [--concurrent N]

Checks the things that would silently rot:
  * /health responds and reports the expected engine
  * a non-streaming completion returns non-empty text and a finish reason
  * a streaming completion delivers >1 chunk (i.e. it really streams)
  * concurrent requests all succeed and return distinct request ids
  * /metrics counts every request that was actually made
Exits non-zero on any failure so it can gate a commit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as c:
        print("\n== health ==")
        r = await c.get("/health")
        health = r.json() if r.status_code == 200 else {}
        check("GET /health -> 200", r.status_code == 200, json.dumps(health))
        check("model loaded", bool(health.get("n_ctx")), f"n_ctx={health.get('n_ctx')}")

        await c.post("/metrics/reset")

        print("\n== single non-streaming completion ==")
        t0 = time.perf_counter()
        r = await c.post(
            "/generate",
            json={
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
        )
        dt = time.perf_counter() - t0
        ok = r.status_code == 200
        body = r.json() if ok else {"error": r.text[:200]}
        check("POST /generate -> 200", ok, f"{dt:.2f}s")
        text = body.get("text", "")
        check("non-empty text", bool(text.strip()), repr(text[:80]))
        check("finish_reason set", bool(body.get("finish_reason")), str(body.get("finish_reason")))
        check(
            "completion_tokens > 0",
            body.get("usage", {}).get("completion_tokens", 0) > 0,
            str(body.get("usage")),
        )
        tm = body.get("timings", {})
        check("ttft recorded", tm.get("ttft_s") is not None, f"ttft={tm.get('ttft_s')}s")

        print("\n== streaming ==")
        chunks: list[str] = []
        first_chunk_t = None
        t0 = time.perf_counter()
        async with c.stream(
            "POST",
            "/generate",
            json={"prompt": "Count: 1 2 3", "max_tokens": args.max_tokens, "temperature": 0.0, "stream": True},
        ) as resp:
            check("stream -> 200", resp.status_code == 200)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if "text" in d:
                    if first_chunk_t is None:
                        first_chunk_t = time.perf_counter() - t0
                    chunks.append(d["text"])
        check("multiple stream chunks", len(chunks) > 1, f"{len(chunks)} chunks")
        check(
            "first chunk before completion",
            first_chunk_t is not None and first_chunk_t < (time.perf_counter() - t0),
            f"client ttft={first_chunk_t:.3f}s" if first_chunk_t else "none",
        )

        print(f"\n== {args.concurrent} concurrent requests ==")
        t0 = time.perf_counter()
        reqs = [
            c.post(
                "/generate",
                json={
                    "messages": [{"role": "user", "content": f"Give me fact number {i} about the ocean."}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
            )
            for i in range(args.concurrent)
        ]
        resps = await asyncio.gather(*reqs, return_exceptions=True)
        wall = time.perf_counter() - t0
        good = [x for x in resps if not isinstance(x, Exception) and x.status_code == 200]
        check("all concurrent requests 200", len(good) == args.concurrent, f"{len(good)}/{args.concurrent} in {wall:.2f}s")
        if good:
            bodies = [x.json() for x in good]
            ids = {b["id"] for b in bodies}
            check("distinct request ids", len(ids) == len(bodies), f"{len(ids)} ids")
            check("all produced text", all(b.get("text", "").strip() for b in bodies))
            ttfts = [b["timings"]["ttft_s"] for b in bodies if b["timings"].get("ttft_s")]
            if ttfts:
                print(f"       ttft: min={min(ttfts):.3f}s max={max(ttfts):.3f}s")
                qts = [b["timings"]["queue_time_s"] for b in bodies if b["timings"].get("queue_time_s") is not None]
                print(f"       queue: min={min(qts):.3f}s max={max(qts):.3f}s")

        print("\n== metrics ==")
        r = await c.get("/metrics")
        m = r.json()
        expected = 2 + args.concurrent  # non-stream + stream + concurrent
        check("GET /metrics -> 200", r.status_code == 200)
        check(
            "metrics counted every request",
            m["requests"]["finished"] == expected,
            f"finished={m['requests']['finished']} expected={expected}",
        )
        check("zero errors", m["requests"]["errored"] == 0, f"errored={m['requests']['errored']}")
        check("generated tokens recorded", m["tokens"]["generated"] > 0, f"{m['tokens']['generated']} tokens")
        print("       " + json.dumps(m["latency"]["ttft_s"]))

    failures = [x for x in _results if x[0] == FAIL]
    print(f"\n{'=' * 60}")
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")
    if failures:
        for _, name, detail in failures:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
