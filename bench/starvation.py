"""Does a low-priority request still finish under unrelenting urgent load?

The scheduler's starvation bound is proved as a unit test in
`tests/test_scheduler.py`, where the mock backend makes the control case cheap
to run. This is the same experiment against the real model, because a bound that
only holds against a simulated backend is a bound about the simulation.

Workload: `--load` concurrent generators that each submit a high-priority
request the instant their previous one finishes, so the queue is never empty.
One low-priority request is submitted once that load is established, and the
time until it completes is measured.

    with protection      it is promoted after `starvation_s` of waiting, becomes
                         immune to preemption, and completes
    without protection   (`--starvation-s 0` on the server) it is outranked by
                         every arrival, forever

Run twice, against a server started each way:

    python -m llama_serve.server --engine continuous --max-seqs 4
    python bench/starvation.py --label protected

    python -m llama_serve.server --engine continuous --max-seqs 4 --starvation-s 0
    python bench/starvation.py --label unprotected
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def generate(client, payload, timeout: float) -> dict:
    t0 = time.perf_counter()
    r = await client.post("/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    b = r.json()
    return {
        "wall_s": time.perf_counter() - t0,
        "generated": b["usage"]["completion_tokens"],
        "timings": b["timings"],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--load", type=int, default=6, help="concurrent urgent generators")
    ap.add_argument("--urgent-tokens", type=int, default=48)
    ap.add_argument("--low-tokens", type=int, default=48)
    ap.add_argument("--deadline", type=float, default=90.0, help="give up after this long")
    ap.add_argument("--out", default="results/starvation.json")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=args.deadline + 30) as c:
        health = (await c.get("/health")).json()
        cfg = (await c.get("/config")).json()
        await c.post("/metrics/reset")
        print(
            f"engine={health['engine']} max_seqs={health['max_seqs']} "
            f"starvation_s={cfg['starvation_s']} preemption={cfg['enable_preemption']}"
        )

        stop = asyncio.Event()
        urgent_done = 0

        async def pressure():
            nonlocal urgent_done
            while not stop.is_set():
                try:
                    await generate(
                        c,
                        {
                            "prompt": "Summarise the rules of chess in one sentence.",
                            "max_tokens": args.urgent_tokens,
                            "temperature": 0.0,
                            "ignore_eos": True,
                            "priority": 0,
                        },
                        args.deadline,
                    )
                    urgent_done += 1
                except Exception:
                    return

        load = [asyncio.create_task(pressure()) for _ in range(args.load)]
        await asyncio.sleep(2.0)  # let the load establish itself

        t0 = time.perf_counter()
        low_task = asyncio.create_task(
            generate(
                c,
                {
                    "prompt": "Describe the water cycle.",
                    "max_tokens": args.low_tokens,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "priority": 9,
                },
                args.deadline,
            )
        )
        try:
            low = await asyncio.wait_for(asyncio.shield(low_task), timeout=args.deadline)
            completed, elapsed = True, time.perf_counter() - t0
        except TimeoutError:
            low, completed, elapsed = None, False, args.deadline
            low_task.cancel()

        stop.set()
        for t in load:
            t.cancel()
        await asyncio.gather(*load, return_exceptions=True)
        stats = (await c.get("/metrics.json")).json().get("engine", {})

    out = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": health["engine"],
        "max_seqs": health["max_seqs"],
        "starvation_s": cfg["starvation_s"],
        "max_preemptions": cfg["max_preemptions"],
        "preemption_enabled": cfg["enable_preemption"],
        "concurrent_urgent_generators": args.load,
        "urgent_requests_completed_meanwhile": urgent_done,
        "low_priority_completed": completed,
        "low_priority_wall_s": round(elapsed, 3),
        "low_priority_tokens": low["generated"] if low else 0,
        "expected_tokens": args.low_tokens,
        "deadline_s": args.deadline,
        "scheduler_stats": stats.get("scheduler", {}),
    }
    verdict = "COMPLETED" if completed else f"STARVED (still waiting at {args.deadline}s)"
    print(f"\n  low-priority request: {verdict} in {out['low_priority_wall_s']}s")
    print(f"  urgent requests served meanwhile: {urgent_done}")
    print(f"  scheduler: {json.dumps(out['scheduler_stats'])}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
