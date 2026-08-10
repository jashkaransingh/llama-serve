"""Where does a decode step actually spend its time?

Before tuning anything, find out what the limit is. This drives the backend
directly — no HTTP, no engine, no scheduler — and measures three things:

  1. **How `llama_decode` scales with batch width.** If a batch of 32 costs the
     same as a batch of 1, the GPU is latency-bound at these widths and more
     concurrency is nearly free. If it scales linearly, it is compute-bound and
     concurrency buys nothing but queueing.
  2. **What a prefill chunk costs**, at the widths the scheduler actually uses.
  3. **The Python tax per step** — sampling, detokenisation and batch marshalling
     around the C call. If that dominates at the widths where decode is flat,
     the bottleneck is this process, not the model.

The answer decides what to tune. Guessing which of the three it is, and then
tuning for the wrong one, is how benchmarks get "optimised" without getting
faster.

    python bench/step_profile.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from llama_serve.backends.base import SamplingParams, TokenSlot
from llama_serve.backends.llama_cpp_backend import LlamaCppBackend
from llama_serve.config import DEFAULT_MODEL


def timeit(fn, repeats: int, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "mean_ms": round(1000 * statistics.fmean(samples), 4),
        "median_ms": round(1000 * statistics.median(samples), 4),
        "min_ms": round(1000 * min(samples), 4),
        "stdev_ms": round(1000 * statistics.stdev(samples), 4) if len(samples) > 1 else 0.0,
        "n": repeats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default="tinyllama-1.1b-q4km")
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--n-ctx-per-seq", type=int, default=256)
    ap.add_argument("--widths", default="1,2,4,8,12,16,24,32,48,64")
    ap.add_argument("--prefill-chunks", default="16,32,64,128,256,512")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="results/step_profile.json")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",")]
    chunks = [int(x) for x in args.prefill_chunks.split(",")]

    backend = LlamaCppBackend(
        model_path=args.model,
        n_ctx_per_seq=args.n_ctx_per_seq,
        max_seqs=args.max_seqs,
        cache_seqs=0,
        block_size=16,
        verbose=False,
    )
    print(f"{args.label}: n_ctx={backend.n_ctx} n_batch={backend.n_batch} max_seqs={backend.max_seqs}")

    prompt = backend.tokenize("The history of maritime navigation is long and detailed.", add_bos=True)

    # --- decode scaling -----------------------------------------------------
    decode = {}
    for w in widths:
        if w > backend.max_seqs:
            continue
        # Give every sequence a real prefix so attention has something to read.
        for s in range(w):
            backend.seq_rm(s, -1, -1)
            backend.decode([TokenSlot(s, t, i, want_logits=False) for i, t in enumerate(prompt)])
        pos = len(prompt)
        state = {"pos": pos}

        def step(width=w, st=state):
            slots = [TokenSlot(s, prompt[-1], st["pos"], want_logits=True) for s in range(width)]
            backend.decode(slots)
            st["pos"] += 1

        r = timeit(step, args.repeats)
        r["per_token_ms"] = round(r["median_ms"] / w, 4)
        r["tokens_per_s"] = round(1000 * w / r["median_ms"], 1)
        decode[w] = r
        print(
            f"  decode width {w:>3}: {r['median_ms']:>7.2f} ms  "
            f"{r['per_token_ms']:>6.3f} ms/token  {r['tokens_per_s']:>7.1f} tok/s"
        )
        for s in range(w):
            backend.seq_rm(s, -1, -1)

    # --- prefill cost -------------------------------------------------------
    prefill = {}
    long_prompt = backend.tokenize(" ".join(["token"] * 600), add_bos=True)
    for c in chunks:
        if c > backend.n_batch or c > len(long_prompt):
            continue

        def chunk_step(size=c):
            backend.seq_rm(0, -1, -1)
            backend.decode(
                [TokenSlot(0, long_prompt[i], i, want_logits=(i == size - 1)) for i in range(size)]
            )

        r = timeit(chunk_step, max(8, args.repeats // 3))
        r["per_token_ms"] = round(r["median_ms"] / c, 4)
        r["tokens_per_s"] = round(1000 * c / r["median_ms"], 1)
        prefill[c] = r
        print(
            f"  prefill {c:>4} tokens: {r['median_ms']:>7.2f} ms  "
            f"{r['per_token_ms']:>6.3f} ms/token  {r['tokens_per_s']:>8.1f} tok/s"
        )
    backend.seq_rm(0, -1, -1)

    # --- the Python tax around the C call -----------------------------------
    # Same work the engine does per decoded token, minus the model: build slots,
    # sample, detokenise. Measured at the width the server actually runs.
    tax = {}
    for w in (8, 16, 32):
        if w > backend.max_seqs:
            continue
        for s in range(w):
            backend.seq_rm(s, -1, -1)
            backend.decode([TokenSlot(s, t, i, want_logits=False) for i, t in enumerate(prompt)])
        samplers = [backend.make_sampler(SamplingParams(temperature=0.0, seed=1)) for _ in range(w)]
        state = {"pos": len(prompt)}

        def full_step(width=w, st=state, sam=samplers):
            slots = [TokenSlot(s, prompt[-1], st["pos"], want_logits=True) for s in range(width)]
            backend.decode(slots)
            for i in range(width):
                tok = sam[i].sample(i)
                sam[i].accept(tok)
                backend.token_to_piece(tok)
            st["pos"] += 1

        def decode_only(width=w, st=state):
            slots = [TokenSlot(s, prompt[-1], st["pos"], want_logits=False) for s in range(width)]
            backend.decode(slots)
            st["pos"] += 1

        full = timeit(full_step, args.repeats)
        bare = timeit(decode_only, args.repeats)
        overhead = round(full["median_ms"] - bare["median_ms"], 4)
        tax[w] = {
            "full_step_ms": full["median_ms"],
            "decode_only_ms": bare["median_ms"],
            "python_overhead_ms": overhead,
            "python_share": round(overhead / full["median_ms"], 4) if full["median_ms"] else 0.0,
        }
        print(
            f"  width {w:>3}: full step {full['median_ms']:>6.2f} ms, "
            f"decode {bare['median_ms']:>6.2f} ms, "
            f"sampling+detokenise {overhead:>6.2f} ms ({100 * tax[w]['python_share']:.0f}%)"
        )
        for s in samplers:
            s.close()
        for s in range(w):
            backend.seq_rm(s, -1, -1)

    # --- sampler: llama.cpp's chain vs the greedy fast path -----------------
    # Both produce the same token (scripts/verify_sampler.py proves that); this
    # is only about what each costs.
    sampler_cmp = {}
    for w in (8, 32):
        if w > backend.max_seqs:
            continue
        for s in range(w):
            backend.seq_rm(s, -1, -1)
            backend.decode([TokenSlot(s, t, i, want_logits=False) for i, t in enumerate(prompt)])
        backend.decode([TokenSlot(s, prompt[-1], len(prompt), True) for s in range(w)])
        params = SamplingParams(temperature=0.0, repeat_penalty=1.1, seed=1)

        def run(sams, width=w):
            for i in range(width):
                sams[i].accept(sams[i].sample(i))

        results = {}
        for mode, flag in (("chain", False), ("fast_greedy", True)):
            backend.fast_greedy = flag
            sams = [backend.make_sampler(params) for _ in range(w)]
            results[mode] = timeit(lambda s=sams: run(s), args.repeats)["median_ms"]
            for s_ in sams:
                s_.close()
        backend.fast_greedy = True
        results["speedup"] = round(results["chain"] / results["fast_greedy"], 2)
        sampler_cmp[w] = results
        print(
            f"  sampler width {w:>3}: chain {results['chain']:>7.3f} ms, "
            f"fast {results['fast_greedy']:>6.3f} ms  ({results['speedup']}x)"
        )
        for s in range(w):
            backend.seq_rm(s, -1, -1)

    backend.close()

    # Where does the curve stop being free? Report the width at which per-token
    # cost has risen 50% above its best.
    per_tok = {w: r["per_token_ms"] for w, r in decode.items()}
    best = min(per_tok.values())
    knee = next((w for w in sorted(per_tok) if per_tok[w] > 1.5 * best), None)

    out = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "platform": platform.platform(),
        "n_ctx": backend.n_ctx,
        "n_batch": backend.n_batch,
        "max_seqs": args.max_seqs,
        "prompt_tokens_used": len(prompt),
        "decode_by_width": {str(k): v for k, v in decode.items()},
        "prefill_by_chunk": {str(k): v for k, v in prefill.items()},
        "step_overhead": {str(k): v for k, v in tax.items()},
        "sampler_chain_vs_fast_ms": {str(k): v for k, v in sampler_cmp.items()},
        "best_per_token_ms": round(best, 4),
        "width_at_best_per_token": min(per_tok, key=per_tok.get),
        "width_where_per_token_rises_50pct": knee,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\nbest per-token {best:.4f} ms at width {out['width_at_best_per_token']}; "
          f"per-token cost rises 50% by width {knee}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
