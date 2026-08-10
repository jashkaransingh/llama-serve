"""Does the greedy fast path pick the same token as llama.cpp's sampler chain?

`GreedySampler` exists because `llama_sampler_sample` materialises a
`llama_token_data_array` over the whole vocabulary on every call — 151,936
entries for Qwen2.5 — and profiling put that at 45% of a decode step. Replacing
it with a numpy argmax is only legitimate if the answer never changes.

This runs both samplers against **the same logits from the same forward pass**
on the real model, decoding a real continuation, and compares the token they
pick at every position. It also compares the two decoded texts end to end.

Both are exercised with a repetition penalty active (the API default, 1.1), so
the penalty arithmetic is under test too and not just the argmax.

    python scripts/verify_sampler.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llama_serve.backends.base import SamplingParams, TokenSlot
from llama_serve.backends.llama_cpp_backend import GreedySampler, LlamaCppBackend, LlamaCppSampler
from llama_serve.config import DEFAULT_MODEL

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue:",
    "List three causes of disk failure in a storage array:",
    "Once upon a time in a very small village,",
    "def fibonacci(n):",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--out", default="results/sampler_equivalence.json")
    args = ap.parse_args()

    backend = LlamaCppBackend(
        model_path=args.model, n_ctx_per_seq=512, max_seqs=2, cache_seqs=0, verbose=False
    )
    params = SamplingParams(temperature=0.0, repeat_penalty=args.repeat_penalty, seed=0)

    rows = []
    for prompt in PROMPTS:
        toks = backend.tokenize(prompt, add_bos=True)
        backend.seq_rm(0, -1, -1)

        chain = LlamaCppSampler(backend._ctx, params)
        fast = GreedySampler(backend._ctx, params, backend.n_vocab)

        # Prefill once; the last position carries the logits both will read.
        backend.decode(
            [TokenSlot(0, t, i, want_logits=(i == len(toks) - 1)) for i, t in enumerate(toks)]
        )

        agree, disagree, chain_text, fast_text = 0, 0, "", ""
        first_divergence = None
        pos = len(toks)
        # Logits live at the batch index of the entry that asked for them: the
        # last entry of the prefill batch, then index 0 of each 1-token decode.
        logits_idx = len(toks) - 1
        for step in range(args.tokens):
            # Same slot, same logits, both samplers. The fast path restores the
            # buffer it borrows, so the chain sees untouched logits.
            a = fast.sample(logits_idx)
            b = chain.sample(logits_idx)
            if a == b:
                agree += 1
            else:
                disagree += 1
                if first_divergence is None:
                    first_divergence = {"step": step, "fast": a, "chain": b}
            fast.accept(a)
            chain.accept(b)
            chain_text += backend.token_to_piece(b)
            fast_text += backend.token_to_piece(a)
            if backend.is_eog(b):
                break
            backend.decode([TokenSlot(0, b, pos, want_logits=True)])
            pos += 1
            logits_idx = 0

        chain.close()
        fast.close()
        rows.append(
            {
                "prompt": prompt,
                "steps": agree + disagree,
                "agree": agree,
                "disagree": disagree,
                "identical_text": chain_text == fast_text,
                "first_divergence": first_divergence,
                "text": chain_text,
            }
        )
        flag = "OK  " if disagree == 0 else "DIFF"
        print(f"  [{flag}] {agree}/{agree + disagree} tokens identical  {prompt[:40]!r}")

    backend.close()

    total_steps = sum(r["steps"] for r in rows)
    total_agree = sum(r["agree"] for r in rows)
    out = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "repeat_penalty": args.repeat_penalty,
        "temperature": 0.0,
        "prompts": len(rows),
        "total_steps": total_steps,
        "total_agree": total_agree,
        "all_identical": total_agree == total_steps and all(r["identical_text"] for r in rows),
        "rows": rows,
    }
    print(f"\n  {total_agree}/{total_steps} sampled tokens identical across {len(rows)} prompts")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[args.label] = out
    path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"wrote {path}")
    return 0 if out["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
