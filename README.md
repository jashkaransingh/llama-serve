# llama-serve

A local LLM inference server built on `llama.cpp`, written to explore the parts of
model serving that sit *between* the HTTP handler and the forward pass: batching,
KV-cache management, and scheduling.

Serving a single request is a `for` loop. Serving a hundred concurrent ones well
is a scheduling problem. This repo implements the scheduling problem.

## Status

Milestones 0-4 work end to end against the real model. Nothing is claimed as
working here until a test or a benchmark in the repo says it
is, with the command used to verify it.

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Scaffolding, backend interface, mock backend | ✅ |
| 1 | Baseline blocking server | ✅ |
| 2 | Static batching | ✅ |
| 3 | Continuous (iteration-level) batching | ✅ |
| 4 | Paged KV-cache allocator + prefix sharing | ✅ |
| 5 | Scheduling policy (priority + preemption) | ⬜ |
| 6 | Observability (`/metrics`) | ⬜ |
| 7 | Load testing harness + measured results | ⬜ |

**No performance number appears in this README unless it was produced by a script
in this repo and its raw output is committed under `results/`.**

## Measured results

Every number below names the script that produced it and the file its raw output
lives in. All were measured on the environment described in the next section.

**Continuous vs static batching** — a short request arriving 1 s into a batch of
eight 300-token generations (`bench/late_arrival.py`,
[`results/late_arrival.json`](results/late_arrival.json)):

| | static | continuous | |
|---|---|---|---|
| late-request TTFT, one slot free | 9.1631 s | **0.0683 s** | 134× faster |
| late-request TTFT, all slots busy | 8.7515 s | 9.1142 s | no better — see below |

The saturated case is reported as measured: rebuilding the batch every step
cannot create capacity that does not exist. Fixing it needs preemption
(milestone 5).

**Paged KV cache with prefix sharing** — 4-request waves over a shared
~250-token preamble, same server run with the cache on and off
(`bench/prefix_cache.py`, [`results/prefix_cache.json`](results/prefix_cache.json)):

| | cache off | cache on | |
|---|---|---|---|
| warm-wave TTFT, mean | 0.82903 s | **0.12407 s** | **85.0 % lower** |
| warm-wave TTFT, median | 1.02972 s | **0.12735 s** | 87.6 % lower |
| prompt tokens served from cache | 0 % | 81.4 % | |
| prefill work skipped | 0 % | 77.35 % | |

Sharing a prefix must not change the answer. With one request in flight at a
time and greedy sampling, a cold pass and a cache-hit pass of the same prompt
produced identical text for **8 of 8 prompts**, with 8 of 8 genuinely hitting
the cache (`bench/prefix_cache_equivalence.py`,
[`results/prefix_cache_equivalence.json`](results/prefix_cache_equivalence.json)).

A caveat worth stating, because it shaped the methodology: llama.cpp on Metal is
not bit-identical across batch shapes. Two cache-off runs at concurrency 4
produced identical output; the same run at concurrency 1 did not. So output
equality is asserted exhaustively against the deterministic mock backend, and on
the real model only in the controlled single-request comparison above.

## Environment this was built on

- Apple M1 Pro, 16 GB unified memory, macOS 15 (Darwin 25.5.0, arm64)
- Python 3.11, `llama-cpp-python` 0.3.34 built from source with
  `-DGGML_METAL=on`; Metal backend active (`Apple M1 Pro`, `MTLGPUFamilyApple7`)
- Model: TinyLlama-1.1B-Chat v1.0, Q4_K_M GGUF (638 MB) — small enough that
  scheduling effects, not raw model latency, dominate the measurements
- Server defaults used for the measurements above: `--engine continuous
  --max-seqs 8`, `n_ctx_per_seq=1024`, `block_size=16`, `cache_seqs=8`

## Quickstart

```bash
python -m llama_serve.server --engine simple --max-seqs 1     # baseline
python scripts/smoke_test.py --concurrent 4                   # verify it
```

```bash
pytest          # 44 tests, no model required (deterministic mock backend)
ruff check .    # lint
```

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":24,"temperature":0}'
# -> {"text":"The capital of France is Paris.","finish_reason":"eos", ...}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # see note below for Apple Silicon
python scripts/download_model.py
```

On Apple Silicon, build the inference backend against Metal:

```bash
CMAKE_ARGS="-DGGML_METAL=on" FORCE_CMAKE=1 pip install --no-binary :all: llama-cpp-python
```

## Design

Every layer of the serving stack is written against a narrow `Backend`
interface (`llama_serve/backends/base.py`) shaped like llama.cpp's low-level C
API — batched `decode(slots)` plus per-sequence KV operations — rather than a
`generate(prompt) -> str` helper. That shape is what makes iteration-level
batching expressible at all, and it lets the whole scheduler run against a
deterministic mock backend in tests without loading a model.

## License

MIT
