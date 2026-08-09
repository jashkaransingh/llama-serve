# llama-serve

A local LLM inference server built on `llama.cpp`, written to explore the parts of
model serving that sit *between* the HTTP handler and the forward pass: batching,
KV-cache management, and scheduling.

Serving a single request is a `for` loop. Serving a hundred concurrent ones well
is a scheduling problem. This repo implements the scheduling problem.

## Status

All eight milestones work end to end against the real model. Nothing is claimed as
working here until a test or a benchmark in the repo says it
is, with the command used to verify it.

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Scaffolding, backend interface, mock backend | ✅ |
| 1 | Baseline blocking server | ✅ |
| 2 | Static batching | ✅ |
| 3 | Continuous (iteration-level) batching | ✅ |
| 4 | Paged KV-cache allocator + prefix sharing | ✅ |
| 5 | Scheduling policy (priority + preemption) | ✅ |
| 6 | Observability (`/metrics`) | ✅ |
| 7 | Load testing harness + measured results | ✅ |

**No performance number appears in this README unless it was produced by a script
in this repo and its raw output is committed under `results/`.**

## Measured results

Every number below names the script that produced it and the file its raw output
lives in. All were measured on the environment described in the next section.

**Load sweep against the milestone-1 baseline** — open-loop Poisson arrivals,
32 output tokens, 3 runs of 20 s per level, 1150 completed requests per engine.
Baseline is the blocking server from milestone 1 (`--engine simple --max-seqs 1`)
on the identical workload and seeds (`bench/load.py`,
[`results/load_sweep.json`](results/load_sweep.json), raw per-request CSVs under
[`results/load_raw/`](results/load_raw), rendered table
[`results/load_comparison.md`](results/load_comparison.md)):

| offered QPS | p50 TTFT base | p50 TTFT now | p99 TTFT base | p99 TTFT now |
|---|---|---|---|---|
| 2 | 0.1768 s | **0.0536 s** | 1.0176 s | **0.1597 s** |
| 3 | 0.5775 s | **0.0597 s** | 2.3476 s | **0.1414 s** |
| 4 *(base saturated)* | 1.7260 s | **0.0678 s** | 5.5115 s | **0.9048 s** |
| 5 *(base saturated)* | 3.9203 s | **0.0849 s** | 10.8430 s | **1.1017 s** |
| 6 *(both saturated)* | 6.8362 s | 2.2966 s | 17.8081 s | 10.0957 s |

Highest load each engine kept up with:

| | achieved QPS | p50 TTFT | p99 TTFT |
|---|---|---|---|
| milestone-1 blocking server | 2.49 | 0.5775 s | 2.3476 s |
| current server | **4.43** | **0.0849 s** | **1.1017 s** |

**1.78× the sustained throughput, at 85 % lower p50 TTFT and 53 % lower p99.**

Read those percentages carefully: most of the gap at 4–5 QPS is *the baseline
degrading*, not the current server improving — its p50 goes 0.58 s → 3.92 s
while the current server moves 0.06 s → 0.08 s. A serving stack's value shows up
as the load at which it stops working. The comparison also bundles every
milestone at once; attribution to individual mechanisms is in the experiments
below.

This machine sustains roughly **5 QPS at 32 output tokens** and **12 QPS at 8**
(`bench/load.py --probe`, [`results/load_probe.json`](results/load_probe.json)).
That is what a 1.1B model on one M1 Pro does; the sweep range was chosen to
bracket it rather than to hit a round number.

**Continuous vs static batching** — a short request arriving 1 s into a batch of
eight 300-token generations (`bench/late_arrival.py`,
[`results/late_arrival.json`](results/late_arrival.json)):

| | static | continuous | |
|---|---|---|---|
| late-request TTFT, one slot free | 9.1631 s | **0.0683 s** | 134× faster |
| late-request TTFT, all slots busy | 8.7515 s | 9.1142 s | no better — see below |

The saturated case is reported as measured: rebuilding the batch every step
cannot create capacity that does not exist. Fixing it needs preemption — which
is the next table.

**Preemption** — the same saturated workload, with the late request marked
high-priority so the scheduler may take a slot from a background generation
(`bench/preemption.py`, 3 runs each, [`results/preemption.json`](results/preemption.json)):

| | preemption off | preemption on |
|---|---|---|
| urgent-request TTFT, mean | 15.8096 s (σ 1.3878) | **0.112 s** (σ 0.0175) |
| longest background request | 16.8175 s (σ 1.389) | 15.2641 s (σ 0.0953) |
| background output length | 300/300 tokens | 300/300 tokens |

141× lower TTFT for the urgent request, with no measurable slowdown for the
background ones and no truncated output. Preemption is pause-and-resume: the
victim keeps its emitted tokens and continues from where it stopped.

**Starvation protection** — 4 slots, 6 generators keeping the queue permanently
full of priority-0 work, one priority-9 request
(`bench/starvation.py`, [`results/starvation.json`](results/starvation.json)):

| | low-priority request | urgent requests served meanwhile |
|---|---|---|
| protection on (`starvation_s=5`) | **completed in 6.428 s** | 23 |
| protection off (`starvation_s=0`) | **still waiting at 60 s** | 172 |

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

## Observability

| endpoint | what it is |
|---|---|
| `GET /metrics` | Prometheus text exposition — counters, gauges, and summaries with exact quantiles |
| `GET /metrics.json` | the same numbers as a JSON snapshot |
| `GET /metrics/requests` | raw per-request rows for real completed requests |
| `GET /dashboard` | dependency-free live view, for watching the scheduler during a benchmark |

Latency is exposed as summaries rather than histograms: the registry keeps raw
samples, so the quantiles are exact rather than bucket-rounded. **A quantile
below its sample floor is omitted, not invented** — p90 needs 10 samples and p99
needs 100, and below that the series is simply absent.

A real scrape is committed at [`results/metrics_sample.txt`](results/metrics_sample.txt);
a test checks the exposition parses with `prometheus_client`'s own parser.

Four lines from that scrape, verbatim (120 completed requests, so p99 is
justified and present):

```
llama_serve_time_to_first_token_seconds{quantile="0.5"} 0.20892
llama_serve_time_to_first_token_seconds{quantile="0.99"} 0.3293
llama_serve_kv_cache_hits_total 121
llama_serve_requests_finished_total 120
```

## Environment this was built on

- Apple M1 Pro, 16 GB unified memory, macOS 26.5.1 (Darwin 25.5.0, arm64) —
  every benchmark JSON records the host it ran on, so this stays checkable
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
pytest          # 87 tests, no model required (deterministic mock backend)
ruff check .    # lint
```

## Reproducing the measurements

Every table above comes from one of these. Each writes its raw output under
`results/`, and each takes a `--label`, so a re-run lands next to the committed
one instead of overwriting it.

```bash
# capacity of this machine, then the sweep and its baseline
python bench/load.py --probe --max-tokens 32
python -m llama_serve.server --engine continuous --max-seqs 8
python bench/load.py --label continuous --qps 1,2,3,4,5,6 --runs 3 --duration 20
python -m llama_serve.server --engine simple --max-seqs 1
python bench/load.py --label simple-baseline --qps 1,2,3,4,5,6 --runs 3 --duration 20
python bench/load.py --compare simple-baseline,continuous

# the per-mechanism experiments
python bench/late_arrival.py --label continuous-headroom --mode headroom
python bench/prefix_cache.py --label cache-on --waves 6
python bench/prefix_cache_equivalence.py
python bench/preemption.py --label preempt-on
python bench/starvation.py --label protected
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
