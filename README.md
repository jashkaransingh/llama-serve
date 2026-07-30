# llama-serve

A local LLM inference server built on `llama.cpp`, written to explore the parts of
model serving that sit *between* the HTTP handler and the forward pass: batching,
KV-cache management, and scheduling.

Serving a single request is a `for` loop. Serving a hundred concurrent ones well
is a scheduling problem. This repo implements the scheduling problem.

## Status

Milestone 1 works end to end against the real model. Nothing is claimed as working here until a smoke test in
the repo says it is, with the command used to verify it.

| # | Milestone | Status |
|---|-----------|--------|
| 0 | Scaffolding, backend interface, mock backend | ✅ |
| 1 | Baseline blocking server | ✅ |
| 2 | Static batching | ✅ |
| 3 | Continuous (iteration-level) batching | ⬜ |
| 4 | Paged KV-cache allocator + prefix sharing | ⬜ |
| 5 | Scheduling policy (priority + preemption) | ⬜ |
| 6 | Observability (`/metrics`) | ⬜ |
| 7 | Load testing harness + measured results | ⬜ |

**No performance number appears in this README unless it was produced by a script
in this repo and its raw output is committed under `results/`.**

## Environment this was built on

- Apple M1 Pro, 16 GB unified memory, macOS 15
- `llama-cpp-python` 0.3.34 built from source with `-DGGML_METAL=on`
- Model: TinyLlama-1.1B-Chat v1.0, Q4_K_M GGUF (638 MB) — small enough that
  scheduling effects, not raw model latency, dominate the measurements

## Quickstart

```bash
python -m llama_serve.server --engine simple --max-seqs 1     # baseline
python scripts/smoke_test.py --concurrent 4                   # verify it
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
