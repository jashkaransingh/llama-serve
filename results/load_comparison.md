# Load sweep: `continuous` vs `simple-baseline`

Generated from `results/load_sweep.json` by `bench/load.py --compare`. 
Workload: shared ~40-token preamble + a distinct question per request, 32 output tokens, open-loop Poisson arrivals, 3 runs of 20.0 s per level.

Model: `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` on macOS-26.5.1-arm64-arm-64bit.

TTFT percentiles are pooled over every completed request at a level (p90 needs 10 samples, p99 needs 100; below that the cell is `—`). Achieved QPS is mean ± stdev across runs. **SAT** marks a level where the server could not keep up, so its latencies describe a growing backlog rather than a steady state.

| offered QPS | achieved (base) | achieved (cand) | p50 base | p50 cand | p50 lower | p99 base | p99 cand | p99 lower |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.95 ± 0.30 | 0.95 ± 0.30 | 0.0884 | 0.0553 | 37.4 % | — | — | — |
| 2 | 1.71 ± 0.13 | 1.71 ± 0.13 | 0.1768 | 0.0536 | 69.7 % | 1.0176 | 0.1597 | 84.3 % |
| 3 | 2.49 ± 0.37 | 2.57 ± 0.46 | 0.5775 | 0.0597 | 89.7 % | 2.3476 | 0.1414 | 94.0 % |
| 4<br>SAT base | 3.04 ± 0.10 | 3.57 ± 0.36 | 1.7260 | 0.0678 | 96.1 % | 5.5115 | 0.9048 | 83.6 % |
| 5<br>SAT base | 3.05 ± 0.13 | 4.43 ± 0.26 | 3.9203 | 0.0849 | 97.8 % | 10.8430 | 1.1017 | 89.8 % |
| 6<br>SAT base<br>SAT cand | 3.21 ± 0.05 | 4.19 ± 0.42 | 6.8362 | 2.2966 | 66.4 % | 17.8081 | 10.0957 | 43.3 % |

## Highest sustained load (last level the server kept up with)

| | achieved QPS | p50 TTFT | p99 TTFT |
|---|---|---|---|
| `simple-baseline` | 2.49 | 0.5775 s | 2.3476 s |
| `continuous` | 4.43 | 0.0849 s | 1.1017 s |

Capacity ratio: **1.78x**.
