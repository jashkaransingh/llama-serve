# Highest sustained load, by configuration

Generated from `results/load_sweep.json` by `bench/load.py --summary`.

Sustained = the highest offered level whose mean achieved QPS stayed within 10% of the offered rate, across all runs at that level. TTFT percentiles are pooled over every completed request at that level.

| configuration | model | out tok | slots | sustained QPS | p50 TTFT | p99 TTFT | offered | runs |
|---|---|---|---|---|---|---|---|---|
| `qwen2.5-0.5b-q4km-16tok-64seq-fastsampler` | qwen2.5-0.5b-instruct-q4_k_m.gguf | 16 | 64 | **27.95** | 0.54252 s | 1.43663 s | 30 | 3 |
| `qwen2.5-0.5b-q4km-16tok-64seq` | qwen2.5-0.5b-instruct-q4_k_m.gguf | 16 | 64 | **25.81** | 0.22926 s | 1.07725 s | 27 | 3 |
| `tinyllama-1.1b-q4km-16tok-64seq` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 16 | 64 | **14.95** | 0.20001 s | 1.05114 s | 16 | 3 |
| `qwen2.5-0.5b-q4km-32tok-64seq` | qwen2.5-0.5b-instruct-q4_k_m.gguf | 32 | 64 | **14.53** | 0.12502 s | 0.88169 s | 16 | 3 |
| `tinyllama-1.1b-q4km-32tok-64seq-fastsampler` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 32 | 64 | **10.43** | 0.19809 s | 1.3415 s | 12 | 3 |
| `tinyllama-1.1b-q4km-32tok-64seq` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 32 | 64 | **10.39** | 0.19486 s | 1.50618 s | 12 | 3 |
| `continuous` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 32 | 8 | **4.43** | 0.08492 s | 1.10171 s | 5 | 3 |
| `simple-baseline` | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 32 | 1 | **2.49** | 0.57748 s | 2.34761 s | 3 | 3 |

Peak measured: **27.95 QPS** on `qwen2.5-0.5b-q4km-16tok-64seq-fastsampler` (qwen2.5-0.5b-instruct-q4_k_m.gguf, 16 output tokens, 64 sequence slots, macOS-26.5.1-arm64-arm-64bit).
