"""Server configuration. Env-overridable so benchmarks can sweep settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    return type(default)(raw)


@dataclass
class Config:
    # model
    model_path: str = DEFAULT_MODEL
    backend: str = "llama.cpp"  # "llama.cpp" | "mock"
    n_gpu_layers: int = -1
    n_ctx_per_seq: int = 1024
    verbose_llama: bool = False

    # engine
    engine: str = "continuous"  # "simple" | "static" | "continuous"
    max_seqs: int = 8  # concurrent KV sequence slots
    max_queue: int = 256
    prefill_chunk: int = 256  # max prompt tokens admitted per decode step
    block_size: int = 16  # paged KV block size, in tokens

    # scheduling
    policy: str = "priority"  # "fcfs" | "priority"
    enable_preemption: bool = True
    enable_prefix_cache: bool = True
    starvation_s: float = 5.0  # age at which a request is force-promoted

    # mock backend knobs (benchmarking the scheduler without a model)
    mock_token_latency_s: float = 0.0

    # server
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Config":
        d = cls()
        for f in d.__dataclass_fields__:
            d.__dict__[f] = _env(f"LLAMA_SERVE_{f.upper()}", getattr(d, f))
        return d

    def as_dict(self) -> dict:
        return dict(self.__dict__)
