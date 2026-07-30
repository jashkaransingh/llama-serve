"""Entrypoint: `python -m llama_serve.server --engine continuous`."""

from __future__ import annotations

import argparse

import uvicorn

from .api import create_app
from .config import Config


def parse_args(argv=None) -> Config:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(prog="llama-serve")
    p.add_argument("--model", dest="model_path", default=cfg.model_path)
    p.add_argument("--backend", choices=["llama.cpp", "mock"], default=cfg.backend)
    p.add_argument("--engine", choices=["simple", "static", "continuous"], default=cfg.engine)
    p.add_argument("--max-seqs", type=int, default=cfg.max_seqs)
    p.add_argument("--n-ctx-per-seq", type=int, default=cfg.n_ctx_per_seq)
    p.add_argument("--prefill-chunk", type=int, default=cfg.prefill_chunk)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--policy", choices=["fcfs", "priority"], default=cfg.policy)
    p.add_argument("--no-preemption", action="store_true")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--mock-token-latency", type=float, default=cfg.mock_token_latency_s)
    p.add_argument("--host", default=cfg.host)
    p.add_argument("--port", type=int, default=cfg.port)
    p.add_argument("--verbose-llama", action="store_true")
    a = p.parse_args(argv)

    cfg.model_path = a.model_path
    cfg.backend = a.backend
    cfg.engine = a.engine
    cfg.max_seqs = a.max_seqs
    cfg.n_ctx_per_seq = a.n_ctx_per_seq
    cfg.prefill_chunk = a.prefill_chunk
    cfg.block_size = a.block_size
    cfg.policy = a.policy
    cfg.enable_preemption = not a.no_preemption
    cfg.enable_prefix_cache = not a.no_prefix_cache
    cfg.mock_token_latency_s = a.mock_token_latency
    cfg.host, cfg.port = a.host, a.port
    cfg.verbose_llama = a.verbose_llama
    return cfg


def main(argv=None) -> None:
    cfg = parse_args(argv)
    print(f"[llama-serve] engine={cfg.engine} backend={cfg.backend} max_seqs={cfg.max_seqs}")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
