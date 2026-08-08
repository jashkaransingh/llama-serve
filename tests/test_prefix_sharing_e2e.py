"""End-to-end prefix-sharing correctness, through the real engine.

The unit tests assert that blocks are shared. These assert the thing that
actually matters to a user: *sharing a prefix must not change the answer*.

That assertion has teeth here because the mock backend's next token is a pure
function of the KV contents at the two most recent positions, read back through
the block table. If the allocator hands a sequence the wrong block, or lets a
copy-on-write slip, the generated text changes and these tests fail.
"""

from __future__ import annotations

import pytest

from llama_serve.api import build_backend, build_engine
from llama_serve.backends.base import SamplingParams
from llama_serve.config import Config
from llama_serve.engine.request import Request

PREAMBLE = "You are a careful assistant. Follow the rules exactly. Cite your sources.\n"


def cfg(prefix_cache: bool, **kw) -> Config:
    c = Config(
        backend="mock",
        engine="continuous",
        max_seqs=2,
        cache_seqs=4,
        n_ctx_per_seq=512,
        block_size=8,
        enable_prefix_cache=prefix_cache,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


async def run_prompts(config: Config, prompts: list[str], max_tokens: int = 24) -> list[str]:
    """Run prompts one wave at a time and return their generated text.

    Sequential waves, not one burst: a cache can only hit on a prefix some
    earlier request has already published, so firing them all at once would
    measure nothing.
    """
    backend = build_backend(config)
    engine = build_engine(config, backend, None)
    await engine.start()
    out: list[str] = []
    try:
        for p in prompts:
            req = Request(
                prompt=p, params=SamplingParams(max_tokens=max_tokens, ignore_eos=True).normalized()
            )
            engine.submit(req)
            async for _ in engine.stream(req):
                pass
            out.append(req.text)
    finally:
        await engine.stop()
        backend.close()
    return out, engine


@pytest.mark.asyncio
async def test_shared_prefix_gives_identical_output_to_independent_runs():
    """The headline correctness claim for milestone 4."""
    prompts = [PREAMBLE + q for q in ("What is a RAID rebuild?", "Why did throughput drop?")]

    with_cache, engine = await run_prompts(cfg(True), prompts)
    without_cache, _ = await run_prompts(cfg(False), prompts)

    assert engine.block_manager is not None
    assert engine.block_manager.stats()["hits"] >= 1, "the workload never hit the cache"
    assert with_cache == without_cache, "prefix sharing changed the generated text"


@pytest.mark.asyncio
async def test_cache_actually_skips_prefill_work():
    """Correctness is necessary but not sufficient — it has to save something."""
    prompts = [PREAMBLE + q for q in ("Question one?", "Question two?", "Question three?")]
    _, engine = await run_prompts(cfg(True), prompts)
    st = engine.block_manager.stats()
    assert st["prompt_tokens_reused"] > 0
    assert st["prefill_saved_frac"] > 0.0
    assert st["hits"] >= 2, f"only {st['hits']} hits across 3 identical-preamble prompts"


@pytest.mark.asyncio
async def test_divergence_after_a_shared_prefix_is_isolated():
    """Two prompts sharing a preamble must diverge into different answers, and
    each must match what it would have produced alone."""
    a = PREAMBLE + "Alpha alpha alpha?"
    b = PREAMBLE + "Beta beta beta?"

    shared, _ = await run_prompts(cfg(True), [a, b])
    alone_a, _ = await run_prompts(cfg(True), [a])
    alone_b, _ = await run_prompts(cfg(True), [b])

    assert shared[0] != shared[1], "divergent prompts produced identical text"
    assert shared[0] == alone_a[0]
    assert shared[1] == alone_b[0], "the second sequence saw the first one's tokens"


@pytest.mark.asyncio
async def test_output_is_unchanged_when_the_cache_is_forced_to_evict():
    """A tiny block budget makes every publication evict something. Eviction is
    a capacity decision, so it must never change an answer."""
    prompts = [PREAMBLE + f"Question {i} about storage arrays?" for i in range(6)]

    roomy, _ = await run_prompts(cfg(True, cache_seqs=4), prompts)
    cramped, engine = await run_prompts(cfg(True, cache_seqs=1), prompts)
    none, _ = await run_prompts(cfg(False), prompts)

    assert engine.block_manager.stats()["evictions"] > 0, "the budget was not actually tight"
    assert roomy == none
    assert cramped == none, "eviction changed the generated text"


@pytest.mark.asyncio
async def test_concurrent_requests_with_a_shared_prefix_stay_correct():
    """Same prefix, in flight at the same time, sharing sequence slots."""
    import asyncio

    config = cfg(True, max_seqs=4)
    backend = build_backend(config)
    engine = build_engine(config, backend, None)
    await engine.start()

    async def one(q: str) -> str:
        req = Request(
            prompt=PREAMBLE + q,
            params=SamplingParams(max_tokens=20, ignore_eos=True).normalized(),
        )
        engine.submit(req)
        async for _ in engine.stream(req):
            pass
        return req.text

    questions = [f"Concurrent question {i}?" for i in range(8)]
    try:
        # A first wave to populate the cache, then a burst that will hit it.
        await one(questions[0])
        burst = await asyncio.gather(*(one(q) for q in questions))
    finally:
        await engine.stop()
        backend.close()

    serial, _ = await run_prompts(cfg(False), [PREAMBLE + q for q in questions], max_tokens=20)
    assert burst == serial, "concurrent shared-prefix requests diverged from serial ones"
    engine.block_manager.pool.check_invariants()
    backend.pages.check_invariants()
