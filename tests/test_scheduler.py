"""Scheduling policy: priority, preemption, starvation protection.

Two kinds of test live here.

*Unit* tests drive `Scheduler` directly with hand-built requests. They pin the
selection rules, which are pure functions of priority and clock.

*Engine* tests run the real `ContinuousBatchEngine` on the mock backend with a
simulated per-token latency, so preemption genuinely happens against a running
batch and is observed the way a user would observe it — through completion
order, request counters and wall-clock deadlines. The mock backend's determinism
is what lets the strongest assertion here exist at all: that a request which was
preempted mid-generation produces *exactly* the text it would have produced if
nobody had touched it.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from llama_serve.api import build_backend, build_engine
from llama_serve.backends.base import SamplingParams
from llama_serve.config import Config
from llama_serve.engine.request import Request, RequestMetrics, Stage
from llama_serve.engine.scheduler import Scheduler, SchedulerConfig

# --- helpers ----------------------------------------------------------------


def mkreq(priority=0, age=0.0, now=None, preemptions=0, stage=Stage.QUEUED) -> Request:
    now = now if now is not None else time.perf_counter()
    r = Request(prompt="x", params=SamplingParams(), priority=priority)
    r.metrics = RequestMetrics(arrival_t=now - age, last_queued_t=now - age)
    r.metrics.preemptions = preemptions
    r.stage = stage
    return r


def sched(**kw) -> Scheduler:
    return Scheduler(SchedulerConfig(**kw))


def cfg(**kw) -> Config:
    c = Config(
        backend="mock",
        engine="continuous",
        max_seqs=2,
        cache_seqs=4,
        n_ctx_per_seq=1024,
        block_size=16,
        mock_token_latency_s=0.004,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class Harness:
    """Runs an engine and lets a test submit requests and await them."""

    def __init__(self, config: Config):
        self.config = config
        self.backend = build_backend(config)
        self.engine = build_engine(config, self.backend, None)

    async def __aenter__(self):
        await self.engine.start()
        return self

    async def __aexit__(self, *exc):
        await self.engine.stop()
        self.backend.close()

    def submit(self, priority=0, max_tokens=40, prompt="hello world") -> Request:
        req = Request(
            prompt=prompt,
            params=SamplingParams(max_tokens=max_tokens, ignore_eos=True).normalized(),
            priority=priority,
        )
        self.engine.submit(req)
        return req

    async def drain(self, req: Request) -> Request:
        async for _ in self.engine.stream(req):
            pass
        return req


# --- unit: priority ordering ------------------------------------------------
def test_lower_priority_value_is_served_first():
    s = sched()
    now = time.perf_counter()
    low = mkreq(priority=5, age=1.0, now=now)
    high = mkreq(priority=0, age=0.0, now=now)
    assert s.pick([low, high], now) is high


def test_equal_priority_falls_back_to_arrival_order():
    s = sched()
    now = time.perf_counter()
    first = mkreq(priority=1, age=2.0, now=now)
    second = mkreq(priority=1, age=1.0, now=now)
    assert s.pick([second, first], now) is first


def test_fcfs_policy_ignores_priority():
    s = sched(policy="fcfs")
    now = time.perf_counter()
    old_low = mkreq(priority=9, age=3.0, now=now)
    new_high = mkreq(priority=0, age=0.1, now=now)
    assert s.pick([new_high, old_low], now) is old_low


# --- unit: starvation promotion --------------------------------------------
def test_a_starving_request_outranks_every_priority_class():
    s = sched(starvation_s=1.0)
    now = time.perf_counter()
    starved = mkreq(priority=9, age=2.0, now=now)
    urgent = mkreq(priority=0, age=0.0, now=now)
    assert s.pick([urgent, starved], now) is starved
    assert s.stats()["promotions"] == 1


def test_promotion_makes_a_request_immune_to_preemption():
    """Without this, a promoted request is admitted and instantly evicted again,
    and it makes no progress no matter how long it waits."""
    s = sched(starvation_s=1.0)
    now = time.perf_counter()
    starved = mkreq(priority=9, age=2.0, now=now)
    s.pick([starved], now)  # promotes it
    starved.stage = Stage.DECODE
    urgent = mkreq(priority=0, age=0.0, now=now)
    assert s.choose_victim([starved], urgent, now) is None


def test_repeated_preemption_also_confers_immunity():
    s = sched(max_preemptions=2)
    now = time.perf_counter()
    thrashed = mkreq(priority=9, preemptions=2, stage=Stage.DECODE, now=now)
    urgent = mkreq(priority=0, now=now)
    assert s.choose_victim([thrashed], urgent, now) is None


def test_starvation_is_measured_from_the_last_time_it_queued():
    """A just-preempted request must not immediately look like the most starved
    one, or the scheduler preempts it and re-picks it forever."""
    s = sched(starvation_s=1.0)
    now = time.perf_counter()
    req = mkreq(priority=9, age=10.0, now=now)
    req.metrics.last_queued_t = now  # just preempted
    assert not s.is_starving(req, now)


# --- unit: victim selection -------------------------------------------------
def test_victim_is_the_lowest_priority_running_request():
    s = sched()
    now = time.perf_counter()
    a = mkreq(priority=1, stage=Stage.DECODE, now=now)
    b = mkreq(priority=7, stage=Stage.DECODE, now=now)
    c = mkreq(priority=3, stage=Stage.DECODE, now=now)
    assert s.choose_victim([a, b, c], mkreq(priority=0, now=now), now) is b


def test_no_preemption_without_a_priority_advantage():
    s = sched()
    now = time.perf_counter()
    running = mkreq(priority=2, stage=Stage.DECODE, now=now)
    assert s.choose_victim([running], mkreq(priority=2, now=now), now) is None
    assert s.choose_victim([running], mkreq(priority=1, now=now), now) is not None


def test_preemption_can_be_switched_off_entirely():
    s = sched(enable_preemption=False)
    now = time.perf_counter()
    running = mkreq(priority=9, stage=Stage.DECODE, now=now)
    assert s.choose_victim([running], mkreq(priority=0, now=now), now) is None


def test_a_request_still_prefilling_is_not_a_victim():
    """Preempting mid-prefill would throw away work without freeing a token of
    generated output, which is the worst of both."""
    s = sched()
    now = time.perf_counter()
    prefilling = mkreq(priority=9, stage=Stage.PREFILL, now=now)
    assert s.choose_victim([prefilling], mkreq(priority=0, now=now), now) is None


# --- engine: priority ordering under load -----------------------------------
@pytest.mark.asyncio
async def test_high_priority_requests_finish_first_under_load():
    """Queue six low-priority requests, then two urgent ones, with two slots."""
    async with Harness(cfg(max_seqs=2, enable_preemption=False)) as h:
        order: list[tuple[int, int]] = []

        async def run(req, tag):
            await h.drain(req)
            order.append((tag, req.rid))

        low = [h.submit(priority=5, max_tokens=30) for _ in range(6)]
        await asyncio.sleep(0.05)  # let the first two occupy the slots
        high = [h.submit(priority=0, max_tokens=30) for _ in range(2)]

        await asyncio.gather(*(run(r, 5) for r in low), *(run(r, 0) for r in high))

    finish_positions = [i for i, (tag, _) in enumerate(order) if tag == 0]
    # The two urgent requests cannot beat the two already running, but they must
    # beat the four still queued behind them.
    assert max(finish_positions) <= 3, f"urgent requests finished late: {order}"


# --- engine: preemption actually happens ------------------------------------
@pytest.mark.asyncio
async def test_preemption_happens_and_is_visible_in_timing_and_counters():
    """A single slot, occupied by a long low-priority generation, and then an
    urgent short request arrives."""
    async with Harness(cfg(max_seqs=1)) as h:
        long_req = h.submit(priority=9, max_tokens=400)
        task = asyncio.ensure_future(h.drain(long_req))
        await asyncio.sleep(0.15)  # it is definitely running now

        t0 = time.perf_counter()
        urgent = h.submit(priority=0, max_tokens=5)
        await h.drain(urgent)
        urgent_wall = time.perf_counter() - t0

        await task
        stats = h.engine.stats()

    assert stats["scheduler"]["preemptions"] >= 1, "nothing was preempted"
    assert long_req.metrics.preemptions >= 1, "the long request was never paused"
    assert stats["resumes"] >= 1, "the victim was never resumed"
    # 400 tokens at ~4 ms is ~1.6 s of work; the urgent request must not wait it out.
    assert urgent_wall < 0.6, f"urgent request waited {urgent_wall:.3f}s for a free slot"
    assert long_req.finish_reason is not None, "the victim never completed"


@pytest.mark.asyncio
async def test_preemption_is_disabled_by_configuration():
    """The control: with preemption off, the same workload makes the urgent
    request wait, which is what shows the mechanism is doing the work."""
    async with Harness(cfg(max_seqs=1, enable_preemption=False, starvation_s=0.0)) as h:
        long_req = h.submit(priority=9, max_tokens=400)
        task = asyncio.ensure_future(h.drain(long_req))
        await asyncio.sleep(0.15)

        t0 = time.perf_counter()
        urgent = h.submit(priority=0, max_tokens=5)
        await h.drain(urgent)
        urgent_wall = time.perf_counter() - t0
        await task
        stats = h.engine.stats()

    assert stats["scheduler"]["preemptions"] == 0
    assert urgent_wall > 0.6, (
        f"urgent request only waited {urgent_wall:.3f}s with preemption off; "
        "the comparison in the preemption test is not measuring anything"
    )


# --- engine: preemption must not corrupt the victim -------------------------
@pytest.mark.asyncio
async def test_a_preempted_request_produces_exactly_the_same_text():
    """The correctness claim for pause-and-resume.

    Same prompt, same length, same backend — once run alone, once interrupted
    mid-generation by urgent traffic. Pausing a request is only acceptable if
    the client cannot tell it happened.
    """
    prompt = "The quick brown fox jumps over the lazy dog and keeps running"

    async with Harness(cfg(max_seqs=1, enable_preemption=False)) as h:
        clean = await h.drain(h.submit(priority=0, max_tokens=120, prompt=prompt))
        expected = clean.text

    async with Harness(cfg(max_seqs=1)) as h:
        victim = h.submit(priority=9, max_tokens=120, prompt=prompt)
        task = asyncio.ensure_future(h.drain(victim))
        for _ in range(3):  # interrupt it repeatedly
            await asyncio.sleep(0.08)
            await h.drain(h.submit(priority=0, max_tokens=3))
        await task
        preemptions = victim.metrics.preemptions

    assert preemptions >= 1, "the victim was never actually preempted"
    assert len(victim.output_tokens) == 120, "resume lost or duplicated tokens"
    assert victim.text == expected, "preemption changed the generated text"


# --- engine: the starvation guarantee ---------------------------------------
@pytest.mark.asyncio
async def test_a_low_priority_request_completes_under_continuous_urgent_load():
    """The bounded-time guarantee, measured rather than asserted by inspection.

    A single sequence slot, one low-priority request, and a generator that keeps
    an urgent request in the queue at all times for the whole run. Without
    starvation protection the low-priority request never runs; with it, it must
    finish within the documented bound.
    """
    starvation_s = 0.4
    config = cfg(max_seqs=1, starvation_s=starvation_s, max_preemptions=2)

    async with Harness(config) as h:
        stop = False

        async def urgent_load():
            while not stop:
                await h.drain(h.submit(priority=0, max_tokens=25))

        pressure = [asyncio.ensure_future(urgent_load()) for _ in range(2)]
        await asyncio.sleep(0.05)

        t0 = time.perf_counter()
        low = h.submit(priority=9, max_tokens=30)
        await asyncio.wait_for(h.drain(low), timeout=15.0)
        elapsed = time.perf_counter() - t0

        stop = True
        for p in pressure:
            p.cancel()
        await asyncio.gather(*pressure, return_exceptions=True)
        stats = h.engine.stats()

    assert low.finish_reason is not None
    assert len(low.output_tokens) == 30, "completed, but not with the output it was owed"
    assert stats["scheduler"]["promotions"] >= 1, "it completed without ever being promoted"

    # Bound: (max_preemptions + 1) waits of at most `starvation_s`, plus the
    # generation itself. 30 tokens at ~4 ms/token is ~0.12 s of decode; the
    # allowance below is generous because the urgent load shares the same slot.
    bound = (config.max_preemptions + 1) * starvation_s + 2.0
    assert elapsed < bound, f"took {elapsed:.2f}s, bound was {bound:.2f}s"


@pytest.mark.asyncio
async def test_without_protection_the_same_workload_starves_it():
    """The control case. This is what makes the guarantee above a guarantee
    rather than a description of a machine that happened not to be busy."""
    config = cfg(max_seqs=1, starvation_s=0.0)  # 0 disables promotion

    async with Harness(config) as h:
        stop = False

        async def urgent_load():
            while not stop:
                await h.drain(h.submit(priority=0, max_tokens=25))

        pressure = [asyncio.ensure_future(urgent_load()) for _ in range(2)]
        await asyncio.sleep(0.05)
        low = h.submit(priority=9, max_tokens=30)

        starved = False
        try:
            await asyncio.wait_for(h.drain(low), timeout=2.0)
        except TimeoutError:
            starved = True

        stop = True
        for p in pressure:
            p.cancel()
        await asyncio.gather(*pressure, return_exceptions=True)

    assert starved, (
        "the low-priority request finished even without starvation protection; "
        "the load was not heavy enough for the guarantee test to mean anything"
    )


# --- sampler history contract -----------------------------------------------
def test_restore_history_replays_every_token():
    """Resume depends on it: a sampler that starts with a blank penalty history
    generates a different continuation than the one it is supposed to resume."""
    from llama_serve.backends.mock import MockBackend

    backend = MockBackend(max_seqs=1)
    sampler = backend.make_sampler(SamplingParams(seed=1))
    seen = []
    sampler.accept = seen.append  # type: ignore[method-assign]
    sampler.restore_history([5, 7, 9])
    assert seen == [5, 7, 9]


def test_engine_resume_uses_restore_history_not_accept():
    """`accept` is a no-op for llama.cpp's sampler chain, because
    `llama_sampler_sample` already accepted the token it returned. Resume has to
    go through `restore_history`, which is not a no-op — otherwise a resumed
    request silently loses its repetition-penalty history."""
    import inspect

    from llama_serve.engine import continuous

    src = inspect.getsource(continuous.ContinuousBatchEngine._admit_locked)
    assert "restore_history" in src
    assert "sampler.accept" not in src
