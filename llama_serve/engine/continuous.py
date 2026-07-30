"""Milestone 3: continuous (iteration-level) batching.

The difference from static batching is the unit of scheduling. Static batching
schedules *a batch* and runs it to completion. This engine schedules *one
decode step* at a time, and rebuilds the batch on every single step. A request
that arrives while eight others are mid-generation joins on the next step —
typically within one token time — instead of waiting for the batch to drain.

Each step builds one `llama_batch` that mixes two kinds of work:

    decode slots   one token per running sequence, at its next position
    prefill slots  a chunk of prompt tokens for a sequence being admitted

llama.cpp allows both in a single `llama_decode` because every entry in the
batch carries its own `(seq_id, pos)`. That is the whole trick: prefill and
decode are the same operation at different widths, so a newly arrived request
can be folded into the same forward pass that is advancing everyone else.

Two ordering rules matter, and both are deliberate:

1. **Decodes are scheduled before prefills.** Decode slots are what keep
   already-admitted requests streaming; if a large prefill consumed the batch
   budget first, every running request would stall for that step. Filling
   decodes first bounds the inter-token latency of running requests by the
   step time, independent of how much prefill work is queued behind them.

2. **Prefill is chunked against the *remaining* budget.** A 900-token prompt
   does not get to occupy an entire forward pass. It is spread over several
   steps, so admitting a long prompt slows running requests slightly rather
   than pausing them completely.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator

from ..backends.base import TokenSlot
from ._util import Emitter, SENTINEL, accept_token, check_stop, finish, is_stop_token
from .base import ContextOverflow, Engine, QueueFull
from .request import FinishReason, Request, Stage


class ContinuousBatchEngine(Engine):
    name = "continuous"

    def __init__(self, backend, config, metrics=None):
        self.backend = backend
        self.config = config
        self.metrics = metrics
        self.max_seqs = min(config.max_seqs, backend.max_seqs)

        self._waiting: deque[Request] = deque()
        self._active: list[Request] = []
        self._free_slots: list[int] = list(range(self.max_seqs))
        self._samplers: dict[int, object] = {}
        self._eog: dict[int, object] = {}
        self._pending_token: dict[int, int] = {}

        self._cv = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._emitter: Emitter | None = None

        # instrumentation
        self.completed = 0
        self.failed = 0
        self.steps = 0
        self.decode_slots_total = 0
        self.prefill_slots_total = 0
        self.batch_widths: deque[int] = deque(maxlen=4000)
        self.decode_widths: deque[int] = deque(maxlen=4000)
        self.step_times: deque[float] = deque(maxlen=4000)
        self.admissions_mid_flight = 0

        # milestone 4 hooks in here; None means "no paged allocator".
        self.block_manager = None

    async def start(self) -> None:
        self._emitter = Emitter(asyncio.get_running_loop())
        self._thread = threading.Thread(target=self._run, name="continuous-batch", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=15)

    # --- admission --------------------------------------------------------
    def submit(self, req: Request) -> None:
        with self._cv:
            if len(self._waiting) + len(self._active) >= self.config.max_queue:
                raise QueueFull(f"queue full ({len(self._waiting)} waiting)")
        self.prepare(req, self.backend)
        budget = self.config.n_ctx_per_seq
        if len(req.prompt_tokens) + req.params.max_tokens > budget:
            raise ContextOverflow(
                f"prompt ({len(req.prompt_tokens)}) + max_tokens ({req.params.max_tokens}) "
                f"exceeds per-sequence context ({budget})"
            )
        req.queue = asyncio.Queue()
        req.done = asyncio.Event()
        with self._cv:
            self._waiting.append(req)
            self._cv.notify()

    async def stream(self, req: Request) -> AsyncIterator[str]:
        while True:
            item = await req.queue.get()
            if item is SENTINEL:
                break
            yield item

    # --- scheduler loop ---------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._active and not self._waiting:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    self._drain()
                    return
            try:
                self._step()
            except Exception as e:  # pragma: no cover - never let the loop die
                self._fail_all(e)

    def _step(self) -> None:
        t0 = time.perf_counter()
        budget = self.backend.n_batch
        slots: list[TokenSlot] = []
        sample_targets: list[tuple[Request, int]] = []

        # --- 1. decode slots for every running sequence ---
        with self._cv:
            running = [r for r in self._active if r.stage is Stage.DECODE]
        for req in running:
            if budget <= 0:
                break
            slots.append(
                TokenSlot(req.seq_id, self._pending_token[req.rid], req.n_computed, want_logits=True)
            )
            sample_targets.append((req, len(slots) - 1))
            req.n_computed += 1
            budget -= 1
        n_decode = len(slots)

        # --- 2. continue in-progress prefills, then admit new requests ---
        budget = self._fill_prefills(slots, sample_targets, budget)

        if not slots:
            return

        self.steps += 1
        self.batch_widths.append(len(slots))
        self.decode_widths.append(n_decode)
        self.decode_slots_total += n_decode
        self.prefill_slots_total += len(slots) - n_decode

        self.backend.decode(slots)

        # --- 3. sample one token for each sequence that produced logits ---
        for req, idx in sample_targets:
            self._advance(req, idx)
        # Timed across the whole step, including sampling and detokenisation:
        # per-step Python overhead is real serving cost and hiding it here
        # would make avg_step_ms disagree with observed throughput.
        self.step_times.append(time.perf_counter() - t0)

    def _fill_prefills(self, slots, sample_targets, budget: int) -> int:
        """Spend the remaining batch budget on prompt tokens."""
        while budget > 0:
            with self._cv:
                req = next((r for r in self._active if r.stage is Stage.PREFILL), None)
                if req is None:
                    req = self._admit_locked()
                if req is None:
                    break

            toks = req.prompt_tokens
            take = min(budget, self.config.prefill_chunk, len(toks) - req.n_computed)
            if take <= 0:  # nothing left to prefill; shouldn't happen, but don't spin
                req.stage = Stage.DECODE
                continue
            start = req.n_computed
            is_final = start + take == len(toks)
            for i in range(take):
                slots.append(
                    TokenSlot(
                        req.seq_id,
                        toks[start + i],
                        start + i,
                        want_logits=(is_final and i == take - 1),
                    )
                )
            req.n_computed += take
            budget -= take
            if is_final:
                sample_targets.append((req, len(slots) - 1))
                req.stage = Stage.DECODE
        return budget

    def _admit_locked(self) -> Request | None:
        """Move one waiting request into a free sequence slot. Caller holds _cv."""
        if not self._waiting or not self._free_slots:
            return None
        req = self._pick_waiting_locked()
        if req is None:
            return None
        self._waiting.remove(req)
        req.seq_id = self._free_slots.pop(0)
        req.stage = Stage.PREFILL
        req.metrics.first_scheduled_t = time.perf_counter()
        self.backend.seq_rm(req.seq_id, -1, -1)

        req.n_computed = 0
        if self.block_manager is not None:
            # Milestone 4: reuse any cached prefix instead of recomputing it.
            req.n_computed = self.block_manager.on_admit(req)

        self._samplers[req.rid] = self.backend.make_sampler(req.params)
        self._eog[req.rid] = is_stop_token(self.backend, req)
        self._active.append(req)
        if self._active and len(self._active) > 1:
            self.admissions_mid_flight += 1
        return req

    def _pick_waiting_locked(self) -> Request | None:
        """Selection policy. Milestone 5 replaces this with a real scheduler."""
        if not self._waiting:
            return None
        if self.config.policy == "priority":
            return min(self._waiting, key=lambda r: (r.priority, r.metrics.arrival_t))
        return self._waiting[0]

    def _advance(self, req: Request, logits_index: int) -> None:
        """Sample one token for `req` and decide whether it is finished."""
        sampler = self._samplers[req.rid]
        tok = sampler.sample(logits_index)

        if self._eog[req.rid](tok):
            self._retire(req, FinishReason.EOS)
            return
        sampler.accept(tok)
        piece = self.backend.token_to_piece(tok)
        accept_token(req, tok, piece)
        self._emitter.piece(req, piece)

        if check_stop(req):
            self._retire(req, FinishReason.STOP)
            return
        if len(req.output_tokens) >= req.params.max_tokens:
            self._retire(req, FinishReason.LENGTH)
            return
        if req.n_computed >= self.config.n_ctx_per_seq:
            self._retire(req, FinishReason.LENGTH)
            return
        self._pending_token[req.rid] = tok

    def _retire(self, req: Request, reason: FinishReason) -> None:
        """Free the sequence slot immediately — this is what lets the next
        waiting request start on the very next step."""
        finish(req, reason)
        with self._cv:
            if req in self._active:
                self._active.remove(req)
            if req.seq_id is not None:
                if self.block_manager is not None:
                    self.block_manager.on_finish(req)
                else:
                    self.backend.seq_rm(req.seq_id, -1, -1)
                self._free_slots.append(req.seq_id)
        s = self._samplers.pop(req.rid, None)
        if s is not None:
            s.close()
        self._eog.pop(req.rid, None)
        self._pending_token.pop(req.rid, None)
        if reason in (FinishReason.ERROR, FinishReason.ABORT):
            self.failed += 1
        else:
            self.completed += 1
        self._emitter.close(req)

    def _fail_all(self, exc: Exception) -> None:  # pragma: no cover
        with self._cv:
            victims = list(self._active)
        for req in victims:
            req.error = f"{type(exc).__name__}: {exc}"
            self._retire(req, FinishReason.ERROR)

    def _drain(self) -> None:
        with self._cv:
            victims = list(self._active) + list(self._waiting)
            self._waiting.clear()
        for req in victims:
            if not req.is_finished:
                if req.seq_id is None:  # never admitted
                    finish(req, FinishReason.ABORT)
                    self._emitter.close(req)
                else:
                    self._retire(req, FinishReason.ABORT)

    # --- introspection ----------------------------------------------------
    def stats(self) -> dict:
        with self._cv:
            waiting, active = len(self._waiting), len(self._active)
            free = len(self._free_slots)
        bw, dw, st = list(self.batch_widths), list(self.decode_widths), list(self.step_times)
        out = {
            "engine": self.name,
            "pending": waiting,
            "running": active,
            "free_slots": free,
            "max_concurrent_seqs": self.max_seqs,
            "completed": self.completed,
            "failed": self.failed,
            "steps": self.steps,
            "admissions_into_running_batch": self.admissions_mid_flight,
            "decode_slots_total": self.decode_slots_total,
            "prefill_slots_total": self.prefill_slots_total,
            "avg_batch_width": round(sum(bw) / len(bw), 3) if bw else 0,
            "avg_decode_width": round(sum(dw) / len(dw), 3) if dw else 0,
            "avg_slot_utilization": round(sum(dw) / (len(dw) * self.max_seqs), 4) if dw else 0,
            "avg_step_ms": round(1000 * sum(st) / len(st), 3) if st else 0,
        }
        if self.block_manager is not None:
            out["kv_cache"] = self.block_manager.stats()
        return out
