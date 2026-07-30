"""Milestone 2: static batching.

Requests that arrive close together are collected into one batch, prefilled,
and then decoded *together* — one `llama_decode` call carries one token for
every sequence in the batch, so N requests cost roughly one forward pass
instead of N.

The defining limitation, which is the entire reason milestone 3 exists:

    **The batch is fixed once it starts.** A request arriving one millisecond
    after the batch begins waits for every member of that batch to finish,
    including the one that was asked for 512 tokens. And as sequences finish,
    the batch shrinks — the GPU spends the tail of every batch running at a
    fraction of the width it was sized for.

The first effect is measured in bench/late_arrival.py rather than asserted.
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


class StaticBatchEngine(Engine):
    name = "static"

    def __init__(self, backend, config, metrics=None):
        self.backend = backend
        self.config = config
        self.metrics = metrics
        self.max_seqs = min(config.max_seqs, backend.max_seqs)

        self._waiting: deque[Request] = deque()
        self._cv = threading.Condition()
        self._running_n = 0
        self._stop = False
        self._thread: threading.Thread | None = None
        self._emitter: Emitter | None = None

        self.completed = 0
        self.failed = 0
        self.batches_run = 0
        self.batch_sizes: list[int] = []
        self.decode_steps = 0
        self.slot_utilization: list[float] = []

    async def start(self) -> None:
        self._emitter = Emitter(asyncio.get_running_loop())
        self._thread = threading.Thread(target=self._loop, name="static-batch", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=10)

    # --- admission --------------------------------------------------------
    def submit(self, req: Request) -> None:
        with self._cv:
            if len(self._waiting) + self._running_n >= self.config.max_queue:
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

    # --- worker thread ----------------------------------------------------
    def _loop(self) -> None:
        while True:
            batch = self._form_batch()
            if batch is None:
                return  # stopping
            try:
                self._run_batch(batch)
            except Exception as e:  # pragma: no cover - one bad batch must not kill the loop
                for r in batch:
                    if not r.is_finished:
                        r.error = f"{type(e).__name__}: {e}"
                        finish(r, FinishReason.ERROR)
                        self.failed += 1
                        self._release(r)

    def _form_batch(self) -> list[Request] | None:
        """Block for the first request, then briefly wait for stragglers.

        The short window matters: without it, two requests submitted
        microseconds apart would land in consecutive batches and see no
        batching benefit at all.
        """
        with self._cv:
            while not self._waiting and not self._stop:
                self._cv.wait(timeout=0.5)
            if self._stop:
                return None
            deadline = time.perf_counter() + self.config.static_batch_wait_s
            while len(self._waiting) < self.max_seqs:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
                if self._stop:
                    break
            batch = [self._waiting.popleft() for _ in range(min(self.max_seqs, len(self._waiting)))]
            self._running_n += len(batch)
        return batch

    def _run_batch(self, batch: list[Request]) -> None:
        now = time.perf_counter()
        for slot, req in enumerate(batch):
            req.seq_id = slot
            req.metrics.first_scheduled_t = now
            req.stage = Stage.PREFILL
            self.backend.seq_rm(slot, -1, -1)

        self.batches_run += 1
        self.batch_sizes.append(len(batch))
        del self.batch_sizes[:-200]

        samplers = {}
        logits_idx: dict[int, int] = {}
        try:
            # --- prefill: one sequence at a time, chunked ---
            for req in batch:
                samplers[req.rid] = self.backend.make_sampler(req.params)
                toks = req.prompt_tokens
                last = len(toks) - 1
                chunk = max(1, min(self.config.prefill_chunk, self.backend.n_batch))
                for start in range(0, len(toks), chunk):
                    part = toks[start : start + chunk]
                    self.backend.decode(
                        [
                            TokenSlot(req.seq_id, t, start + i, want_logits=(start + i == last))
                            for i, t in enumerate(part)
                        ]
                    )
                    logits_idx[req.rid] = len(part) - 1
                req.n_computed = len(toks)
                req.stage = Stage.DECODE

            # --- joint decode: one forward pass per token, across all seqs ---
            eog = {r.rid: is_stop_token(self.backend, r) for r in batch}
            active = list(batch)
            while active:
                still: list[Request] = []
                slots: list[TokenSlot] = []
                for req in active:
                    tok = samplers[req.rid].sample(logits_idx[req.rid])
                    if eog[req.rid](tok):
                        finish(req, FinishReason.EOS)
                        continue
                    samplers[req.rid].accept(tok)
                    piece = self.backend.token_to_piece(tok)
                    accept_token(req, tok, piece)
                    self._emitter.piece(req, piece)

                    if check_stop(req):
                        finish(req, FinishReason.STOP)
                        continue
                    if len(req.output_tokens) >= req.params.max_tokens:
                        finish(req, FinishReason.LENGTH)
                        continue
                    slots.append(TokenSlot(req.seq_id, tok, req.n_computed, want_logits=True))
                    req.n_computed += 1
                    still.append(req)

                for req in active:
                    if req.is_finished:
                        self.completed += 1
                        self._release(req)

                if not slots:
                    break
                # Slot utilisation: how much of the batch width this pass used.
                self.slot_utilization.append(len(slots) / self.max_seqs)
                del self.slot_utilization[:-2000]
                self.decode_steps += 1
                self.backend.decode(slots)
                for i, req in enumerate(still):
                    logits_idx[req.rid] = i
                active = still
        finally:
            for s in samplers.values():
                s.close()
            for req in batch:
                self.backend.seq_rm(req.seq_id, -1, -1)
                if not req.is_finished:
                    finish(req, FinishReason.ABORT)
                    self._release(req)

    def _release(self, req: Request) -> None:
        with self._cv:
            self._running_n -= 1
        self._emitter.close(req)

    # --- introspection ----------------------------------------------------
    def stats(self) -> dict:
        with self._cv:
            waiting = len(self._waiting)
            running = self._running_n
        util = self.slot_utilization
        return {
            "engine": self.name,
            "pending": waiting,
            "running": running,
            "max_concurrent_seqs": self.max_seqs,
            "completed": self.completed,
            "failed": self.failed,
            "batches_run": self.batches_run,
            "avg_batch_size": round(sum(self.batch_sizes) / len(self.batch_sizes), 2)
            if self.batch_sizes
            else 0,
            "decode_steps": self.decode_steps,
            # The number milestone 3 is designed to raise.
            "avg_slot_utilization": round(sum(util) / len(util), 4) if util else 0,
        }
