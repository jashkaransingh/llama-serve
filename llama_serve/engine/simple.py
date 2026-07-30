"""Milestone 1: the baseline engine.

One request in flight at a time. Deliberately the dumbest correct thing:
prefill the whole prompt in one decode, then loop one token per decode.
Concurrent callers serialise behind a semaphore, so the second request's
queue time is the first request's entire generation time.

This is the control group. Every later milestone is measured against it, and
it stays in the repo as the fallback that is known to work.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from ..backends.base import TokenSlot
from .base import ContextOverflow, Engine, QueueFull
from .request import FinishReason, Request, Stage

_SENTINEL = object()


class SimpleEngine(Engine):
    name = "simple"

    def __init__(self, backend, config):
        self.backend = backend
        self.config = config
        self._gate = asyncio.Semaphore(1)
        self._pending = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self.completed = 0
        self.failed = 0
        self._history: list[dict] = []

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        pass

    def submit(self, req: Request) -> None:
        if self._pending >= self.config.max_queue:
            raise QueueFull(f"queue full ({self._pending})")
        self.prepare(req, self.backend)
        budget = self.backend.n_ctx_per_seq if hasattr(self.backend, "n_ctx_per_seq") else self.backend.n_ctx
        if len(req.prompt_tokens) + req.params.max_tokens > budget:
            raise ContextOverflow(
                f"prompt ({len(req.prompt_tokens)}) + max_tokens ({req.params.max_tokens}) "
                f"exceeds per-sequence context ({budget})"
            )
        req.queue = asyncio.Queue()
        req.done = asyncio.Event()
        self._pending += 1

    async def stream(self, req: Request) -> AsyncIterator[str]:
        try:
            async with self._gate:
                task = asyncio.get_running_loop().run_in_executor(None, self._run_blocking, req)
                while True:
                    item = await req.queue.get()
                    if item is _SENTINEL:
                        break
                    yield item
                await task
        finally:
            self._pending -= 1
            self._history.append(req.snapshot())
            del self._history[:-500]

    # --- blocking generation, runs in a worker thread ---------------------
    def _run_blocking(self, req: Request) -> None:
        loop = self._loop
        assert loop is not None

        def emit(piece: str) -> None:
            loop.call_soon_threadsafe(req.queue.put_nowait, piece)

        seq_id = 0
        sampler = None
        try:
            req.metrics.first_scheduled_t = time.perf_counter()
            req.stage = Stage.PREFILL
            self.backend.seq_rm(seq_id, -1, -1)
            sampler = self.backend.make_sampler(req.params)

            toks = req.prompt_tokens
            last = len(toks) - 1
            # Prefill in chunks so a long prompt cannot exceed n_batch.
            chunk = max(1, min(self.config.prefill_chunk, self.backend.n_batch))
            for start in range(0, len(toks), chunk):
                part = toks[start : start + chunk]
                slots = [
                    TokenSlot(seq_id, t, start + i, want_logits=(start + i == last))
                    for i, t in enumerate(part)
                ]
                self.backend.decode(slots)
                logits_index = len(slots) - 1

            req.n_computed = len(toks)
            req.stage = Stage.DECODE

            pos = len(toks)
            eos = self.backend.eos_tokens
            is_eog = getattr(self.backend, "is_eog", None) or (lambda t: t in eos)
            if req.params.ignore_eos:
                is_eog = lambda t: False  # noqa: E731 - benchmarking mode

            for _ in range(req.params.max_tokens):
                tok = sampler.sample(logits_index)
                if is_eog(tok):
                    req.finish_reason = FinishReason.EOS
                    break
                sampler.accept(tok)
                req.output_tokens.append(tok)
                if req.metrics.first_token_t is None:
                    req.metrics.first_token_t = time.perf_counter()

                piece = self.backend.token_to_piece(tok)
                req.text += piece
                emit(piece)

                if self._hit_stop(req):
                    req.finish_reason = FinishReason.STOP
                    break

                self.backend.decode([TokenSlot(seq_id, tok, pos, want_logits=True)])
                logits_index = 0
                pos += 1
            else:
                req.finish_reason = FinishReason.LENGTH

            req.stage = Stage.DONE
            self.completed += 1
        except Exception as e:  # pragma: no cover - surfaced to the client
            req.error = f"{type(e).__name__}: {e}"
            req.finish_reason = FinishReason.ERROR
            req.stage = Stage.ABORTED
            self.failed += 1
        finally:
            if sampler is not None:
                sampler.close()
            self.backend.seq_rm(seq_id, -1, -1)
            req.metrics.generated_tokens = len(req.output_tokens)
            req.metrics.finished_t = time.perf_counter()
            loop.call_soon_threadsafe(req.queue.put_nowait, _SENTINEL)
            loop.call_soon_threadsafe(req.done.set)

    def _hit_stop(self, req: Request) -> bool:
        for s in req.params.stop:
            idx = req.text.find(s)
            if idx >= 0:
                req.text = req.text[:idx]
                return True
        return False

    def stats(self) -> dict:
        return {
            "engine": self.name,
            "pending": self._pending,
            "running": 1 if self._gate.locked() else 0,
            "max_concurrent_seqs": 1,
            "completed": self.completed,
            "failed": self.failed,
        }
