"""Helpers shared by the batching engines.

Both the static and continuous engines run their scheduling loop on a dedicated
worker thread — llama.cpp's `llama_decode` is a blocking C call (ctypes drops
the GIL for its duration), so driving it from the event loop would stall every
other connection. These helpers are the thread -> asyncio boundary.
"""

from __future__ import annotations

import time

from .request import FinishReason, Request, Stage

SENTINEL = object()


class Emitter:
    """Pushes generated pieces from the worker thread into a request's asyncio queue."""

    def __init__(self, loop):
        self.loop = loop

    def piece(self, req: Request, text: str) -> None:
        self.loop.call_soon_threadsafe(req.queue.put_nowait, text)

    def close(self, req: Request) -> None:
        self.loop.call_soon_threadsafe(req.queue.put_nowait, SENTINEL)
        self.loop.call_soon_threadsafe(req.done.set)


def check_stop(req: Request) -> bool:
    """Truncate at the first stop string, if any has appeared."""
    for s in req.params.stop:
        idx = req.text.find(s)
        if idx >= 0:
            req.text = req.text[:idx]
            return True
    return False


def accept_token(req: Request, tok: int, piece: str) -> None:
    if req.metrics.first_token_t is None:
        req.metrics.first_token_t = time.perf_counter()
    req.output_tokens.append(tok)
    req.text += piece


def finish(req: Request, reason: FinishReason) -> None:
    req.finish_reason = reason
    req.stage = Stage.ABORTED if reason in (FinishReason.ABORT, FinishReason.ERROR) else Stage.DONE
    req.metrics.generated_tokens = len(req.output_tokens)
    req.metrics.finished_t = time.perf_counter()


def eog_fn(backend):
    eos = backend.eos_tokens
    return getattr(backend, "is_eog", None) or (lambda t: t in eos)


def is_stop_token(backend, req: Request) -> callable:
    """EOG predicate for one request, honouring `ignore_eos`."""
    base = eog_fn(backend)
    if req.params.ignore_eos:
        return lambda t: False
    return base
