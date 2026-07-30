"""Engine interface.

Three engines implement this, and the API layer cannot tell them apart:

  simple      one request at a time, blocking          (milestone 1)
  static      concurrent requests grouped into a batch (milestone 2)
  continuous  iteration-level batching                 (milestone 3+)

Keeping them behind one interface is what makes the load-test comparison
meaningful: the same harness hits the same endpoint, and only the scheduler
underneath changes.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from .request import Request


class Engine(abc.ABC):
    name: str = "engine"

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def submit(self, req: Request) -> None:
        """Admit a request. Raises `QueueFull` if the queue is saturated."""

    @abc.abstractmethod
    async def stream(self, req: Request) -> AsyncIterator[str]:
        """Yield generated text pieces for an already-submitted request."""

    @abc.abstractmethod
    def stats(self) -> dict: ...

    # --- shared helpers ---------------------------------------------------
    def prepare(self, req: Request, backend) -> None:
        """Tokenize and record prompt length. Common to every engine."""
        req.prompt_tokens = backend.tokenize(req.prompt, add_bos=True)
        req.metrics.prompt_tokens = len(req.prompt_tokens)


class QueueFull(Exception):
    """The engine is at capacity; the API layer turns this into HTTP 503."""


class ContextOverflow(Exception):
    """Prompt + max_tokens does not fit in a sequence's context window."""
