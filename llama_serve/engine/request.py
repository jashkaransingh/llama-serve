"""Request lifecycle types shared by every scheduler and engine."""

from __future__ import annotations

import asyncio
import enum
import itertools
import time
from dataclasses import dataclass, field

from ..backends.base import SamplingParams

_ids = itertools.count(1)


class Stage(str, enum.Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    PREEMPTED = "preempted"
    DONE = "done"
    ABORTED = "aborted"


class FinishReason(str, enum.Enum):
    LENGTH = "length"
    EOS = "eos"
    STOP = "stop"
    ABORT = "abort"
    ERROR = "error"


@dataclass
class RequestMetrics:
    """Per-request timing. Every field is measured, never estimated."""

    arrival_t: float = field(default_factory=time.perf_counter)
    first_scheduled_t: float | None = None
    first_token_t: float | None = None
    finished_t: float | None = None

    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0  # prefill skipped thanks to the prefix cache
    generated_tokens: int = 0
    preemptions: int = 0

    @property
    def queue_time_s(self) -> float | None:
        if self.first_scheduled_t is None:
            return None
        return self.first_scheduled_t - self.arrival_t

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_t is None:
            return None
        return self.first_token_t - self.arrival_t

    @property
    def total_time_s(self) -> float | None:
        if self.finished_t is None:
            return None
        return self.finished_t - self.arrival_t

    @property
    def decode_time_s(self) -> float | None:
        if self.finished_t is None or self.first_token_t is None:
            return None
        return self.finished_t - self.first_token_t

    @property
    def output_tps(self) -> float | None:
        """Tokens/sec measured over the decode phase (excludes TTFT)."""
        dt = self.decode_time_s
        if not dt or self.generated_tokens <= 1:
            return None
        return (self.generated_tokens - 1) / dt

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "preemptions": self.preemptions,
            "queue_time_s": _r(self.queue_time_s),
            "ttft_s": _r(self.ttft_s),
            "total_time_s": _r(self.total_time_s),
            "output_tps": _r(self.output_tps),
        }


def _r(v: float | None, nd: int = 5) -> float | None:
    return None if v is None else round(v, nd)


@dataclass
class Request:
    """A single generation request as the engine sees it."""

    prompt: str
    params: SamplingParams
    priority: int = 0  # lower value == higher priority
    rid: int = field(default_factory=lambda: next(_ids))

    prompt_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    text: str = ""

    stage: Stage = Stage.QUEUED
    finish_reason: FinishReason | None = None
    error: str | None = None
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    # Runtime placement, owned by the engine.
    seq_id: int | None = None
    n_computed: int = 0  # prompt+output tokens already in the KV cache
    block_ids: list[int] = field(default_factory=list)

    # Streaming plumbing.
    queue: asyncio.Queue | None = field(default=None, repr=False)
    done: asyncio.Event | None = field(default=None, repr=False)

    @property
    def n_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.output_tokens

    @property
    def is_finished(self) -> bool:
        return self.stage in (Stage.DONE, Stage.ABORTED)

    @property
    def needs_prefill(self) -> bool:
        return self.n_computed < len(self.prompt_tokens)

    def snapshot(self) -> dict:
        return {
            "id": self.rid,
            "stage": self.stage.value,
            "priority": self.priority,
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
            **self.metrics.as_dict(),
        }
