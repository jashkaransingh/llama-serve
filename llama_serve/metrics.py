"""Server-side metrics.

Records one row per completed request and derives percentiles from the raw
samples rather than from a running estimate — with request counts in the
hundreds, exact percentiles over a bounded ring buffer are cheaper and more
honest than a sketch. p99 of 200 samples is the 198th value; saying "p99" when
you have 20 samples is meaningless, so `_pct` returns None below a floor.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .engine.request import Request

_WINDOW = 2000  # requests retained for percentile computation


def _pct(values: list[float], q: float, min_n: int = 1) -> float | None:
    if len(values) < min_n or not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 5)
    # Nearest-rank: the smallest value at or above the q-th percentile.
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 5)


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._t0 = time.perf_counter()
            self._rows: deque[dict] = deque(maxlen=_WINDOW)
            self.arrived = 0
            self.finished = 0
            self.errored = 0
            self.prompt_tokens = 0
            self.cached_prompt_tokens = 0
            self.generated_tokens = 0
            self.preemptions = 0

    # --- hooks ------------------------------------------------------------
    def on_arrival(self, req: Request) -> None:
        with self._lock:
            self.arrived += 1

    def on_finish(self, req: Request) -> None:
        m = req.metrics
        with self._lock:
            self.finished += 1
            if req.error:
                self.errored += 1
            self.prompt_tokens += m.prompt_tokens
            self.cached_prompt_tokens += m.cached_prompt_tokens
            self.generated_tokens += m.generated_tokens
            self.preemptions += m.preemptions
            self._rows.append(
                {
                    "id": req.rid,
                    "t": time.perf_counter() - self._t0,
                    "queue_time_s": m.queue_time_s,
                    "ttft_s": m.ttft_s,
                    "total_time_s": m.total_time_s,
                    "output_tps": m.output_tps,
                    "prompt_tokens": m.prompt_tokens,
                    "cached_prompt_tokens": m.cached_prompt_tokens,
                    "generated_tokens": m.generated_tokens,
                    "preemptions": m.preemptions,
                    "priority": req.priority,
                    "finish_reason": req.finish_reason.value if req.finish_reason else None,
                }
            )

    # --- readout ----------------------------------------------------------
    def snapshot(self, engine=None) -> dict:
        with self._lock:
            rows = list(self._rows)
            elapsed = max(1e-9, time.perf_counter() - self._t0)
            arrived, finished, errored = self.arrived, self.finished, self.errored
            ptok, ctok, gtok = self.prompt_tokens, self.cached_prompt_tokens, self.generated_tokens
            preempt = self.preemptions

        def col(k):
            return [r[k] for r in rows if r.get(k) is not None]

        ttft, qt, tt, tps = col("ttft_s"), col("queue_time_s"), col("total_time_s"), col("output_tps")

        out = {
            "window_s": round(elapsed, 3),
            "requests": {
                "arrived": arrived,
                "finished": finished,
                "errored": errored,
                "in_flight": arrived - finished,
                "throughput_rps": round(finished / elapsed, 4),
            },
            "tokens": {
                "prompt": ptok,
                "generated": gtok,
                "prompt_cache_hits": ctok,
                "prompt_cache_hit_rate": round(ctok / ptok, 4) if ptok else 0.0,
                "generated_tps": round(gtok / elapsed, 3),
                "preemptions": preempt,
            },
            "latency": {
                "ttft_s": _summary(ttft),
                "queue_time_s": _summary(qt),
                "total_time_s": _summary(tt),
                "output_tps": _summary(tps),
            },
        }
        if engine is not None:
            out["engine"] = engine.stats()
        return out

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._rows)


def _summary(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "mean": round(sum(v) / len(v), 5),
        "p50": _pct(v, 0.50),
        "p90": _pct(v, 0.90, min_n=10),
        "p99": _pct(v, 0.99, min_n=100),  # below 100 samples p99 is noise
        "max": round(max(v), 5),
    }
