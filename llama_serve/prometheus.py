"""Milestone 6: Prometheus exposition for the metrics the server already keeps.

**Prometheus text format, not JSON, and why.** The JSON snapshot
(`MetricsRegistry.snapshot`) is convenient to read and useless to scrape: every
number in it is a point-in-time value with no type, so nothing downstream can
tell a counter from a gauge, and `rate()` over it is meaningless. The text
format costs one module and makes the server work with the thing people
actually run. It is served at `/metrics`, which is the path scrapers default to.
The JSON snapshot did not go away — it moved to `/metrics.json`, and the raw
per-request rows are at `/metrics/requests`.

**Summaries, not histograms, for latency.** A histogram needs buckets chosen in
advance, and the honest bucket layout for TTFT on this machine (tens of
milliseconds when a request walks into a free slot, tens of *seconds* when it
waits out a saturated batch) spans three orders of magnitude. The registry
already keeps the raw samples in a bounded ring buffer and computes exact
percentiles from them, so a summary reports what was actually measured instead
of what a bucket edge rounds it to. The trade-off is real and worth naming:
summary quantiles cannot be aggregated across replicas. For a single-node
server that is not a cost.

**A quantile that cannot be justified is not emitted.** `MetricsRegistry`
refuses to report p90 below 10 samples and p99 below 100, because "p99" over six
requests is noise. That rule is kept here: the corresponding `quantile` line is
simply absent rather than filled in with a plausible-looking number. A missing
series is a fact a dashboard can show; a fabricated one is not.
"""

from __future__ import annotations

from .metrics import MetricsRegistry, _pct

_PREFIX = "llama_serve"


def _fmt(v: float | int) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    return repr(round(float(v), 6))


class _Doc:
    """Accumulates exposition lines, one metric family at a time."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def metric(self, name: str, kind: str, help_text: str) -> None:
        self.lines.append(f"# HELP {_PREFIX}_{name} {help_text}")
        self.lines.append(f"# TYPE {_PREFIX}_{name} {kind}")

    def sample(self, name: str, value, labels: str = "") -> None:
        if value is None:
            return
        self.lines.append(f"{_PREFIX}_{name}{labels} {_fmt(value)}")

    def counter(self, name: str, value, help_text: str) -> None:
        self.metric(name, "counter", help_text)
        self.sample(name, value)

    def gauge(self, name: str, value, help_text: str) -> None:
        if value is None:
            return
        self.metric(name, "gauge", help_text)
        self.sample(name, value)

    def summary(self, name: str, values: list[float], help_text: str) -> None:
        """A summary over the raw samples the registry retained.

        Quantiles below their sample-count floor are omitted rather than
        guessed, so a dashboard shows a gap instead of a fiction.
        """
        self.metric(name, "summary", help_text)
        if values:
            for q, floor in ((0.5, 1), (0.9, 10), (0.99, 100)):
                v = _pct(values, q, min_n=floor)
                if v is not None:
                    self.sample(name, v, f'{{quantile="{q}"}}')
            self.sample(f"{name}_sum", sum(values))
        self.sample(f"{name}_count", len(values))

    def render(self) -> str:
        return "\n".join(self.lines) + "\n"


def render(metrics: MetricsRegistry, engine=None, config=None) -> str:
    """Render everything the server knows in Prometheus text format."""
    snap = metrics.snapshot(engine)
    rows = metrics.rows()
    d = _Doc()

    def col(key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    # --- request lifecycle ------------------------------------------------
    req = snap["requests"]
    d.counter("requests_arrived_total", req["arrived"], "Requests accepted by the server.")
    d.counter("requests_finished_total", req["finished"], "Requests that reached a terminal state.")
    d.counter("requests_errored_total", req["errored"], "Requests that ended in an error.")
    d.gauge("requests_in_flight", req["in_flight"], "Requests accepted but not yet finished.")

    # --- tokens -----------------------------------------------------------
    tok = snap["tokens"]
    d.counter("prompt_tokens_total", tok["prompt"], "Prompt tokens across all finished requests.")
    d.counter("generated_tokens_total", tok["generated"], "Tokens generated across all requests.")
    d.counter(
        "prompt_tokens_from_cache_total",
        tok["prompt_cache_hits"],
        "Prompt tokens served from the prefix cache instead of being prefilled.",
    )
    d.counter(
        "preemptions_total", tok["preemptions"], "Times a running request was paused for another."
    )

    # --- latency, from real completed requests ----------------------------
    d.summary(
        "queue_time_seconds",
        col("queue_time_s"),
        "Time from arrival until the request was first given a sequence slot.",
    )
    d.summary(
        "time_to_first_token_seconds",
        col("ttft_s"),
        "Time from arrival until the first generated token was emitted.",
    )
    d.summary(
        "request_duration_seconds", col("total_time_s"), "Arrival to completion, per request."
    )
    d.summary(
        "output_tokens_per_second",
        col("output_tps"),
        "Per-request decode throughput, measured over the decode phase only.",
    )

    # --- server-wide throughput ------------------------------------------
    d.gauge(
        "throughput_requests_per_second",
        req["throughput_rps"],
        "Finished requests per second over the current metrics window.",
    )
    d.gauge(
        "throughput_generated_tokens_per_second",
        tok["generated_tps"],
        "Generated tokens per second over the current metrics window.",
    )
    d.gauge("metrics_window_seconds", snap["window_s"], "Age of the current metrics window.")

    # --- engine -----------------------------------------------------------
    eng = snap.get("engine")
    if eng:
        d.gauge("queue_depth", eng.get("pending"), "Requests waiting for a sequence slot.")
        d.gauge("running_sequences", eng.get("running"), "Requests currently occupying a slot.")
        d.gauge("free_sequence_slots", eng.get("free_slots"), "Unoccupied sequence slots.")
        d.gauge(
            "max_sequence_slots", eng.get("max_concurrent_seqs"), "Configured concurrent slots."
        )
        d.counter("decode_steps_total", eng.get("steps"), "Batched forward passes executed.")
        d.gauge(
            "slot_utilization_ratio",
            eng.get("avg_slot_utilization"),
            "Mean fraction of sequence slots carrying a decode token per step.",
        )
        d.gauge(
            "decode_step_seconds_avg",
            (eng.get("avg_step_ms") or 0) / 1000.0,
            "Mean wall time per scheduler step, including sampling and detokenisation.",
        )
        d.gauge("batch_width_avg", eng.get("avg_batch_width"), "Mean entries per batched decode.")
        d.counter(
            "admissions_into_running_batch_total",
            eng.get("admissions_into_running_batch"),
            "Requests folded into a batch that was already running.",
        )
        d.counter(
            "resumes_total", eng.get("resumes"), "Preempted requests re-admitted and resumed."
        )
        d.counter(
            "resume_recomputed_tokens_total",
            eng.get("resumed_tokens_recomputed"),
            "Tokens re-prefilled by resumed requests that the prefix cache did not supply.",
        )

        sch = eng.get("scheduler") or {}
        d.counter(
            "scheduler_promotions_total",
            sch.get("promotions"),
            "Requests force-promoted after waiting past the starvation threshold.",
        )
        d.gauge(
            "scheduler_promoted_requests",
            sch.get("currently_promoted"),
            "Promoted requests currently in the system; these cannot be preempted.",
        )

        kv = eng.get("kv_cache") or {}
        if kv:
            d.gauge("kv_blocks_total", kv.get("total_blocks"), "Blocks in the prefix-cache pool.")
            d.gauge("kv_blocks_used", kv.get("used_blocks"), "Blocks currently referenced.")
            d.gauge(
                "kv_blocks_shared",
                kv.get("shared_blocks"),
                "Blocks referenced by more than one cached prefix.",
            )
            d.gauge("kv_cached_prefixes", kv.get("cached_prefixes"), "Resident cached prefixes.")
            d.counter("kv_cache_hits_total", kv.get("hits"), "Prefix-cache lookups that matched.")
            d.counter("kv_cache_misses_total", kv.get("misses"), "Prefix-cache lookups that did not.")
            d.counter("kv_cache_evictions_total", kv.get("evictions"), "Cached prefixes evicted.")
            d.counter(
                "kv_prefill_tokens_reused_total",
                kv.get("prompt_tokens_reused"),
                "Prompt tokens that did not have to be prefilled.",
            )
            d.counter(
                "kv_prefill_tokens_computed_total",
                kv.get("prompt_tokens_computed"),
                "Prompt tokens actually pushed through the model.",
            )

    # --- build info -------------------------------------------------------
    if config is not None:
        labels = (
            f'{{engine="{config.engine}",backend="{config.backend}",'
            f'policy="{config.policy}",block_size="{config.block_size}",'
            f'max_seqs="{config.max_seqs}"}}'
        )
        d.metric("build_info", "gauge", "Server configuration, as labels on a constant 1.")
        d.sample("build_info", 1, labels)

    return d.render()


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
