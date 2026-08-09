"""Milestone 5: scheduling policy — priority, preemption, starvation protection.

Milestone 3 could admit a waiting request the instant a sequence slot freed.
That is only useful if a slot *does* free. The measured failure case is in
`bench/late_arrival.py`: with all eight slots holding 300-token generations, a short
request arriving 1 s in still waited 9.1 s, because iteration-level batching
cannot create capacity that does not exist. Taking capacity away from someone is
what this module adds.

Three policies, and each of them is a decision with a cost:

**Priority.** Requests carry an integer `priority`, lower meaning more urgent.
Selection is `(effective_priority, arrival)`, so ties fall back to FIFO and the
scheduler never reorders equal-priority work.

**Preemption is pause-and-resume, implemented by recompute.** When a
higher-priority request needs a slot and none is free, the lowest-priority
running request is *paused*: its sequence slot is released, the tokens it has
already emitted are kept, and it returns to the queue. When it is re-admitted it
re-prefills prompt + tokens-generated-so-far and carries on from exactly where
it stopped. The client sees a pause in its stream — never an error, never a
duplicated or dropped token.

    Why not abort-and-retry? Because the tokens already streamed to the client
    cannot be unsent. Aborting means either lying to the client or throwing away
    work it has already seen.

    Why not swap the KV to host memory? It is possible — llama.cpp exposes
    `llama_state_seq_get_data` / `llama_state_seq_set_data`, and for TinyLlama a
    300-token sequence is only ~7 MB. It was not done because recompute is
    strictly simpler and its cost here is small: re-prefilling N tokens is one
    batched forward pass over N tokens, the same shape as the original prefill,
    and the prompt part of it is usually served by the prefix cache from
    milestone 4 — the preempted request publishes prompt + generated tokens on
    its way out, so the resume is largely a cache hit rather than real work.
    Swap would win for very long sequences; that trade is not measured here, so
    it is not claimed.

**Starvation protection has to be a hard guarantee, not a nudge.** Aging alone
is not enough: a low-priority request whose effective priority improves with age
can still be admitted and then immediately preempted again by the next urgent
arrival, forever. So promotion comes in two parts:

  1. *Promotion* — once a request has waited `starvation_s` (measured from the
     last time it entered the queue, so a just-preempted request does not
     instantly look starved), it jumps ahead of every non-starving request
     regardless of priority, oldest arrival first.
  2. *Immunity* — a request admitted while promoted can never be preempted. So
     can a request that has already been preempted `max_preemptions` times,
     which bounds thrashing for requests that never quite reach the age
     threshold.

Together these give a real bound. Let `S = starvation_s`, `M` the request's
`max_tokens`, `T` the decode step time, and `K` the number of sequence slots.

  * The request waits at most `S` before promotion. It can be preempted at most
    `max_preemptions` times first, and each preemption restarts that wait, so
    the queueing term is at most `(max_preemptions + 1)·S`.
  * Once promoted it is picked next. If a slot is free it is admitted
    immediately. If not, the scheduler preempts the lowest-priority preemptible
    running request, so it is admitted on the next step. If *every* running
    request is itself immune, it waits at most until the shortest of them
    finishes, which is at most `M·T` — immune requests cannot be preempted, but
    they also cannot run forever, because `max_tokens` bounds them.
  * Once admitted it is immune, so it runs to completion in at most `M` steps.

  Total: **bounded by `(max_preemptions + 1)·S + 2·M·T`**, independent of how
  much high-priority load arrives afterwards. `tests/test_scheduler.py` asserts this empirically,
  and asserts the control case — with protection disabled the same workload
  starves the low-priority request indefinitely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .request import Request, Stage


@dataclass
class SchedulerConfig:
    policy: str = "priority"  # "fcfs" | "priority"
    enable_preemption: bool = True
    starvation_s: float = 5.0  # age at which a waiting request is force-promoted
    max_preemptions: int = 2  # after this many, a request becomes immune


class Scheduler:
    """Chooses what to admit next, and who to take a slot from.

    Pure policy: it never touches the backend or the KV cache. The engine calls
    it while holding its own lock and performs whatever it decides.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.preemptions = 0
        self.promotions = 0
        self.promoted_rids: set[int] = set()

    # --- admission --------------------------------------------------------
    def is_starving(self, req: Request, now: float) -> bool:
        """Has this request waited past the starvation threshold?

        Measured from `last_queued_t`, not from arrival: a request that has just
        been preempted has not been waiting, and treating it as the most starved
        thing in the queue would have the scheduler preempt it and immediately
        re-select it, forever.
        """
        s = self.config.starvation_s
        return s > 0 and (now - req.metrics.last_queued_t) >= s

    def pick(self, waiting: list[Request] | None, now: float) -> Request | None:
        """The next request to admit, or None if nothing is waiting."""
        if not waiting:
            return None
        if self.config.policy == "fcfs":
            return min(waiting, key=lambda r: r.metrics.arrival_t)

        starving = [r for r in waiting if self.is_starving(r, now)]
        if starving:
            # Oldest first, and ahead of every priority class. This is the only
            # rule in the module that ignores `priority`, and it is the one that
            # turns "usually fine" into a bound.
            chosen = min(starving, key=lambda r: r.metrics.arrival_t)
            if chosen.rid not in self.promoted_rids:
                self.promoted_rids.add(chosen.rid)
                self.promotions += 1
            return chosen
        return min(waiting, key=lambda r: (r.priority, r.metrics.arrival_t))

    # --- preemption -------------------------------------------------------
    def is_immune(self, req: Request) -> bool:
        """Immune requests are never preempted. Both cases exist to stop livelock.

        A request promoted for age that could still be preempted would make no
        progress under sustained load; a request preempted repeatedly would
        spend all its time re-prefilling.
        """
        return (
            req.rid in self.promoted_rids
            or req.metrics.preemptions >= self.config.max_preemptions
        )

    def choose_victim(
        self, active: list[Request], candidate: Request, now: float
    ) -> Request | None:
        """Who should give up their slot so `candidate` can run — if anyone.

        Returns None unless preemption is genuinely warranted: the candidate has
        to outrank the victim (or be starving), and the victim has to be a
        running request that is not immune.
        """
        if not self.config.enable_preemption:
            return None

        victims = [
            r
            for r in active
            if r.stage is Stage.DECODE and not self.is_immune(r) and r is not candidate
        ]
        if not victims:
            return None

        # Lowest priority first; among equals, the one admitted most recently,
        # because it has the least generated work to recompute on resume.
        victim = max(
            victims, key=lambda r: (r.priority, r.metrics.first_scheduled_t or 0.0)
        )
        starving = self.is_starving(candidate, now)
        if not starving and candidate.priority >= victim.priority:
            return None  # not urgent enough to cost someone else their progress
        return victim

    def note_preemption(self) -> None:
        self.preemptions += 1

    def forget(self, req: Request) -> None:
        """Drop per-request state once the request is gone."""
        self.promoted_rids.discard(req.rid)

    # --- introspection ----------------------------------------------------
    def stats(self) -> dict:
        return {
            "policy": self.config.policy,
            "preemption_enabled": self.config.enable_preemption,
            "starvation_s": self.config.starvation_s,
            "max_preemptions": self.config.max_preemptions,
            "preemptions": self.preemptions,
            "promotions": self.promotions,
            "currently_promoted": len(self.promoted_rids),
        }


def wait_time_s(req: Request, now: float | None = None) -> float:
    return (now or time.perf_counter()) - req.metrics.arrival_t
