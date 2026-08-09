"""Milestone 4: the server-side prefix cache, built on the paged allocator.

Goal: when two requests begin with the same tokens — a shared system prompt, a
few-shot preamble, a chat history being continued — the second one should not
recompute the attention state for that prefix.

**Two layers, because two different things own memory.**

`PagedKVCache` (see `paged_kv.py`) is the allocator: fixed-size blocks, a free
list, reference counts, copy-on-write, LRU reclamation. This class is the
*policy* on top of it — what to cache, which cached prefix to serve a new
request from, and what to throw away when the budget runs out.

The split exists because the backend, not this process, owns the physical KV
tensors. llama.cpp's cache is unified: every cell carries a *set* of sequence
ids, and `llama_memory_seq_cp` adds `dst` to a range of cells rather than
copying their contents. Sharing a prefix there is genuinely zero-copy, exactly
as a block table with refcounts would be. So this class supplies the
block-level policy and llama.cpp supplies the mechanism, behind the backend's
`seq_share_prefix` operation. Against
the mock backend the mechanism is `PagedKVCache` too — the backend stores its
KV in one — which is what lets prefix sharing be tested for correctness, as an
output-equality assertion, without a GPU.

**Donor sequences.** A prefix can only be shared while some sequence still
holds it, and the backend addresses *sequences*, not block ids. So when a
request finishes, its block-aligned prompt prefix is handed to a dedicated
*cache sequence id* drawn from a pool beyond the running slots, and the
request's own slot is released. Those donor sequences are what the allocator's
reference counts are counting.

**The invalidation rule.** A cached prefix is evicted only when the block
budget cannot otherwise satisfy a new publication, and then strictly in LRU
order over whole entries. Entries are the eviction unit rather than blocks
because the backend addresses sequences: evicting one block out of the middle
of a donor sequence would leave a hole no backend can express. Within an entry the
blocks are still refcounted, so evicting entry A does not disturb the blocks A
shares with entry B — those simply drop to one reference and stay resident.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .paged_kv import OutOfBlocks, PagedKVCache


@dataclass
class PrefixEntry:
    """A cached prompt prefix, kept alive in the backend by `seq_id`."""

    seq_id: int  # donor sequence in the backend
    hashes: tuple[int, ...]  # chained per-block hashes
    tokens: tuple[int, ...]  # block-aligned prompt prefix, for exact match
    last_used: float = field(default_factory=time.perf_counter)
    hits: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.hashes)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


class BlockManager:
    """Prefix-cache policy over a `PagedKVCache` block budget."""

    def __init__(self, backend, block_size: int, cache_seq_ids: list[int], total_blocks: int):
        self.backend = backend
        self.block_size = max(1, block_size)
        # track_cells=False: the backend holds the actual KV. This pool exists
        # to allocate, refcount and reclaim *blocks*, which is what bounds how
        # much of the backend's cache the prefix cache is allowed to pin.
        self.pool = PagedKVCache(
            num_blocks=max(1, total_blocks), block_size=self.block_size, track_cells=False
        )
        self._free_cache_seqs: list[int] = list(cache_seq_ids)
        self._entries: dict[int, PrefixEntry] = {}  # donor seq_id -> entry
        self._lock = threading.RLock()

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.tokens_saved = 0
        self.tokens_prefilled = 0
        self.publications = 0

    # --- hashing ----------------------------------------------------------
    def block_hashes(self, tokens: list[int]) -> tuple[int, ...]:
        return tuple(self.pool.block_hashes(tokens))

    # --- engine hooks -----------------------------------------------------
    def on_admit(self, req) -> int:
        """Populate `req.seq_id`'s KV from the cache. Returns tokens already computed.

        Matches on everything the request needs back in the cache, which for a
        request resumed after preemption is prompt + tokens already generated.
        """
        tokens = req.prefill_src or req.prompt_tokens
        hashes = self.block_hashes(tokens)
        if not hashes:
            with self._lock:
                self.misses += 1
            return 0

        with self._lock:
            entry, n_match = self._longest_match_locked(hashes, tokens)
            if entry is None or n_match == 0:
                self.misses += 1
                return 0
            n_tokens = n_match * self.block_size
            # Always leave at least one prompt token to prefill: the model must
            # run at least one position to produce logits to sample from.
            if n_tokens >= len(tokens):
                n_tokens -= self.block_size
                n_match -= 1
            if n_tokens <= 0:
                self.misses += 1
                return 0
            entry.last_used = time.perf_counter()
            entry.hits += 1
            donor = entry.seq_id

        if not self.backend.seq_share_prefix(donor, req.seq_id, n_tokens):
            with self._lock:
                self.misses += 1
            return 0

        with self._lock:
            self.hits += 1
            self.tokens_saved += n_tokens
        # Reported against the prompt, so the server-level hit rate stays a
        # fraction: a resumed request can legitimately reuse more tokens than
        # its prompt contains, and that surplus is counted as resume work.
        req.metrics.cached_prompt_tokens = min(n_tokens, len(req.prompt_tokens))
        return n_tokens

    def on_finish(self, req) -> bool:
        """Retire `req`: publish its prompt prefix to the cache, free its slot.

        Returns True if a new cache entry was created.
        """
        return self._retire(req, req.prompt_tokens)

    def on_preempt(self, req) -> bool:
        """Retire a *paused* request, publishing prompt + generated tokens.

        This is what makes preempt-by-recompute cheap. The request will be
        re-admitted needing exactly these tokens back in the cache, and the
        entry published here is the one it will match, so the resume costs a
        cache lookup and a few uncached tail tokens rather than a full prefill.
        """
        return self._retire(req, req.resume_tokens)

    def _retire(self, req, tokens: list[int]) -> bool:
        published = False
        try:
            published = self._publish(req, tokens)
        finally:
            # The request's own sequence is always released, published or not.
            self.backend.seq_rm(req.seq_id, -1, -1)
        return published

    def _publish(self, req, tokens: list[int]) -> bool:
        hashes = self.block_hashes(tokens)
        if not hashes:
            return False
        aligned = tuple(tokens[: len(hashes) * self.block_size])

        with self._lock:
            if self._exact_locked(hashes) is not None:
                # Already cached by an earlier request; refresh its recency and
                # do not spend a second donor slot on identical content.
                self._exact_locked(hashes).last_used = time.perf_counter()
                return False
            cache_seq = self._acquire_cache_seq_locked(list(aligned))
            if cache_seq is None:
                return False
            # Blocks this request computed itself are distinct physical cells in
            # the backend even where their content hash matches; only the prefix
            # it was *given* at admit time is genuinely shared downstream.
            share_limit = req.metrics.cached_prompt_tokens
            try:
                self.pool.allocate(cache_seq, list(aligned), share_limit=share_limit)
            except OutOfBlocks:
                self._free_cache_seqs.append(cache_seq)
                return False

        if not self.backend.seq_share_prefix(req.seq_id, cache_seq, len(aligned)):
            with self._lock:  # backend refused; hand the resources back
                self.pool.free_seq(cache_seq)
                self._free_cache_seqs.append(cache_seq)
            return False

        with self._lock:
            self._entries[cache_seq] = PrefixEntry(
                seq_id=cache_seq, hashes=hashes, tokens=aligned
            )
            self.publications += 1
        return True

    def note_prefill(self, n: int) -> None:
        with self._lock:
            self.tokens_prefilled += n

    # --- internals --------------------------------------------------------
    def _longest_match_locked(self, hashes: tuple[int, ...], tokens: list[int]):
        """The entry sharing the longest block-aligned prefix with `hashes`.

        The hash chain narrows the search; token equality decides it, so a hash
        collision costs a cache miss instead of producing corrupt output.
        """
        best, best_n = None, 0
        for e in self._entries.values():
            n = 0
            for a, b in zip(e.hashes, hashes, strict=False):  # shortest chain wins
                if a != b:
                    break
                n += 1
            if n > best_n:
                B = self.block_size
                if e.tokens[: n * B] != tuple(tokens[: n * B]):
                    continue
                best, best_n = e, n
        return best, best_n

    def _exact_locked(self, hashes: tuple[int, ...]) -> PrefixEntry | None:
        for e in self._entries.values():
            if e.hashes == hashes:
                return e
        return None

    def _acquire_cache_seq_locked(self, tokens: list[int]) -> int | None:
        """Get a donor sequence id, evicting whole LRU entries to make room.

        Two resources have to be available at once: a donor sequence id in the
        backend, and enough blocks in the pool. Both are freed by evicting an
        entry, so one loop covers both.
        """
        while not self._free_cache_seqs or not self.pool.would_fit(tokens):
            if not self._entries:
                return None
            victim = min(self._entries.values(), key=lambda e: e.last_used)
            self._evict_locked(victim)
        return self._free_cache_seqs.pop(0)

    def _evict_locked(self, entry: PrefixEntry) -> None:
        """Drop one cached prefix.

        Blocks it shares with a surviving entry keep a reference and stay
        resident; blocks unique to it fall to zero references and become
        reclaimable. Its backend sequence is removed, which is what actually
        returns KV cells to the model.
        """
        self._entries.pop(entry.seq_id, None)
        self.pool.free_seq(entry.seq_id)
        self.backend.seq_rm(entry.seq_id, -1, -1)
        self._free_cache_seqs.append(entry.seq_id)
        self.evictions += 1

    # --- introspection ----------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            attempted = self.tokens_saved + self.tokens_prefilled
            pool = self.pool.stats()
            return {
                "block_size": self.block_size,
                "total_blocks": pool["total_blocks"],
                "used_blocks": pool["used_blocks"],
                "free_blocks": pool["free_blocks"],
                "shared_blocks": pool["shared_blocks"],
                "cached_prefixes": len(self._entries),
                "free_cache_seqs": len(self._free_cache_seqs),
                "lookups": total,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "evictions": self.evictions,
                "publications": self.publications,
                "prompt_tokens_reused": self.tokens_saved,
                "prompt_tokens_computed": self.tokens_prefilled,
                # The number that matters: fraction of prompt work skipped.
                "prefill_saved_frac": (
                    round(self.tokens_saved / attempted, 4) if attempted else 0.0
                ),
            }
