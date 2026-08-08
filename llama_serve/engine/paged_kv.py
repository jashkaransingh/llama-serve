"""Milestone 4: a real paged KV-cache allocator.

This is the data structure, not a wrapper around a dict. It owns a fixed pool
of `num_blocks` physical blocks, each holding exactly `block_size` token
positions, and hands them out through a free list. A sequence does not own a
contiguous region of the cache; it owns a *block table* — an ordered list of
physical block ids — so logical position `p` of sequence `s` lives at

    block  = table[s][p // block_size]
    offset = p % block_size

That indirection is the entire point. It is what allows two sequences to share
the physical blocks of a common prefix instead of each storing its own copy,
and what allows a sequence's cache to grow without a contiguous reservation.

Three policies live here, and each is a deliberate choice:

**Sharing is by content hash, verified by tokens.** A full block is hashed as a
chain over all preceding blocks:

    h_0 = hash(tokens[0:B])
    h_i = hash(h_{i-1}, tokens[i*B:(i+1)*B])

Chaining makes a match *positional*: identical tokens at a different offset
produce a different hash, so a prefix match is a true prefix and never a
coincidental mid-prompt collision. On a hash hit the candidate block's stored
token ids are compared for equality before reuse, so a hash collision degrades
into a cache miss rather than into silently wrong output.

**Invalidation is by reference count, not by timer.** See `free_seq` and
`_evict_one`: a block with `ref_count > 0` is never evicted, ever. Only blocks
that no sequence references are eviction candidates, taken in LRU order. This
is the whole cache-invalidation policy, and it is safe because of the next rule.

**Full blocks are immutable; divergence is handled by copy-on-write.** A block
is registered in the hash index only once it is completely filled, and a filled
block is never written to again while shared. Writing into a block whose
`ref_count > 1` therefore triggers `_cow`: allocate a private block, copy the
cells, redirect this sequence's table entry, drop the old reference. The only
block that can be partially filled is a sequence's last one, so COW is
precisely the mechanism that lets two sequences share a prompt and then diverge
into different continuations.

`num_blocks * block_size` cells of physical storage are allocated up front and
stored in `self.cells`. In a production engine those cells are slices of the
key/value tensors; here each cell holds the token id that produced it, which is
what lets the mock backend use this class as its actual KV store and lets the
tests assert on real paged storage rather than on bookkeeping counters.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field

EMPTY = -1  # sentinel stored in an unwritten cell


class OutOfBlocks(RuntimeError):
    """The block pool is exhausted and nothing is evictable."""


@dataclass
class Block:
    """One fixed-size physical block."""

    block_id: int
    ref_count: int = 0
    n_filled: int = 0  # cells written, 0..block_size
    content_hash: int | None = None  # set only when full; None => not shareable
    tokens: tuple[int, ...] = ()  # token ids that produced this block, for exact match
    last_used: int = 0  # logical clock, for LRU among ref_count == 0 blocks

    @property
    def is_full(self) -> bool:
        return self.content_hash is not None


@dataclass
class AllocResult:
    """What `allocate` did, in terms callers can assert on."""

    n_cached_tokens: int = 0  # prefix served from already-resident blocks
    n_new_tokens: int = 0  # tokens that still need to be computed
    shared_blocks: list[int] = field(default_factory=list)
    new_blocks: list[int] = field(default_factory=list)


class PagedKVCache:
    """A fixed pool of KV blocks with refcounted sharing and copy-on-write.

    Every public method takes the lock: the engine's scheduler thread mutates
    the tables while the event loop reads `stats()`.
    """

    def __init__(self, num_blocks: int, block_size: int, track_cells: bool = True):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.track_cells = track_cells

        self._blocks = [Block(i) for i in range(num_blocks)]
        # Physical storage. Real engines put tensor slices here.
        self.cells: list[list[int]] = (
            [[EMPTY] * block_size for _ in range(num_blocks)] if track_cells else []
        )

        self._free: deque[int] = deque(range(num_blocks))
        self._index: dict[int, int] = {}  # content_hash -> block_id
        # ref_count == 0 and hashed: reusable by hash, evictable in LRU order.
        self._evictable: OrderedDict[int, None] = OrderedDict()

        self._tables: dict[int, list[int]] = {}  # seq_id -> block ids
        self._lens: dict[int, int] = {}  # seq_id -> logical token count

        self._clock = 0
        self._lock = threading.RLock()

        # counters
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.cow_copies = 0
        self.tokens_reused = 0
        self.blocks_allocated = 0

    # --- hashing ----------------------------------------------------------
    def block_hashes(self, tokens: list[int] | tuple[int, ...]) -> list[int]:
        """Chained hashes over the block-aligned prefix of `tokens`.

        A trailing partial block gets no hash: it is still mutable, and hashing
        it would let a half-written block be shared.
        """
        B = self.block_size
        out: list[int] = []
        h = 0
        for start in range(0, len(tokens) - B + 1, B):
            h = hash((h, tuple(tokens[start : start + B])))
            out.append(h)
        return out

    # --- allocation -------------------------------------------------------
    def allocate(self, seq_id: int, tokens: list[int], share_limit: int | None = None) -> AllocResult:
        """Give `seq_id` a block table covering `tokens`, sharing what it can.

        Returns how many leading tokens came from already-resident blocks. The
        caller must still compute the remaining ones; this class does not know
        how to run a model.

        `share_limit` caps how many leading tokens may be satisfied by sharing.
        Callers that sit above a backend which owns the physical cells use it to
        keep this pool's accounting exact: a block this sequence computed for
        itself is a distinct physical allocation downstream even when its
        content hash matches a resident block, and counting it as shared would
        understate real cache occupancy.
        """
        with self._lock:
            if seq_id in self._tables:
                self._free_locked(seq_id)
            B = self.block_size
            hashes = self.block_hashes(tokens)
            max_shared = len(hashes) if share_limit is None else share_limit // B

            table: list[int] = []
            res = AllocResult()

            # 1. share the longest run of leading blocks already in the pool
            for i, h in enumerate(hashes):
                if i >= max_shared:
                    break
                bid = self._index.get(h)
                if bid is None:
                    break
                blk = self._blocks[bid]
                want = tuple(tokens[i * B : (i + 1) * B])
                if blk.tokens != want:  # hash collision: refuse rather than corrupt
                    break
                self._acquire_locked(bid)
                table.append(bid)
                res.shared_blocks.append(bid)

            n_shared_blocks = len(table)
            res.n_cached_tokens = n_shared_blocks * B
            res.n_new_tokens = len(tokens) - res.n_cached_tokens

            # 2. private blocks for the rest, rolled back atomically on OOM
            try:
                for start in range(res.n_cached_tokens, len(tokens), B):
                    chunk = tokens[start : start + B]
                    bid = self._alloc_block_locked()
                    table.append(bid)
                    res.new_blocks.append(bid)
                    self._write_block_locked(bid, 0, chunk)
                    if len(chunk) == B:
                        self._publish_locked(bid, hashes[start // B], tuple(chunk))
            except OutOfBlocks:
                for bid in table:
                    self._release_locked(bid)
                raise

            self._tables[seq_id] = table
            self._lens[seq_id] = len(tokens)
            if n_shared_blocks:
                self.hits += 1
                self.tokens_reused += res.n_cached_tokens
            else:
                self.misses += 1
            return res

    def fork(self, src: int, dst: int) -> None:
        """Point `dst` at every block of `src` (copy-on-write).

        Nothing is copied here. The two sequences share physical blocks until
        one of them writes, at which point `_cow` gives the writer a private
        copy of just the block being written.
        """
        with self._lock:
            if src not in self._tables:
                raise KeyError(f"unknown sequence {src}")
            if dst in self._tables:
                self._free_locked(dst)
            table = list(self._tables[src])
            for bid in table:
                self._acquire_locked(bid)
            self._tables[dst] = table
            self._lens[dst] = self._lens[src]

    def share_prefix(self, src: int, dst: int, n_tokens: int) -> int:
        """Give `dst` the first `n_tokens` of `src`'s cache.

        Whole blocks are shared by reference. A trailing partial block cannot be
        shared — it is still mutable — so it is copied into a private block for
        `dst`. Returns the number of tokens `dst` ended up holding.
        """
        with self._lock:
            if src not in self._tables:
                return 0
            n_tokens = min(n_tokens, self._lens[src])
            if n_tokens <= 0:
                return 0
            if dst in self._tables:
                self._free_locked(dst)

            B = self.block_size
            src_table = self._tables[src]
            n_whole, tail = divmod(n_tokens, B)
            table: list[int] = []
            try:
                for bid in src_table[:n_whole]:
                    self._acquire_locked(bid)
                    table.append(bid)
                if tail:
                    bid = self._alloc_block_locked()
                    table.append(bid)
                    src_bid = src_table[n_whole]
                    if self.track_cells:
                        self.cells[bid][:tail] = self.cells[src_bid][:tail]
                    self._blocks[bid].n_filled = tail
                    self.cow_copies += 1
            except OutOfBlocks:
                for bid in table:
                    self._release_locked(bid)
                raise
            self._tables[dst] = table
            self._lens[dst] = n_tokens
            return n_tokens

    # --- writing ----------------------------------------------------------
    def write(self, seq_id: int, pos: int, token: int) -> None:
        """Write one token at logical position `pos`, growing the table as needed.

        This is the copy-on-write trigger: if the target block is shared with
        another sequence, the writer gets a private copy first.
        """
        with self._lock:
            if pos < 0:
                raise ValueError("pos must be >= 0")
            table = self._tables.setdefault(seq_id, [])
            self._lens.setdefault(seq_id, 0)
            bi, off = divmod(pos, self.block_size)
            grown: list[int] = []
            try:
                while len(table) + len(grown) <= bi:
                    grown.append(self._alloc_block_locked())
            except OutOfBlocks:  # leave the sequence exactly as it was
                for gid in grown:
                    self._release_locked(gid)
                raise
            table.extend(grown)
            bid = table[bi]
            if self._blocks[bid].ref_count > 1:
                bid = self._cow_locked(seq_id, bi)
            self._write_block_locked(bid, off, [token])
            self._lens[seq_id] = max(self._lens[seq_id], pos + 1)

    def read(self, seq_id: int, pos: int) -> int:
        """Token stored at logical position `pos`, or EMPTY."""
        with self._lock:
            table = self._tables.get(seq_id)
            if not table or pos < 0 or pos >= self._lens.get(seq_id, 0):
                return EMPTY
            bi, off = divmod(pos, self.block_size)
            if bi >= len(table):
                return EMPTY
            return self.cells[table[bi]][off] if self.track_cells else EMPTY

    def tokens(self, seq_id: int) -> list[int]:
        """The full logical token sequence, gathered through the block table."""
        with self._lock:
            table = self._tables.get(seq_id)
            if not table or not self.track_cells:
                return []
            n = self._lens.get(seq_id, 0)
            out: list[int] = []
            for bid in table:
                out.extend(self.cells[bid])
            return out[:n]

    # --- release ----------------------------------------------------------
    def free_seq(self, seq_id: int) -> None:
        """Drop `seq_id`'s reference to every block it holds.

        Blocks still referenced by another sequence survive untouched — that is
        the invalidation rule. Blocks that fall to zero references and carry a
        content hash become *evictable*: still resident, still reusable by a
        future prefix match, but now candidates for reclamation under pressure.
        """
        with self._lock:
            self._free_locked(seq_id)

    def truncate(self, seq_id: int, n_tokens: int) -> None:
        """Keep only the first `n_tokens`; release blocks past the cut."""
        with self._lock:
            table = self._tables.get(seq_id)
            if table is None:
                return
            n_tokens = max(0, min(n_tokens, self._lens.get(seq_id, 0)))
            keep = (n_tokens + self.block_size - 1) // self.block_size
            for bid in table[keep:]:
                self._release_locked(bid)
            del table[keep:]
            self._lens[seq_id] = n_tokens
            if not table:
                self._tables.pop(seq_id, None)
                self._lens.pop(seq_id, None)

    # --- introspection ----------------------------------------------------
    def n_tokens(self, seq_id: int) -> int:
        with self._lock:
            return self._lens.get(seq_id, 0)

    def block_table(self, seq_id: int) -> list[int]:
        with self._lock:
            return list(self._tables.get(seq_id, []))

    def ref_count(self, block_id: int) -> int:
        with self._lock:
            return self._blocks[block_id].ref_count

    @property
    def free_blocks(self) -> int:
        with self._lock:
            return len(self._free) + len(self._evictable)

    @property
    def used_blocks(self) -> int:
        with self._lock:
            return self.num_blocks - len(self._free) - len(self._evictable)

    def would_fit(self, tokens: list[int]) -> bool:
        """True if `allocate(tokens)` can succeed without evicting a live block.

        Accounts for blocks the prefix would share rather than allocate.
        """
        with self._lock:
            B = self.block_size
            shared = 0
            for i, h in enumerate(self.block_hashes(tokens)):
                bid = self._index.get(h)
                if bid is None or self._blocks[bid].tokens != tuple(tokens[i * B : (i + 1) * B]):
                    break
                shared += 1
            need = (len(tokens) - shared * B + B - 1) // B
            return need <= len(self._free) + len(self._evictable)

    def stats(self) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "block_size": self.block_size,
                "total_blocks": self.num_blocks,
                "used_blocks": self.num_blocks - len(self._free) - len(self._evictable),
                "free_blocks": len(self._free),
                "evictable_blocks": len(self._evictable),
                "shared_blocks": sum(1 for b in self._blocks if b.ref_count > 1),
                "indexed_blocks": len(self._index),
                "sequences": len(self._tables),
                "lookups": lookups,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 4) if lookups else 0.0,
                "evictions": self.evictions,
                "cow_copies": self.cow_copies,
                "tokens_reused": self.tokens_reused,
                "blocks_allocated": self.blocks_allocated,
            }

    def check_invariants(self) -> None:
        """Assert the allocator's internal consistency. Used by the tests.

        Cheap enough to call in a loop, and it catches the failure mode that
        matters: a refcount that drifts from the tables it is supposed to
        summarise, which shows up much later as either a leak or as a block
        being handed out while somebody still reads it.
        """
        with self._lock:
            expected: dict[int, int] = {}
            for table in self._tables.values():
                for bid in table:
                    expected[bid] = expected.get(bid, 0) + 1
            for blk in self._blocks:
                want = expected.get(blk.block_id, 0)
                assert blk.ref_count == want, (
                    f"block {blk.block_id}: ref_count={blk.ref_count} but {want} table entries"
                )
                if blk.ref_count > 0:
                    assert blk.block_id not in self._free, f"block {blk.block_id} live but free"
                    assert blk.block_id not in self._evictable, (
                        f"block {blk.block_id} live but evictable"
                    )
                if blk.block_id in self._evictable:
                    assert blk.ref_count == 0
                    assert blk.content_hash is not None, "unhashed block is not reusable"
            assert len(set(self._free)) == len(self._free), "duplicate id in free list"
            for h, bid in self._index.items():
                assert self._blocks[bid].content_hash == h, f"stale index entry for block {bid}"
            total = len(self._free) + len(self._evictable) + sum(
                1 for b in self._blocks if b.ref_count > 0
            )
            assert total == self.num_blocks, f"block accounting lost blocks: {total}/{self.num_blocks}"

    # --- internals --------------------------------------------------------
    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _alloc_block_locked(self) -> int:
        """Take a block from the free list, evicting an unreferenced one if needed."""
        if not self._free:
            self._evict_one_locked()
        if not self._free:
            raise OutOfBlocks("no free blocks and nothing evictable")
        bid = self._free.popleft()
        blk = self._blocks[bid]
        blk.ref_count = 1
        blk.n_filled = 0
        blk.content_hash = None
        blk.tokens = ()
        blk.last_used = self._tick()
        if self.track_cells:
            cells = self.cells[bid]
            for i in range(self.block_size):
                cells[i] = EMPTY
        self.blocks_allocated += 1
        return bid

    def _evict_one_locked(self) -> None:
        """Reclaim the least recently used block that nobody references.

        A referenced block is never a candidate. If every block is referenced,
        this does nothing and the caller raises OutOfBlocks — dropping a live
        sequence's cache silently would corrupt its output.
        """
        if not self._evictable:
            return
        bid, _ = self._evictable.popitem(last=False)  # LRU end
        blk = self._blocks[bid]
        assert blk.ref_count == 0
        if blk.content_hash is not None:
            self._index.pop(blk.content_hash, None)
        blk.content_hash = None
        blk.tokens = ()
        blk.n_filled = 0
        self._free.append(bid)
        self.evictions += 1

    def _acquire_locked(self, bid: int) -> None:
        blk = self._blocks[bid]
        if blk.ref_count == 0:
            self._evictable.pop(bid, None)
        blk.ref_count += 1
        blk.last_used = self._tick()

    def _release_locked(self, bid: int) -> None:
        blk = self._blocks[bid]
        blk.ref_count -= 1
        assert blk.ref_count >= 0, f"block {bid} released more times than acquired"
        if blk.ref_count == 0:
            blk.last_used = self._tick()
            if blk.content_hash is not None:
                # Resident and reusable, but reclaimable under pressure.
                self._evictable[bid] = None
            else:
                # A partial block can never be matched again; recycle it now.
                blk.tokens = ()
                blk.n_filled = 0
                self._free.append(bid)

    def _write_block_locked(self, bid: int, offset: int, toks: list[int] | tuple[int, ...]) -> None:
        blk = self._blocks[bid]
        if blk.content_hash is not None:
            # Mutating a published block would invalidate every sharer's view.
            self._index.pop(blk.content_hash, None)
            blk.content_hash = None
            blk.tokens = ()
        if self.track_cells:
            self.cells[bid][offset : offset + len(toks)] = list(toks)
        blk.n_filled = max(blk.n_filled, offset + len(toks))
        blk.last_used = self._tick()

    def _publish_locked(self, bid: int, content_hash: int, toks: tuple[int, ...]) -> None:
        """Mark a now-full block immutable and shareable."""
        blk = self._blocks[bid]
        blk.content_hash = content_hash
        blk.tokens = toks
        blk.n_filled = self.block_size
        self._index[content_hash] = bid

    def _cow_locked(self, seq_id: int, block_index: int) -> int:
        """Give `seq_id` a private copy of its `block_index`-th block."""
        table = self._tables[seq_id]
        old = table[block_index]
        new = self._alloc_block_locked()
        if self.track_cells:
            self.cells[new][:] = list(self.cells[old])
        self._blocks[new].n_filled = self._blocks[old].n_filled
        table[block_index] = new
        self._release_locked(old)
        self.cow_copies += 1
        return new

    def _free_locked(self, seq_id: int) -> None:
        table = self._tables.pop(seq_id, None)
        self._lens.pop(seq_id, None)
        if not table:
            return
        for bid in table:
            self._release_locked(bid)
