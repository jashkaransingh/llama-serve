"""Correctness tests for the paged KV allocator.

These are the tests that matter most in the whole repo, because the allocator
fails *silently*. A block shared when it should have been copied does not raise;
it produces subtly wrong attention state and therefore subtly wrong text. So
every test here asserts on observable content — the tokens a sequence reads back
through its block table — and on the allocator's own invariants, not on counters.
"""

from __future__ import annotations

import pytest

from llama_serve.engine.paged_kv import EMPTY, OutOfBlocks, PagedKVCache

B = 4


def cache(num_blocks=16, block_size=B):
    return PagedKVCache(num_blocks=num_blocks, block_size=block_size)


# --- allocation / free / reuse ---------------------------------------------
def test_allocate_lays_tokens_out_through_the_block_table():
    c = cache()
    toks = list(range(10, 20))  # 10 tokens -> 2 full blocks + 1 partial
    res = c.allocate(1, toks)
    assert res.n_cached_tokens == 0
    assert res.n_new_tokens == 10
    assert len(c.block_table(1)) == 3
    assert c.tokens(1) == toks
    assert c.n_tokens(1) == 10
    c.check_invariants()


def test_blocks_are_not_contiguous_and_that_is_fine():
    """Two sequences interleaved: neither owns a contiguous region."""
    c = cache()
    c.allocate(1, list(range(8)))
    c.allocate(2, list(range(100, 108)))
    c.write(1, 8, 999)
    c.write(2, 8, 888)
    assert c.tokens(1) == list(range(8)) + [999]
    assert c.tokens(2) == list(range(100, 108)) + [888]
    assert set(c.block_table(1)).isdisjoint(c.block_table(2))
    c.check_invariants()


def test_free_returns_blocks_to_the_pool_and_they_are_reused():
    c = cache(num_blocks=4)
    c.allocate(1, list(range(16)))  # exactly 4 blocks: pool is full
    assert c.free_blocks == 0
    first = set(c.block_table(1))
    c.free_seq(1)
    assert c.free_blocks == 4
    c.allocate(2, list(range(200, 216)))
    assert set(c.block_table(2)) == first, "freed blocks were not reused"
    assert c.tokens(2) == list(range(200, 216))
    c.check_invariants()


def test_partial_block_is_recycled_immediately_on_free():
    """A never-filled block can never be matched by hash, so holding it in the
    reusable set would just waste capacity."""
    c = cache(num_blocks=4)
    c.allocate(1, [1, 2])  # one partial block
    c.free_seq(1)
    assert c.stats()["evictable_blocks"] == 0
    assert c.stats()["free_blocks"] == 4
    c.check_invariants()


def test_allocation_fails_cleanly_when_every_block_is_referenced():
    c = cache(num_blocks=2)
    c.allocate(1, list(range(8)))  # both blocks, live
    with pytest.raises(OutOfBlocks):
        c.allocate(2, list(range(100, 108)))
    # The failed allocation must not have leaked or half-registered anything.
    assert c.n_tokens(2) == 0
    assert c.tokens(1) == list(range(8))
    c.check_invariants()


def test_write_grows_the_block_table_on_demand():
    c = cache()
    c.allocate(1, [0, 1])
    for pos in range(2, 12):
        c.write(1, pos, pos)
    assert c.tokens(1) == list(range(12))
    assert len(c.block_table(1)) == 3
    c.check_invariants()


def test_truncate_releases_only_the_blocks_past_the_cut():
    c = cache(num_blocks=8)
    c.allocate(1, list(range(16)))  # 4 blocks
    c.truncate(1, 5)  # keep blocks 0 and 1
    assert len(c.block_table(1)) == 2
    assert c.n_tokens(1) == 5
    assert c.tokens(1) == [0, 1, 2, 3, 4]
    c.check_invariants()


def test_read_outside_the_sequence_is_empty():
    c = cache()
    c.allocate(1, [7, 8, 9])
    assert c.read(1, 0) == 7
    assert c.read(1, 3) == EMPTY
    assert c.read(99, 0) == EMPTY


# --- prefix sharing ---------------------------------------------------------
def test_identical_prefix_shares_the_same_physical_blocks():
    c = cache()
    shared = list(range(8))
    c.allocate(1, shared + [50, 51, 52, 53])
    res = c.allocate(2, shared + [60, 61, 62, 63])

    assert res.n_cached_tokens == 8, "two full blocks of identical prefix should be shared"
    assert res.n_new_tokens == 4
    t1, t2 = c.block_table(1), c.block_table(2)
    assert t1[:2] == t2[:2], "prefix blocks are not the same physical blocks"
    assert t1[2] != t2[2], "divergent block must be private"
    assert c.ref_count(t1[0]) == 2
    # Sharing must not disturb what either sequence reads.
    assert c.tokens(1) == shared + [50, 51, 52, 53]
    assert c.tokens(2) == shared + [60, 61, 62, 63]
    c.check_invariants()


def test_sharing_stops_at_the_first_divergent_block():
    c = cache()
    c.allocate(1, list(range(12)))
    res = c.allocate(2, list(range(8)) + [99, 99, 99, 99])
    assert res.n_cached_tokens == 8
    c.check_invariants()


def test_a_partial_block_is_never_shared():
    """Only full blocks are immutable, so only full blocks are shareable."""
    c = cache()
    c.allocate(1, [1, 2, 3, 4, 5, 6])  # one full block + a partial
    res = c.allocate(2, [1, 2, 3, 4, 5, 6])
    assert res.n_cached_tokens == 4, "the trailing partial block must be recomputed"
    assert c.block_table(1)[1] != c.block_table(2)[1]
    c.check_invariants()


def test_hash_collision_falls_back_to_a_miss_not_to_corruption():
    c = cache()
    c.allocate(1, [1, 2, 3, 4])
    bid = c.block_table(1)[0]
    # Forge a collision: a different block claiming a resident block's hash.
    forged = c.block_hashes([9, 9, 9, 9])[0]
    c._index[forged] = bid
    res = c.allocate(2, [9, 9, 9, 9])
    assert res.n_cached_tokens == 0, "matched on hash alone; tokens were never verified"
    assert c.tokens(2) == [9, 9, 9, 9]
    c.check_invariants()


def test_share_prefix_shares_whole_blocks_and_copies_the_partial_tail():
    c = cache()
    c.allocate(1, list(range(10)))
    n = c.share_prefix(1, 2, 6)
    assert n == 6
    assert c.tokens(2) == list(range(6))
    t1, t2 = c.block_table(1), c.block_table(2)
    assert t1[0] == t2[0], "the whole block should be shared by reference"
    assert t1[1] != t2[1], "the partial tail must be a private copy"
    c.check_invariants()


def test_reallocating_a_live_sequence_releases_its_old_blocks():
    c = cache(num_blocks=4)
    c.allocate(1, list(range(8)))
    c.allocate(1, list(range(100, 108)))  # same seq, new content
    assert c.tokens(1) == list(range(100, 108))
    c.check_invariants()


# --- invalidation: eviction and copy-on-write -------------------------------
def test_referenced_blocks_are_never_evicted():
    """The invalidation rule, stated as a test: liveness is reference count."""
    c = cache(num_blocks=4)
    c.allocate(1, list(range(8)))  # 2 blocks, live
    c.allocate(2, list(range(100, 104)))  # 1 block, live
    c.free_seq(2)  # that block is now evictable
    live = list(c.block_table(1))

    c.allocate(3, list(range(200, 208)))  # needs 2 blocks: 1 free + 1 evicted
    assert set(live).isdisjoint(c.block_table(3)), "evicted a block that was still referenced"
    assert c.tokens(1) == list(range(8)), "a live sequence's content changed under it"
    assert c.stats()["evictions"] >= 1
    c.check_invariants()


def test_freed_blocks_stay_reusable_by_hash_until_actually_evicted():
    c = cache(num_blocks=8)
    toks = list(range(8))
    c.allocate(1, toks)
    blocks = list(c.block_table(1))
    c.free_seq(1)
    res = c.allocate(2, toks)  # nothing has reclaimed them yet
    assert res.n_cached_tokens == 8, "a freed-but-resident prefix should still hit"
    assert c.block_table(2) == blocks
    c.check_invariants()


def test_eviction_is_least_recently_used():
    c = cache(num_blocks=3)
    c.allocate(1, [0, 1, 2, 3])
    c.allocate(2, [4, 5, 6, 7])
    c.allocate(3, [8, 9, 10, 11])
    c.free_seq(1)
    c.free_seq(2)
    c.free_seq(3)
    c.allocate(4, [0, 1, 2, 3])  # touches seq 1's block, making it most recent
    c.allocate(5, [90, 91, 92, 93])  # must evict, and must not take seq 1's block

    assert c.allocate(6, [4, 5, 6, 7]).n_cached_tokens == 0, "expected the LRU block to be gone"
    assert c.tokens(4) == [0, 1, 2, 3]
    c.check_invariants()


def test_divergence_after_fork_triggers_copy_on_write():
    """Two sequences share a prompt, then generate different continuations.

    The shared full blocks must stay shared; the partial block they both write
    into must be split, or one sequence would see the other's tokens.
    """
    c = cache()
    prompt = [1, 2, 3, 4, 5, 6]  # one full block + a 2-token partial
    c.allocate(1, prompt)
    c.fork(1, 2)
    shared_full = c.block_table(1)[0]
    shared_partial = c.block_table(1)[1]
    assert c.block_table(2)[1] == shared_partial
    assert c.ref_count(shared_partial) == 2

    c.write(1, 6, 111)  # sequence 1 diverges
    c.write(2, 6, 222)  # sequence 2 diverges differently

    assert c.tokens(1) == prompt + [111]
    assert c.tokens(2) == prompt + [222], "copy-on-write did not isolate the writers"
    assert c.block_table(1)[0] == shared_full == c.block_table(2)[0], (
        "the immutable full block should still be shared after divergence"
    )
    assert c.block_table(1)[1] != c.block_table(2)[1]
    assert c.stats()["cow_copies"] >= 1
    c.check_invariants()


def test_cow_leaves_the_original_untouched_for_every_other_holder():
    c = cache()
    base = list(range(6))
    c.allocate(1, base)
    c.fork(1, 2)
    c.fork(1, 3)
    c.write(2, 6, 77)
    assert c.tokens(1) == base
    assert c.tokens(3) == base
    assert c.tokens(2) == base + [77]
    c.check_invariants()


def test_writing_into_a_published_block_unpublishes_it():
    """A block whose content changed must stop answering hash lookups, or a
    later sequence would be handed a prefix that no longer matches its tokens."""
    c = cache()
    c.allocate(1, [1, 2, 3, 4])
    assert c.allocate(2, [1, 2, 3, 4]).n_cached_tokens == 4
    c.free_seq(2)
    c.write(1, 0, 42)  # mutate the block in place (ref_count is 1 -> no COW)
    res = c.allocate(3, [1, 2, 3, 4])
    assert res.n_cached_tokens == 0, "stale block still served a hash hit"
    assert c.tokens(3) == [1, 2, 3, 4]
    c.check_invariants()


def test_would_fit_accounts_for_shared_blocks():
    c = cache(num_blocks=3)
    c.allocate(1, list(range(8)))  # 2 blocks live, 1 free
    assert not c.would_fit(list(range(100, 108))), "2 fresh blocks should not fit in 1"
    assert c.would_fit(list(range(8)) + [9, 9, 9, 9]), "sharing 2 blocks leaves room for 1"
    c.check_invariants()


# --- stress ------------------------------------------------------------------
def test_invariants_hold_under_a_randomised_workload():
    """The failure mode this catches is a refcount that drifts from the tables
    it summarises — which surfaces much later as a leak or as a live block being
    handed out twice."""
    import random

    rng = random.Random(1234)
    c = cache(num_blocks=24, block_size=4)
    shared_prefixes = [list(range(k * 100, k * 100 + 8)) for k in range(4)]
    live: dict[int, list[int]] = {}

    for step in range(600):
        seq = rng.randrange(50)
        action = rng.random()
        if seq in live and action < 0.35:
            c.free_seq(seq)
            live.pop(seq)
        elif seq in live and action < 0.6:
            toks = live[seq]
            try:
                c.write(seq, len(toks), rng.randrange(1000))
            except OutOfBlocks:
                continue  # a full pool must refuse, not corrupt
            toks.append(c.read(seq, len(toks)))
        elif seq not in live:
            toks = list(rng.choice(shared_prefixes)) + [rng.randrange(1000) for _ in range(rng.randrange(6))]
            try:
                c.allocate(seq, toks)
            except OutOfBlocks:
                continue
            live[seq] = toks
        if step % 25 == 0:
            c.check_invariants()
            for s, toks in live.items():
                assert c.tokens(s) == toks, f"seq {s} read back wrong content at step {step}"

    c.check_invariants()
    for s, toks in live.items():
        assert c.tokens(s) == toks
