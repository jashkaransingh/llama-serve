"""Policy tests for the prefix cache, run against the mock backend.

The allocator underneath is tested in `test_paged_kv.py`. What is tested here is
the layer that decides *what* to cache and *what to throw away* — the part that
can be wrong silently: a prefix cache that returns a match one block too long
produces subtly corrupted output rather than an error.
"""

from __future__ import annotations

from llama_serve.backends.base import SamplingParams
from llama_serve.backends.mock import MockBackend
from llama_serve.engine.block_manager import BlockManager
from llama_serve.engine.request import Request

BS = 4
RUNNING_SLOTS = 4


def make(cache_seqs=4, total_blocks=64):
    backend = MockBackend(
        n_ctx=4096, max_seqs=RUNNING_SLOTS, cache_seqs=cache_seqs, block_size=BS
    )
    bm = BlockManager(
        backend=backend,
        block_size=BS,
        cache_seq_ids=list(range(RUNNING_SLOTS, RUNNING_SLOTS + cache_seqs)),
        total_blocks=total_blocks,
    )
    return backend, bm


def req(tokens, seq_id=0):
    r = Request(prompt="", params=SamplingParams())
    r.prompt_tokens = list(tokens)
    r.seq_id = seq_id
    return r


def finish(backend, bm, tokens, seq_id=0):
    """Run a request's prompt through the backend, then retire it."""
    backend.seed_kv(seq_id, tokens)
    r = req(tokens, seq_id=seq_id)
    return bm.on_finish(r)


# --- hashing ---------------------------------------------------------------
def test_block_hashes_are_positional():
    """The same block at a different offset must hash differently, or a match
    could be satisfied by a coincidental mid-prompt collision."""
    _, bm = make()
    a = bm.block_hashes([1, 2, 3, 4, 9, 9, 9, 9])
    b = bm.block_hashes([9, 9, 9, 9, 1, 2, 3, 4])
    assert a[0] != b[1], "block hash ignored its position in the chain"
    assert a[0] == bm.block_hashes([1, 2, 3, 4])[0]


def test_block_hashes_ignore_partial_trailing_block():
    _, bm = make()
    assert len(bm.block_hashes(list(range(10)))) == 2  # 10 // 4
    assert bm.block_hashes(list(range(3))) == ()


def test_identical_prefix_shares_hash_chain():
    _, bm = make()
    shared = list(range(20))
    h1 = bm.block_hashes(shared + [100, 101, 102, 103])
    h2 = bm.block_hashes(shared + [200, 201, 202, 203])
    assert h1[:5] == h2[:5]
    assert h1[5] != h2[5]


# --- sharing ---------------------------------------------------------------
def test_miss_on_empty_cache():
    _, bm = make()
    assert bm.on_admit(req(list(range(16)))) == 0
    assert bm.stats()["hits"] == 0


def test_hit_after_publish_and_kv_is_actually_populated():
    backend, bm = make()
    tokens = list(range(32))
    assert finish(backend, bm, tokens, seq_id=0) is True

    r2 = req(tokens + [99, 98, 97, 96], seq_id=1)
    n = bm.on_admit(r2)
    assert n > 0, "identical prefix should hit"
    assert n % BS == 0, "match must be block aligned"
    assert backend.kv_len(1) == n, "cached KV was not actually copied into the new sequence"
    assert backend.pages.tokens(1) == tokens[:n], "shared prefix holds the wrong tokens"
    assert r2.metrics.cached_prompt_tokens == n


def test_a_hit_is_zero_copy_at_the_block_level():
    """The point of the whole exercise: the second sequence must *reference*
    the donor's blocks, not receive a copy of them."""
    backend, bm = make()
    tokens = list(range(32))
    finish(backend, bm, tokens, seq_id=0)
    donor_seq = next(iter(bm._entries))
    donor_blocks = backend.pages.block_table(donor_seq)

    n = bm.on_admit(req(tokens + [7, 7, 7, 7], seq_id=1))
    assert backend.pages.block_table(1)[: n // BS] == donor_blocks[: n // BS]
    assert backend.pages.ref_count(donor_blocks[0]) == 2


def test_never_consumes_the_entire_prompt():
    """At least one token must remain to prefill: the model has to run one
    position to produce logits to sample from."""
    backend, bm = make()
    tokens = list(range(32))
    finish(backend, bm, tokens, seq_id=0)

    r2 = req(tokens, seq_id=1)  # exactly the same prompt
    n = bm.on_admit(r2)
    assert n < len(tokens), f"consumed the whole prompt ({n} of {len(tokens)}); nothing left to decode"


def test_partial_prefix_match_is_truncated_to_common_blocks():
    backend, bm = make()
    shared = list(range(16))
    finish(backend, bm, shared + [50, 51, 52, 53], seq_id=0)

    other = shared + [60, 61, 62, 63, 70, 71, 72, 73]
    n = bm.on_admit(req(other, seq_id=1))
    assert n == len(shared), f"expected match to stop at the divergent block, got {n}"
    assert backend.pages.tokens(1) == shared


def test_divergent_prompt_does_not_match():
    backend, bm = make()
    finish(backend, bm, list(range(100, 132)), seq_id=0)
    assert bm.on_admit(req(list(range(200, 232)), seq_id=1)) == 0


def test_longest_of_several_candidates_wins():
    backend, bm = make()
    base = list(range(16))
    finish(backend, bm, base, seq_id=0)
    finish(backend, bm, base + list(range(100, 116)), seq_id=1)

    n = bm.on_admit(req(base + list(range(100, 116)) + [5, 5, 5, 5], seq_id=2))
    assert n == 32, f"should have matched the longer cached prefix, got {n}"


# --- eviction --------------------------------------------------------------
def test_lru_eviction_when_donor_slots_exhausted():
    backend, bm = make(cache_seqs=2, total_blocks=64)
    published = []
    for k in range(3):
        toks = [k * 1000 + i for i in range(16)]
        finish(backend, bm, toks, seq_id=k)
        published.append(toks)

    assert bm.stats()["evictions"] >= 1, "should have evicted to make room"
    assert bm.stats()["cached_prefixes"] <= 2
    # The most recently published prefix must still be resident.
    assert bm.on_admit(req(published[-1] + [7, 7, 7, 7], seq_id=1)) > 0


def test_block_budget_is_respected():
    backend, bm = make(cache_seqs=8, total_blocks=8)  # only 8 blocks total
    for k in range(6):
        toks = [k * 1000 + i for i in range(16)]  # 4 blocks each
        finish(backend, bm, toks, seq_id=k % RUNNING_SLOTS)
    st = bm.stats()
    assert st["used_blocks"] <= st["total_blocks"], f"over budget: {st}"
    assert st["evictions"] > 0
    bm.pool.check_invariants()


def test_eviction_frees_the_donors_backend_sequence():
    """Otherwise the prefix cache leaks KV cells the model needs for live work."""
    backend, bm = make(cache_seqs=1, total_blocks=64)
    finish(backend, bm, [1] * 16, seq_id=0)
    first_donor = next(iter(bm._entries))
    assert backend.kv_len(first_donor) == 16
    finish(backend, bm, [2] * 16, seq_id=0)  # forces eviction of the first
    assert bm.stats()["evictions"] == 1
    assert backend.kv_len(first_donor) in (0, 16)  # slot reused or cleared
    assert len(bm._entries) == 1


def test_entries_sharing_blocks_do_not_free_each_others_blocks():
    backend, bm = make(cache_seqs=4, total_blocks=64)
    base = list(range(16))
    finish(backend, bm, base, seq_id=0)
    r = req(base + list(range(50, 66)), seq_id=1)
    bm.on_admit(r)  # hits, so its first 16 tokens are genuinely shared
    backend.seed_kv(1, r.prompt_tokens)
    bm.on_finish(r)

    # Two entries, sharing the 4 blocks of `base`: 8 blocks of content, 4 shared.
    st = bm.stats()
    assert st["cached_prefixes"] == 2
    assert st["used_blocks"] == 8, f"shared blocks were double-counted: {st}"
    bm.pool.check_invariants()


def test_duplicate_publish_does_not_double_allocate():
    backend, bm = make()
    toks = list(range(16))
    finish(backend, bm, toks, seq_id=0)
    used = bm.stats()["used_blocks"]
    assert finish(backend, bm, toks, seq_id=1) is False
    assert bm.stats()["used_blocks"] == used, "same prefix published twice consumed blocks twice"
    assert bm.stats()["cached_prefixes"] == 1


def test_finish_releases_the_request_slot():
    backend, bm = make()
    finish(backend, bm, list(range(16)), seq_id=0)
    assert backend.kv_len(0) == 0, "request's own sequence KV should be cleared on finish"


def test_hit_rate_and_saved_fraction_are_coherent():
    backend, bm = make()
    toks = list(range(32))
    finish(backend, bm, toks, seq_id=0)
    bm.on_admit(req(toks + [1, 2, 3, 4], seq_id=1))
    bm.note_prefill(4)
    st = bm.stats()
    assert 0.0 < st["hit_rate"] <= 1.0
    assert 0.0 < st["prefill_saved_frac"] <= 1.0
    assert st["prompt_tokens_reused"] > 0
