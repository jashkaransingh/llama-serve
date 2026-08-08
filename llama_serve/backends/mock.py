"""Deterministic mock backend.

Exists so the serving architecture can be exercised (and unit-tested) without
loading a real model: CI has no GPU, and tests should not take 30s of prefill
to assert that the scheduler evicts the right sequence.

It is a real implementation of the `Backend` contract, not a stub:
  * a byte-level tokenizer (reversible, so round-trip assertions are meaningful)
  * a KV cache that is *actually paged*: storage is a `PagedKVCache`, so every
    read and write goes through a block table, and `seq_cp` shares physical
    blocks by reference exactly as llama.cpp's unified cache does. Prefix
    sharing, copy-on-write and eviction are therefore exercised against real
    paged storage in tests, not against a dict pretending to be one.
  * a deterministic next-token rule (hash of the last two tokens) so generated
    output is stable across runs and comparable between milestones
  * an optional simulated per-token latency, so load tests of the scheduler
    can be run without a model

Because the next token is a pure function of the KV contents at the two most
recent positions, a bug in the paged allocator — a block shared when it should
have been copied, a stale block reused after eviction — changes the generated
text. That is deliberate: it makes prefix-sharing correctness observable end to
end, as an output-equality assertion rather than an internal counter.
"""

from __future__ import annotations

import time

from ..engine.paged_kv import EMPTY, PagedKVCache
from .base import Backend, Sampler, SamplingParams, TokenSlot

_BOS = 1
_EOS = 2
_OFFSET = 3  # tokens 0..2 reserved for control, byte b -> b + _OFFSET


class MockSampler(Sampler):
    def __init__(self, backend: MockBackend, params: SamplingParams):
        self._backend = backend
        self._params = params
        self._state = (params.seed or 0) & 0xFFFFFFFF

    def sample(self, slot_index: int) -> int:
        return self._backend._logits[slot_index]

    def accept(self, token: int) -> None:
        self._state = (self._state * 31 + token) & 0xFFFFFFFF


class MockBackend(Backend):
    name = "mock"
    supports_prefix_sharing = True

    def __init__(
        self,
        n_ctx: int = 4096,
        max_seqs: int = 8,
        cache_seqs: int = 0,
        block_size: int = 16,
        token_latency_s: float = 0.0,
        eos_after: int = 0,
    ):
        self.n_ctx = n_ctx
        self.n_vocab = 259
        self.max_seqs = max_seqs
        self.cache_seqs = cache_seqs
        self.n_seq_total = max_seqs + cache_seqs
        self.n_batch = 512
        self.block_size = block_size
        self.token_latency_s = token_latency_s
        self.eos_after = eos_after  # force EOS after N generated tokens (0 = never)
        # The KV cache. n_ctx cells, paged into fixed-size blocks.
        self.pages = PagedKVCache(
            num_blocks=max(1, n_ctx // max(1, block_size)), block_size=max(1, block_size)
        )
        self._logits: list[int] = []
        self.decode_calls = 0
        self.tokens_decoded = 0

    # --- tokenizer --------------------------------------------------------
    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        toks = [b + _OFFSET for b in text.encode("utf-8")]
        return [_BOS] + toks if add_bos else toks

    def detokenize(self, tokens: list[int]) -> str:
        raw = bytes(t - _OFFSET for t in tokens if t >= _OFFSET)
        return raw.decode("utf-8", errors="replace")

    def token_to_piece(self, token: int) -> str:
        if token < _OFFSET:
            return ""
        return bytes([token - _OFFSET]).decode("utf-8", errors="replace")

    @property
    def eos_tokens(self) -> frozenset[int]:
        return frozenset({_EOS})

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        return "".join(f"{m.get('role','user')}: {m.get('content','')}\n" for m in messages)

    # --- inference --------------------------------------------------------
    def decode(self, slots: list[TokenSlot]) -> None:
        if not slots:
            return
        self.decode_calls += 1
        self.tokens_decoded += len(slots)
        if self.token_latency_s:
            time.sleep(self.token_latency_s)

        self._logits = [0] * len(slots)
        for i, s in enumerate(slots):
            # Writing through the block table is what makes the paging real:
            # if this position falls in a block shared with another sequence,
            # the allocator copies it first.
            self.pages.write(s.seq_id, s.pos, s.token)
            if s.want_logits:
                self._logits[i] = self._next_token(s.seq_id, s.pos)

    def _next_token(self, seq_id: int, pos: int) -> int:
        if self.eos_after and pos >= self.eos_after:
            return _EOS
        prev = self.pages.read(seq_id, pos)
        prev2 = self.pages.read(seq_id, pos - 1)
        prev = _BOS if prev == EMPTY else prev
        prev2 = _BOS if prev2 == EMPTY else prev2
        # Deterministic printable ASCII in [32, 126] -> stable, readable output.
        h = (prev * 1103515245 + prev2 * 12345 + pos) & 0x7FFFFFFF
        return _OFFSET + 32 + (h % 95)

    def make_sampler(self, params: SamplingParams) -> Sampler:
        return MockSampler(self, params)

    # --- KV cache ---------------------------------------------------------
    def seq_rm(self, seq_id: int, p0: int = -1, p1: int = -1) -> None:
        """Drop KV for `seq_id`. Matches llama.cpp's position-range semantics.

        Only the two forms the engines actually use are supported: drop
        everything (`-1, -1`), or drop a suffix (`p0, -1`). A hole punched in
        the middle of a sequence has no meaning for a block table, and no
        caller wants one.
        """
        if p0 < 0 and p1 < 0:
            self.pages.free_seq(seq_id)
            return
        if p1 < 0:
            self.pages.truncate(seq_id, max(0, p0))
            return
        raise NotImplementedError("mid-sequence KV removal is not supported by the block table")

    def seq_cp(self, src: int, dst: int, p0: int = -1, p1: int = -1) -> bool:
        """Share `src`'s cached prefix with `dst`.

        Whole blocks are shared by reference — no cell is copied — which is the
        same zero-copy behaviour llama.cpp's unified cache gives via
        `llama_memory_seq_cp`.
        """
        if self.pages.n_tokens(src) == 0 or p0 > 0:
            return False  # only prefix sharing is meaningful for a block table
        n = self.pages.n_tokens(src) if p1 < 0 else p1
        return self.pages.share_prefix(src, dst, n) > 0

    def seq_share_prefix(self, src: int, dst: int, n_tokens: int) -> bool:
        if n_tokens <= 0 or self.pages.n_tokens(src) < n_tokens:
            return False
        return self.pages.share_prefix(src, dst, n_tokens) > 0

    # --- test helpers -----------------------------------------------------
    def kv_len(self, seq_id: int) -> int:
        return self.pages.n_tokens(seq_id)

    def seed_kv(self, seq_id: int, tokens: list[int]) -> None:
        """Populate a sequence's cache directly, as a prefill would have."""
        for i, t in enumerate(tokens):
            self.pages.write(seq_id, i, t)
