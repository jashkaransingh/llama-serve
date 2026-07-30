"""Deterministic mock backend.

Exists so the serving architecture can be exercised (and unit-tested) without
loading a real model: CI has no GPU, and tests should not take 30s of prefill
to assert that the scheduler evicts the right sequence.

It is a real implementation of the `Backend` contract, not a stub:
  * a byte-level tokenizer (reversible, so round-trip assertions are meaningful)
  * a per-sequence KV cache dict that honours `seq_rm` / `seq_cp` semantics,
    so paged-allocator and prefix-sharing logic is genuinely tested
  * a deterministic next-token rule (hash of the last two tokens) so generated
    output is stable across runs and comparable between milestones
  * an optional simulated per-token latency, so load tests of the scheduler
    can be run without a model
"""

from __future__ import annotations

import time

from .base import Backend, Sampler, SamplingParams, TokenSlot

_BOS = 1
_EOS = 2
_OFFSET = 3  # tokens 0..2 reserved for control, byte b -> b + _OFFSET


class MockSampler(Sampler):
    def __init__(self, backend: "MockBackend", params: SamplingParams):
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
        block_size: int = 16,
        token_latency_s: float = 0.0,
        eos_after: int = 0,
    ):
        self.n_ctx = n_ctx
        self.n_vocab = 259
        self.max_seqs = max_seqs
        self.n_batch = 512
        self.block_size = block_size
        self.token_latency_s = token_latency_s
        self.eos_after = eos_after  # force EOS after N generated tokens (0 = never)
        # seq_id -> {pos: token}. Mirrors llama.cpp's per-sequence KV cache.
        self._kv: dict[int, dict[int, int]] = {}
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
            cache = self._kv.setdefault(s.seq_id, {})
            cache[s.pos] = s.token
            if s.want_logits:
                self._logits[i] = self._next_token(cache, s.pos)

    def _next_token(self, cache: dict[int, int], pos: int) -> int:
        if self.eos_after and pos >= self.eos_after:
            return _EOS
        prev = cache.get(pos, _BOS)
        prev2 = cache.get(pos - 1, _BOS)
        # Deterministic printable ASCII in [32, 126] -> stable, readable output.
        h = (prev * 1103515245 + prev2 * 12345 + pos) & 0x7FFFFFFF
        return _OFFSET + 32 + (h % 95)

    def make_sampler(self, params: SamplingParams) -> Sampler:
        return MockSampler(self, params)

    # --- KV cache ---------------------------------------------------------
    def seq_rm(self, seq_id: int, p0: int = -1, p1: int = -1) -> None:
        cache = self._kv.get(seq_id)
        if cache is None:
            return
        if p0 < 0 and p1 < 0:
            self._kv.pop(seq_id, None)
            return
        lo = 0 if p0 < 0 else p0
        hi = 1 << 62 if p1 < 0 else p1
        for pos in [p for p in cache if lo <= p < hi]:
            del cache[pos]

    def seq_cp(self, src: int, dst: int, p0: int = -1, p1: int = -1) -> bool:
        source = self._kv.get(src)
        if source is None:
            return False
        lo = 0 if p0 < 0 else p0
        hi = 1 << 62 if p1 < 0 else p1
        target = self._kv.setdefault(dst, {})
        for pos, tok in source.items():
            if lo <= pos < hi:
                target[pos] = tok
        return True

    # --- test helpers -----------------------------------------------------
    def kv_len(self, seq_id: int) -> int:
        return len(self._kv.get(seq_id, {}))
