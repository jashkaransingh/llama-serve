"""Model backend interface.

The serving stack (scheduler, block manager, continuous-batching engine) is
written entirely against this interface so it can run on top of real llama.cpp
inference or a deterministic mock used by the test suite.

The interface is deliberately shaped like llama.cpp's low-level C API rather
than like a `generate(prompt) -> str` helper, because iteration-level batching
needs to drive the model one *decode step* at a time across many sequences:

    slots = [TokenSlot(seq_id=0, token=15043, pos=0, want_logits=False), ...]
    backend.decode(slots)
    tok = backend.sample(slot_index, sampler)

`seq_id` indexes a KV-cache sequence slot inside the model context. The engine
owns the mapping from user requests to seq_ids; the backend only has to honour
the per-sequence KV operations (`seq_rm`, `seq_cp`).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(slots=True)
class TokenSlot:
    """One (token, position, sequence) entry inside a single batched decode."""

    seq_id: int
    token: int
    pos: int
    want_logits: bool = False


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 0
    max_tokens: int = 128
    stop: tuple[str, ...] = ()
    # Benchmarking knob: forces an exact output length so measurements compare
    # like with like. Without it a model that emits EOS early makes a "250
    # token" load test secretly a 30-token one.
    ignore_eos: bool = False

    def normalized(self) -> SamplingParams:
        """Clamp values into ranges llama.cpp accepts."""
        return SamplingParams(
            temperature=max(0.0, float(self.temperature)),
            top_p=min(1.0, max(0.0, float(self.top_p))),
            top_k=int(self.top_k) if self.top_k and self.top_k > 0 else 0,
            repeat_penalty=max(0.0, float(self.repeat_penalty)),
            seed=int(self.seed),
            max_tokens=max(1, int(self.max_tokens)),
            stop=tuple(self.stop or ()),
            ignore_eos=bool(self.ignore_eos),
        )


class Sampler(abc.ABC):
    """Per-request sampling state (holds RNG + penalty history)."""

    @abc.abstractmethod
    def sample(self, slot_index: int) -> int:
        """Sample a token from the logits produced at `slot_index` of the last decode."""

    @abc.abstractmethod
    def accept(self, token: int) -> None:
        """Feed a chosen token back into the sampler (for repetition penalties)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class Backend(abc.ABC):
    """A loaded model that can be driven one batched decode step at a time."""

    # --- capability / shape metadata -------------------------------------
    name: str = "backend"
    n_ctx: int = 2048
    n_vocab: int = 32000
    max_seqs: int = 1
    n_batch: int = 512
    block_size: int = 16
    supports_prefix_sharing: bool = False

    # --- tokenizer --------------------------------------------------------
    @abc.abstractmethod
    def tokenize(self, text: str, add_bos: bool = True) -> list[int]: ...

    @abc.abstractmethod
    def detokenize(self, tokens: list[int]) -> str: ...

    @abc.abstractmethod
    def token_to_piece(self, token: int) -> str: ...

    @property
    @abc.abstractmethod
    def eos_tokens(self) -> frozenset[int]: ...

    # --- prompt formatting ------------------------------------------------
    def format_chat(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages with the model's prompt template."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"<|{role}|>\n{m.get('content', '')}</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # --- inference --------------------------------------------------------
    @abc.abstractmethod
    def decode(self, slots: list[TokenSlot]) -> None:
        """Run one forward pass over `slots`, which may mix many sequences."""

    @abc.abstractmethod
    def make_sampler(self, params: SamplingParams) -> Sampler: ...

    # --- KV cache ---------------------------------------------------------
    @abc.abstractmethod
    def seq_rm(self, seq_id: int, p0: int = -1, p1: int = -1) -> None:
        """Drop KV entries for `seq_id` in position range [p0, p1)."""

    def seq_cp(self, src: int, dst: int, p0: int = -1, p1: int = -1) -> bool:
        """Copy cached KV from one sequence to another. Returns False if unsupported."""
        return False

    def seq_share_prefix(self, src: int, dst: int, n_tokens: int) -> bool:
        """Give `dst` the first `n_tokens` positions of `src`'s KV cache.

        This is the one KV operation the prefix cache needs, and it is a
        separate method rather than a `seq_cp(src, dst, 0, n)` call because
        backends differ in what ranges they will accept. llama.cpp 0.3.34
        asserts `seq_cp() is only supported for full KV buffers` — a partial
        range aborts the process — so its implementation copies the whole
        sequence and then trims the destination. Hiding that behind a named
        operation keeps the workaround where it belongs, in the backend.

        Returns False if the backend cannot share.
        """
        return False

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
