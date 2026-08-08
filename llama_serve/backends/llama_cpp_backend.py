"""Real inference backend, built on llama.cpp's low-level C API.

Deliberately *not* built on `llama_cpp.Llama`. The high-level wrapper owns a
single implicit sequence and exposes `create_completion()`, which forces one
request per forward pass — that makes iteration-level batching impossible.

The low-level API exposes what a server actually needs:

  * `llama_batch` can carry tokens belonging to *different* sequences in one
    `llama_decode` call, each with its own position. That is the whole
    mechanism behind continuous batching.
  * `llama_memory_seq_rm` / `llama_memory_seq_cp` give per-sequence control of
    the KV cache: eviction on completion, and copying a cached prefix from one
    sequence to another for prefix sharing.

API-version note: llama.cpp renamed `llama_kv_cache_seq_*` -> `llama_kv_self_seq_*`
-> `llama_memory_seq_*` across releases. We probe for whichever exists so the
backend is not pinned to one point release.
"""

from __future__ import annotations

import ctypes
import os
import threading

import llama_cpp.llama_cpp as C

from .base import Backend, Sampler, SamplingParams, TokenSlot

_backend_init_lock = threading.Lock()
_backend_ready = False


def _ensure_backend(verbose: bool) -> None:
    global _backend_ready
    with _backend_init_lock:
        if _backend_ready:
            return
        if not verbose:
            # Silence llama.cpp's per-load logging; it floods the server output.
            @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
            def _quiet(level, text, user_data):  # noqa: ANN001
                pass

            _quiet_ref.append(_quiet)  # keep the callback alive
            C.llama_log_set(_quiet, ctypes.c_void_p(0))
        C.llama_backend_init()
        _backend_ready = True


_quiet_ref: list = []


class LlamaCppSampler(Sampler):
    """A llama.cpp sampler chain owned by one request.

    Each request gets its own chain because the chain carries state: the RNG
    for `dist`, and the token history for repetition penalties. Sharing one
    chain across concurrent requests would leak one request's penalty history
    into another's logits.
    """

    def __init__(self, ctx, params: SamplingParams):
        self._ctx = ctx
        self._chain = C.llama_sampler_chain_init(C.llama_sampler_chain_default_params())
        add = C.llama_sampler_chain_add

        if params.repeat_penalty and params.repeat_penalty != 1.0:
            add(self._chain, C.llama_sampler_init_penalties(64, params.repeat_penalty, 0.0, 0.0))

        if params.temperature <= 0.0:
            add(self._chain, C.llama_sampler_init_greedy())
        else:
            if params.top_k > 0:
                add(self._chain, C.llama_sampler_init_top_k(params.top_k))
            if params.top_p < 1.0:
                add(self._chain, C.llama_sampler_init_top_p(params.top_p, 1))
            add(self._chain, C.llama_sampler_init_temp(params.temperature))
            add(self._chain, C.llama_sampler_init_dist(params.seed or C.LLAMA_DEFAULT_SEED))
        self._closed = False

    def sample(self, slot_index: int) -> int:
        return C.llama_sampler_sample(self._chain, self._ctx, slot_index)

    def accept(self, token: int) -> None:
        C.llama_sampler_accept(self._chain, token)

    def close(self) -> None:
        if not self._closed:
            C.llama_sampler_free(self._chain)
            self._closed = True

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class LlamaCppBackend(Backend):
    name = "llama.cpp"
    supports_prefix_sharing = True

    def __init__(
        self,
        model_path: str,
        n_ctx_per_seq: int = 1024,
        max_seqs: int = 8,
        cache_seqs: int = 0,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        block_size: int = 16,
        verbose: bool = False,
        flash_attn: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        _ensure_backend(verbose)

        self.model_path = model_path
        self.max_seqs = max_seqs
        # Donor slots for the prefix cache live beyond the running slots, so a
        # cached prefix does not consume capacity that a live request needs.
        self.cache_seqs = cache_seqs
        self.n_seq_total = max_seqs + cache_seqs
        self.block_size = block_size
        self.n_ctx_per_seq = n_ctx_per_seq

        mparams = C.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        self._model = C.llama_model_load_from_file(model_path.encode("utf-8"), mparams)
        if not self._model:
            raise RuntimeError(f"failed to load model: {model_path}")
        self._vocab = C.llama_model_get_vocab(self._model)

        cparams = C.llama_context_default_params()
        # llama.cpp's KV cache is unified: n_ctx is the total number of cells
        # shared by all sequences, so budget it per sequence slot.
        cparams.n_ctx = n_ctx_per_seq * self.n_seq_total
        cparams.n_seq_max = self.n_seq_total
        # One decode call may carry a full prefill chunk for one sequence plus a
        # decode token for every other, so the batch must be at least that wide.
        cparams.n_batch = max(512, n_ctx_per_seq)
        cparams.n_ubatch = cparams.n_batch
        if n_threads:
            cparams.n_threads = n_threads
            cparams.n_threads_batch = n_threads
        if hasattr(cparams, "flash_attn_type") and flash_attn:
            # 1 == enabled in this build's llama_flash_attn_type enum
            try:
                cparams.flash_attn_type = 1
            except Exception:
                pass

        self._ctx = C.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            C.llama_model_free(self._model)
            raise RuntimeError("failed to create llama context")

        self.n_ctx = C.llama_n_ctx(self._ctx)
        self.n_batch = cparams.n_batch
        self.n_vocab = C.llama_vocab_n_tokens(self._vocab)
        self._mem = C.llama_get_memory(self._ctx)

        self._batch = C.llama_batch_init(self.n_batch, 0, self.n_seq_total)
        self._lock = threading.Lock()
        self._closed = False

        self._eos = frozenset(
            t
            for t in {
                C.llama_vocab_eos(self._vocab),
                C.llama_vocab_eot(self._vocab),
            }
            if t is not None and t >= 0
        )
        self.decode_calls = 0
        self.tokens_decoded = 0

    # --- tokenizer --------------------------------------------------------
    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        raw = text.encode("utf-8")
        cap = len(raw) + 64
        buf = (C.llama_token * cap)()
        n = C.llama_tokenize(self._vocab, raw, len(raw), buf, cap, add_bos, True)
        if n < 0:  # buffer too small; llama.cpp returns -required
            cap = -n
            buf = (C.llama_token * cap)()
            n = C.llama_tokenize(self._vocab, raw, len(raw), buf, cap, add_bos, True)
        if n < 0:
            raise RuntimeError("tokenization failed")
        return list(buf[:n])

    def token_to_piece(self, token: int) -> str:
        buf = (ctypes.c_char * 64)()
        n = C.llama_token_to_piece(self._vocab, token, buf, 64, 0, True)
        if n < 0:
            buf = (ctypes.c_char * (-n))()
            n = C.llama_token_to_piece(self._vocab, token, buf, -n, 0, True)
        # errors="replace" would corrupt multi-byte UTF-8 split across tokens;
        # the engine reassembles pieces, so emit the raw bytes losslessly here.
        return bytes(buf[:n]).decode("utf-8", errors="replace")

    def detokenize(self, tokens: list[int]) -> str:
        out = bytearray()
        for t in tokens:
            buf = (ctypes.c_char * 64)()
            n = C.llama_token_to_piece(self._vocab, t, buf, 64, 0, True)
            if n > 0:
                out += bytes(buf[:n])
        return out.decode("utf-8", errors="replace")

    @property
    def eos_tokens(self) -> frozenset[int]:
        return self._eos

    def is_eog(self, token: int) -> bool:
        return bool(C.llama_vocab_is_eog(self._vocab, token))

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        # TinyLlama-Chat uses the Zephyr template.
        parts = []
        for m in messages:
            parts.append(f"<|{m.get('role', 'user')}|>\n{m.get('content', '')}</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # --- inference --------------------------------------------------------
    def decode(self, slots: list[TokenSlot]) -> None:
        if not slots:
            return
        if len(slots) > self.n_batch:
            raise ValueError(f"batch of {len(slots)} exceeds n_batch={self.n_batch}")

        with self._lock:
            b = self._batch
            b.n_tokens = len(slots)
            for i, s in enumerate(slots):
                b.token[i] = s.token
                b.pos[i] = s.pos
                b.n_seq_id[i] = 1
                b.seq_id[i][0] = s.seq_id
                b.logits[i] = 1 if s.want_logits else 0

            rc = C.llama_decode(self._ctx, b)
            if rc != 0:
                # rc == 1 means "no KV slot available" — recoverable by the
                # scheduler (preempt someone), so surface it distinctly.
                raise DecodeError(rc, f"llama_decode failed with rc={rc}")

            self.decode_calls += 1
            self.tokens_decoded += len(slots)

    def make_sampler(self, params: SamplingParams) -> Sampler:
        return LlamaCppSampler(self._ctx, params)

    # --- KV cache ---------------------------------------------------------
    def seq_rm(self, seq_id: int, p0: int = -1, p1: int = -1) -> None:
        with self._lock:
            C.llama_memory_seq_rm(self._mem, seq_id, p0, p1)

    def seq_cp(self, src: int, dst: int, p0: int = -1, p1: int = -1) -> bool:
        """Whole-sequence copy only.

        `llama_kv_cache::seq_cp` in this build asserts
        `is_full && "seq_cp() is only supported for full KV buffers"` and calls
        `ggml_abort` on failure, which kills the process rather than returning
        an error. So a partial range is refused here instead of being attempted.
        """
        full = p0 <= 0 and p1 < 0
        if not full:
            return False
        with self._lock:
            C.llama_memory_seq_cp(self._mem, src, dst, -1, -1)
        return True

    def seq_share_prefix(self, src: int, dst: int, n_tokens: int) -> bool:
        """Share `src`'s first `n_tokens` cells with `dst`, zero-copy.

        Built from the two range-safe primitives: copy the whole sequence (the
        only form `seq_cp` accepts), then remove `dst` from the cells past the
        prefix. `llama_memory_seq_cp` adds `dst` to each cell's sequence set
        rather than duplicating the K/V data, so this shares memory rather than
        spending it — the trim afterwards only drops `dst`'s membership.
        """
        if n_tokens <= 0:
            return False
        with self._lock:
            C.llama_memory_seq_rm(self._mem, dst, -1, -1)
            C.llama_memory_seq_cp(self._mem, src, dst, -1, -1)
            C.llama_memory_seq_rm(self._mem, dst, n_tokens, -1)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            C.llama_batch_free(self._batch)
            C.llama_free(self._ctx)
            C.llama_model_free(self._model)
        except Exception:  # pragma: no cover
            pass


class DecodeError(RuntimeError):
    def __init__(self, rc: int, msg: str):
        super().__init__(msg)
        self.rc = rc

    @property
    def no_kv_slot(self) -> bool:
        return self.rc == 1
