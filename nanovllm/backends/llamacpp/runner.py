from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from nanovllm.backends.base import BackendExecutionPlan
from nanovllm.config import Config


class _NanoLlamaBackendParams(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("n_ctx", ctypes.c_int32),
        ("n_batch", ctypes.c_int32),
        ("n_ubatch", ctypes.c_int32),
        ("n_seq_max", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("n_gpu_layers", ctypes.c_int32),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
    ]


class _NanoLlamaKVPlan(ctypes.Structure):
    _fields_ = [
        ("is_prefill", ctypes.c_bool),
        ("n_tokens", ctypes.c_int32),
        ("n_seqs", ctypes.c_int32),
        ("block_size", ctypes.c_int32),
        ("block_table_cols", ctypes.c_int32),
        ("tokens", ctypes.POINTER(ctypes.c_int32)),
        ("positions", ctypes.POINTER(ctypes.c_int32)),
        ("token_seq_ids", ctypes.POINTER(ctypes.c_int32)),
        ("seq_ids", ctypes.POINTER(ctypes.c_int32)),
        ("scheduled_token_counts", ctypes.POINTER(ctypes.c_int32)),
        ("slot_mapping", ctypes.POINTER(ctypes.c_int32)),
        ("block_tables", ctypes.POINTER(ctypes.c_int32)),
        ("context_lens", ctypes.POINTER(ctypes.c_int32)),
        ("num_cached_tokens", ctypes.POINTER(ctypes.c_int32)),
    ]


class LlamaCppRunner:
    create_symbol = "nano_llama_backend_create"
    destroy_symbol = "nano_llama_backend_destroy"
    run_symbol = "nano_llama_backend_run"
    release_blocks_symbol = "nano_llama_backend_release_blocks"
    shared_kv_bytes_symbol: str | None = None

    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.lib = self._load_library(config)
        self._bind_functions()

        device_config = config.device_config or {}
        backend_id = 1 if config.backend in ("llamacpp_vulkan", "llamacpp_pd") else 0
        default_ubatch = min(config.max_num_batched_tokens, 512) if backend_id == 1 else config.max_num_batched_tokens
        n_ubatch = device_config.get("n_ubatch") or default_ubatch
        params = _NanoLlamaBackendParams(
            model_path=os.fsencode(config.gguf_model),
            n_ctx=int(config.max_model_len),
            n_batch=int(config.max_num_batched_tokens),
            n_ubatch=int(n_ubatch),
            n_seq_max=int(config.max_num_seqs),
            n_threads=int(device_config.get("n_threads", max(os.cpu_count() or 1, 1))),
            n_threads_batch=int(device_config.get("n_threads_batch", max(os.cpu_count() or 1, 1))),
            n_gpu_layers=int(device_config.get("n_gpu_layers", -1 if backend_id == 1 else 0)),
            offload_kqv=bool(device_config.get("offload_kqv", backend_id == 1)),
            flash_attn=bool(device_config.get("flash_attn", False)),
        )
        self.ctx = self._create(ctypes.byref(params))
        if not self.ctx:
            raise RuntimeError("failed to create nano llama.cpp backend context")
        self.vocab_size = int(self.lib.nano_llama_backend_vocab_size(self.ctx))
        self.shared_kv_bytes = int(self._shared_kv_bytes(self.ctx)) if self._shared_kv_bytes else 0

    @staticmethod
    def _load_library(config: Config):
        device_config = config.device_config or {}
        lib_path = (
            device_config.get("library_path")
            or os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB")
            or str(Path(__file__).with_name("libnanollama_backend.so"))
        )
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                "llama.cpp backend library not found. Set device_config['library_path'] "
                f"or NANOVLLM_LLAMA_BACKEND_LIB. Tried: {lib_path}"
            )
        return ctypes.CDLL(lib_path)

    def _bind_functions(self):
        def bind_symbol(name: str):
            try:
                return getattr(self.lib, name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"llama.cpp backend library does not export {name}. "
                    "Rebuild libnanollama_backend.so with the matching nano-vLLM llama.cpp sources."
                ) from exc

        self._create = bind_symbol(self.create_symbol)
        self._create.argtypes = [ctypes.POINTER(_NanoLlamaBackendParams)]
        self._create.restype = ctypes.c_void_p
        self._destroy = bind_symbol(self.destroy_symbol)
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None
        self.lib.nano_llama_backend_vocab_size.argtypes = [ctypes.c_void_p]
        self.lib.nano_llama_backend_vocab_size.restype = ctypes.c_int32
        self.lib.nano_llama_backend_eos_token.argtypes = [ctypes.c_void_p]
        self.lib.nano_llama_backend_eos_token.restype = ctypes.c_int32
        self.lib.nano_llama_backend_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        self.lib.nano_llama_backend_tokenize.restype = ctypes.c_int32
        self.lib.nano_llama_backend_detokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        self.lib.nano_llama_backend_detokenize.restype = ctypes.c_int32
        self._run = bind_symbol(self.run_symbol)
        self._run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NanoLlamaKVPlan),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._run.restype = ctypes.c_int32
        self._release_blocks = bind_symbol(self.release_blocks_symbol)
        self._release_blocks.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._release_blocks.restype = None
        self._shared_kv_bytes = None
        if self.shared_kv_bytes_symbol is not None:
            self._shared_kv_bytes = bind_symbol(self.shared_kv_bytes_symbol)
            self._shared_kv_bytes.argtypes = [ctypes.c_void_p]
            self._shared_kv_bytes.restype = ctypes.c_size_t

    def call(self, method_name, *args):
        method = getattr(self, method_name)
        return method(*args)

    def eos_token_id(self) -> int:
        return int(self.lib.nano_llama_backend_eos_token(self.ctx))

    def tokenize(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        needed = self.lib.nano_llama_backend_tokenize(self.ctx, data, None, 0, True, True)
        if needed < 0:
            needed = -needed
        tokens = (ctypes.c_int32 * needed)()
        n_tokens = self.lib.nano_llama_backend_tokenize(self.ctx, data, tokens, needed, True, True)
        if n_tokens < 0:
            raise RuntimeError("llama.cpp tokenization buffer was too small")
        return list(tokens[:n_tokens])

    def detokenize(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        tokens = (ctypes.c_int32 * len(token_ids))(*token_ids)
        needed = self.lib.nano_llama_backend_detokenize(self.ctx, tokens, len(token_ids), None, 0, True, True)
        if needed < 0:
            needed = -needed
        buf = ctypes.create_string_buffer(needed + 1)
        n_bytes = self.lib.nano_llama_backend_detokenize(self.ctx, tokens, len(token_ids), buf, len(buf), True, True)
        if n_bytes < 0:
            raise RuntimeError("llama.cpp detokenization buffer was too small")
        return bytes(buf.raw[:n_bytes]).decode("utf-8", errors="replace")

    def allocate_kv_cache(self, num_blocks: int, block_size: int):
        return None

    def run(self, plan: BackendExecutionPlan) -> list[int]:
        c_plan, buffers = self._make_c_plan(plan)
        logits = np.empty((len(plan.seq_ids), self.vocab_size), dtype=np.float32)
        ret = self._run(
            self.ctx,
            ctypes.byref(c_plan),
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logits.shape[0],
            self.vocab_size,
        )
        if ret != 0:
            raise RuntimeError(f"llama.cpp backend execution failed with code {ret}")
        return self._sample(logits, np.asarray(plan.temperatures, dtype=np.float32)).tolist()

    def _make_c_plan(self, plan: BackendExecutionPlan):
        n_tokens = len(plan.input_ids)
        n_seqs = len(plan.seq_ids)
        max_block_table_len = max((len(block_table) for block_table in plan.block_tables), default=0)
        flat_block_tables = []
        for block_table in plan.block_tables:
            flat_block_tables.extend(block_table + [-1] * (max_block_table_len - len(block_table)))

        buffers = {
            "tokens": (ctypes.c_int32 * n_tokens)(*plan.input_ids),
            "positions": (ctypes.c_int32 * n_tokens)(*plan.positions),
            "seq_ids": (ctypes.c_int32 * n_seqs)(*plan.seq_ids),
            "scheduled_token_counts": (ctypes.c_int32 * n_seqs)(*plan.scheduled_token_counts),
            "slot_mapping": (ctypes.c_int32 * max(1, len(plan.slot_mapping)))(*plan.slot_mapping or [-1]),
            "block_tables": (ctypes.c_int32 * max(1, len(flat_block_tables)))(*flat_block_tables or [-1]),
            "context_lens": (ctypes.c_int32 * n_seqs)(*plan.context_lens),
            "num_cached_tokens": (ctypes.c_int32 * n_seqs)(*plan.num_cached_tokens),
        }
        c_plan = _NanoLlamaKVPlan(
            is_prefill=plan.is_prefill,
            n_tokens=n_tokens,
            n_seqs=n_seqs,
            block_size=self.block_size,
            block_table_cols=max_block_table_len,
            tokens=buffers["tokens"],
            positions=buffers["positions"],
            token_seq_ids=None,
            seq_ids=buffers["seq_ids"],
            scheduled_token_counts=buffers["scheduled_token_counts"],
            slot_mapping=buffers["slot_mapping"],
            block_tables=buffers["block_tables"],
            context_lens=buffers["context_lens"],
            num_cached_tokens=buffers["num_cached_tokens"],
        )
        return c_plan, buffers

    @staticmethod
    def _sample(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
        scaled = logits.astype(np.float32, copy=False) / temperatures[:, None]
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        noise = np.random.exponential(size=probs.shape).astype(np.float32)
        return np.argmax(probs / np.maximum(noise, 1e-10), axis=1).astype(np.int32)

    def release_blocks(self, block_ids: list[int], seq_ids: list[int] | None = None):
        if not block_ids:
            return
        c_blocks = (ctypes.c_int32 * len(block_ids))(*block_ids)
        seq_ids = seq_ids or []
        c_seq_ids = (ctypes.c_int32 * max(1, len(seq_ids)))(*seq_ids or [-1])
        self._release_blocks(
            self.ctx,
            c_blocks,
            len(block_ids),
            c_seq_ids,
            len(seq_ids),
            self.block_size,
        )

    def shutdown(self):
        if getattr(self, "ctx", None):
            self._destroy(self.ctx)
            self.ctx = None

    def exit(self):
        self.shutdown()


class LlamaCppPDRunner(LlamaCppRunner):
    create_symbol = "nano_llama_pd_backend_create"
    destroy_symbol = "nano_llama_pd_backend_destroy"
    run_symbol = "nano_llama_pd_backend_run"
    release_blocks_symbol = "nano_llama_pd_backend_release_blocks"
    shared_kv_bytes_symbol = "nano_llama_pd_backend_shared_kv_bytes"
