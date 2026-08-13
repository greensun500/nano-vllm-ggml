from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from nanovllm.backends.base import BackendExecutionPlan, BackendExecutionResult
from nanovllm.config import Config


class NativeRunner:
    """nano-vLLM scheduler adapter for the in-tree Qwen3.5 runtime.

    The native extension owns the model, GGML graphs and cache storage.  This
    class deliberately keeps the scheduling-side responsibilities in Python:
    it assigns the bounded native sequence slots, converts an execution plan
    to the extension ABI, and turns greedy/MTP outputs back into the common
    :class:`BackendExecutionResult` contract.

    ``nanovllm._C.Qwen35Runtime`` is intentionally looked up at construction
    time.  The lower-level native-runtime smoke-test extension can therefore
    continue to build before the full Qwen3.5 runtime type is available, while
    attempts to use this runner fail with an actionable error.
    """

    def __init__(self, config: Config):
        self.config = config
        self.block_size = int(config.kvcache_block_size)
        self._seq_slots: dict[int, int] = {}
        self._free_seq_slots = list(reversed(range(int(config.max_num_seqs))))
        self.mtp_drafted_tokens = 0
        self.mtp_accepted_tokens = 0
        self.mtp_verification_steps = 0

        runtime_type = self._load_runtime_type()
        device_config = config.device_config or {}
        backend = self._backend_kind(config.backend)
        model_path = config.gguf_model or config.model
        if not model_path:
            raise ValueError("the native Qwen3.5 runtime requires a GGUF model path")

        # Keep this list explicit: in particular, library_path and the old
        # llama.cpp context/offload options must never become part of the
        # in-tree runtime ABI.
        self.runtime = runtime_type(
            model_path=os.fspath(model_path),
            backend=backend,
            max_model_len=int(config.max_model_len),
            max_num_batched_tokens=int(config.max_num_batched_tokens),
            max_num_seqs=int(config.max_num_seqs),
            block_size=self.block_size,
            num_blocks=int(config.num_kvcache_blocks),
            n_threads=int(device_config.get("n_threads", max(os.cpu_count() or 1, 1))),
            device_index=int(device_config.get("device_index", 0)),
            enable_mtp=bool(config.enable_mtp),
            mtp_max_draft_tokens=int(config.mtp_max_draft_tokens),
            enable_graph_reuse=bool(getattr(config, "enable_graph_reuse", True)),
            enable_vulkan_graph_reuse=bool(
                getattr(config, "native_vulkan_graph_reuse", False)
            ),
            attention_impl=str(getattr(config, "native_attention_impl", "math")),
            enable_batched_recurrent_snapshots=bool(
                getattr(config, "native_batched_recurrent_snapshots", False)
            ),
            enable_mtp_prefill_fusion=bool(
                getattr(config, "native_mtp_prefill_fusion", False)
            ),
        )

    @staticmethod
    def _load_runtime_type():
        try:
            extension = importlib.import_module("nanovllm._C")
        except ImportError as exc:
            raise RuntimeError(
                "nano-vLLM's native Qwen3.5 runtime is not built. "
                "Run scripts/build_native_runtime.sh first."
            ) from exc

        runtime_type = getattr(extension, "Qwen35Runtime", None)
        if not callable(runtime_type):
            raise RuntimeError(
                "nanovllm._C does not expose Qwen35Runtime. Rebuild the native "
                "extension with the Qwen3.5 model runtime enabled; the current "
                "extension only provides the lower-level native-runtime API."
            )
        return runtime_type

    @staticmethod
    def _backend_kind(backend: str) -> str:
        if backend.endswith("_cpu"):
            return "cpu"
        if backend.endswith("_vulkan"):
            return "vulkan"
        if backend.endswith("_cuda"):
            return "cuda"
        raise ValueError(f"the native Qwen3.5 runner cannot use backend {backend!r}")

    def call(self, method_name, *args):
        method = getattr(self, method_name)
        return method(*args)

    def _optional_tokenizer_method(self, name: str) -> Callable[..., Any]:
        runtime = getattr(self, "runtime", None)
        method = getattr(runtime, name, None)
        if not callable(method):
            raise RuntimeError(
                f"the native Qwen3.5 runtime does not provide {name}(); "
                "configure tokenizer_backend='hf' or rebuild the extension "
                "with tokenizer support"
            )
        return method

    def eog_token_ids(self) -> tuple[int, ...]:
        tokens = tuple(int(token) for token in self._optional_tokenizer_method("eog_token_ids")())
        if not tokens:
            raise RuntimeError("the native Qwen3.5 runtime returned no EOG token IDs")
        return tokens

    def tokenize(self, text: str) -> list[int]:
        tokens = self._optional_tokenizer_method("tokenize")(text)
        return [int(token) for token in tokens]

    def detokenize(self, token_ids: list[int]) -> str:
        return str(self._optional_tokenizer_method("detokenize")([int(token) for token in token_ids]))

    def allocate_kv_cache(self, num_blocks: int, block_size: int):
        # Cache sizes are construction parameters of Qwen35Runtime.  Retain the
        # common runner method without creating a second allocation path.
        return None

    def run(self, plan: BackendExecutionPlan) -> BackendExecutionResult:
        self._validate_greedy(plan.temperatures)
        native_plan = self._make_native_plan(plan)

        if (
            self.config.enable_mtp
            and not plan.is_prefill
            and self._mtp_window_fits(native_plan)
        ):
            return self._run_mtp(native_plan, len(plan.seq_ids))

        runtime = self._require_live_runtime()
        token_ids = self._as_int32_vector(runtime.run(native_plan), "run token IDs")
        if token_ids.size != len(plan.seq_ids):
            raise RuntimeError(
                "native Qwen3.5 runtime returned "
                f"{token_ids.size} greedy tokens for {len(plan.seq_ids)} sequences"
            )
        return BackendExecutionResult([[int(token_id)] for token_id in token_ids])

    def _mtp_window_fits(self, native_plan: dict[str, Any]) -> bool:
        """Return whether K draft slots fit after every pending decode token.

        Near the configured context boundary the target path remains valid even
        when a full speculative verification window does not. In that case we
        execute one ordinary target step and keep the bundled MTP cache caught
        up, instead of failing the request solely because speculation is on.
        """

        positions = native_plan["positions"]
        if positions.size == 0:
            return False
        required_end = int(np.max(positions)) + 1 + int(self.config.mtp_max_draft_tokens)
        return required_end <= int(self.config.max_model_len)

    def _run_mtp(self, native_plan: dict[str, Any], n_seqs: int) -> BackendExecutionResult:
        runtime = self._require_live_runtime()
        capacity = int(self.config.mtp_max_draft_tokens) + 1
        raw_result = runtime.run_mtp(native_plan, capacity)
        try:
            flat_token_ids, output_counts, draft_counts = raw_result
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "native Qwen3.5 MTP runtime must return "
                "(flat_token_ids, output_counts, draft_counts)"
            ) from exc

        flat_token_ids = self._as_int32_vector(flat_token_ids, "MTP token IDs")
        output_counts = self._as_int32_vector(output_counts, "MTP output counts")
        draft_counts = self._as_int32_vector(draft_counts, "MTP draft counts")
        expected_tokens = n_seqs * capacity
        if flat_token_ids.size != expected_tokens:
            raise RuntimeError(
                "native Qwen3.5 MTP runtime returned "
                f"{flat_token_ids.size} token slots; expected {expected_tokens}"
            )
        if output_counts.size != n_seqs or draft_counts.size != n_seqs:
            raise RuntimeError(
                "native Qwen3.5 MTP count arrays must contain one value per sequence"
            )
        if np.any(output_counts < 1) or np.any(output_counts > capacity):
            raise RuntimeError(
                f"native Qwen3.5 MTP output counts must be in [1, {capacity}]"
            )
        if np.any(draft_counts < 0) or np.any(draft_counts >= capacity):
            raise RuntimeError(
                f"native Qwen3.5 MTP draft counts must be in [0, {capacity - 1}]"
            )

        token_rows = flat_token_ids.reshape(n_seqs, capacity)
        outputs = [
            token_rows[row, : int(output_counts[row])].astype(np.int32, copy=False).tolist()
            for row in range(n_seqs)
        ]
        drafts = draft_counts.tolist()
        self.mtp_drafted_tokens += sum(drafts)
        self.mtp_accepted_tokens += sum(len(output) - 1 for output in outputs)
        self.mtp_verification_steps += n_seqs
        return BackendExecutionResult(outputs, drafts)

    def mtp_stats(self) -> dict[str, int]:
        return {
            "drafted_tokens": self.mtp_drafted_tokens,
            "accepted_tokens": self.mtp_accepted_tokens,
            "verification_steps": self.mtp_verification_steps,
        }

    def mtp_profile_stats(self) -> dict[str, int]:
        runtime = self._require_live_runtime()
        method = getattr(runtime, "mtp_profile_stats", None)
        if not callable(method):
            return {}
        return {key: int(value) for key, value in method().items()}

    def graph_reuse_stats(self) -> dict[str, int]:
        runtime = self._require_live_runtime()
        method = getattr(runtime, "graph_reuse_stats", None)
        if not callable(method):
            return {"hits": 0, "misses": 0, "evictions": 0, "active_entries": 0}
        stats = method()
        return {
            "hits": int(stats.get("hits", 0)),
            "misses": int(stats.get("misses", 0)),
            "evictions": int(stats.get("evictions", 0)),
            "active_entries": int(stats.get("active_entries", 0)),
        }

    def memory_stats(self) -> dict[str, int]:
        runtime = self._require_live_runtime()
        method = getattr(runtime, "memory_stats", None)
        if not callable(method):
            return {}
        return {key: int(value) for key, value in method().items()}

    def _make_native_plan(self, plan: BackendExecutionPlan) -> dict[str, Any]:
        n_tokens = len(plan.input_ids)
        n_seqs = len(plan.seq_ids)
        if len(plan.positions) != n_tokens or len(plan.slot_mapping) != n_tokens:
            raise ValueError("native execution plan requires one position and KV slot per input token")

        sequence_fields = {
            "scheduled_token_counts": plan.scheduled_token_counts,
            "block_tables": plan.block_tables,
            "context_lens": plan.context_lens,
            "num_cached_tokens": plan.num_cached_tokens,
            "temperatures": plan.temperatures,
        }
        for name, values in sequence_fields.items():
            if len(values) != n_seqs:
                raise ValueError(
                    f"native execution plan field {name!r} has {len(values)} entries; "
                    f"expected {n_seqs}"
                )

        block_table_cols = max((len(table) for table in plan.block_tables), default=0)
        flat_block_tables = [
            block_id
            for table in plan.block_tables
            for block_id in table + [-1] * (block_table_cols - len(table))
        ]
        native_seq_ids = [self._native_seq_id(seq_id) for seq_id in plan.seq_ids]

        return {
            "is_prefill": bool(plan.is_prefill),
            "n_tokens": n_tokens,
            "n_seqs": n_seqs,
            "block_size": self.block_size,
            "block_table_cols": block_table_cols,
            "tokens": self._int32_array(plan.input_ids),
            "positions": self._int32_array(plan.positions),
            "seq_ids": self._int32_array(native_seq_ids),
            "scheduled_token_counts": self._int32_array(plan.scheduled_token_counts),
            "slot_mapping": self._int32_array(plan.slot_mapping),
            "block_tables": self._int32_array(flat_block_tables),
            "context_lens": self._int32_array(plan.context_lens),
            "num_cached_tokens": self._int32_array(plan.num_cached_tokens),
        }

    @staticmethod
    def _int32_array(values) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.int32).reshape(-1)

    @classmethod
    def _as_int32_vector(cls, values, label: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise RuntimeError(f"native Qwen3.5 {label} must be a one-dimensional array")
        if array.dtype.kind not in "iu":
            raise RuntimeError(f"native Qwen3.5 {label} must contain integer values")
        return cls._int32_array(array)

    @staticmethod
    def _validate_greedy(temperatures) -> None:
        values = np.asarray(temperatures, dtype=np.float32)
        if np.any(values != 0):
            raise ValueError("the native Qwen3.5 runtime only supports greedy decoding")

    def _native_seq_id(self, seq_id: int) -> int:
        native_seq_id = self._seq_slots.get(seq_id)
        if native_seq_id is not None:
            return native_seq_id
        if not self._free_seq_slots:
            raise RuntimeError("native Qwen3.5 sequence slots are exhausted")
        native_seq_id = self._free_seq_slots.pop()
        self._seq_slots[seq_id] = native_seq_id
        return native_seq_id

    def release_blocks(self, block_ids: list[int], seq_ids: list[int] | None = None):
        seq_ids = seq_ids or []
        if not seq_ids:
            raise ValueError("native Qwen3.5 KV release requires sequence IDs")
        missing = [seq_id for seq_id in seq_ids if seq_id not in self._seq_slots]
        if missing:
            raise RuntimeError(f"native Qwen3.5 sequence slots are not allocated: {missing}")

        native_seq_ids = [self._seq_slots[seq_id] for seq_id in seq_ids]
        runtime = self._require_live_runtime()
        result = runtime.release_blocks(
            block_ids=self._int32_array(block_ids),
            seq_ids=self._int32_array(native_seq_ids),
            block_size=self.block_size,
        )
        if isinstance(result, (int, np.integer)) and int(result) != 0:
            raise RuntimeError(f"native Qwen3.5 sequence release failed with code {result}")

        for seq_id, native_seq_id in zip(seq_ids, native_seq_ids):
            del self._seq_slots[seq_id]
            self._free_seq_slots.append(native_seq_id)

    def _require_live_runtime(self):
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            raise RuntimeError("the native Qwen3.5 runtime has been shut down")
        return runtime

    def shutdown(self):
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return
        runtime.shutdown()
        self.runtime = None

    def exit(self):
        self.shutdown()
