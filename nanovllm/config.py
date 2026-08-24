"""Public configuration for the native Qwen3.5 GGUF runtime.

The first public release deliberately supports one live sequence. Keeping this
contract explicit is more honest than exposing vLLM-style batch controls while
the native target graph is still executed per sequence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(slots=True)
class Config:
    model: str
    backend: str = "native_cpu"
    gguf_model: str | None = None
    device_config: dict | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_model_len: int = 4096
    eos_token_ids: tuple[int, ...] = ()
    # Internal scheduler invariants for the single-sequence native release.
    enable_prefix_cache: bool = False
    enable_preemption: bool = False
    enable_mtp: bool = False
    mtp_max_draft_tokens: int = 1
    # Mali graph reuse is retained for hardware investigation only. CPU,
    # NVIDIA Vulkan and CUDA graph reuse are selected internally by runtime.
    mali_experimental_graph_reuse: bool = False
    enable_session_cache: bool = True
    max_retained_sessions: int = 1
    max_consecutive_prefill_rounds: int = 4
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self) -> None:
        _require(
            self.backend in ("native_cpu", "native_vulkan", "native_cuda"),
            f"unsupported native backend: {self.backend}",
        )
        if self.gguf_model is None:
            self.gguf_model = self.model
        _require(bool(self.gguf_model), "native runtime requires a GGUF model path")
        _require(os.path.isfile(self.gguf_model), f"GGUF model does not exist: {self.gguf_model}")

        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = 2048
        if self.max_num_seqs is None:
            self.max_num_seqs = 1
        _require(self.max_num_seqs == 1, "the public native runtime supports max_num_seqs=1")
        _require(self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive")
        _require(self.max_model_len > 0, "max_model_len must be positive")
        _require(self.max_retained_sessions in (0, 1), "max_retained_sessions must be 0 or 1")
        _require(
            self.max_consecutive_prefill_rounds >= 0,
            "max_consecutive_prefill_rounds must be non-negative",
        )
        if self.enable_session_cache:
            _require(self.max_retained_sessions == 1, "session cache requires one retained session")
        _require(self.kvcache_block_size > 0, "kvcache_block_size must be positive")
        _require(
            self.kvcache_block_size % 256 == 0,
            "kvcache_block_size must be a positive multiple of 256",
        )
        _require(
            self.num_kvcache_blocks == -1 or self.num_kvcache_blocks > 0,
            "num_kvcache_blocks must be positive or -1",
        )
        _require(self.mtp_max_draft_tokens > 0, "mtp_max_draft_tokens must be positive")
        _require(not self.enable_prefix_cache, "native runtime does not support prefix caching")
        _require(not self.enable_preemption, "native runtime does not support preemption")
        if self.enable_mtp:
            _require(
                self.max_num_batched_tokens >= self.mtp_max_draft_tokens + 1,
                "max_num_batched_tokens must fit one MTP verification batch",
            )
        _require(
            not self.mali_experimental_graph_reuse or self.backend == "native_vulkan",
            "mali_experimental_graph_reuse requires backend='native_vulkan'",
        )
        if self.num_kvcache_blocks == -1:
            self.num_kvcache_blocks = (
                self.max_model_len + self.kvcache_block_size - 1
            ) // self.kvcache_block_size
        _require(
            self.num_kvcache_blocks * self.kvcache_block_size >= self.max_model_len,
            "num_kvcache_blocks * kvcache_block_size must cover max_model_len",
        )
        if self.device_config is None:
            self.device_config = {}
