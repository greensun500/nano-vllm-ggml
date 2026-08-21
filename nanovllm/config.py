import os
from dataclasses import dataclass, field
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(slots=True)
class Config:
    model: str
    backend: str = "cuda"
    model_format: str = "hf"
    gguf_model: str | None = None
    tokenizer: str | None = None
    tokenizer_backend: str = "hf"
    device_config: dict | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any | None = None
    eos_token_ids: tuple[int, ...] = ()
    enable_prefix_cache: bool | None = None
    enable_preemption: bool | None = None
    # None means that the native runner may select a measured, device-specific
    # default before it creates the scheduler.  True/False is always an
    # explicit user override.
    enable_mtp: bool | None = None
    mtp_max_draft_tokens: int | None = None
    enable_graph_reuse: bool = True
    native_vulkan_graph_reuse: bool | None = None
    native_performance_profile: str = "auto"
    native_effective_performance_profile: str = field(init=False, default="unresolved")
    # Probe FlashAttention for ordinary native target execution. The runtime
    # deliberately retains math attention for MTP because draft/verification
    # shapes must stay numerically aligned for acceptance.
    native_attention_impl: str = "auto"
    native_batched_recurrent_snapshots: bool = False
    native_mtp_prefill_fusion: bool = False
    native_mtp_verification_kv_fusion: bool = False
    # Explicit, default-off Mali-G720 MTP-prefill experiments.  The runtime
    # rejects these modes on non-Mali-G720 Vulkan devices.
    native_mali_mtp_prefill_strategy: str = "whole"
    enable_session_cache: bool = True
    max_retained_sessions: int = 1
    max_consecutive_prefill_rounds: int = 4
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        is_native = self.backend in ("native_cpu", "native_vulkan", "native_cuda")
        # Only native backends can identify the active GGML device at runner
        # creation time.  Keep historical concrete defaults for every other
        # backend rather than leaking the tri-state native configuration there.
        if not is_native:
            if self.enable_mtp is None:
                self.enable_mtp = False
            if self.mtp_max_draft_tokens is None:
                self.mtp_max_draft_tokens = 3
            if self.native_vulkan_graph_reuse is None:
                self.native_vulkan_graph_reuse = False
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = 2048 if is_native else 16384
        if self.max_num_seqs is None:
            # Recurrent state is allocated per native sequence slot.  A small
            # default keeps an edge-device launch from reserving hundreds of
            # state rows before it has received its first request.
            self.max_num_seqs = 1 if is_native else 512
        _require(self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive")
        _require(self.max_num_seqs > 0, "max_num_seqs must be positive")
        _require(self.max_model_len > 0, "max_model_len must be positive")
        _require(self.max_retained_sessions >= 0, "max_retained_sessions must be non-negative")
        _require(
            self.max_consecutive_prefill_rounds >= 0,
            "max_consecutive_prefill_rounds must be non-negative",
        )
        if self.enable_session_cache:
            _require(
                self.max_retained_sessions > 0,
                "max_retained_sessions must be positive when session cache is enabled",
            )
            _require(
                self.max_retained_sessions <= self.max_num_seqs,
                "max_retained_sessions must not exceed max_num_seqs",
            )
        _require(self.kvcache_block_size > 0, "kvcache_block_size must be positive")
        _require(
            self.num_kvcache_blocks == -1 or self.num_kvcache_blocks > 0,
            "num_kvcache_blocks must be positive or -1",
        )
        _require(self.backend in (
            "cuda",
            "llamacpp_cpu",
            "llamacpp_vulkan",
            "native_cpu",
            "native_vulkan",
            "native_cuda",
        ), f"unsupported backend: {self.backend}")
        _require(self.model_format in ("hf", "gguf"), "model_format must be 'hf' or 'gguf'")
        # A native runtime can take its tokenizer solely from the GGUF. Keep
        # an explicitly supplied HF directory as a compatibility override.
        if is_native and self.tokenizer_backend == "hf" and not self.tokenizer:
            self.tokenizer_backend = "native"
        _require(
            self.tokenizer_backend in ("hf", "llamacpp", "native"),
            "tokenizer_backend must be 'hf', 'llamacpp', or 'native'",
        )
        _require(
            self.kvcache_block_size % 256 == 0,
            "kvcache_block_size must be a positive multiple of 256",
        )
        _require(
            self.max_num_batched_tokens >= self.max_num_seqs,
            "max_num_batched_tokens must be at least max_num_seqs",
        )
        _require(
            self.native_attention_impl in ("math", "auto", "flash", "paged"),
            "native_attention_impl must be 'math', 'auto', 'flash', or 'paged'",
        )
        _require(
            self.native_performance_profile in (
                "auto",
                "baseline",
                "arm_a720_cpu",
                "mali_g720_vulkan",
                "nvidia_a100_vulkan",
            ),
            "native_performance_profile must be 'auto', 'baseline', "
            "'arm_a720_cpu', 'mali_g720_vulkan', or 'nvidia_a100_vulkan'",
        )
        _require(
            self.native_mali_mtp_prefill_strategy in (
                "whole",
                "chunked_legacy",
                "chunked_staged",
            ),
            "native_mali_mtp_prefill_strategy must be 'whole', "
            "'chunked_legacy', or 'chunked_staged'",
        )
        if self.native_mtp_prefill_fusion:
            _require(is_native, "native_mtp_prefill_fusion requires a native backend")
            _require(self.enable_mtp is True, "native_mtp_prefill_fusion requires enable_mtp=True")
        if self.native_mtp_verification_kv_fusion:
            _require(
                is_native,
                "native_mtp_verification_kv_fusion requires a native backend",
            )
            _require(
                self.enable_mtp is True,
                "native_mtp_verification_kv_fusion requires enable_mtp=True",
            )
        if self.native_mali_mtp_prefill_strategy != "whole":
            _require(
                self.backend == "native_vulkan",
                "chunked Mali MTP prefill requires backend='native_vulkan'",
            )
            _require(
                self.enable_mtp is True,
                "chunked Mali MTP prefill requires enable_mtp=True",
            )
        if self.native_vulkan_graph_reuse is True:
            _require(
                self.backend == "native_vulkan",
                "native_vulkan_graph_reuse requires backend='native_vulkan'",
            )
        _require(
            1 <= self.tensor_parallel_size <= 8,
            "tensor_parallel_size must be between 1 and 8",
        )
        if self.enable_prefix_cache is None:
            self.enable_prefix_cache = self.backend == "cuda"
        if self.enable_preemption is None:
            self.enable_preemption = self.backend == "cuda"
        if self.backend == "cuda":
            _require(
                not self.enable_mtp,
                "built-in MTP is only supported by the GGUF CPU/Vulkan backends",
            )
            _require(
                self.tokenizer_backend == "hf",
                "CUDA backend requires tokenizer_backend='hf'",
            )
            from transformers import AutoConfig

            _require(os.path.isdir(self.model), "CUDA model must be a local Hugging Face directory")
            _require(self.model_format == "hf", "CUDA backend requires model_format='hf'")
            self.hf_config = AutoConfig.from_pretrained(self.model)
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        else:
            if is_native:
                _require(
                    not self.enable_prefix_cache,
                    "native backend does not support prefix caching yet",
                )
                _require(
                    not self.enable_preemption,
                    "native backend does not support preemption yet",
                )
                _require(
                    self.tokenizer_backend in ("native", "hf"),
                    "native backend tokenizer_backend must be 'native' or 'hf'",
                )
                if self.tokenizer_backend == "hf":
                    _require(
                        bool(self.tokenizer),
                        "native HF tokenizer requires tokenizer=/path/to/the matching tokenizer",
                    )
                    _require(
                        os.path.isdir(self.tokenizer),
                        "native HF tokenizer must be a local Hugging Face tokenizer directory",
                    )
                backend_label = "native backend"
                process_label = "native backend"
            else:
                _require(
                    not self.enable_prefix_cache,
                    "llama.cpp staged backend does not support prefix caching yet",
                )
                _require(
                    not self.enable_preemption,
                    "llama.cpp paged KV does not support preemption yet",
                )
                backend_label = "llama.cpp backend"
                process_label = "llama.cpp backend v1"
            if self.gguf_model is None and self.model.endswith(".gguf"):
                self.gguf_model = self.model
                self.model_format = "gguf"
            _require(
                self.gguf_model is not None,
                f"{backend_label} requires gguf_model=/path/to/model.gguf",
            )
            _require(os.path.isfile(self.gguf_model), f"GGUF model does not exist: {self.gguf_model}")
            _require(self.tensor_parallel_size == 1, f"{process_label} is single-process")
            if self.enable_mtp is True:
                _require(
                    self.mtp_max_draft_tokens is None or self.mtp_max_draft_tokens > 0,
                    "mtp_max_draft_tokens must be positive",
                )
                if self.mtp_max_draft_tokens is not None:
                    _require(
                        self.max_num_batched_tokens >= self.max_num_seqs * (self.mtp_max_draft_tokens + 1),
                        "max_num_batched_tokens must fit one MTP verification batch",
                    )
            if self.num_kvcache_blocks == -1:
                self.num_kvcache_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
            _require(
                self.num_kvcache_blocks * self.kvcache_block_size >= self.max_model_len,
                "num_kvcache_blocks * kvcache_block_size must cover max_model_len",
            )
        if self.device_config is None:
            self.device_config = {}
