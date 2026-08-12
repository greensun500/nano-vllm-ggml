import os
from dataclasses import dataclass
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
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any | None = None
    eos_token_ids: tuple[int, ...] = ()
    enable_prefix_cache: bool | None = None
    enable_preemption: bool | None = None
    enable_mtp: bool = False
    mtp_max_draft_tokens: int = 3
    enable_graph_reuse: bool = True
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        _require(self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive")
        _require(self.max_num_seqs > 0, "max_num_seqs must be positive")
        _require(self.max_model_len > 0, "max_model_len must be positive")
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
        ), f"unsupported backend: {self.backend}")
        _require(self.model_format in ("hf", "gguf"), "model_format must be 'hf' or 'gguf'")
        _require(
            self.tokenizer_backend in ("hf", "llamacpp"),
            "tokenizer_backend must be 'hf' or 'llamacpp'",
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
            from transformers import AutoConfig

            _require(os.path.isdir(self.model), "CUDA model must be a local Hugging Face directory")
            _require(self.model_format == "hf", "CUDA backend requires model_format='hf'")
            self.hf_config = AutoConfig.from_pretrained(self.model)
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        else:
            is_native = self.backend in ("native_cpu", "native_vulkan")
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
                    self.tokenizer_backend == "hf",
                    "native backend requires tokenizer_backend='hf'; it does not use llama.cpp tokenization",
                )
                _require(
                    bool(self.tokenizer),
                    "native backend requires tokenizer=/path/to/the matching Hugging Face tokenizer",
                )
                _require(
                    os.path.isdir(self.tokenizer),
                    "native backend tokenizer must be a local Hugging Face tokenizer directory",
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
            if self.enable_mtp:
                _require(self.mtp_max_draft_tokens > 0, "mtp_max_draft_tokens must be positive")
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
