import os
from dataclasses import dataclass
from typing import Any


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
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert self.backend in ("cuda", "llamacpp_cpu", "llamacpp_vulkan")
        assert self.model_format in ("hf", "gguf")
        assert self.tokenizer_backend in ("hf", "llamacpp")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.backend == "cuda":
            from transformers import AutoConfig

            assert os.path.isdir(self.model)
            assert self.model_format == "hf"
            self.hf_config = AutoConfig.from_pretrained(self.model)
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        else:
            if self.gguf_model is None and self.model.endswith(".gguf"):
                self.gguf_model = self.model
                self.model_format = "gguf"
            assert self.gguf_model is not None, "llama.cpp backend requires gguf_model=/path/to/model.gguf"
            assert os.path.isfile(self.gguf_model)
            assert self.tensor_parallel_size == 1, "llama.cpp backend v1 is single-process"
            if self.num_kvcache_blocks <= 0:
                self.num_kvcache_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
            self.max_model_len = self.num_kvcache_blocks * self.kvcache_block_size
        if self.device_config is None:
            self.device_config = {}
