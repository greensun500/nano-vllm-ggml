<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

中文源码学习路线与执行流程见 [`LEARNING_GUIDE.zh.md`](LEARNING_GUIDE.zh.md)。

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Qwen3.5 llama.cpp CPU/Vulkan backend

Stage 3 supports text-only Qwen3.5 GGUF inference on CPU and Vulkan. Full-attention layers use nano-vLLM paged KV slots, recurrent layers keep one native state per sequence, and the GGUF's built-in MTP layer can perform greedy speculative decoding without a separate draft model. The backend also supports the Qwen3.5 non-thinking chat template and all model EOG tokens.

MTP is opt-in. `mtp_max_draft_tokens` defaults to 3, matching the current llama.cpp default. An MTP verification batch requires `max_num_batched_tokens >= max_num_seqs * (mtp_max_draft_tokens + 1)`.

The integration is based on official llama.cpp commit `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`. Build the dedicated llama.cpp checkout:

```bash
./scripts/build_llamacpp_cpu.sh
./scripts/build_llamacpp_vulkan.sh
```

Run CPU chat:

```bash
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_cpu \
  --gguf-model /home/cix/nano-vllm/models/Qwen3.5-2B-Q4_0.gguf \
  --library-path /home/cix/nano-vllm/llama.cpp-qwen35/build_nanovllm_cpu/bin/libnanollama_backend.so \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

Run Vulkan chat:

```bash
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/nano-vllm/models/Qwen3.5-2B-Q4_0.gguf \
  --library-path /home/cix/nano-vllm/llama.cpp-qwen35/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --gpu-layers -1 \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

The staged backend intentionally disables prefix caching and preemption, and MTP v1 only accepts greedy sampling. It does not expose the previous PD backend. The older PD design documents remain as historical references for their original branches.

See `QWEN35_STAGED_IMPLEMENTATION.zh.md` for stage boundaries, validation evidence, and reflection notes.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
