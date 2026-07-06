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

## llama.cpp CPU/Vulkan/PD backend

The experimental llama.cpp backend keeps nano-vLLM's scheduler and paged KV block manager, while llama.cpp executes the model on CPU, Vulkan, or PD-separated Vulkan-prefill/CPU-decode.

```bash
LLAMA_CPP_DIR=/home/cix/nano-vlm/llama.cpp ./scripts/build_llamacpp_cpu.sh
export NANOVLLM_LLAMA_BACKEND_LIB=/home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so
export NANOVLLM_GGUF_MODEL=/home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf
PYTHONPATH=. python examples/llamacpp_backend.py
```

Use `NANOVLLM_BACKEND=llamacpp_vulkan` with `scripts/build_llamacpp_vulkan.sh` for Vulkan.

Use `NANOVLLM_BACKEND=llamacpp_pd` with the Vulkan build for strict zero-copy PD separation. In this mode prefill runs on Vulkan, decode runs on CPU, and both contexts share one Vulkan host-visible KV pool. If llama.cpp cannot create a `Vulkan_Host` KV buffer, initialization fails instead of silently falling back to KV copy.

Interactive chat test:

```bash
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_cpu \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so
```

For Vulkan:

```bash
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --ubatch-size 512
```

For PD zero-copy:

```bash
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_pd \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --ubatch-size 512
```

Inside the chat CLI, use `/reset` to clear history, `/system <prompt>` to change the system prompt, and `/exit` to quit.

Benchmark CPU/Vulkan/PD performance:

```bash
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --ubatch-size 512
```

PD benchmark:

```bash
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_pd \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 512 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --ubatch-size 512
```

The benchmark reports prefill input throughput, decode-step throughput, end-to-end generated-token throughput, load-time RSS, and load time. For PD it also reports the shared KV pool size. By default it perturbs synthetic prompts between runs to avoid measuring prefix-cache hits; add `--use-prefix-cache` when you want to benchmark cache reuse. Use `--json` for machine-readable output.

See `NANOVLLM_LLAMA_CPP_V2.0_PD_ZERO_COPY.zh.md` for the v2.0 PD zero-copy architecture, source-reading guide, benchmark interpretation, and optimization directions.

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
