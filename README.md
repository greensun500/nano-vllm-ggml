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

The base Python dependencies are sufficient for the native CPU/Vulkan path;
the in-tree extension must still be compiled from a source checkout with the
build script below. Install the CUDA-specific stack explicitly when using the
original CUDA backend from a checkout:

```bash
pip install '.[cuda]'
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

## In-tree Qwen3.5 CPU/Vulkan runtime

The new runtime keeps request scheduling and cache ownership in nano-vLLM and
statically embeds only the official GGML CPU/Vulkan implementation. It does not
link `llama`, construct a `llama_context`, or require `--library-path`.

```bash
git submodule update --init --recursive
./scripts/build_native_runtime.sh

# Build one module containing CPU plus Vulkan:
NANOVLLM_NATIVE_VULKAN=ON ./scripts/build_native_runtime.sh
```

The native path implements the 24-layer Qwen3.5-2B target graph, six
full-attention Paged-KV layers, 18 recurrent Gated Delta Net layers, and the
GGUF's bundled MTP layer. MTP v1 is greedy-only and opt-in. Start with one
sequence while validating the correctness-first F32 state implementation:

```bash
PYTHONPATH=. python3 -m nanovllm.cli.chat \
  /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_cpu \
  --tokenizer /path/to/Qwen3.5-2B \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

After a Vulkan-enabled build, change only the execution backend:

```bash
PYTHONPATH=. python3 -m nanovllm.cli.chat \
  /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_vulkan \
  --tokenizer /path/to/Qwen3.5-2B \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

For an NVIDIA machine, build the same native extension with GGML CUDA and then
select `native_cuda`:

```bash
NANOVLLM_NATIVE_CUDA=ON scripts/build_native_runtime.sh
python -m nanovllm.cli.chat /path/to/Qwen3.5.gguf \
  --backend native_cuda \
  --tokenizer /path/to/Qwen3.5-tokenizer
```

The earlier `llamacpp_cpu`/`llamacpp_vulkan` external-library path remains only
as a migration oracle. It is not used by either native backend. The current
runtime flow is documented in
[`CURRENT_RUNTIME_FLOW.zh.md`](CURRENT_RUNTIME_FLOW.zh.md); current-version
changes, correctness evidence and remote benchmarks are recorded in
[`CURRENT_VERSION_CHANGES.zh.md`](CURRENT_VERSION_CHANGES.zh.md).

## Benchmark

Use the backend-aware CLI. Its JSON includes the pinned GGML commit, Vulkan
build flag, physical KV block count, and MTP drafted/accepted/verification
statistics:

```bash
PYTHONPATH=. python3 -m nanovllm.cli.bench \
  /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_cpu \
  --tokenizer /path/to/Qwen3.5-2B \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --num-kvcache-blocks 1 \
  --prompt-len 32 \
  --gen-len 16 \
  --enable-mtp \
  --temperature 0 \
  --json
```

`num_kvcache_blocks` is total physical capacity and does not change the
per-sequence `max_model_len`. Increase it explicitly when concurrent sequences
need more pages.

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
