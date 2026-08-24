<p align="center">
  <img src="assets/nanovllm-ggml-hero.png" alt="Token scheduling flowing through a GGML graph to CPU and GPU backends">
</p>

<h1 align="center">nanovllm-ggml</h1>

<p align="center">
  A readable, native GGUF inference runtime for Qwen3.5 — from Python request scheduling to GGML execution on CPU, Vulkan, and CUDA.
</p>

<p align="center">
  <a href="#why-nanovllm-ggml">Why nanovllm-ggml</a> ·
  <a href="#measured-performance">Performance</a> ·
  <a href="QUICKSTART.md">Quick Start</a>
</p>

## Why nanovllm-ggml

Most lightweight inference projects stop at wrapping an existing runtime.
nanovllm-ggml keeps the serving control plane small and visible, while implementing
the Qwen3.5 execution path directly on GGML:

```text
LLM / Scheduler / Sequence / BlockManager  (Python)
                    ↓ execution plan
Qwen35Runtime / cache / GGML graph           (C++)
                    ↓
              CPU · Vulkan · CUDA
```

- **One GGUF, no tokenizer checkout.** The tokenizer is read from GGUF
  metadata, so chat and benchmark start from a single model file.
- **Native Qwen3.5 runtime.** The qualified model path has 24 target layers:
  6 full-attention layers with paged KV cache, 18 recurrent Gated Delta Net
  layers with persistent state, plus Qwen3.5's bundled MTP layer.
- **A compact codebase with real runtime boundaries.** Python handles requests,
  scheduling and logical blocks; C++ owns weights, cache/state, graph creation,
  backend placement and execution.
- **Speculative decoding without hiding correctness.** Greedy MTP draft tokens
  are verified by the target model; rollback, recurrent snapshots and cache
  lifetime are part of the native runtime rather than a demo-only shortcut.
- **Device-aware execution.** CUDA Graphs are enabled in CUDA builds; persistent
  graph reuse is enabled for CPU, CUDA and non-Mali Vulkan. Mali-G720 uses its
  qualified raw-MMQ path by default, while Mali graph reuse stays opt-in.

The v4.0 public release deliberately focuses on **one active sequence** and
**greedy Qwen3.5 GGUF inference**. That makes the current performance and
correctness envelope concrete instead of presenting unqualified vLLM-like
features.

## Measured performance

All numbers below are real runs on `Qwen3.5-2B-Q4_0.gguf`, batch size 1 and
greedy decoding. `PP` is prompt length and `TG` is generated-token length.
`llama.cpp upstream` is the native upstream Vulkan/CPU baseline; the
nanovllm-ggml rows use the native runtime on the same platform.

| Platform | Framework | Configuration | PP + TG | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| NVIDIA A100 Vulkan | llama.cpp upstream | Native Vulkan, MTP off | 2048 + 512 | 11908.7 | 260.7 |
| NVIDIA A100 Vulkan | nanovllm-ggml | Native Vulkan, MTP off | 2048 + 512 | **12208.9** | 210.0 |
| NVIDIA A100 Vulkan | nanovllm-ggml | Native Vulkan, MTP K=3 + graph reuse | 2048 + 512 | 11469.7 | **279.6** |
| Arm Cortex-A720 CPU | llama.cpp | Native CPU, MTP off | 512 + 128 | **140.9** | 25.4 |
| Arm Cortex-A720 CPU | llama.cpp | Native CPU, MTP K=1 | 512 + 128 | 133.5 | 22.6 |
| Arm Cortex-A720 CPU | nanovllm-ggml | Native CPU, MTP off | 512 + 128 | 88.46 | 27.90 |
| Arm Cortex-A720 CPU | nanovllm-ggml | Native CPU, MTP K=1 | 512 + 128 | 87.60 | **35.58** |
| Mali-G720 Vulkan | llama.cpp | Native Vulkan, MTP off | 512 + 128 | 18.0 | 19.7 |
| Mali-G720 Vulkan | llama.cpp | Native Vulkan, MTP K=3 | 512 + 128 | 17.7 | 11.8 |
| Mali-G720 Vulkan | nanovllm-ggml v4.0 | Native Vulkan, MTP off | 512 + 128 | **167.65** | 21.55 |
| Mali-G720 Vulkan | nanovllm-ggml v4.0 | Native Vulkan, MTP K=3 | 512 + 128 | 166.13 | **24.55** |

The A100 result uses an NVIDIA A100-SXM4-80GB, full Vulkan offload and eight
CPU threads. The upstream llama.cpp A100 baseline has no MTP result. For
Arm/Mali llama.cpp rows, retain the workload and build settings supplied with
the original benchmark when making strict claims; cross-platform rates are not
directly comparable.

## Build and run

```bash
git clone https://github.com/greensun500/nano-vllm-ggml.git
cd nano-vllm-ggml

# CPU build
./scripts/build_native_runtime.sh

# Run with a single complete GGUF file
PYTHONPATH=. python3 -m nanovllm.cli.chat /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_cpu --max-model-len 8092 --num-kvcache-blocks 32 \
  --threads 8 --temperature 0
```

For Vulkan/CUDA builds, MTP, benchmarks, backend switching, and common
platform-specific settings, see **[QUICKSTART.md](QUICKSTART.md)**.

## Scope

| Included | Not a release claim yet |
| --- | --- |
| Native CPU, Vulkan and CUDA backends | Multi-sequence throughput |
| GGUF model loading and embedded tokenizer | Non-greedy sampling |
| Qwen3.5-2B-Q4_0 correctness qualification | General model-architecture coverage |
| Greedy MTP with target verification | PyTorch or external llama.cpp execution |

The historical PyTorch and external llama.cpp runners were intentionally
removed so the repository has one supported inference path.

## License and provenance

This project is MIT licensed. The fixed upstream GGML and tokenizer subset in
`third_party/llama.cpp` retains its own license and recorded
[vendor revision](third_party/llama.cpp/NANOVLLM_VENDOR_REVISION.md).
