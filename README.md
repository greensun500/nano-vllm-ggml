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

All numbers below are real end-to-end runs on `Qwen3.5-2B-Q4_0.gguf`, batch
size 1, greedy decoding, `repeat=3`, `warmup=1`. `pp` is prompt length and
`tg` is generated-token length. Compare configurations on the same device, not
absolute rates across devices.

| Platform | Workload | Best measured configuration | Prefill tok/s | Decode tok/s | Result |
| --- | --- | --- | ---: | ---: | --- |
| NVIDIA A100-SXM4-80GB, Vulkan | pp128 / tg128 | MTP K=3 + graph reuse | 4327.7 | **288.4** | **1.85×** baseline decode |
| NVIDIA A100-SXM4-80GB, Vulkan | pp2048 / tg512 | MTP K=3 + graph reuse | 11469.7 | **279.6** | **1.84×** baseline decode |
| Arm Cortex-A720/A520, CPU | pp186 / tg541 | MTP K=1 | 87.60 | **35.58** | **+27.5%** decode, 100% acceptance |
| Mali-G720-Immortalis, Vulkan | pp186 / tg541 | Stable default: MTP off | **70.04** | **23.12** | Best end-to-end interactive path |

Mali MTP K=3 reaches `25.03 tok/s` pure decode in this workload, but its
prefill path is currently slower; it is therefore not the recommended
interactive default.

> These are development validation results. Release tags should rerun the same
> matrix on the stated hardware before making a tagged performance claim.

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
