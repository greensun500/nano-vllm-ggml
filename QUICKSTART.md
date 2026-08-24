# Quick Start

This guide is the operational companion to the short `nanovllm-ggml` project overview in the
[README](README.md). The qualified reference model is
`Qwen3.5-2B-Q4_0.gguf`; a complete GGUF includes all tokenizer metadata needed
by nanovllm-ggml.

## Prerequisites

- Python 3.10–3.12
- CMake 3.19+, a C++17 compiler, and Python development headers
- For Vulkan: Vulkan loader/development files and `glslc`
- For CUDA: a CUDA-capable build environment supported by the vendored GGML

Clone the repository; the fixed GGML source is vendored in-tree:

```bash
git clone https://github.com/greensun500/nano-vllm-ggml.git
cd nano-vllm-ggml
python3 -m pip install -e .
```

The Python import and CLI module remain `nanovllm`:
`from nanovllm import LLM` and `python3 -m nanovllm.cli.chat`. An editable or
wheel install also exposes `nanovllm-ggml-chat` and `nanovllm-ggml-bench`.

## Build one backend

The Python extension is emitted into `nanovllm/`, so build one GPU backend at a
time. Use a separate CMake build directory for each backend.

### CPU

```bash
NANOVLLM_NATIVE_BUILD_DIR=build/cpu \
./scripts/build_native_runtime.sh
```

### Vulkan

```bash
NANOVLLM_NATIVE_VULKAN=ON \
NANOVLLM_NATIVE_BUILD_DIR=build/vulkan \
NANOVLLM_VULKAN_GLSLC=/path/to/glslc \
./scripts/build_native_runtime.sh
```

If CMake cannot find Vulkan automatically, append the appropriate
`-DVulkan_INCLUDE_DIR=... -DVulkan_LIBRARY=...` arguments to the build script.

### CUDA

```bash
NANOVLLM_NATIVE_CUDA=ON \
NANOVLLM_NATIVE_BUILD_DIR=build/cuda \
./scripts/build_native_runtime.sh
```

CUDA Graph support is enabled automatically in a CUDA build.

## Chat

```bash
MODEL=/path/to/Qwen3.5-2B-Q4_0.gguf

PYTHONPATH=. python3 -m nanovllm.cli.chat "$MODEL" \
  --backend native_cpu \
  --max-model-len 8092 \
  --num-kvcache-blocks 32 \
  --threads 8 \
  --max-tokens 256 \
  --temperature 0
```

Replace `native_cpu` with `native_vulkan` or `native_cuda` after building that
backend. The interactive commands are `/help`, `/history`, `/reset`,
`/system <prompt>`, and `/exit`. Every completed turn also prints its prefill
and decode timing; the first generated token is produced by the prefill graph
and is shown separately in the generated-token breakdown.

## Enable MTP

MTP is optional and greedy-only. The target model verifies every proposal, so
the final output remains target-model output; acceptance rate determines
whether it improves throughput.

```bash
PYTHONPATH=. python3 -m nanovllm.cli.chat "$MODEL" \
  --backend native_cpu \
  --max-model-len 8092 --num-kvcache-blocks 32 --threads 8 \
  --enable-mtp --mtp-max-draft-tokens 1 --temperature 0
```

Measured starting points:

| Platform | Suggested first setting |
| --- | --- |
| Arm Cortex-A720 CPU | `--enable-mtp --mtp-max-draft-tokens 1` |
| NVIDIA Vulkan | Test `--enable-mtp --mtp-max-draft-tokens 3` |
| Mali-G720 Vulkan | Start MTP off; assess long-generation-only workloads separately |

Mali-G720 uses the qualified tile-tuned raw-MMQ prefill path by default. To
force the upstream pipeline while bisecting a driver or correctness issue, set
the explicit safety fallback:

```bash
GGML_VK_DISABLE_MALI_MMQ_TUNE=1 PYTHONPATH=. python3 -m nanovllm.cli.chat "$MODEL" \
  --backend native_vulkan --max-model-len 8092 --num-kvcache-blocks 32
```

Mali Vulkan graph reuse is a diagnostic experiment, not a generic default:

```bash
--mali-experimental-graph-reuse
```

## Benchmark

```bash
PYTHONPATH=. python3 -m nanovllm.cli.bench "$MODEL" \
  --backend native_cpu \
  --max-model-len 8092 --num-kvcache-blocks 32 --threads 8 \
  --prompt-len 512 --gen-len 128 \
  --repeat 3 --warmup 1 --temperature 0 --json
```

For an MTP A/B run, append:

```bash
--enable-mtp --mtp-max-draft-tokens 1
```

The JSON reports prefill/decode/processed/generated rates, MTP drafted and
accepted tokens, verification steps, graph-cache statistics, and compiled GGML
build information. Keep model, clock/power policy, thread affinity, driver,
prompt/generation lengths and repeat/warmup fixed across comparisons.

## Verify a source build

```bash
python3 -m pytest -q tests
ctest --test-dir build/cpu --output-on-failure
```

Real-model oracle tests are opt-in because they load the GGUF:

```bash
env PYTHONPATH=. \
  NANOVLLM_TEST_QWEN35_GGUF="$MODEL" \
  NANOVLLM_TEST_QWEN35_TOKENIZER=/optional/reference/tokenizer/path \
  NANOVLLM_TEST_CPU_THREADS=8 \
  python3 -m pytest -q tests/test_native_qwen35_cpu_oracle.py
```

The reference tokenizer path is used only to compare historical token traces;
normal runtime use does not require it.

## Common issues

- **`nanovllm._C` is missing or has the wrong backend**: rerun the build for the
  backend you are about to use. The extension in `nanovllm/` is shared.
- **GGUF validation fails**: this release qualifies Qwen3.5-2B-Q4_0. Other
  Qwen3.5 variants may not yet satisfy every native graph/cache contract.
- **MTP is slower**: compare `MTP-off`, K=1 and K=3 on your device. Acceptance
  alone is not a performance metric; verification and prefill cost matter.
