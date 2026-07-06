# nano-vLLM + llama.cpp CPU/Vulkan v1.0 说明

本文档记录当前 v1.0 版本所需源码、构建要求、部署方式、运行命令和验证方法。
# v1.0性能：
metric              tokens        time(s)      tok/s
prefill                  384      23.995      16.00
decode steps              93       5.984      15.54
generated total           96      29.979       3.20

## 1. 版本目标

本版本把 llama.cpp 的 CPU/Vulkan 执行能力接入 nano-vLLM，但不是把 llama.cpp 当作完整黑盒推理器使用。

当前架构保持 nano-vLLM 统一管理：

- `Sequence` 请求生命周期
- `Scheduler` prefill/decode 调度
- `BlockManager` paged KV block 分配、释放、prefix cache
- `BackendExecutionPlan` 统一执行计划
- Python 侧 sampling
- CLI chat 和 benchmark

llama.cpp 后端负责：

- 加载 GGUF 模型
- 使用 GGUF tokenizer 分词/反分词
- 创建 CPU 或 Vulkan context
- 按 nano-vLLM 指定的 KV cell 写入/读取 KV cache
- 返回 logits 给 nano-vLLM 采样

## 2. 源码目录

本版本涉及两个仓库：

```text
/home/kevin/kevin_prj/nano-vllm
/home/kevin/kevin_prj/llama.cpp
```

远程开发板建议部署目录：

```text
/home/cix/nano-vlm/nano-vllm
/home/cix/nano-vlm/llama.cpp
```

## 3. nano-vLLM 关键文件

后端抽象：

```text
nanovllm/backends/base.py
nanovllm/backends/__init__.py
nanovllm/backends/llamacpp/runner.py
```

Engine 和调度：

```text
nanovllm/config.py
nanovllm/engine/llm_engine.py
nanovllm/engine/model_runner.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
```

命令行工具：

```text
nanovllm/cli/chat.py
nanovllm/cli/bench.py
```

示例和脚本：

```text
examples/llamacpp_backend.py
scripts/build_llamacpp_cpu.sh
scripts/build_llamacpp_vulkan.sh
```

文档：

```text
README.md
LEARNING_GUIDE.zh.md
NANOVLLM_LLAMA_CPP_V1.0.zh.md
```

## 4. llama.cpp 关键文件

新增 nano-vLLM shim：

```text
src/nano-vllm-ext.h
src/nano-vllm-ext.cpp
src/nano-vllm-backend.cpp
```

改造 llama.cpp 内部 KV/cache/context：

```text
src/CMakeLists.txt
src/llama-context.cpp
src/llama-kv-cache.h
src/llama-kv-cache.cpp
```

Vulkan descriptor set 修复：

```text
ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

## 5. Python 运行要求

推荐 Python：

```text
Python 3.10 - 3.12
```

nano-vLLM 依赖：

```text
torch>=2.4.0
triton>=3.0.0
transformers>=4.51.0
flash-attn
numpy
xxhash
```

说明：

- CUDA 后端仍需要 PyTorch、Triton、FlashAttention。
- llama.cpp CPU/Vulkan 后端主要依赖 `numpy` 和 Python 标准库 `ctypes`。
- 若远程板没有 `xxhash`，当前代码会 fallback 到 `hashlib.blake2b`。
- llama.cpp tokenizer 后端不依赖 transformers。

## 6. C/C++ 构建要求

远程 ARM 开发板建议原生编译，不建议交叉编译。

基础工具：

```text
cmake
gcc/g++
make 或 ninja
python3
```

CPU 后端建议硬件/编译能力：

```text
ARMv9-A
i8mm
dotprod
sve
Kleidiai
```

Vulkan 后端要求：

```text
libvulkan.so
Vulkan headers
glslc
Mali Vulkan driver
```

当前验证设备：

```text
CPU: ARMv9.2
GPU: Mali-G720-Immortalis
OS: Linux aarch64
```

## 7. llama.cpp CPU 构建

在远程板执行：

```bash
cd /home/cix/nano-vlm/nano-vllm
LLAMA_CPP_DIR=/home/cix/nano-vlm/llama.cpp ./scripts/build_llamacpp_cpu.sh
```

脚本核心参数：

```bash
cmake .. \
  -DLLAMA_CURL=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_VULKAN=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF \
  -DGGML_F16C=OFF \
  -DGGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod+sve \
  -DGGML_CPU_KLEIDIAI=ON
```

输出库：

```text
/home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so
```

## 8. llama.cpp Vulkan 构建

在远程板执行：

```bash
cd /home/cix/nano-vlm/nano-vllm
LLAMA_CPP_DIR=/home/cix/nano-vlm/llama.cpp ./scripts/build_llamacpp_vulkan.sh
```

脚本核心参数：

```bash
cmake .. \
  -DLLAMA_CURL=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_VULKAN=ON \
  -DVulkan_LIBRARY=/usr/lib/aarch64-linux-gnu/libvulkan.so \
  -DVulkan_INCLUDE_DIR=/usr/include \
  -DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv9-a+sve
```

输出库：

```text
/home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so
```

## 9. 模型要求

llama.cpp 后端使用 GGUF 模型。

当前验证模型：

```text
/home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf
```

CUDA 后端仍使用 HF safetensors 格式。

## 10. Chat CLI

CPU：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_cpu \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so \
  --threads 8 \
  --threads-batch 8
```

Vulkan：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --threads 8 \
  --threads-batch 8 \
  --ubatch-size 512
```

交互命令：

```text
/reset              清空历史
/system <prompt>   修改 system prompt
/history            查看历史
/exit 或 /quit      退出
```

## 11. Benchmark CLI

CPU：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_cpu \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --threads 8 \
  --threads-batch 8
```

Vulkan：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --threads 8 \
  --threads-batch 8 \
  --ubatch-size 512
```

JSON 输出：

```bash
PYTHONPATH=. python -m nanovllm.cli.bench ... --json
```

Benchmark 指标：

```text
prefill_tok_s       prompt/prefill 输入 token 吞吐
decode_tok_s        decode step token 吞吐
generated_tok_s     端到端生成 token 吞吐
load_s              模型加载时间
```

默认 benchmark 会扰动 synthetic prompt，避免 prefix cache 影响 prefill 结果。

如需测试 prefix cache：

```bash
--use-prefix-cache
```

## 12. 关键运行参数

常用环境变量：

```text
NANOVLLM_GGUF_MODEL
NANOVLLM_BACKEND
NANOVLLM_LLAMA_BACKEND_LIB
NANOVLLM_THREADS
NANOVLLM_THREADS_BATCH
NANOVLLM_GPU_LAYERS
NANOVLLM_UBATCH_SIZE
NANOVLLM_MAX_MODEL_LEN
NANOVLLM_MAX_NUM_SEQS
NANOVLLM_MAX_NUM_BATCHED_TOKENS
```

Vulkan 默认：

```text
n_gpu_layers = -1
offload_kqv = true
n_ubatch = min(max_num_batched_tokens, 512)
```

Mali-G720 上建议显式使用：

```text
--ubatch-size 512
```

## 13. v1.0 已验证项目

本地验证：

```text
python -m compileall -q nanovllm examples
cmake configure llama.cpp nanollama_backend
cmake --build target nanollama_backend
```

远程板验证：

```text
CPU libnanollama_backend.so 编译通过
Vulkan libnanollama_backend.so 编译通过
CPU chat smoke test 通过
Vulkan chat smoke test 通过
CPU benchmark smoke test 通过
Vulkan benchmark smoke test 通过
```

Vulkan 验证日志中应出现：

```text
Mali-G720-Immortalis
using device Vulkan0
offloaded 37/37 layers to GPU
```

## 14. 已知限制

- llama.cpp 后端 v1.0 只支持 single-process，不支持 nano-vLLM tensor parallel。
- CUDA 后端保留原 nano-vLLM 路径。
- llama.cpp 后端使用 GGUF；HF safetensors 仍走 CUDA 后端。
- 当前 sampling 在 Python 侧完成，主要支持 temperature sampling。
- prefix cache 当前按完整 block 复用。
- 长上下文、preemption、chunked prefill 已接入架构，但仍建议继续补更完整的压力测试。
- Vulkan 后端在 Mali 上建议控制 `n_ubatch`，避免驱动/descriptor 资源压力。

## 15. v1.0 Git 标记建议

本版本需要同时标记两个仓库：

```text
nano-vllm: v1.0
llama.cpp: v1.0
```

两个 tag 共同组成当前 nano-vLLM + llama.cpp CPU/Vulkan 集成版。
