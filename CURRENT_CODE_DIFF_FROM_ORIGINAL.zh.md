# 当前代码相对原始 nano-vLLM 的完整修改导览

本文档用于逐点学习当前代码和最原始 nano-vLLM 代码之间的差异。

对比基线：

- 原始 nano-vLLM 基线：`origin/main` / `native-upstream`，commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- 当前分支：`qwen35-native-runtime`，v3.73 CPU 权重 repack 加载优化工作区（上一个版本提交 `b413650 v3.72: trim MTP draft hidden lifetime`）

总体规模：

- 相对原始基线的具体规模会随文档、测试和子模块状态变化；学习时应以 `git diff origin/main...HEAD --stat` 重新计算。

一句话总结：原始 nano-vLLM 是一个以 CUDA/PyTorch/Triton 为主的轻量 vLLM 实现；当前 v3.7 在保留原 CUDA 路径的同时，形成了 in-tree Qwen3.5 native runtime，支持 GGUF 加载、GGML CPU/Vulkan/CUDA backend、PagedKV、hybrid recurrent state、内置 MTP、会话保留、公平调度、benchmark/chat CLI、CPU graph reuse、可选 FlashAttention/GDN snapshot/MTP prefill fusion、可选 CUDA Graph 和 MTP/memory profiling。

## 1. 修改时间线

这些提交展示了功能逐步演进的路线，学习时建议按这个顺序看。

| 阶段 | commit | 核心内容 |
|---|---|---|
| v1.0 | `01f1373` | 加入外部 llama.cpp CPU/Vulkan backend，Python scheduler 仍在 nano-vLLM，执行通过 `libnanollama_backend.so` |
| v1.0.1 | `e1b519f` | 增加执行流程文档 `run_pipeline.md` |
| v2.0 | `850d6fa` | llama.cpp paged KV / zero-copy 方向改造，减少 Python 和 llama.cpp KV 之间的语义断层 |
| Qwen3.5 stage1 | `03bd202` | 先让 Qwen3.5-2B 通过 llama.cpp staged backend 跑通基础推理 |
| Qwen3.5 stage2 | `17cc6c1` | 加入 Qwen3.5 hybrid paged KV 阶段 |
| Qwen3.5 stage3 | `4d45f6b` | 加入 Qwen3.5 内置 MTP 的 staged 支持 |
| v3.0 | `a27ba58` | 切到 in-tree Qwen3.5 native runtime，不再依赖外部 `libllama`/`llama_context` |
| v3.1 | `fb5a028` | 优化 MTP KV maintenance，把 MTP 维护拆成 KV-only 路径 |
| v3.2 | `65f8b79` | 引入 `TargetChunkGraph`，批量执行 target prefill/verification |
| v3.3 | `a7d4b24` | Arm/Mali native 执行优化，包括 prefill chunk policy 和 Arm CPU 构建参数 |
| v3.4 | `0220adb` | llama.cpp 子模块切到带 Mali Q4_0 DMMV shape routing 的修订 |
| v3.5 | `71bd889` | CPU 单序列 decode/MTP persistent graph bucket reuse；Vulkan fallback |
| v3.6 | `12272ae` | native session cache、decode 饥饿保护、native CUDA、memory metrics |
| v3.6 fix | `cd6ced0` | 恢复 Qwen3.5 attention query gate |
| v3.6 profile | `aabd733` | 暴露 draft/verification/KV catch-up 分段计时 |
| v3.7 | `044e92e` | opt-in FlashAttention、连续 GDN snapshot 写回、MTP prefill 合图、opt-in CUDA Graph |
| v3.71 | `21474dd` | 去除 fallback graph 的重复同步；plan speculative-tail 校验改为无 slot 向量分配的逻辑范围检查 |
| v3.72 | `b413650` | 仅在 MTP draft 的调用方需要时将 hidden 声明为 GGML output；cache key 区分其活跃区间 |
| v3.73 working tree | 未提交 | CPU loader 将符合 GGML 原生 ISA/shape 条件的 Q4_0/Q6_K 二维矩阵放入 CPU_REPACK，其余 tensor 保持 default buffer |

## 2. 目录级总览

### 2.1 新增的 C++ native runtime

原始 nano-vLLM 没有 `csrc/`。当前新增了完整 C++ extension：

```text
csrc/
  native_module.cpp
  python/qwen35_runtime_binding.{h,cpp}
  runtime/
    backend.{h,cpp}
    gguf_loader.{h,cpp}
    graph_executor.{h,cpp}
    paged_kv.{h,cpp}
    qwen35_model.{h,cpp}
    qwen35_runtime.{h,cpp}
    recurrent_state.{h,cpp}
  models/qwen35/
    graph.{h,cpp}
    ops.{h,cpp}
    weights.{h,cpp}
```

这是当前版本和原始版本最大的差异。原始项目的模型执行基本在 Python/PyTorch/CUDA 内；现在 Qwen3.5 native 路径把 GGUF 权重、GGML graph、CPU/Vulkan/CUDA backend、KV cache、recurrent state 都放到了 C++ extension 里。

### 2.2 新增 backend 抽象

新增：

```text
nanovllm/backends/
  __init__.py
  base.py
  llamacpp/
    __init__.py
    runner.py
  native/
    __init__.py
    runner.py
    runtime.py
```

原始代码里 `LLMEngine` 直接创建 `ModelRunner`。当前改成：

```text
LLMEngine
  -> create_backend(config)
       cuda             -> ModelRunner
       llamacpp_cpu     -> LlamaCppRunner
       llamacpp_vulkan  -> LlamaCppRunner
       native_cpu       -> NativeRunner
       native_vulkan    -> NativeRunner
```

### 2.3 新增构建系统和第三方 GGML

新增：

```text
.gitmodules
CMakeLists.txt
scripts/build_native_runtime.sh
scripts/build_llamacpp_cpu.sh
scripts/build_llamacpp_vulkan.sh
third_party/llama.cpp
```

当前 native runtime 只编译 llama.cpp 的 GGML 子目录，不链接 `libllama`，不创建 `llama_context`。

### 2.4 新增 CLI、测试和文档

新增：

```text
nanovllm/cli/chat.py
nanovllm/cli/bench.py
nanovllm/cli/__init__.py
examples/llamacpp_backend.py
run_pipeline.md
LEARNING_GUIDE.zh.md
CURRENT_RUNTIME_FLOW.zh.md
CURRENT_VERSION_CHANGES.zh.md
tests/native/test_graph_executor.cpp
tests/test_native_backend_config.py
tests/test_native_cli.py
tests/test_native_qwen35_cpu_oracle.py
tests/test_native_qwen35_long_context_oracle.py
tests/test_native_runner.py
tests/test_native_runtime.py
tests/test_qwen35_stages.py
```

## 3. Python 配置层修改

文件：`nanovllm/config.py`

原始版本：

- 只面向本地 Hugging Face 模型目录；
- 默认只走 CUDA；
- 直接 `AutoConfig.from_pretrained(self.model)`；
- 只有单个 `eos`；
- prefix cache / preemption 默认一直按原 CUDA 路径启用；
- 不支持 GGUF、llama.cpp、native CPU/Vulkan、MTP。

当前版本新增字段：

```python
backend: str = "cuda"
model_format: str = "hf"
gguf_model: str | None = None
tokenizer: str | None = None
tokenizer_backend: str = "hf"
device_config: dict | None = None
eos_token_ids: tuple[int, ...] = ()
enable_prefix_cache: bool | None = None
enable_preemption: bool | None = None
enable_mtp: bool = False
mtp_max_draft_tokens: int = 3
enable_graph_reuse: bool = True
```

新增校验逻辑：

- `backend` 必须是：
  - `cuda`
  - `llamacpp_cpu`
  - `llamacpp_vulkan`
  - `native_cpu`
  - `native_vulkan`
  - `native_cuda`
- CUDA 后端仍要求 HF model；
- GGUF 后端要求 `gguf_model`；
- native 后端要求本地 HF tokenizer，因为 native runtime 不使用 llama.cpp tokenizer；
- staged llama.cpp 后端可以使用 `tokenizer_backend="llamacpp"`；
- native / llama.cpp 后端暂不支持 prefix cache 和 preemption；
- MTP 使用 native GGUF 后端；CPU/Vulkan 已完成实测，CUDA 尚待 NVIDIA 实机验收；
- `max_num_batched_tokens` 必须能容纳 MTP verification window；
- `num_kvcache_blocks` 默认由 `max_model_len` 和 `kvcache_block_size` 推导。

学习重点：

- `Config` 从“CUDA 模型配置”变成“全后端运行配置”；
- `hf_config` 改成延迟/条件加载，避免 GGUF 路径强依赖 transformers model config；
- `eos` 从单 token 改成 `eos_token_ids`，用于兼容 Qwen 的多个结束 token。

## 4. BackendExecutionPlan 抽象

文件：`nanovllm/backends/base.py`

新增核心数据结构：

```python
BackendExecutionPlan:
  mode
  input_ids
  positions
  seq_ids
  scheduled_token_counts
  block_tables
  slot_mapping
  context_lens
  num_cached_tokens
  temperatures
```

原始版本里，CUDA `ModelRunner` 自己从 `Sequence` 对象里拆：

```text
seqs + is_prefill -> prepare_prefill/prepare_decode
```

当前版本改成：

```text
Scheduler 选出 seqs
  -> build_execution_plan(seqs, is_prefill, block_size)
  -> BackendRunner.run(plan)
```

这个抽象是后面所有后端共存的基础。CUDA、llama.cpp、native runtime 都吃同一份 plan。

新增返回结构：

```python
BackendExecutionResult:
  token_ids: list[list[int]]
  draft_token_counts: list[int] | None
```

为什么 `token_ids` 是二维？因为普通 decode 一轮每个请求只返回 1 个 token，而 MTP 一轮可能返回多个 token。

## 5. LLMEngine 修改

文件：`nanovllm/engine/llm_engine.py`

主要变化：

1. 后端创建方式改变

原始：

```python
self.model_runner = ModelRunner(config, 0, self.events)
```

当前：

```python
self.model_runner = create_backend(config, 0, self.events)
```

非 CUDA 后端不会创建 torch multiprocessing 子进程。

2. tokenizer 选择改变

新增两种 tokenizer backend：

- `hf`：使用 `AutoTokenizer`；
- `llamacpp`：通过后端调用 `tokenize` / `detokenize`。

native Qwen3.5 路径当前要求 `hf` tokenizer，但会额外把 Qwen 的 `<|endoftext|>` / `<|im_end|>` 放进 EOG 集合。

3. step 执行方式改变

原始：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

当前：

```python
plan = build_execution_plan(seqs, is_prefill, block_size)
result = self.model_runner.call("run", plan)
self.scheduler.postprocess(seqs, result, is_prefill)
```

4. 增加后端 block release

新增：

```python
flush_backend_releases()
```

Scheduler 释放 Python block 后，会把释放的 block id 和 sequence id 通知 native/llama.cpp 后端，便于后端清理自己的 KV cache 或 sequence slot。

学习重点：

- `LLMEngine` 从“CUDA runner 调用者”变成“通用 backend orchestrator”；
- 生成主循环还在 Python 层，这符合我们“nano-vLLM 作为调度框架”的目标。

## 6. Scheduler 修改

文件：`nanovllm/engine/scheduler.py`

原始版本：

- 每轮 decode 默认每个 running sequence 追加 1 token；
- 只处理单个 `eos`；
- preemption 默认存在；
- block release 只在 Python 内部发生；
- postprocess 只接收 `list[int]`。

当前版本：

1. 支持多个 EOG token

```python
self.eos_token_ids = frozenset(config.eos_token_ids)
```

2. 支持关闭 preemption

native / llama.cpp 后端不支持 prefix cache/preemption，因此 scheduler 在 cache 不够时直接报错，而不是尝试抢占。

3. 为 MTP 预留 speculative KV slots

```python
self.speculative_tokens = config.mtp_max_draft_tokens if config.enable_mtp else 0
```

decode 调度时，`can_append` / `may_append` 会按 `1 + K` 的窗口预留 block，避免 MTP draft/verification 写出当前已分配 KV 范围。

4. postprocess 支持一轮多个 token

普通后端：

```text
[[next_token]]
```

MTP 后端：

```text
[[target_token, accepted_draft_1, accepted_draft_2, ...]]
```

Scheduler 会逐个 append token，并在遇到 EOG 或 max_tokens 时停止。

5. 记录 block release

```python
self.block_releases: list[tuple[list[int], int]]
```

被 preempt 或 finish 的 sequence 会把释放的 block id 记录下来，下一步由 `LLMEngine.flush_backend_releases()` 通知后端。

学习重点：

- MTP 不是只改后端，scheduler 也必须知道“一轮可能产出多个 token”；
- 如果不预留 speculative slots，后端 MTP 可能会写到没有分配的 KV block。

## 7. BlockManager 修改

文件：`nanovllm/engine/block_manager.py`

主要变化：

1. `xxhash` 变成可选依赖

如果没有 `xxhash`，fallback 到 `hashlib.blake2b`。这让非 CUDA / native 开发环境更容易跑测试。

2. 支持关闭 prefix cache

新增构造参数：

```python
enable_prefix_cache: bool = True
```

关闭后：

- `can_allocate()` 不查 hash；
- `hash_blocks()` 直接返回；
- 每个请求都独占自己分配的 block。

3. deallocate 返回释放 block id

原始：

```python
def deallocate(seq): ...
```

当前：

```python
def deallocate(seq) -> list[int]
```

返回的 block id 会传给后端释放 native KV。

4. append 支持 speculative token 数

新增：

```python
can_append(seq, speculative_tokens=0)
may_append(seq, speculative_tokens=0)
```

这是为了 MTP 一轮可能写多个位置。

## 8. CUDA ModelRunner 兼容改造

文件：`nanovllm/engine/model_runner.py`

原始 CUDA runner 的核心功能仍保留：

- torch distributed；
- CUDA model loading；
- KV cache allocation；
- CUDA graph capture；
- sampler；
- 多进程 tensor parallel。

当前修改主要是适配统一 backend contract：

- `run(seqs, is_prefill)` 改为 `run(plan)`；
- `prepare_prefill()` / `prepare_decode()` 改为从 `BackendExecutionPlan` 读取输入；
- `prepare_block_tables()` 从 `plan.block_tables` 读取；
- 返回 `BackendExecutionResult([[token_id], ...])`；
- 增加空实现 `release_blocks()`，保持和其他后端接口一致；
- 增加 `shutdown()` alias。

文件里还增加了一些中文注释，这些不改变逻辑，只是学习辅助。

学习重点：

- CUDA 路径被“接口适配”，不是被重写；
- CUDA graph 逻辑仍在原 `ModelRunner.capture_cudagraph()` 里。

## 9. Sampler 和 SamplingParams 修改

文件：

- `nanovllm/layers/sampler.py`
- `nanovllm/sampling_params.py`

原始版本禁止 greedy：

```python
assert self.temperature > 1e-10
```

当前版本默认 greedy：

```python
temperature: float = 0.0
```

Sampler 新增：

- `temperature == 0` 时直接 `argmax`；
- `temperature > 0` 时仍使用原来的 Gumbel-style sampling。

原因：

- Qwen3.5 MTP 第一版只支持 greedy；
- correctness oracle 也需要 deterministic greedy trace。

## 10. llama.cpp staged backend

文件：`nanovllm/backends/llamacpp/runner.py`

这是 v1/v2/stage1~stage3 的遗留迁移路径，目前 README 里已经标注为 migration oracle。

它的形态：

```text
Python LlamaCppRunner
  -> ctypes load libnanollama_backend.so
  -> nano_llama_backend_create()
  -> nano_llama_backend_run()
  -> nano_llama_backend_run_mtp()
```

新增能力：

- 支持 `llamacpp_cpu` / `llamacpp_vulkan`；
- 支持 GGUF tokenizer 的 tokenize/detokenize；
- 支持 eog token 查询；
- 支持 staged backend 的 MTP 调用；
- 将 `BackendExecutionPlan` 转成 C ABI `NanoLlamaKVPlan`；
- 维护 Python sequence id 到 native sequence slot 的映射；
- 统计 MTP drafted/accepted/verification steps；
- 支持 `release_blocks()` 通知后端释放 paged KV。

学习重点：

- 这是“链接外部 llama.cpp 库”的路线；
- v3.0 之后的 native runtime 不再把它作为主路径，但它仍可用于迁移对照。

## 11. NativeRunner

文件：`nanovllm/backends/native/runner.py`

这是当前主路径的 Python adapter。

职责：

1. 加载 extension 类型

```python
nanovllm._C.Qwen35Runtime
```

2. 把 `Config` 转成 C++ runtime options

包括：

- GGUF model path；
- backend kind：`cpu` / `vulkan`；
- max model len；
- max batched tokens；
- max seqs；
- block size；
- num blocks；
- CPU threads；
- Vulkan device index；
- MTP 开关；
- MTP draft 数；
- graph reuse 开关。

3. 把 `BackendExecutionPlan` 转成 native ABI dict

包括：

- int32 numpy arrays；
- flat block tables；
- native sequence slots；
- slot mapping；
- context lens；
- cached token counts。

4. greedy 校验

native Qwen3.5 当前只支持 greedy，所以 `temperatures` 必须全为 0。

5. MTP window 边界判断

如果快到 `max_model_len` 尾部，`K` 个 draft 放不下，就 fallback 到普通 target decode。

6. MTP 返回值拆分

C++ 返回：

```text
flat_token_ids
output_counts
draft_counts
```

Python 拆成：

```python
BackendExecutionResult(outputs, drafts)
```

7. graph reuse stats

新增：

```python
graph_reuse_stats()
```

用于 benchmark 输出 hits/misses/evictions/active entries。

## 12. Native extension 模块

文件：

- `csrc/native_module.cpp`
- `csrc/python/qwen35_runtime_binding.cpp`
- `csrc/python/qwen35_runtime_binding.h`

### 12.1 `native_module.cpp`

暴露底层 smoke/debug API：

- `build_info()`
- `available_backends()`
- `gguf_info(path)`
- `matmul_smoke()`
- `q4_0_matmul_smoke()`
- `qwen35_weight_load_smoke()`
- `qwen35_model_info()`

作用：

- 检查 extension 是否正确构建；
- 检查 CPU/Vulkan/CUDA backend 是否可用；
- 检查 GGUF 读取；
- 检查 Q4_0 matmul 是否能跑；
- 检查 Qwen3.5 权重 contract 和 tensor binding。

### 12.2 `qwen35_runtime_binding.cpp`

暴露 Python 类型：

```python
nanovllm._C.Qwen35Runtime
```

支持：

- 构造 runtime；
- `run(plan)`；
- `run_mtp(plan, token_capacity)`；
- `release_blocks(block_ids, sequence_ids, block_size)`；
- `shutdown()`；
- `graph_reuse_stats()`。

这个文件重点是 ABI 安全：

- Python dict 字段检查；
- buffer 类型检查；
- int32 范围检查；
- GIL release；
- C++ exception 转 Python exception。

## 13. C++ backend 封装

文件：

- `csrc/runtime/backend.h`
- `csrc/runtime/backend.cpp`

新增概念：

- `BackendKind`
  - CPU
  - Vulkan
  - CUDA
- `BackendConfig`
  - CPU threads
  - Vulkan/CUDA device index
- `Backend`
  - RAII 管理 `ggml_backend_t`
  - 查询 default buffer type
  - synchronize
  - backend kind/name/device info
- `BackendList`
  - CPU-only
  - Vulkan-with-CPU
  - CUDA-with-CPU

为什么 accelerator 模式仍有 CPU：

- GGML graph 里有些输入/metadata/host side tensor 更适合留在 CPU；
- compute node 可以放 Vulkan/CUDA，输入和小张量可以在 CPU 或由 scheduler 复制。

## 14. GraphExecutor

文件：

- `csrc/runtime/graph_executor.h`
- `csrc/runtime/graph_executor.cpp`

原始项目没有这个层。它是我们自己对 GGML graph scheduler 的封装。

职责：

- 创建 `ggml_backend_sched_t`；
- reserve/allocate graph；
- compute graph；
- reset scheduler；
- 设置 tensor backend；
- 处理 view tensor 的真实 source；
- 检查 compute node placement；
- Vulkan/CUDA placement audit；
- 确认所有 compute 节点是否真的在目标 accelerator。

它支撑两个路径：

1. fallback path：每轮构图、placement、allocate、同步 compute、reset；v3.71 在 compute 成功后直接 reset scheduler，避免已同步 graph 再执行一次 backend synchronize；异常路径保留同步 reset；v3.72 仅在调用方需要 host readback 时将 TokenGraph hidden 标为 output；
2. v3.5 persistent graph path：首次构图和 allocate，后续只更新输入 tensor 后 compute；v3.7 的 CUDA Graph 仅在显式开关时让 GGML capture/replay 稳定 CUDA graph。

学习重点：

- 这是“nano-vLLM 接管 GGML graph 生命周期”的关键；
- 不再把 graph 生命周期交给 `llama_context`。

## 15. GGUF loader 和 Qwen3.5 model contract

文件：

- `csrc/runtime/gguf_loader.h`
- `csrc/runtime/gguf_loader.cpp`
- `csrc/runtime/qwen35_model.h`
- `csrc/runtime/qwen35_model.cpp`

### 15.1 GGUF loader

封装：

- 打开 GGUF；
- mmap/读取 tensor data；
- 提供 metadata/tensor descriptor 给后续权重加载。

v3.73 的 CPU load 会先检测 GGML 是否支持 `CPU_REPACK`，并只为符合运行时 ISA 和 shape 条件的二维 Q4_0/Q6_K 权重建立一个 repack buffer；GGML 随后为尚未分配的 descriptor 建立 default buffer。这样既能让 Q4_0/Q6_K `mul_mat` 命中 GGML 已有的重排内核，又不会把没有可用 traits 的普通 tensor 放入 repack buffer。repack 上传要求完整 tensor 的单次写入，因而仅在加载期对这些矩阵临时整块读取；runtime 持有并在销毁前释放两类 persistent buffer，resident bytes 统计两者之和。

### 15.2 Qwen3.5 contract

当前只支持一个明确 profile：Qwen3.5-2B + bundled one-layer MTP。

关键校验：

- `architecture == qwen35`
- `block_count == 25`
- `nextn_predict_layers == 1`
- `main_layers == 24`
- `context_length == 262144`
- `embedding_length == 2048`
- `feed_forward_length == 6144`
- `vocab_size == 248320`
- `head_count == 8`
- `head_count_kv == 2`
- `key_length == 256`
- `value_length == 256`
- `rope.dimension_count == 64`
- `rope.dimension_sections == [11, 11, 10, 0]`
- `rope.freq_base == 10000000`
- `ssm.conv_kernel == 4`
- `ssm.state_size == 128`
- `ssm.group_count == 16`
- `ssm.time_step_rank == 16`
- `ssm.inner_size == 2048`
- `full_attention_interval == 4`

层类型：

```cpp
is_recurrent_layer(layer):
  main layer 且 (layer + 1) % full_attention_interval != 0

is_full_attention_layer(layer):
  main layer 且不是 recurrent

is_mtp_layer(layer):
  layer >= main_layers && layer < block_count
```

因此 24 层 target 中：

- 6 层 full attention；
- 18 层 recurrent Gated Delta Net；
- 第 25 个 block 是 bundled MTP layer。

## 16. Qwen3.5 weights binding

文件：

- `csrc/models/qwen35/weights.h`
- `csrc/models/qwen35/weights.cpp`

职责：

- 根据 `qwen35_model` 产生的 tensor specs 查找 GGUF tensor；
- 验证 shape/type；
- 把 tensor descriptor 绑定成 C++ `Qwen35Weights`；
- 区分 common/full-attention/recurrent/MTP 的权重集合。

支持的量化 profile：

- 当前 strict profile 是：
  - token embedding: `Q6_K`
  - linear weights: `Q4_0`
  - norm/bias/recurrent scalar/conv: `F32`
- 代码里也保留了 mixed supported profile 的识别。

学习重点：

- 这里解决的是“GGUF 文件里 tensor 名字和模型 graph 里的权重指针如何对应”；
- 如果模型文件名字、shape、量化类型不符合，就会在这里或 contract 阶段失败。

## 17. Qwen3.5 ops

文件：

- `csrc/models/qwen35/ops.h`
- `csrc/models/qwen35/ops.cpp`

这里封装了构图时反复使用的小算子组合，例如：

- RMSNorm；
- linear/matmul；
- Q/K norm；
- IMRoPE position 展开；
- gated/delta/recurrent 相关操作；
- greedy argmax 或 head 计算相关辅助。

学习重点：

- `ops` 不直接管理 runtime 状态；
- 它只是“用 GGML primitive 拼模型层”的积木。

## 18. Qwen3.5 graph

文件：

- `csrc/models/qwen35/graph.h`
- `csrc/models/qwen35/graph.cpp`

这是模型结构核心。

新增 graph 类型：

```cpp
TokenGraph
TargetChunkGraph
MtpKvUpdateGraph
```

### 18.1 TokenGraph

用于单 token graph，主要服务：

- target decode 早期路径；
- MTP draft；
- MTP layer 单步执行。

v3.5 后 `TokenGraph` 支持可选 `causal_mask`，用于 bucket graph reuse。

### 18.2 TargetChunkGraph

v3.2 引入，用于 T 个 token 共用一张 target graph。

输入：

```text
tokens[T]
positions[4T]
write_slots[T]
read_slots[C]
causal_mask[C,T]
```

输出模式：

- `None`：不读 token，必要时也不构建 output norm/head；
- `Last`：只读最后一列 greedy token；
- `All`：读所有 T 列 prediction，供 MTP verification。

用途：

- prefill chunk；
- ordinary decode T=1；
- MTP target verification T=K+1。

### 18.3 MtpKvUpdateGraph

v3.1 引入，用于 MTP KV-only maintenance。

它只做：

```text
token + shifted target hidden
  -> MTP merge/norm
  -> MTP K/V projection
  -> K norm + IMRoPE
  -> 写 MTP PagedKV
```

它不做：

- MTP attention；
- FFN；
- vocab head；
- greedy readback。

这是 MTP 性能优化的一部分。

### 18.4 full attention graph 差异

Qwen3.5 full attention 层不是简单自带连续 KV，而是：

- 当前 token 先算 K/V；
- 写入 PagedKV；
- 按 `read_slots` gather 历史 K/V；
- 做 Q/K norm；
- 做 IMRoPE；
- softmax attention；
- output projection。

v3.5 bucket 模式下，`read_slots` 可能被 padding 到 `n_kv_bucket`，并通过 mask 屏蔽 padding KV。

### 18.5 recurrent GDN graph 差异

recurrent 层会在 graph 内逐 token 更新：

- convolution state；
- delta state；
- gated output。

普通路径只提交最新 canonical state；MTP verification 需要生成 newest-first `K+1` snapshot planes。

## 19. PagedKV

文件：

- `csrc/runtime/paged_kv.h`
- `csrc/runtime/paged_kv.cpp`

职责：

- 为 Qwen3.5 full-attention target 层分配 KV；
- 为 MTP layer 分配独立 KV；
- 校验 physical slot；
- 根据 Python block table 展开 context/read/write slots；
- 在 graph 中创建 `SET_ROWS` / gather 相关 tensor；
- 清理 release blocks。

和原始 Python `BlockManager` 的关系：

- Python `BlockManager` 决定逻辑 block 分配；
- C++ `PagedKvCache` 持有真正的 target/MTP KV tensor；
- Python finish/preempt 后通过 `release_blocks()` 通知 C++ 清理。

## 20. RecurrentState

文件：

- `csrc/runtime/recurrent_state.h`
- `csrc/runtime/recurrent_state.cpp`

这是 Qwen3.5 相对 Qwen2.5/原始 Qwen3 dense CUDA 路径新增的核心状态。

状态内容：

- 每个 recurrent layer 有 convolution state；
- 每个 recurrent layer 有 delta state；
- 每个 sequence slot 有 canonical plane；
- MTP 开启时还有 `max_draft_tokens + 1` 个 snapshot plane。

MTP rollback 规则：

```text
plane 0 = verification 后最新状态
plane K = pending/target token 后的状态
接受 a 个 draft -> 选择 plane K-a
```

因此：

- accepted=0：回滚到只接受 pending token 的状态；
- accepted=K：使用最新状态；
- 中间值：选择对应 snapshot。

学习重点：

- 对 Qwen3.5，speculative decoding 不能只 rollback KV；
- recurrent state 必须和 token 接受结果严格同步。

## 21. Qwen35Runtime

文件：

- `csrc/runtime/qwen35_runtime.h`
- `csrc/runtime/qwen35_runtime.cpp`

这是 native 路径总控。

### 21.1 初始化

做的事情：

```text
validate options
create CPU、Vulkan or CUDA backend list
load GGUF
validate Qwen3.5 model contract
bind weights
allocate PagedKV
allocate RecurrentState
create fallback GraphExecutor
prepare sequence slots
prepare optional persistent graph cache
```

### 21.2 plan 校验

`validate_plan()` 会检查：

- prefill/decode 模式；
- n_tokens/n_seqs；
- block table；
- slot mapping；
- context lens；
- sequence slot；
- physical block/slot 是否有效；
- MTP window 是否越界；
- greedy temperature。

v3.71 的 speculative-tail 检查复用 PagedKV 的无分配逻辑范围校验：仍验证所需 block-table 前缀、block ID 和同 sequence block 唯一性，但不为尚未执行的 context 物化/丢弃 physical slot 向量。

### 21.3 普通 run

`run(plan)`：

- prefill：
  - 可能按 chunk 执行；
  - 更新 target KV；
  - 更新 recurrent state；
  - MTP 开启时维护 pending hidden 和 MTP KV；
  - 最后一段返回 greedy token。
- decode：
  - 执行 `TargetChunkGraph(T=1, Last)`；
  - 返回一个 greedy token；
  - MTP 开启但 window 不适合时，也会走普通 target step 并维护状态。

### 21.4 MTP run

`run_mtp(plan, token_capacity)`：

1. 读 pending token 和 old pending hidden；
2. 串行执行 K 步 MTP draft；
3. 构造 verification input：

```text
[pending_token, draft_1, draft_2, ..., draft_K]
```

4. target 一次验证 K+1 个位置；
5. 比较 target prediction 和 draft token；
6. 计算 accepted draft count；
7. recurrent state rollback；
8. accepted draft 的 MTP KV catch-up；
9. 更新 pending token / pending hidden / pending position；
10. 返回 accepted token 序列。

### 21.5 release_blocks

用于清理：

- target PagedKV；
- MTP PagedKV；
- recurrent state；
- sequence slot mapping；
- pending hidden。

### 21.6 v3.5/v3.7 persistent graph reuse

v3.5 新增、v3.7 保留：

- `Qwen35RuntimeOptions.enable_graph_reuse`
- `Qwen35GraphReuseStats`
- `PersistentGraphKind`
- `PersistentGraphKey`
- `PersistentGraphEntry`
- graph cache hits/misses/evictions/active entries

v3.7 只有在 CUDA build 显式具备 `GGML_CUDA_GRAPHS` 时才将同一机制扩展到 native CUDA；默认 build 仍是 CPU reuse、CUDA eager graph。GGML 的 graph capture 是 backend scope 行为，故此处用独立 build variant 而不是 runtime toggle。

固定图类型：

- MTP-off target decode：`TargetChunkGraph(T=1, Last)`
- MTP draft：`TokenGraph(T=1)`
- MTP target verification：`TargetChunkGraph(T=K+1, All, retain_hidden=true)`
- MTP KV catch-up：`MtpKvUpdateGraph(T=1..K)`

bucket：

```text
128, 256, 512, 1024, 2048, 4096, max_model_len
```

当前限制：

- 默认 build 只启用 CPU 单序列 persistent graph reuse；native CUDA 只有在独立 CUDA Graph build 中才允许复用稳定图；
- CUDA Graph build 必须同时显式传入 `NANOVLLM_NATIVE_CUDA=ON` 与 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON`；
- Vulkan 由于 padded-mask bucket attention 行为不稳定，当前 fallback 到旧路径；
- prefill/chunk 暂不强制缓存。

学习重点：

- bucket 不保存 KV cache，不复制权重；
- bucket 保存 graph metadata、scheduler allocation、临时 activation buffer、输入/输出 tensor 指针；
- 命中时只更新输入 tensor/mask/hidden，然后 `compute()`。

## 22. Native graph reuse 与 v3.6/v3.7 连带修改

v3.5 graph reuse 与后续 v3.6/v3.7 共涉及：

- `csrc/models/qwen35/graph.h`
- `csrc/models/qwen35/graph.cpp`
- `csrc/runtime/qwen35_runtime.h`
- `csrc/runtime/qwen35_runtime.cpp`
- `csrc/python/qwen35_runtime_binding.cpp`
- `csrc/native_module.cpp`
- `csrc/runtime/backend.h`
- `csrc/runtime/backend.cpp`
- `nanovllm/backends/native/runner.py`
- `nanovllm/config.py`
- `nanovllm/cli/chat.py`
- `nanovllm/cli/bench.py`

具体新增点：

- graph 支持 `force_causal_mask`；
- T=1 bucket decode 也可带 causal mask；
- `ggml_soft_max_ext()` 接收 mask；
- backend device info 更完整；
- Python binding 暴露 `enable_graph_reuse` 和 `graph_reuse_stats()`；
- CLI 增加 `--no-graph-reuse`；
- benchmark 输出 graph cache 统计。

v3.6 在这条 native 链路上继续增加：

- `BackendKind::Cuda`、`[cuda, cpu]` backend layout 和 CUDA placement audit；
- `Qwen35Runtime.memory_stats()`，统计 weights、PagedKV、recurrent state、graph metadata；
- `Qwen35Runtime.mtp_profile_stats()`，统计 draft、verification、KV catch-up 的 calls/tokens/setup/elapsed；
- Python binding 和 `NativeRunner` 的相应包装；
- attention weighted value 恢复乘 `sigmoid(query_gate)`，再进入 output projection。

## 23. 构建系统修改

文件：

- `CMakeLists.txt`
- `scripts/build_native_runtime.sh`
- `.gitmodules`
- `third_party/llama.cpp`

### 23.1 CMakeLists

新增 native extension target：

```text
Python3_add_library(_C MODULE WITH_SOABI ...)
```

链接：

```text
ggml::ggml
```

编译定义：

- `NANOVLLM_GGML_COMMIT`
- `NANOVLLM_GGML_BASE_COMMIT`
- `NANOVLLM_NATIVE_HAS_VULKAN`

还增加 native C++ test：

```text
nanovllm_graph_executor_test
```

### 23.2 llama.cpp 子模块

新增 submodule：

```text
third_party/llama.cpp -> https://github.com/ggml-org/llama.cpp.git
```

当前 CMake 强制检查 gitlink commit，避免构建时静默使用错误版本。

当前记录：

- 官方 GGML base：`91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- 实际子模块：`5ed33380b4679533243ca45e172804d5ddfe59ec`

### 23.3 build script

`scripts/build_native_runtime.sh` 做了：

- CPU/Vulkan/CUDA build dir 选择；
- 可选 `NANOVLLM_NATIVE_VULKAN=ON` 或 `NANOVLLM_NATIVE_CUDA=ON`（互斥）；
- Arm aarch64 dotprod/i8mm 自动探测；
- 可选 `NANOVLLM_VULKAN_GLSLC`；
- 编译 `_C`；
- 编译并运行 `graph_executor` C++ test；
- 打印 build info 和 available backends。

## 24. CLI 修改

文件：

- `nanovllm/cli/chat.py`
- `nanovllm/cli/bench.py`

原始项目只有简单 `bench.py`，没有 backend-aware CLI。

### 24.1 chat CLI

支持：

- `--backend cuda|llamacpp_cpu|llamacpp_vulkan|native_cpu|native_vulkan`
- `--gguf-model`
- `--tokenizer`
- `--library-path`
- `--max-model-len`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--num-kvcache-blocks`
- `--threads`
- `--device-index`
- `--enable-mtp`
- `--mtp-max-draft-tokens`
- `--no-graph-reuse`
- `--temperature`

### 24.2 bench CLI

支持：

- pp/tg/pp+tg 风格的测试；
- repeat/warmup；
- RSS 统计；
- MTP drafted/accepted/verification stats；
- graph cache hits/misses/evictions/active entries；
- GGML commit / Vulkan compiled 信息；
- JSON 输出。

学习重点：

- benchmark 不再只面向 CUDA；
- 它已经成为 native CPU/Vulkan/CUDA correctness/performance 验收入口；当前实测数据主要来自 CPU 与 Mali Vulkan。

## 25. README 和学习文档

修改/新增：

- `README.md`
- `LEARNING_GUIDE.zh.md`
- `run_pipeline.md`
- `CURRENT_RUNTIME_FLOW.zh.md`
- `CURRENT_VERSION_CHANGES.zh.md`

作用：

- README 增加 native Qwen3.5 CPU/Vulkan/CUDA runtime 说明；
- `LEARNING_GUIDE.zh.md` 是原始 nano-vLLM 源码学习路线；
- `run_pipeline.md` 记录早期 llama.cpp backend 执行流程；
- `CURRENT_RUNTIME_FLOW.zh.md` 记录当前 v3.7 runtime、session、MTP 和图优化流程；
- `CURRENT_VERSION_CHANGES.zh.md` 记录 v3.5 到 v3.7 的文件变更、测试和性能结果。

历史上曾有更多中间文档，例如 v1/v2/v3.0 专文，v3.1 后被合并整理到当前两个 `CURRENT_*` 文档中。

## 26. 测试修改

新增：

```text
tests/native/test_graph_executor.cpp
tests/test_native_backend_config.py
tests/test_native_cli.py
tests/test_native_qwen35_cpu_oracle.py
tests/test_native_qwen35_long_context_oracle.py
tests/test_native_runner.py
tests/test_native_runtime.py
tests/test_qwen35_stages.py
```

覆盖内容：

- backend config 校验；
- CLI 参数构造；
- native extension build/smoke API；
- graph executor placement；
- NativeRunner plan 转换；
- MTP result 处理；
- Qwen3.5 CPU oracle；
- 长上下文 bucket 升档 oracle；
- staged Qwen3.5 逻辑；
- MTP 预留 block、greedy-only、prefix/preemption 禁用等行为。

`tests/test_native_qwen35_long_context_oracle.py` 已纳入版本控制；v3.6 继续扩展 native backend、session 生命周期、调度公平性、memory stats 和 attention gate 的测试覆盖。

## 27. pyproject 修改

文件：`pyproject.toml`

原始依赖里 CUDA 相关依赖是默认依赖：

- torch
- triton
- flash-attn

当前改成：

- base dependencies：
  - transformers
  - numpy
  - xxhash
- optional `cuda` dependencies：
  - torch
  - triton
  - flash-attn

新增 console scripts：

```toml
nanovllm-chat = "nanovllm.cli.chat:main"
nanovllm-bench = "nanovllm.cli.bench:main"
```

原因：

- CPU/Vulkan native path 不应该强制安装 CUDA 栈；native CUDA 使用单独构建环境。
- CLI 需要可安装入口。

## 28. example.py 修改

文件：`example.py`

只增加了一些中文解释性注释，不影响核心逻辑。

## 29. 当前版本相比原始版本的核心执行流变化

### 29.1 原始 CUDA 路径

```text
LLM.generate()
  -> LLMEngine
  -> Scheduler.schedule()
  -> ModelRunner.run(seqs, is_prefill)
  -> PyTorch Qwen3ForCausalLM
  -> Sampler
  -> Scheduler.postprocess(token_ids)
```

### 29.2 当前通用路径

```text
LLM.generate()
  -> LLMEngine
  -> Scheduler.schedule()
  -> build_execution_plan()
  -> BackendRunner.run(plan)
       cuda            -> ModelRunner
       llamacpp_*      -> external libnanollama_backend.so
       native_*        -> nanovllm._C.Qwen35Runtime
  -> BackendExecutionResult
  -> Scheduler.postprocess(result)
  -> flush_backend_releases()
```

### 29.3 当前 native Qwen3.5 路径

```text
Python scheduler
  -> NativeRunner
  -> Qwen35Runtime
  -> TargetChunkGraph / TokenGraph / MtpKvUpdateGraph
  -> PagedKV + RecurrentState
  -> GGML CPU/Vulkan/CUDA backend
```

## 30. Qwen2.5 / staged 路径和 Qwen3.5 native 路径的区别

之前 Qwen2.5/早期 staged 支持更像：

```text
nano-vLLM scheduler
  -> ctypes
  -> llama.cpp external backend
  -> llama_context 内部执行
```

当前 Qwen3.5 native 支持是：

```text
nano-vLLM scheduler
  -> in-tree C++ runtime
  -> 自己加载 GGUF
  -> 自己构建 Qwen3.5 graph
  -> 自己管理 target/MTP PagedKV
  -> 自己管理 recurrent state
  -> 只借 GGML CPU/Vulkan/CUDA backend 和算子
```

所以 Qwen3.5 的支持工作不是简单“换模型名”，而是实现了一个专用 runtime。

## 31. 建议学习顺序

如果你要一点点学，建议顺序如下。

### 第一组：先理解 Python 框架改造

1. `nanovllm/config.py`
2. `nanovllm/backends/base.py`
3. `nanovllm/backends/__init__.py`
4. `nanovllm/engine/llm_engine.py`
5. `nanovllm/engine/scheduler.py`
6. `nanovllm/engine/block_manager.py`
7. `nanovllm/engine/model_runner.py`

看完这组，你会理解为什么需要 `BackendExecutionPlan`，以及 MTP 为什么会影响 scheduler。

### 第二组：看 legacy llama.cpp backend

1. `nanovllm/backends/llamacpp/runner.py`
2. `run_pipeline.md`
3. `scripts/build_llamacpp_cpu.sh`
4. `scripts/build_llamacpp_vulkan.sh`

这组帮助理解早期“链接外部库”的路线。

### 第三组：看 native runtime 基础设施

1. `CMakeLists.txt`
2. `scripts/build_native_runtime.sh`
3. `csrc/runtime/backend.{h,cpp}`
4. `csrc/runtime/graph_executor.{h,cpp}`
5. `csrc/runtime/gguf_loader.{h,cpp}`
6. `csrc/native_module.cpp`

这组帮助理解怎么把 GGML 嵌进 nano-vLLM。

### 第四组：看 Qwen3.5 模型结构

1. `csrc/runtime/qwen35_model.{h,cpp}`
2. `csrc/models/qwen35/weights.{h,cpp}`
3. `csrc/models/qwen35/ops.{h,cpp}`
4. `csrc/models/qwen35/graph.{h,cpp}`

这组是模型支持的核心。

### 第五组：看状态管理

1. `csrc/runtime/paged_kv.{h,cpp}`
2. `csrc/runtime/recurrent_state.{h,cpp}`

这组解释 Qwen3.5 为什么不仅有 KV cache，还有 recurrent state。

### 第六组：看 runtime 总控和 MTP

1. `csrc/runtime/qwen35_runtime.{h,cpp}`
2. `csrc/python/qwen35_runtime_binding.cpp`
3. `nanovllm/backends/native/runner.py`

重点看：

- `run()`
- `run_mtp()`
- `execute_target_chunk()`
- `execute_mtp()`
- `execute_mtp_kv_update()`
- `release_blocks()`
- v3.5 `PersistentGraphCache`

### 第七组：看测试和 CLI

1. `tests/test_native_backend_config.py`
2. `tests/test_native_runner.py`
3. `tests/test_native_runtime.py`
4. `tests/test_native_qwen35_cpu_oracle.py`
5. `tests/test_native_qwen35_long_context_oracle.py`
6. `tests/native/test_graph_executor.cpp`
7. `nanovllm/cli/chat.py`
8. `nanovllm/cli/bench.py`

这组用于验证你对流程的理解是否能落到可运行命令。

## 32. 文件级修改清单

### 32.1 新增文件

```text
.gitmodules
CMakeLists.txt
CURRENT_RUNTIME_FLOW.zh.md
CURRENT_VERSION_CHANGES.zh.md
LEARNING_GUIDE.zh.md
csrc/models/qwen35/graph.cpp
csrc/models/qwen35/graph.h
csrc/models/qwen35/ops.cpp
csrc/models/qwen35/ops.h
csrc/models/qwen35/weights.cpp
csrc/models/qwen35/weights.h
csrc/native_module.cpp
csrc/python/qwen35_runtime_binding.cpp
csrc/python/qwen35_runtime_binding.h
csrc/runtime/backend.cpp
csrc/runtime/backend.h
csrc/runtime/gguf_loader.cpp
csrc/runtime/gguf_loader.h
csrc/runtime/graph_executor.cpp
csrc/runtime/graph_executor.h
csrc/runtime/paged_kv.cpp
csrc/runtime/paged_kv.h
csrc/runtime/qwen35_model.cpp
csrc/runtime/qwen35_model.h
csrc/runtime/qwen35_runtime.cpp
csrc/runtime/qwen35_runtime.h
csrc/runtime/recurrent_state.cpp
csrc/runtime/recurrent_state.h
examples/llamacpp_backend.py
nanovllm/backends/__init__.py
nanovllm/backends/base.py
nanovllm/backends/llamacpp/__init__.py
nanovllm/backends/llamacpp/runner.py
nanovllm/backends/native/__init__.py
nanovllm/backends/native/runner.py
nanovllm/backends/native/runtime.py
nanovllm/cli/__init__.py
nanovllm/cli/bench.py
nanovllm/cli/chat.py
run_pipeline.md
scripts/build_llamacpp_cpu.sh
scripts/build_llamacpp_vulkan.sh
scripts/build_native_runtime.sh
tests/native/test_graph_executor.cpp
tests/test_native_backend_config.py
tests/test_native_cli.py
tests/test_native_qwen35_cpu_oracle.py
tests/test_native_runner.py
tests/test_native_runtime.py
tests/test_qwen35_stages.py
third_party/llama.cpp
```

未跟踪但当前存在：

```text
tests/test_native_qwen35_long_context_oracle.py
```

### 32.2 修改了原始文件

```text
README.md
example.py
nanovllm/config.py
nanovllm/engine/block_manager.py
nanovllm/engine/llm_engine.py
nanovllm/engine/model_runner.py
nanovllm/engine/scheduler.py
nanovllm/layers/sampler.py
nanovllm/sampling_params.py
pyproject.toml
```

### 32.3 当前 v3.7 重点修改文件

```text
CMakeLists.txt
scripts/build_native_runtime.sh
csrc/models/qwen35/graph.cpp
csrc/native_module.cpp
csrc/python/qwen35_runtime_binding.cpp
csrc/runtime/backend.{h,cpp}
csrc/runtime/graph_executor.{h,cpp}
csrc/runtime/qwen35_runtime.{h,cpp}
nanovllm/backends/{__init__,base}.py
nanovllm/backends/native/runner.py
nanovllm/cli/{bench,chat}.py
nanovllm/config.py
nanovllm/engine/{llm_engine,scheduler,sequence}.py
tests/test_native_backend_config.py
tests/test_native_runner.py
tests/test_qwen35_stages.py
```

这些包含已提交的 v3.5/v3.6 代码，以及 v3.7 工作区图优化：v3.5 graph reuse，v3.6 session/fair scheduling/CUDA/memory+MTP profile 和 attention gate 正确性修复；v3.7 opt-in FlashAttention、连续 GDN snapshot 写回、MTP prefill 合图和 CUDA Graph build variant。

## 33. 最容易遗漏但很重要的小改动

1. `SamplingParams.temperature` 默认从 `1.0` 改成 `0.0`，这是 MTP greedy correctness 的基础。
2. `Sampler` 支持 `temperature=0`，否则原始代码会除以 0 或禁止 greedy。
3. `eos` 变成 `eos_token_ids`，否则 Qwen 的多个结束 token 处理不完整。
4. `BlockManager.deallocate()` 返回 block id，这是后端释放 KV 的桥。
5. `Scheduler.postprocess()` 支持一轮多个 token，这是 MTP 能接入 Python 主循环的桥。
6. `can_append()` 增加 speculative token 预留，否则 MTP 可能写越界。
7. native backend 禁用 prefix cache/preemption，不是因为永远不需要，而是当前 C++ KV/recurrent state 还没有对应的安全语义。
8. native path 要求 HF tokenizer，不走 llama.cpp tokenizer，这是为了避免 native runtime 额外承担 tokenizer ABI。
9. `pyproject.toml` 把 CUDA 依赖移到 optional，是为了 CPU/Vulkan 开发环境可以轻量安装。
10. v3.5 graph reuse 当前只开 CPU 单序列，Vulkan fallback 是刻意的 correctness 保护。

## 34. 之后继续学习时可以提出的逐点问题

建议每次只挑一个点深入，例如：

- `BackendExecutionPlan` 是怎么从 `Sequence` 变成 flat arrays 的？
- MTP 为什么要求 scheduler 预留 `K` 个 speculative slots？
- Qwen3.5 的 18 层 recurrent state 在 graph 里怎么更新？
- `TargetChunkGraph` 为什么能同时服务 prefill、decode 和 verification？
- MTP catch-up 为什么能做成 KV-only？
- v3.5 graph reuse 的 bucket 里到底缓存了什么？
- Vulkan graph reuse 为什么在 masked bucket attention 上失败？

这样逐点学，会比直接从 1.4 万行 diff 里硬啃舒服很多。
