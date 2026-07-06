# nano-vLLM + llama.cpp PD zero-copy v2.0 说明

本文档记录当前 v2.0 版本的目标、执行流程、关键源码、benchmark 使用方法、日志判读和后续优化方向。

v2.0 的核心变化是新增：

```text
backend="llamacpp_pd"
```

即：

```text
prefill 阶段使用 Vulkan / Mali GPU
decode 阶段使用 ARM CPU
KV cache 使用 Vulkan_Host shared KV pool
nano-vLLM 仍统一管理 Sequence / Scheduler / BlockManager / BackendExecutionPlan / sampling
```

## 1. 版本目标

v2.0 不是把 llama.cpp 当成完整黑盒推理器使用，而是保留 nano-vLLM 的统一调度框架。

nano-vLLM 负责：

- `Sequence` 请求生命周期。
- `Scheduler` 决定本轮 prefill 或 decode。
- `BlockManager` 负责 paged KV block 分配、释放、prefix cache。
- `BackendExecutionPlan` 描述本轮执行的 token、position、block table、slot mapping。
- Python 侧 sampling。
- CLI chat 和 benchmark。

llama.cpp PD backend 负责：

- 加载 GGUF 模型。
- 使用 GGUF tokenizer 分词/反分词。
- 创建两个 llama context：
  - Vulkan prefill context。
  - CPU decode context。
- 让两个 context 的 K/V tensor 指向同一份 `Vulkan_Host` KV pool。
- 按 nano-vLLM 的 `slot_mapping` 写入 KV。
- 按 nano-vLLM 的 `block_table + context_lens` 读取 KV。
- 返回 logits 给 nano-vLLM 采样。

## 2. 总体执行流程

```text
用户运行 nanovllm.cli.bench/chat
  ↓
LLM(... backend="llamacpp_pd", model_format="gguf")
  ↓
Config 校验 GGUF / single process / llama.cpp tokenizer
  ↓
create_backend()
  ↓
LlamaCppPDRunner
  ↓
ctypes 加载 Vulkan-enabled libnanollama_backend.so
  ↓
nano_llama_pd_backend_create()
  ├─ 加载 prefill model，n_gpu_layers=-1，Vulkan offload
  ├─ 创建 prefill context，KV tensor 分配到 Vulkan_Host
  ├─ 记录 shared KV base pointer / size / buffer name
  ├─ 加载 decode model，n_gpu_layers=0，CPU
  └─ 创建 decode context，KV tensor 使用 CPU_Mapped wrapper 指向同一份 shared KV
  ↓
LLM.generate() / bench loop
  ↓
Scheduler.schedule()
  ├─ prefill：分配 block，处理 prompt token，尝试 prefix cache
  └─ decode：每个 running sequence 每轮生成 1 token
  ↓
build_execution_plan()
  ↓
LlamaCppPDRunner.run(plan)
  ├─ plan.is_prefill=True  → nano_llama_pd_backend_run() → Vulkan context
  └─ plan.is_prefill=False → nano_llama_pd_backend_run() → CPU context
  ↓
llama.cpp external KV plan
  ├─ cell_idx = block_id * block_size + offset
  ├─ prefill 写入 shared KV
  ├─ prefill 后同步 Vulkan backend
  ├─ mirror metadata 到 CPU context
  └─ decode 从同一份 shared KV 读取历史 K/V，并写入新 token K/V
  ↓
nano-vLLM Python sampler
  ↓
Scheduler.postprocess()
  ├─ append token
  ├─ 判断 EOS / max_tokens
  └─ 请求结束后释放 block，并调用 release_blocks 清理 llama.cpp 两个 context metadata
```

## 3. 源码目录

本地源码：

```text
/home/kevin/kevin_prj/nano-vllm
/home/kevin/kevin_prj/llama.cpp
```

远程开发板源码：

```text
/home/cix/nano-vlm/nano-vllm
/home/cix/nano-vlm/llama.cpp
```

模型：

```text
/home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf
```

## 4. nano-vLLM 关键源码

后端入口：

```text
nanovllm/config.py
nanovllm/backends/__init__.py
nanovllm/backends/base.py
nanovllm/backends/llamacpp/runner.py
```

重点看：

- `Config.__post_init__()`：允许 `llamacpp_pd`，GGUF 后端设置 `num_kvcache_blocks`。
- `create_backend()`：`llamacpp_pd` 创建 `LlamaCppPDRunner`。
- `BackendExecutionPlan`：统一描述 prefill/decode 的输入。
- `build_execution_plan()`：把 `Sequence` 状态转换成后端执行计划。
- `LlamaCppPDRunner`：复用普通 `LlamaCppRunner` 的 plan 打包、sampling、tokenizer 逻辑，但绑定 `nano_llama_pd_*` C ABI。

调度和 KV block：

```text
nanovllm/engine/llm_engine.py
nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
```

重点看：

- `LLMEngine.generate()`：请求入口。
- `LLMEngine.step()`：调度、执行、postprocess 主循环。
- `Scheduler.schedule()`：决定 prefill / decode。
- `Scheduler.postprocess()`：追加 token、结束请求、释放 block。
- `BlockManager`：paged KV block、prefix cache、引用计数。
- `Sequence.block_table`：逻辑 token 到物理 KV block 的映射。

CLI：

```text
nanovllm/cli/chat.py
nanovllm/cli/bench.py
```

重点看：

- `bench.py::run_generation_once()`：手动展开 step，用于分别统计 prefill/decode 时间。
- `BenchResult.shared_kv_mib`：PD shared KV pool 大小。
- `BenchResult.load_rss_mib`：模型加载后进程 RSS。

## 5. llama.cpp 关键源码

nano-vLLM shim：

```text
src/nano-vllm-ext.h
src/nano-vllm-ext.cpp
src/nano-vllm-backend.cpp
```

重点看：

- `nano_llama_backend_create()`：普通 CPU/Vulkan 后端。
- `nano_llama_pd_backend_create()`：PD 双 context 初始化。
- `nano_llama_pd_backend_run()`：根据 `plan->is_prefill` 分发到 Vulkan 或 CPU context。
- `nano_llama_pd_backend_release_blocks()`：同时清理 prefill/decode context metadata。
- `nano_llama_pd_backend_shared_kv_bytes()`：向 Python 暴露 shared KV 大小。
- `nano_llama_mark_plan_cells()`：prefill 后把已写入的 KV cell metadata mirror 到 CPU context。
- `nano_llama_backend_run_ctx()`：把 `nano_llama_kv_plan` 转成 `llama_batch` 并执行 `llama_decode()`。

KV cache 改造：

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-context.cpp
```

重点看：

- `nano_llama_alloc_external_kv_buffer()`：external KV allocation path。
- `nano_llama_kv_alloc_plan::OWNER`：prefill context 创建 `Vulkan_Host` KV buffer。
- `nano_llama_kv_alloc_plan::MIRROR`：decode context 用 `ggml_backend_cpu_buffer_from_ptr()` 指向同一地址。
- `llama_kv_cache::external_mark_cell()`：外部标记 KV cell 的 `pos/seq_id`。
- `llama_kv_cache::external_rm_cell_seq()`：释放 block 时移除 seq metadata。
- `nano_llama_init_external_memory()`：让 llama.cpp graph 使用 nano-vLLM 提供的 block table / slot mapping。

Vulkan 相关：

```text
ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

重点看：

- Vulkan host buffer 类型。
- descriptor set 分配。
- graph compute / synchronize。
- `Vulkan_Host` buffer 对 prefill 性能的影响。

## 6. PD zero-copy 的内存结构

v2.0 使用双 context / 双模型加载：

```text
prefill model  : Vulkan offload，权重大部分在 Vulkan0
decode model   : CPU，权重在 CPU_Mapped
prefill KV     : Vulkan_Host shared KV pool
decode KV      : CPU_Mapped wrapper，base pointer 指向同一份 shared KV pool
```

初始化日志中应该同时看到两段模型加载。

prefill Vulkan 模型：

```text
load_tensors: offloading output layer to GPU
load_tensors: offloading 35 repeating layers to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   243.43 MiB
load_tensors:      Vulkan0 model buffer size =  1732.72 MiB
```

decode CPU 模型：

```text
done_getting_tensors: tensor 'token_embd.weight' ... cannot be used with preferred buffer type Vulkan_Host, using CPU instead
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/37 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1732.72 MiB
```

第二段是正常的，它表示 CPU decode context 不上 GPU，不代表 PD 失败。

strict zero-copy 成功标志：

```text
nano_llama_pd: shared KV buffer name=Vulkan_Host base=0x... size=...
nano_llama_pd: strict zero-copy shared KV total=...
```

如果没有 `Vulkan_Host`，则不满足本版本目标。

## 7. 构建方式

PD 后端必须使用 Vulkan-enabled backend library。

在远程板构建：

```bash
cd /home/cix/nano-vlm/nano-vllm
LLAMA_CPP_DIR=/home/cix/nano-vlm/llama.cpp ./scripts/build_llamacpp_vulkan.sh
```

输出：

```text
/home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so
```

不要使用 CPU-only so 跑 PD：

```text
/home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so
```

如果误用 CPU-only so，会报：

```text
llama.cpp backend library does not export nano_llama_pd_backend_create.
```

## 8. Benchmark 使用方法

快速 smoke test：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_pd \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 128 \
  --gen-len 8 \
  --warmup 0 \
  --repeat 1 \
  --max-model-len 2048 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 4 \
  --threads 8 \
  --threads-batch 8 \
  --ubatch-size 128 \
  --json 2>&1 | tee pd_bench.log
```

正式对比测试：

```bash
cd /home/cix/nano-vlm/nano-vllm
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_pd \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 512 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --threads 8 \
  --threads-batch 8 \
  --ubatch-size 512 \
  --json 2>&1 | tee pd_bench_512_32.log
```

同样参数对比 CPU：

```bash
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_cpu \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_cpu/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 512 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --threads 8 \
  --threads-batch 8 \
  --json 2>&1 | tee cpu_bench_512_32.log
```

同样参数对比 Vulkan：

```bash
PYTHONPATH=. python -m nanovllm.cli.bench \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so \
  --batch-size 1 \
  --prompt-len 512 \
  --gen-len 32 \
  --warmup 1 \
  --repeat 3 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --threads 8 \
  --threads-batch 8 \
  --ubatch-size 512 \
  --json 2>&1 | tee vulkan_bench_512_32.log
```

建议测试矩阵：

```text
prompt_len = 128 / 512 / 1024 / 2048
gen_len    = 8 / 32 / 64
batch      = 1 / 2 / 4
backend    = llamacpp_cpu / llamacpp_vulkan / llamacpp_pd
```

## 9. Benchmark 指标解释

输出字段：

```text
load_s             模型加载耗时
load_rss_mib       模型加载后进程 RSS
prefill_s          prefill 累计耗时
decode_s           decode 累计耗时
prefill_tokens     prefill 输入 token 数
decode_tokens      decode step token 数
generated_tokens   最终生成 token 数
prefill_tok_s      prefill_tokens / prefill_s
decode_tok_s       decode_tokens / decode_s
generated_tok_s    generated_tokens / total_s
shared_kv_mib      PD shared KV pool 大小
```

注意：

- `gen_len=1` 基本只测 prefill，因为第一个输出 token 来自 prefill logits。
- 想看 CPU decode 性能，`gen_len` 至少设为 `8`，更推荐 `32/64`。
- bench 默认最后才打印结果，中间没有进度条，长 prompt 会看起来像卡住。
- `warmup=1 repeat=3` 实际会跑 4 轮。
- PD 目前使用 `Vulkan_Host` KV，可能明显拖慢 Vulkan prefill，需要用数据判断收益。

## 10. 日志判读

检查 PD 是否真正启用 Vulkan prefill：

```bash
grep -E "offloaded|model buffer size|shared KV|strict zero-copy" pd_bench.log
```

正常应看到：

```text
load_tensors: offloaded 37/37 layers to GPU
load_tensors:      Vulkan0 model buffer size = ...
load_tensors: offloaded 0/37 layers to GPU
load_tensors:   CPU_Mapped model buffer size = ...
nano_llama_pd: shared KV buffer name=Vulkan_Host ...
nano_llama_pd: strict zero-copy shared KV total=...
```

含义：

- 第一段 `37/37` 是 prefill Vulkan model。
- 第二段 `0/37` 是 decode CPU model。
- `Vulkan_Host` 是 strict zero-copy shared KV pool。

如果只有 `0/37`，没有 `37/37`，说明 prefill 没有正确使用 Vulkan。

如果没有 `Vulkan_Host`，说明不是本版本要求的 zero-copy PD。

## 11. 当前已知限制

- 只支持 GGUF。
- 只支持 single process。
- 只支持 dense non-SWA attention 模型。
- 不支持 tensor parallel。
- 不支持 CUDA PD。
- 不支持 KV copy fallback。
- 双 context / 双模型加载会增加内存占用。
- `Vulkan_Host` KV 可能降低 prefill 的 GPU 侧吞吐。
- CPU decode context 和 Vulkan prefill context 只共享 KV data，不共享完整 llama.cpp runtime metadata。
- prefill 后需要 mirror metadata 到 CPU context。
- CPU decode 写入 generated token KV 后，Vulkan context 不读取这些 generated token KV；如果后续要做复杂 preemption/re-prefill，需要重新确认 metadata 和 KV 覆盖策略。

## 12. 后续源码阅读顺序

建议按这条线读：

1. `nanovllm/cli/bench.py`

   先理解 benchmark 怎么拆分 prefill/decode 时间。

2. `nanovllm/engine/llm_engine.py`

   看 `generate()`、`add_request()`、`step()`。

3. `nanovllm/engine/scheduler.py`

   看 prefill/decode 切换、chunked prefill、preemption。

4. `nanovllm/engine/block_manager.py`

   看 block 分配、释放、prefix cache、引用计数。

5. `nanovllm/backends/base.py`

   看 `BackendExecutionPlan` 和 `build_execution_plan()`。

6. `nanovllm/backends/llamacpp/runner.py`

   看 Python 如何把 plan 打包成 C struct。

7. `llama.cpp/src/nano-vllm-backend.cpp`

   看 PD create/run/release 的完整控制流。

8. `llama.cpp/src/llama-kv-cache.cpp`

   看 external KV allocation path。

9. `llama.cpp/src/nano-vllm-ext.cpp`

   看 external plan 如何接入 llama.cpp 内部 attention/KV。

10. `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`

    看 Vulkan buffer、descriptor、graph compute 和 synchronize。

## 13. 优化方向索引

### 13.1 确认 `Vulkan_Host` KV 对 prefill 的影响

当前 strict zero-copy 要求 prefill KV 写到 `Vulkan_Host`。这避免了 CPU decode 前的 KV copy，但可能让 GPU prefill 访问 KV 的路径变慢。

建议对比：

```text
llamacpp_vulkan: device-local KV
llamacpp_pd:     Vulkan_Host shared KV
```

重点看：

```text
prefill_tok_s
decode_tok_s
generated_tok_s
```

如果 PD prefill 明显低于 Vulkan-only，说明 `Vulkan_Host` KV 是核心瓶颈之一。

### 13.2 降低双模型加载内存

当前 PD 加载两份权重：

```text
Vulkan prefill model
CPU decode model
```

优化方向：

- 研究 llama.cpp 是否可让两个 context 共享同一个 `llama_model` 元数据。
- CPU decode 是否可以只加载 CPU 所需权重。
- 对 mmap / buffer ownership 做更细粒度复用。

### 13.3 减少 prefill 后 metadata mirror 成本

当前 prefill 后会把本轮写入 cell 的 metadata mirror 到 CPU context。

优化方向：

- 批量 mark cell，减少逐 cell 操作。
- 将 `block_table + context_lens` 作为 CPU context 的 external view，减少重复 metadata。
- prefix cache 命中时只 attach block 粒度 metadata。

### 13.4 改进 bench 可观测性

当前 bench 最后才输出结果。

建议增加：

```text
--progress
--print-each-run
--dump-phase-times
```

每轮输出：

```text
run_id
phase
scheduled tokens
elapsed
tok/s
```

这样长 prompt 不会误判为卡住。

### 13.5 decode CPU 优化

CPU decode 仍依赖 llama.cpp CPU 后端。

可以继续观察：

- `--threads`
- `--threads-batch`
- `GGML_CPU_KLEIDIAI`
- `GGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod+sve`
- Q4_0 / Q4_K / Q5_K / Q8_0 等 GGUF 量化格式对 ARM decode 的影响。

### 13.6 Vulkan prefill 优化

方向：

- `--ubatch-size` 对 prefill 的影响。
- `max_num_batched_tokens` 与 graph reserve / descriptor set 的关系。
- Mali-G720 上 `Vulkan_Host` buffer 的读写带宽。
- 是否可以只让 K/V cache 使用 shared host buffer，而中间 compute buffer 保持 device-local。

## 14. 当前 v2.0 验证状态

已验证：

```text
本地 python -m compileall nanovllm
本地 nanollama_backend C++ build
远程 Vulkan libnanollama_backend.so build
远程 llamacpp_pd smoke bench
远程 llamacpp_vulkan regression bench
远程 llamacpp_cpu regression bench
CPU-only so 运行 llamacpp_pd 时快速报错
```

远程 PD zero-copy 成功日志示例：

```text
nano_llama_pd: shared KV buffer name=Vulkan_Host base=0x7fdf92400000 size=18.00 MiB
nano_llama_pd: strict zero-copy shared KV total=18.00 MiB
```

你本地测试中 `max_model_len=2048` 时可能看到：

```text
nano_llama_pd: shared KV buffer name=Vulkan_Host base=0x... size=72.00 MiB
nano_llama_pd: strict zero-copy shared KV total=72.00 MiB
```

这符合 Qwen2.5-3B 在当前 block/context 配置下的 shared KV 大小。

## 15. v2.0 结论

当前版本已经达成：

```text
nano-vLLM 统一调度
llama.cpp Vulkan prefill
llama.cpp CPU decode
strict zero-copy shared KV
bench/chat 可用
```

下一步优化重点不是继续扩大接口，而是基于 benchmark 数据确认：

```text
Vulkan_Host KV 是否拖慢 prefill
CPU decode 是否成为瓶颈
双模型加载内存是否可接受
metadata mirror 是否有明显开销
batch / prompt_len / gen_len 在 PD 下的收益边界
```

源码阅读时建议始终围绕这条主线：

```text
Scheduler 产出计划
  -> BackendExecutionPlan
  -> Python ctypes plan
  -> nano_llama_pd_backend_run
  -> external KV slot_mapping
  -> shared KV pool
  -> logits
  -> Python sampling
  -> Scheduler.postprocess
```
