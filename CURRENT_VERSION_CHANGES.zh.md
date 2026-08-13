# nano-vLLM 当前版本修改说明：v3.71 Native fallback 生命周期优化

## 1. 版本定位

本文件只描述当前代码版本；旧版本细节由 Git 历史保存。

- v3.5 基线：`71bd889 Add Qwen3.5 v3.5 graph reuse baseline`
- v3.6 功能提交：`12272ae Add native session scheduling and CUDA backend`
- v3.6 正确性修复：`cd6ced0 Restore Qwen3.5 chunk attention gate`
- v3.6 当前提交：`aabd733 v3.6: Add native MTP stage profiling`
- v3.7：`044e92e v3.7: add native runtime optimization controls`
- v3.71 工作区：同步生命周期与 execution-plan 校验分配优化，待本次提交固化
- 模型：Qwen3.5-2B-Q4_0 GGUF，24 层 target（6 attention + 18 recurrent）和 bundled 单层 MTP
- GGML：官方基线 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`，项目固定修订 `5ed33380b4679533243ca45e172804d5ddfe59ec`
- Native 后端：`native_cpu`、`native_vulkan`、`native_cuda`；仍不创建 `llama_context`、不调用 `llama_decode`

v3.6 的主题不是改变 Qwen3.5 图结构，而是让 native runtime 更接近端侧可用形态：多轮对话不重复 prefill、调度不会无限压住 decode、GGML CUDA 可作为第三个 native 后端、MTP 的时间与常驻内存可以直接测量。`cd6ced0` 同时补回 attention 的 query gate，保证 Qwen3.5 attention 图与模型结构一致。

v3.7 在此基础上只加入可独立 A/B 的图级优化，默认仍走 v3.6 math-attention 和逐 snapshot 写回路径：非 MTP 的 FlashAttention 用 runtime capability probe 选择；MTP 的 `auto` 固定 math，避免不同 token-shape 的 Flash 舍入差异降低 draft acceptance；GDN delta snapshot 改为可选连续写回；MTP prefill 的 hidden-to-KV maintenance 可进入同一张 target graph；CUDA Graph 仅在显式 CUDA Graph build variant 中使用。没有实现自定义 Q4_0 `lm_head + argmax` kernel，避免在缺乏目标 GPU profile 的情况下引入高风险的量化算子分叉。

v3.71 不改变模型图、权重、缓存布局或 greedy token 语义，只去除两个已确认的运行时实现损耗：GGML `graph_compute` 成功后重复的 scheduler synchronize，以及 execution plan 校验中为 speculative tail 创建后即丢弃的完整 context slot 向量。

## 2. 从 v3.5 到 v3.7 的文件与改动

### 2.1 Python 会话、调度与容量默认值

| 文件 | v3.6 改动 |
| --- | --- |
| `nanovllm/engine/sequence.py` | 新增 `PARKED` 状态、`retain_cache`、按 turn 计数的 completion；`begin_turn()` 将新用户输入追加到原 sequence。 |
| `nanovllm/engine/scheduler.py` | 新增 parked session 的 LRU 管理、恢复/关闭/淘汰流程；缺 block 或 native sequence slot 时先淘汰最旧 idle session；加入连续 prefill 轮数上限，避免 decode 饥饿。 |
| `nanovllm/engine/llm_engine.py` | 新增 `ChatSession`、`start_session()`、`session.generate()`、`close()`；按阶段记录 tokenize/schedule/plan/native-run/postprocess 时间，并汇总 native memory stats。 |
| `nanovllm/cli/chat.py` | native chat CLI 使用 `ChatSession`；首轮完整 prefill，后续只提交新 turn 的 token；`/reset` 和 `/system` 会关闭旧 session 并释放其 native state。 |
| `nanovllm/config.py` | native 默认 `max_num_seqs=1`、`max_num_batched_tokens=2048`，避免端侧启动时为大量 recurrent slot 预留状态；新增 `enable_session_cache`、`max_retained_sessions`、`max_consecutive_prefill_rounds`。 |

会话保留的是同一条 sequence：Python `block_table`、native sequence slot、target/MTP PagedKV、recurrent conv/delta state 和 `pending_hidden` 都继续存在。它不是跨请求 prefix cache：不同请求之间仍不会共享 KV 或 recurrent state。idle session 超过上限或显式 `close()` 时，Scheduler 收集 block id，`LLMEngine.flush_backend_releases()` 调用 native `release_blocks()`，同时释放 PagedKV 和 recurrent slot。

调度公平策略保持简单：waiting 队列优先 prefill；但当 waiting 和 running 同时非空，且已经连续执行 `max_consecutive_prefill_rounds`（默认 4）轮 prefill，就强制执行一轮 decode。端侧低并发仍以低开销 FIFO 为主，不引入复杂的服务端 QoS 算法。

### 2.2 Native CUDA 后端

| 文件 | v3.6 改动 |
| --- | --- |
| `CMakeLists.txt` | 新增 `NANOVLLM_NATIVE_CUDA`，将 GGML CUDA backend 静态链接到 `nanovllm._C`。 |
| `scripts/build_native_runtime.sh` | 新增 `NANOVLLM_NATIVE_CUDA=ON`；CUDA 与 Vulkan 互斥构建，分别使用 `build/native-cuda` 与 `build/native-vulkan`。 |
| `csrc/native_module.cpp` | `build_info()` 和 `available_backends()` 枚举 CUDA；smoke API 可创建 CUDA backend。 |
| `csrc/runtime/backend.{h,cpp}` | `BackendKind`、设备发现、`BackendConfig` 和 `BackendList` 增加 CUDA，布局为 `[cuda, cpu]`。 |
| `csrc/runtime/graph_executor.{h,cpp}` | placement audit 增加 CUDA compute/storage 节点计数，并接受 `[cuda, cpu]` scheduler 布局。 |
| `nanovllm/backends/{__init__,base}.py`、`nanovllm/backends/native/runner.py`、`nanovllm/config.py`、CLI | Python backend 名称、配置校验和 native runner 路由增加 `native_cuda`。 |

CUDA、Vulkan 都让 accelerator 排在 GGML scheduler 的第一个 backend，CPU 保留为输入、传输和 fallback backend。当前分支在 CPU/Mali 上完成了实测；CUDA 路径已完成构建、发现、placement 和配置接入，仍需要 NVIDIA 实机的模型正确性与性能验收。

### 2.3 Native runtime 可观测性

| 文件 | v3.6 改动 |
| --- | --- |
| `csrc/runtime/qwen35_runtime.{h,cpp}` | 新增累计 `Qwen35MtpProfileStats`：draft、target verification、MTP KV catch-up 的 calls/tokens、graph setup 时间和总 wall time；新增 weights/PagedKV/recurrent/graph metadata 的 persistent memory accounting。 |
| `csrc/python/qwen35_runtime_binding.cpp` | Python `Qwen35Runtime` 暴露 `mtp_profile_stats()` 和 `memory_stats()`。 |
| `nanovllm/backends/native/runner.py` | 增加对应的 Python 包装方法。 |
| `nanovllm/engine/llm_engine.py` | `runtime_metrics()` 汇总 Python 各阶段耗时和 native 常驻内存。 |

计时覆盖真实路径：graph 构建/allocate、host-backend tensor set/get、同步 compute 都包含在 stage elapsed 中；因此可直接用来拆解 MTP，而不是把性能问题都归到 GGML kernel。

### 2.4 Qwen3.5 图正确性修复

| 文件 | v3.6 改动 |
| --- | --- |
| `csrc/models/qwen35/graph.cpp` | 在 full-attention 的 weighted value 合并后恢复 `attended * sigmoid(query_gate.gate)`，再进入 `attn_output` 线性层。 |

Qwen3.5 attention 不只是标准 Q/K/V attention：query projection 还拆出 gate。漏掉该乘法会改变 attention 输出和 greedy token，因此它是正确性修复，不是可选优化。

### 2.5 测试、配置和使用文档

`tests/test_native_backend_config.py`、`tests/test_native_runner.py`、`tests/test_qwen35_stages.py` 扩展了 CUDA backend、session 生命周期、prefill/decode 公平性、native memory stats 和 query gate 的覆盖。README 增加 native CUDA build/example；`CURRENT_RUNTIME_FLOW.zh.md`、`CURRENT_CODE_DIFF_FROM_ORIGINAL.zh.md` 同步到 v3.6。

### 2.6 v3.7 图级优化实验

| 文件 | v3.7 改动 |
| --- | --- |
| `csrc/models/qwen35/graph.{h,cpp}` | 添加 `AttentionImplementation`；保留 math attention，同时可构建 `ggml_flash_attn_ext` 路径。MTP prefill 可把 target 的 normalized hidden 右移后直接做 MTP KV-only 部分；不再读回整块 `[E,T]` hidden。 |
| `csrc/runtime/recurrent_state.{h,cpp}`、`csrc/runtime/qwen35_runtime.cpp` | 为相邻 delta snapshot plane 创建连续 `[D,1,K,1]` view；启用时一个 `ggml_cpy` 写回全部 snapshot，CUDA 后端可匹配 GGML 现有的 GDN-to-cache 融合。 |
| `csrc/runtime/qwen35_runtime.{h,cpp}` | 增加 `math/auto/flash` 选择、按真实 shape/backend probe Flash 能力、F32/F16 mask 上传转换；MTP 的 `auto` 固定 math，避免 Flash 的 shape-dependent rounding 压低 greedy acceptance；MTP prefill fusion 禁止复用旧 bucket，避免图结构与输入/输出语义混淆。 |
| `CMakeLists.txt`、`scripts/build_native_runtime.sh`、`csrc/native_module.cpp` | 增加默认关闭的 `NANOVLLM_NATIVE_CUDA_GRAPHS` 构建开关和 build info；CUDA Graph 是 GGML backend scope 行为，因此以独立 build variant 作为 A/B 边界；若未同时开启 CUDA，CMake 和脚本都会明确拒绝该配置。 |
| `nanovllm/config.py`、native runner、chat/bench CLI | 暴露 `native_attention_impl`、`native_batched_recurrent_snapshots`、`native_mtp_prefill_fusion`；全部默认关闭，CLI/JSON 会记录运行时 A/B 配置。 |
| native tests | 增加精确 GQA FlashAttention layout/F16 causal-mask oracle、连续 GDN snapshot CPY plane-order 测试、Python 配置/ABI 传递测试，以及 opt-in real-model MTP trace oracle。 |

关键边界：Flash 路径仍先 `SET_ROWS` 到 PagedKV、再 `GET_ROWS` gather，所以它不是新的 paged-attention kernel；只是在 gather 后把 QK、softmax、PV 的中间分数/概率留在 backend kernel 内。MTP verification 的 accepted 数量必须在 host 比较后才能知道，故其 catch-up 仍是独立 KV-only graph。Mali 实测显示 Flash 在正确性 oracle 中可用，但 draft `T=1` 与 verification `T=K+1` 的舍入差异会把 K=1 acceptance 从 100% 降至 50%，并让 decode 吞吐退化；因此 MTP 下 `auto` 保守选择 math，显式 `flash` 只作实验。`lm_head + argmax` 的全词表量化 matmul 仍是第二阶段课题；当前没有为了“融合”而添加不成熟的自定义 Q4 kernel。

### 2.7 v3.71 fallback 生命周期与 plan 校验优化

| 文件 | v3.71 改动 |
| --- | --- |
| `csrc/runtime/graph_executor.{h,cpp}` | 新增只用于“成功同步 compute 之后”的 reset；保留通用 `reset()` 的同步和异常路径保护。 |
| `csrc/runtime/qwen35_runtime.cpp` | fallback target、MTP draft、MTP KV catch-up 的 RAII guard 仅在成功 compute 后走无重复同步的 reset；构图或执行异常仍走同步 reset。 |
| `csrc/runtime/paged_kv.{h,cpp}` | 提供无分配 `validate_logical_range()`；`physical_indices()` 继续保持原有展开语义。 |
| `tests/native/test_graph_executor.cpp` | 覆盖成功 compute 后 reset、立即重新 allocate/compute 的 scheduler 生命周期。 |

GGML 当前 `ggml_backend_sched_graph_compute()` 已按同步语义完成 backend work，因此 fallback graph 在作用域退出时再次调用 `ggml_backend_sched_synchronize()` 是额外等待。v3.71 只在该同步成功返回后跳过第二次等待；若 allocate、上传或 compute 抛异常，guard 仍使用原始同步 reset，避免在未知 backend 状态下重用 scheduler。plan 校验仍验证 block table 的最大逻辑范围、block ID 合法性与同 sequence block 唯一性，只是不再物化本轮尚未参与计算的 context slot 列表。

## 3. 当前运行链路的新增部分

```text
native chat 首轮
  prompt -> Sequence(WAITING) -> prefill/decode -> PARKED

native chat 后续轮
  新 turn token -> parked Sequence.begin_turn()
  -> Scheduler.resume()
  -> 只对新增 token prefill
  -> 继续使用原 block_table / native slot / PagedKV / recurrent state
  -> decode -> PARKED

close / LRU eviction
  -> BlockManager.deallocate()
  -> NativeRunner.release_blocks(block_ids, sequence_slot)
  -> 清空 target/MTP KV、recurrent state、pending hidden
```

普通 `LLM.generate()` 的一次性请求语义不变。`enable_session_cache` 只影响显式 `start_session()` 创建的保留会话。

v3.5 的 persistent graph bucket 仍保留：CPU 单 sequence 的 decode/MTP 高频路径默认启用；native CUDA 只有在独立 CUDA build 带 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 时才复用稳定 graph 并让 GGML 捕获/回放。Vulkan、多 sequence、prefill/chunk 仍按轮构图。bucket 不复制模型权重、PagedKV 或 recurrent state，只缓存 graph metadata、scheduler allocation、输入/输出 tensor 和临时 activation buffer。

## 4. 当前性能结论

统一 long benchmark：Qwen3.5-2B-Q4_0、`prompt_len=186`、`gen_len=541`、`repeat=3`、`warmup=1`、`threads=8`、`max_model_len=8092`、32 个 KV blocks。Vulkan 使用与 v3.5 一致的 `glslc-2025.2`，运行日志确认 `int dot=1, matrix cores=KHR_coopmat`。

| 后端/模式 | v3.5 decode tok/s | v3.6 decode tok/s | 结论 |
| --- | ---: | ---: | --- |
| CPU MTP off | 27.71 | 27.84 | +0.5%，持平 |
| CPU MTP K=3 | 32.54 | 32.53 | -0.03%，持平 |
| Vulkan MTP off | 20.84 | 20.88 | +0.2%，持平 |
| Vulkan MTP K=1 | 18.55 | 18.51 | -0.2%，持平 |

CPU 上 MTP K=3 在本模型/工作负载有约 17% decode 收益。Vulkan 固定高性能构建下，K=1/2/3 为 `18.51/20.62/20.66 tok/s`，均未超过 MTP off 的 `20.88 tok/s`；K=1 虽然接受率 100%，仍被大量小 verification、额外 MTP vocab head 和 host/backend 往返拖慢。Vulkan 长 decode 当前应默认关闭 MTP，或依据实测吞吐自适应启用，而不能仅根据 acceptance rate 固定 K。

llama.cpp 参考数据的 decode 长度与 nano-vLLM 不同，适合作为量级对比，不作为 v3.5/v3.6 回归判断。v3.7 已在 Armv9.2 Cortex-A720/A520 + Mali-G720-Immortalis 远机完成真实 GGUF oracle 和统一长 benchmark；GCC 12 使用兼容拼写 `armv9-a+dotprod+i8mm`，Vulkan 日志确认 `int dot=1, matrix cores=KHR_coopmat`。相对 v3.6，v3.7 CPU MTP-off / K=3 decode 为 `27.75 / 32.93 tok/s`，Vulkan Flash MTP-off / math MTP K=1 为 `21.24 / 18.59 tok/s`。MTP 的 `auto` 固定 math 已避免 Mali Flash 的 shape-dependent rounding 将 K=1 acceptance 从 100% 降至 50%、decode 降至 `16.08 tok/s` 的回退。完整 benchmark、K sweep 和 profile 记录在开发笔记 `8-mtp升级.md`。

## 5. 构建与使用

```bash
# CPU
scripts/build_native_runtime.sh

# Vulkan：在 Mali 上应固定能生成 coopmat/int-dot shader 的 glslc
NANOVLLM_NATIVE_VULKAN=ON \
NANOVLLM_VULKAN_GLSLC=/path/to/glslc-2025.2 \
scripts/build_native_runtime.sh

# CUDA：不能与 Vulkan 同时打开
NANOVLLM_NATIVE_CUDA=ON scripts/build_native_runtime.sh
```

Native chat 默认启用 session cache。首轮由 CLI 创建 session；后续输入只追加本 turn：

```bash
PYTHONPATH=. python3 -m nanovllm.cli.chat "$MODEL" \
  --backend native_cpu \
  --tokenizer "$TOKENIZER" \
  --max-model-len 8092 \
  --max-num-seqs 1
```

可通过 `enable_session_cache=False`、`max_retained_sessions` 和 `max_consecutive_prefill_rounds` 调整会话保留与公平策略。图优化 A/B 使用 `--native-attention-impl {math,auto,flash}`、`--native-batched-recurrent-snapshots`、`--native-mtp-prefill-fusion`；CUDA Graph 的 A/B 则使用独立 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 构建。benchmark 继续使用 `--enable-mtp --mtp-max-draft-tokens K`；性能比较必须同时记录 GGML commit、shader compiler、Mali capability 日志、warmup/repeat 和模型上下文长度。

## 6. 当前边界与下一步

1. Native 不支持跨请求 prefix cache；hybrid recurrent state 不能像纯 attention KV 一样廉价地共享。
2. Native 不支持 preemption；容量不足时 session LRU 释放 idle state，active request 仍沿用原有的显式错误/调度路径。
3. graph reuse 默认只覆盖 CPU 单 sequence 高频 decode/MTP；native CUDA 需使用 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 的独立构建且尚未在 NVIDIA 实机验收，Vulkan 仍按轮构图。
4. Native 路径使用 HF tokenizer；当前 MTP 热路径并不调用 Python tokenizer，验证耗时主要在 native graph 和 LM head。
5. v3.7 已减少 prefill hidden 往返和 recurrent snapshot copy；Vulkan MTP 仍需目标设备实测。后续高风险第二阶段是 Q4_0 LM head + argmax 融合、真正 paged FlashAttention 和自适应 K 策略。
