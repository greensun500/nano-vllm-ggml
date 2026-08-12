# nano-vLLM 当前版本修改说明：v3.6 Native 会话、CUDA 与 MTP 可观测性

## 1. 版本定位

本文件只描述当前代码版本；旧版本细节由 Git 历史保存。

- v3.5 基线：`71bd889 Add Qwen3.5 v3.5 graph reuse baseline`
- v3.6 功能提交：`12272ae Add native session scheduling and CUDA backend`
- v3.6 正确性修复：`cd6ced0 Restore Qwen3.5 chunk attention gate`
- v3.6 当前提交：`aabd733 v3.6: Add native MTP stage profiling`
- 模型：Qwen3.5-2B-Q4_0 GGUF，24 层 target（6 attention + 18 recurrent）和 bundled 单层 MTP
- GGML：官方基线 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`，项目固定修订 `5ed33380b4679533243ca45e172804d5ddfe59ec`
- Native 后端：`native_cpu`、`native_vulkan`、`native_cuda`；仍不创建 `llama_context`、不调用 `llama_decode`

v3.6 的主题不是改变 Qwen3.5 图结构，而是让 native runtime 更接近端侧可用形态：多轮对话不重复 prefill、调度不会无限压住 decode、GGML CUDA 可作为第三个 native 后端、MTP 的时间与常驻内存可以直接测量。`cd6ced0` 同时补回 attention 的 query gate，保证 Qwen3.5 attention 图与模型结构一致。

## 2. 从 v3.5 到 v3.6 的文件与改动

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

v3.5 的 persistent graph bucket 仍保留：只在 CPU、单 sequence 的 decode/MTP 高频路径启用；Vulkan/CUDA、多 sequence 和 prefill/chunk 仍按轮构图。bucket 不复制模型权重、PagedKV 或 recurrent state，只缓存 graph metadata、scheduler allocation、输入/输出 tensor 和临时 activation buffer。

## 4. 当前性能结论

统一 long benchmark：Qwen3.5-2B-Q4_0、`prompt_len=186`、`gen_len=541`、`repeat=3`、`warmup=1`、`threads=8`、`max_model_len=8092`、32 个 KV blocks。Vulkan 使用与 v3.5 一致的 `glslc-2025.2`，运行日志确认 `int dot=1, matrix cores=KHR_coopmat`。

| 后端/模式 | v3.5 decode tok/s | v3.6 decode tok/s | 结论 |
| --- | ---: | ---: | --- |
| CPU MTP off | 27.71 | 27.84 | +0.5%，持平 |
| CPU MTP K=3 | 32.54 | 32.53 | -0.03%，持平 |
| Vulkan MTP off | 20.84 | 20.88 | +0.2%，持平 |
| Vulkan MTP K=1 | 18.55 | 18.51 | -0.2%，持平 |

CPU 上 MTP K=3 在本模型/工作负载有约 17% decode 收益。Vulkan 固定高性能构建下，K=1/2/3 为 `18.51/20.62/20.66 tok/s`，均未超过 MTP off 的 `20.88 tok/s`；K=1 虽然接受率 100%，仍被大量小 verification、额外 MTP vocab head 和 host/backend 往返拖慢。Vulkan 长 decode 当前应默认关闭 MTP，或依据实测吞吐自适应启用，而不能仅根据 acceptance rate 固定 K。

llama.cpp 参考数据的 decode 长度与 nano-vLLM 不同，适合作为量级对比，不作为 v3.5/v3.6 回归判断。完整 benchmark、K sweep 和 profile 记录在开发笔记 `8-mtp升级.md`。

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

可通过 `enable_session_cache=False`、`max_retained_sessions` 和 `max_consecutive_prefill_rounds` 调整会话保留与公平策略。benchmark 继续使用 `--enable-mtp --mtp-max-draft-tokens K`；性能比较必须同时记录 GGML commit、shader compiler、Mali capability 日志、warmup/repeat 和模型上下文长度。

## 6. 当前边界与下一步

1. Native 不支持跨请求 prefix cache；hybrid recurrent state 不能像纯 attention KV 一样廉价地共享。
2. Native 不支持 preemption；容量不足时 session LRU 释放 idle state，active request 仍沿用原有的显式错误/调度路径。
3. graph reuse 只覆盖 CPU 单 sequence 高频 decode/MTP，Vulkan/CUDA 尚无 persistent graph reuse。
4. Native 路径使用 HF tokenizer；当前 MTP 热路径并不调用 Python tokenizer，验证耗时主要在 native graph 和 LM head。
5. Vulkan MTP 的首要优化是减少小图/host 往返、融合 Q4_0 LM head + argmax，再决定自适应 K 策略。
