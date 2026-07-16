# nano-vLLM 3.0 原生 CPU/Vulkan 推理流程

本文档描述当前 3.0 版本的 Qwen3.5-2B GGUF 推理流程。这个版本的核心变化是：nano-vLLM 不再把 llama.cpp 当作一个外部推理库调用，而是在项目内静态编入官方 GGML 的 CPU/Vulkan backend、GGUF 读取和量化算子，由 nano-vLLM 自己完成模型 graph、PagedKV、recurrent state 和 MTP 调度。

当前 3.0 支持范围：

- 模型：严格匹配 `Qwen3.5-2B-Q4_0.gguf`
- 后端：`native_cpu`、`native_vulkan`
- 解码：greedy only，`temperature=0`
- KV：nano-vLLM Python 侧分配逻辑页；C++ native runtime 持有物理 GGML buffer
- MTP：使用 GGUF 内置 `blk.24`，不需要外部 draft model
- correctness-first：性能和内存还不是当前验收标准

## 1. 总体分层

3.0 的推理链路分成三层：

```text
用户 API / CLI
    |
    v
Python 调度层
    LLMEngine / Scheduler / BlockManager / Sequence
    |
    v
native backend 适配层
    NativeRunner / BackendExecutionPlan / BackendExecutionResult
    |
    v
C++ 原生 runtime
    Qwen35Runtime / GGUF loader / Qwen35 graph / PagedKvCache
    RecurrentStateCache / GraphExecutor / GGML CPU/Vulkan backend
```

各层职责边界：

- Python 侧仍然是主调度框架，负责请求生命周期、batch 组织、逻辑 KV block 分配、停止条件和输出整理。
- `NativeRunner` 负责把 Python 调度计划转换成 C++ ABI，并管理 Python sequence id 到 native sequence slot 的映射。
- C++ runtime 负责 GGUF 权重、GGML graph、物理 KV/recurrent/MTP buffer、CPU/Vulkan backend 选择和实际算子执行。
- llama.cpp 的 `llama_model`、`llama_context`、`llama_decode`、llama KV cache 都不在 native path 中使用。

## 2. 构建路径

入口脚本：

```bash
./scripts/build_native_runtime.sh
NANOVLLM_NATIVE_VULKAN=ON ./scripts/build_native_runtime.sh
```

根 `CMakeLists.txt` 做几件关键事情：

- 固定官方 llama.cpp 子模块提交：
  `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- 只加入 `third_party/llama.cpp/ggml`
- 强制关闭 CUDA、动态 backend 加载和 llama.cpp 高层 runtime
- 将 GGML、GGUF、CPU/Vulkan backend 静态链接进 `nanovllm/_C*.so`
- 编译 `nanovllm_graph_executor_test`，验证 graph placement 语义

最终 Python 扩展是：

```text
nanovllm/_C.cpython-*.so
```

这个扩展提供：

- `build_info()`
- `available_backends()`
- `gguf_info()`
- `qwen35_model_info()`
- `Qwen35Runtime`
- F32/Q4_0/Vulkan smoke test helper

## 3. 初始化流程

用户从 `LLM` 或 CLI 进入：

```python
llm = LLM(
    "/path/to/Qwen3.5-2B-Q4_0.gguf",
    backend="native_vulkan",
    model_format="gguf",
    gguf_model="/path/to/Qwen3.5-2B-Q4_0.gguf",
    tokenizer="/path/to/Qwen3.5-2B",
    tokenizer_backend="hf",
    enable_mtp=True,
    mtp_max_draft_tokens=3,
)
```

`Config.__post_init__` 会为 native backend 做硬约束：

- backend 必须是 `native_cpu` 或 `native_vulkan`
- `gguf_model` 必须存在
- tokenizer 必须是本地 Hugging Face tokenizer 目录
- 不支持 prefix cache
- 不支持 preemption
- 单进程运行
- MTP 只允许 greedy，并要求 `max_num_batched_tokens >= max_num_seqs * (K + 1)`
- `num_kvcache_blocks * kvcache_block_size` 必须覆盖 `max_model_len`

随后 `LLMEngine` 调用 `create_backend(config)`，native backend 会创建 `NativeRunner`。

`NativeRunner` 初始化时会创建 C++ `Qwen35Runtime`：

```text
NativeRunner
    -> nanovllm._C.Qwen35Runtime(
           model_path,
           backend="cpu" | "vulkan",
           max_model_len,
           max_num_batched_tokens,
           max_num_seqs,
           block_size,
           num_blocks,
           n_threads,
           device_index,
           enable_mtp,
           mtp_max_draft_tokens,
       )
```

C++ runtime 初始化时会：

- 创建 CPU 或 Vulkan `BackendList`
- 读取 GGUF metadata 和 tensor manifest
- 校验 Qwen3.5-2B Q4_0 合同
- 加载 335 个 tensor 到目标 backend buffer
- 构造 target graph 所需权重视图
- 分配 PagedKV cache
- 分配 recurrent state cache
- 如果开启 MTP，再分配 MTP KV cache 和 MTP graph 所需状态

## 4. 请求进入与 tokenizer

`LLMEngine.add_request()` 接收字符串或 token id 列表：

- 字符串 prompt：使用 HF tokenizer 编码
- token id prompt：直接进入 `Sequence`

native backend 默认使用 HF tokenizer，不使用 llama.cpp tokenizer。Qwen3.5 的停止 token 会额外包含：

- `<|endoftext|>`：`248044`
- `<|im_end|>`：`248046`

每个请求被封装为 `Sequence`，加入 `Scheduler.waiting`。

## 5. Python 调度流程

每一轮 `LLMEngine.step()` 做固定四步：

```text
Scheduler.schedule()
    -> build_execution_plan()
    -> NativeRunner.run()
    -> Scheduler.postprocess()
```

### 5.1 Prefill

prefill 阶段会从 `waiting` 队列取新请求：

- 通过 `BlockManager.can_allocate()` 检查逻辑 KV block 是否足够
- 通过 `BlockManager.allocate()` 给 `Sequence.block_table` 分配物理 block id
- 支持 chunked prefill，但当前 native path 主要以 correctness-first 的小 batch 验证
- 生成本轮需要处理的 prompt token 范围

`build_execution_plan()` 会把多个 sequence 的 prefill 数据展平成一个 `BackendExecutionPlan`：

```text
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

其中 `slot_mapping` 是 Python 侧逻辑 block table 到物理 KV slot 的展开结果。

### 5.2 Decode

decode 阶段每个 running sequence 默认调度 1 个 pending token，也就是 `seq.last_token`。

如果开启 MTP，Scheduler 会额外为 speculative window 预留 KV slot：

```text
speculative_tokens = mtp_max_draft_tokens
```

但只有在 `len(seq) + K <= max_model_len` 时才预留。靠近上下文边界时，Scheduler 不再强行预留 K 个 draft slot，而是让 native runtime fallback 到普通 target greedy。这样开启 MTP 不会让原本合法的边界 decode 失败。

### 5.3 Postprocess

backend 返回 `BackendExecutionResult`：

```python
BackendExecutionResult(
    token_ids=[[...], ...],
    draft_token_counts=[...] | None,
)
```

对于普通 decode，每个 sequence 返回 1 个 token。

对于 MTP decode，每个 sequence 可能返回多个 token：

```text
[accepted_draft_0, accepted_draft_1, ..., final_target_token]
```

Scheduler 会逐个 `seq.append_token()`，检查 EOS 和 `max_tokens`，并释放完成请求的 block。

释放流程：

```text
Scheduler.block_releases
    -> LLMEngine.flush_backend_releases()
    -> NativeRunner.release_blocks(block_ids, seq_ids)
    -> Qwen35Runtime.release_blocks()
```

这样 Python 的逻辑页释放和 C++ 的物理 buffer 清理保持同步。

## 6. NativeRunner 适配层

`NativeRunner` 是 Python 和 C++ runtime 之间的边界。

它负责：

- 校验 greedy：`temperature` 必须全为 `0`
- 把 `BackendExecutionPlan` 转成 numpy int32 ABI
- 把 Python `seq_id` 映射为连续 native sequence slot
- 判断当前 decode 是否适合跑 MTP
- 将 C++ 返回值转回 `BackendExecutionResult`
- 统计 MTP draft/accepted/verification 次数
- 在 sequence 释放时归还 native slot

native sequence slot 是必要的，因为 recurrent state 不是按 Python `seq_id` 稀疏分配，而是按 `[0, max_num_seqs)` 的连续 slot 存储。

MTP 分支判断：

```text
enable_mtp
and not plan.is_prefill
and max(position) + 1 + K <= max_model_len
```

满足时调用：

```text
Qwen35Runtime.run_mtp(plan, token_capacity=K+1)
```

否则调用：

```text
Qwen35Runtime.run(plan)
```

## 7. C++ Qwen35Runtime

C++ runtime 是 native path 的执行核心。它的私有实现持有：

- GGML backend handles
- GGUF model metadata
- 权重 tensors 和 backend buffers
- target token graph 构建逻辑
- MTP token graph 构建逻辑
- PagedKvCache
- RecurrentStateCache
- GraphExecutor

模型合同固定为：

```text
architecture=qwen35
block_count=25
main_layers=24
nextn_predict_layers=1
vocab_size=248320
embedding_length=2048
tensor_count=335
strict_q4_0_profile=true
```

24 个 target layer 中：

```text
full attention layers = [3, 7, 11, 15, 19, 23]
recurrent layers      = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
MTP layer             = blk.24
```

## 8. Target Graph 执行

target graph 每次处理本轮调度的 token。

输入包括：

- token ids
- positions
- slot mapping
- context indices
- sequence slot ids
- active recurrent row ids

执行过程：

```text
token embedding
    -> 24-layer Qwen3.5 trunk
       recurrent layer:
           read conv/delta state
           run ssm_conv + gated_delta_net
           write new recurrent snapshots
       full attention layer:
           write current K/V to PagedKV
           gather context K/V from PagedKV
           run attention
    -> final norm
    -> output head
    -> greedy argmax
```

full attention 层使用 PagedKV；recurrent 层使用 RecurrentStateCache。

CPU 和 Vulkan 共用同一套 graph 构造代码。差异只在 backend：

- `native_cpu`：所有计算节点放 CPU
- `native_vulkan`：所有非纯 metadata-view 计算节点必须放 Vulkan

`GraphExecutor` 会区分 compute placement 和 storage placement。`CPY`、`SET_ROWS` 虽然结果可能 alias 目标 tensor，但仍然是计算节点，不能当作普通 view 忽略。

## 9. PagedKV 设计

Python `BlockManager` 决定逻辑页：

```text
Sequence.block_table = [physical_block_id, ...]
```

C++ `PagedKvCache` 持有真实物理 buffer：

```text
physical_slots = num_blocks * block_size
target K/V cache: full attention layers * 2
MTP K/V cache:    blk.24 * 2, only when enable_mtp
```

当前 cache 数据类型为 F32，优先保证跨 CPU/Vulkan 的正确性和可调试性。

每次 graph 中：

- `SET_ROWS` 把当前 token 的 K/V 写入物理 slot
- `GET_ROWS` 按 context indices 取出历史 K/V
- `validate_write_indices()` 保证同一 graph 内没有重复写
- `validate_read_indices()` 保证读取 slot 有效

Python 释放 block 后，C++ runtime 会清理对应物理 block，并释放 native sequence slot 的 recurrent state。

## 10. Recurrent State 设计

Qwen3.5 的 recurrent 层需要保存两类状态：

```text
conv  state
delta state
```

`RecurrentStateCache` 为每个 native sequence slot 保存状态。开启 MTP 时，每个 slot 有 `K + 1` 个 snapshot plane：

```text
plane 0 = 最新 draft 后状态
plane K = pending/target token 后状态
```

MTP verify 后，如果接受了 `a` 个 draft，则选择：

```text
rollback_plane = K - a
```

例如 `K=3`：

```text
accept 3 -> plane 0
accept 1 -> plane 2
accept 0 -> plane 3
```

这让 recurrent state 能跟 PagedKV 一起回滚到“已提交 token 边界”。

## 11. MTP 推理流程

MTP 只在 decode 阶段运行。当前版本固定 greedy。

一次 MTP decode 的目标是：在一个 step 中最多提交多个 token。

流程：

```text
1. target 先处理 pending token，得到 target token T0 和 hidden state
2. MTP graph 用 hidden state + token embedding draft 出 D0
3. target 用 D0 做下一步 verify，得到 T1
4. 如果 D0 == T0，则接受 D0；否则立即停止，提交 T0
5. 重复直到 K 个 draft 或第一次 mismatch
6. 根据 accepted_drafts 回滚 recurrent state 和 PagedKV
7. MTP cache catch-up 到新的已提交边界
8. 返回 accepted drafts + final target token
```

返回形态：

```text
flat_token_ids  = [n_seqs, K+1]
output_counts   = 每个 sequence 实际返回 token 数
draft_counts    = 每个 sequence 实际 draft 数
```

Python `NativeRunner` 会把它转成：

```python
BackendExecutionResult(
    token_ids=[[...]],
    draft_token_counts=[draft_count],
)
```

注意：MTP 是执行优化，不改变 greedy 语义。开启或关闭 MTP 时，最终提交 token 必须与 target greedy 一致。

## 12. Vulkan Placement

Vulkan path 的关键不是“有些 tensor 在 Vulkan”，而是所有真实计算节点都必须在 Vulkan。

3.0 的 placement 规则：

- `VIEW`、`RESHAPE`、`PERMUTE`、`TRANSPOSE` 是纯 metadata view
- `CPY`、`SET_ROWS` 是计算节点，必须检查 compute backend
- Vulkan graph 构建后，runtime 调用 `set_all_compute_nodes_backend(..., Vulkan)`
- allocate/compute 后调用 `assert_all_compute_nodes_on_backend(..., Vulkan)`
- persistent weights、PagedKV、recurrent state、MTP KV 都分配在目标 backend 默认 buffer type

CIX 实机上已验证：

```text
Vulkan device = Mali-G720-Immortalis
weight buffer = Vulkan0
tensor_count  = 335
vulkan_compiled = true
```

## 13. 一次完整生成的时序

以 `enable_mtp=True` 的 decode 为例：

```text
LLM.generate()
    |
    v
LLMEngine.add_request()
    -> tokenizer.encode()
    -> Sequence(prompt_ids, sampling_params)
    -> Scheduler.waiting

while not finished:
    Scheduler.schedule()
        prefill:
            BlockManager.allocate()
            build prefill plan
        decode:
            BlockManager.may_append(speculative_tokens=K)
            build decode plan

    build_execution_plan()
        -> flatten token/position/block/slot info

    NativeRunner.run(plan)
        -> greedy validation
        -> seq_id -> native slot
        -> run_mtp() if decode and window fits

    Qwen35Runtime.run_mtp()
        -> target graph
        -> MTP graph
        -> target verify
        -> rollback state/cache
        -> return accepted tokens

    Scheduler.postprocess()
        -> append returned tokens
        -> stop on EOS/max_tokens
        -> release finished blocks

    NativeRunner.release_blocks()
        -> clear native KV/recurrent state

LLM.generate()
    -> tokenizer.decode()
    -> [{"text": ..., "token_ids": ...}]
```

## 14. 与 2.0 路径的差异

2.0 路径是 staged llama.cpp backend：

```text
nano-vLLM Python scheduler
    -> ctypes / external libnanollama_backend.so
    -> llama.cpp runtime
```

3.0 路径是 in-tree native runtime：

```text
nano-vLLM Python scheduler
    -> nanovllm._C
    -> Qwen35Runtime
    -> embedded GGML CPU/Vulkan backend
```

主要差异：

- 不需要 `--library-path`
- 不加载外部 `libnanollama_backend.so`
- 不构造 `llama_context`
- PagedKV 和 recurrent state 是 nano-vLLM runtime 自己的结构
- GGML 只作为 backend、tensor、graph、GGUF 和算子层使用
- CPU/Vulkan 是同一套 Qwen3.5 graph，仅 backend placement 不同

## 15. 当前验收记录

本机 CPU 已通过：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
NANOVLLM_TEST_QWEN35_GGUF=/home/kevin/model_raw/Qwen3.5-2B-Q4_0.gguf \
NANOVLLM_TEST_QWEN35_TOKENIZER=/home/kevin/model_raw/Qwen3.5-2B \
PYTHONPATH=. python3 -m unittest tests.test_native_qwen35_cpu_oracle -v
```

CIX Vulkan 已通过：

- `NANOVLLM_NATIVE_VULKAN=ON ./scripts/build_native_runtime.sh`
- `ctest` placement gate
- direct CPU/Vulkan target + MTP oracle
- full `LLM -> Scheduler -> BlockManager -> NativeRunner` CPU/Vulkan 路径
- `native_vulkan` CLI benchmark smoke

代表性输出：

```text
build_info:
  runtime=nanovllm_native
  ggml_commit=91c631b21d6e5d09e9c6659efdf6baeef5a44ddb
  cpu=True
  vulkan=True
  uses_llama_context=False

LLM output token ids:
  [0, 353, 1044, 264, 5286, 314]
```

## 16. 后续可演进点

3.0 当前把框架先跑通。后续优化可以分几条线继续：

- 多序列和更大 batch 的真实压力测试
- PagedKV 数据类型与布局优化
- recurrent state 内存压缩
- Vulkan graph 分配和复用优化
- MTP acceptance 统计、调度策略和更大 K 的验证
- wheel/安装流程自动化
- 更多 GGUF 合同和模型族支持
