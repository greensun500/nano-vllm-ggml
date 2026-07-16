# Nano-vLLM 3.0 MTP 推理性能审计

本文档记录当前 3.0 版本 native CPU/Vulkan 推理路径的静态代码审计结果，并结合 Qwen3.5-2B-Q4_0.gguf 的实际测试数据，分析开启 MTP 后推理变慢的原因。

> 术语修正：本文中的“没有独立 MTP head”只表示 GGUF 没有单独的 `blk.24.nextn.shared_head_head.weight` 词表输出矩阵，并不表示模型没有 MTP。该 GGUF 明确包含 Qwen3.5 的 MTP block（`qwen35.nextn_predict_layers=1`，额外的 `blk.24` 及 `blk.24.nextn.*` 权重）。

## 结论摘要

当前 3.0 的 MTP 是“正确性优先”的串行实现。开启 MTP 后变慢，主要不是因为接受率过低，而是因为每生成一个有效 token，平均执行了约 3 个计算图，并且重复执行约 2 次完整的 248,320 词表输出头。

实际测试中：

- Vulkan decode 时间从 8.9957 s 增加到 14.0660 s，增加约 56.4%。
- CPU decode 时间从 8.9343 s 增加到 13.2540 s，增加约 48.4%。
- Vulkan MTP 接受率约 82.42%，CPU MTP 接受率约 82.14%，接受率本身不是主要问题。

根据模型权重规模估算，MTP 每个输出 token 的权重访问量增加约 51%，与实测 decode 变慢幅度高度吻合。

## 已提出因素的核验结果

| 因素 | 结论 | 影响 |
| --- | --- | ---: |
| Target 验证串行执行 | 成立 | 极高 |
| 每 token 重建、分配计算图 | 成立 | 高 |
| 重复 synchronize/reset | 成立 | 高，Vulkan 更明显 |
| CPU/Vulkan 隐状态传输 | 成立 | 中高，主要是同步延迟 |
| MTP 导致 prefill 额外计算 | 成立 | 中 |
| rollback/catch-up 额外执行 | 成立 | 高 |
| Vulkan 算子意外回退 CPU | 基本排除 | 非主因 |
| 请求末尾多生成少量 token | 成立 | 低 |
| MTP 接受率过低 | 排除 | 当前约 82%，不是主因 |

## 1. 每轮执行的计算图过多

当 `draft_count=3` 时，一轮 MTP 固定执行：

1. 3 次 MTP draft 图；
2. 4 次完整 target 图；
3. `accepted + 1` 次 MTP catch-up 图。

对应实现位于 [qwen35_runtime.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/qwen35_runtime.cpp:597)。

以 Vulkan 测试数据为例：

- MTP rounds：55；
- drafted：165；
- accepted：136；
- target verification：`55 × 4 = 220`；
- catch-up：`136 + 55 = 191`；
- 总计算图数：`165 + 220 + 191 = 576`；
- 最终有效 decode token：189。

因此：

- 不开 MTP：约 1 个图/token；
- 开 MTP：约 3.05 个图/token；
- 完整词表 head：约 2.04 次/token。

当前实现减少的是 Python 调度轮数，并没有减少底层模型执行工作量。

## 2. 共享词表输出头是核心瓶颈

该 GGUF 包含完整的 Qwen3.5 MTP block，包括 `blk.24.nextn.eh_proj.weight`、`enorm`、`hnorm`、MTP attention/FFN 权重以及 `nextn.shared_head_norm.weight`。但是，它没有单独的 MTP 词表输出矩阵 `blk.24.nextn.shared_head_head.weight`；因此 MTP 的最后词表投影回退使用全局 `token_embd.weight`。这是当前模型允许的 tied-head 结构，绑定逻辑见 [weights.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/models/qwen35/weights.cpp:338)。

模型权重规模约为：

- 主模型 24 层：738.5 MiB；
- MTP 层：32.7 MiB；
- 共享词表 head：397.9 MiB；
- 词表大小：248,320；
- head 类型：Q6_K。

每次 `emit_greedy=true` 都执行完整 head matmul 和 argmax，见 [graph.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/models/qwen35/graph.cpp:400)。

按 Vulkan 平均每轮接受 2.47 个 draft、输出 3.47 个 token 估算：

```text
4 × target(738.5 + 397.9)
+ 3 × draft(32.7 + 397.9)
+ 3.47 × catchup(32.7)
≈ 5950 MiB/轮

5950 / 3.47 ≈ 1714 MiB/输出 token
```

不开 MTP 时约为：

```text
738.5 + 397.9 ≈ 1136 MiB/token
```

理论权重访问量增加约 50.9%，与 CPU 约 48.4%、Vulkan 约 56.4% 的 decode 时间增幅基本一致。

因此，当前 MTP 性能优化的第一优先级不是继续提高接受率，而是减少重复 target/head 执行，并将多个 hidden 行合并执行 head GEMM。

## 3. 每次都重建计算图，没有复用

每次 target/MTP 调用都执行以下流程：

1. `executor->reset()`；
2. 创建新的 32 MiB `GraphContext`；
3. 重新 build graph；
4. 重新分配 transient graph；
5. 遍历并验证 placement；
6. compute；
7. synchronize；
8. RAII guard 析构时再次 reset/synchronize。

实现见 [qwen35_runtime.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/qwen35_runtime.cpp:423) 和 [graph_executor.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/graph_executor.cpp:157)。

`GraphExecutor::reserve()` 虽然已经实现，但当前运行路径没有使用。MTP 每轮包含多个小图，因此图构建、分配和临时内存管理成本会被放大。

## 4. 同步次数过多

GGML 的 `ggml_backend_sched_graph_compute()` 本身已经是同步调用，但 runtime 随后又显式调用 `executor->synchronize()`。

此外还包括：

- graph 开始前 reset/synchronize；
- graph 结束时 RAII reset/synchronize；
- hidden tensor get 同步读回；
- greedy token get；
- 下一张图开始前再次 reset/synchronize。

CPU 上主要表现为函数和线程同步开销；Vulkan 上可能进一步变成 queue/fence 等待。

## 5. hidden 状态被频繁读回再上传

每个 target/MTP 图都会把 2048 个 F32 hidden 读回 CPU，大小约 8 KiB：

- target hidden：backend → CPU；
- MTP hidden input：CPU → backend；
- MTP hidden output：backend → CPU。

数据量本身不大，但同步成本较高。尤其 catch-up 图的返回 hidden 最终没有使用，却仍然执行了读回。

每张图还会执行 4 次公共输入 `tensor_set`，MTP 再增加 1 次 hidden 上传，见 [qwen35_runtime.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/qwen35_runtime.cpp:399)。如果 Vulkan 输入 buffer 不是 host-visible，每次写入还会经过 staging submit 和 fence。

## 6. Prefill 开启 MTP 后增加额外计算

普通 prefill 按 token 执行 target 图。开启 MTP 后，每个 prefill token 还会执行一次不带 head 的 MTP 图，用于建立 MTP 状态，见 [qwen35_runtime.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/qwen35_runtime.cpp:520)。

这与实测一致：

- Vulkan prefill 时间增加约 8.7%；
- CPU prefill 时间增加约 5.6%。

## 7. rollback/catch-up 开销

draft 阶段写入的是 draft-conditioned MTP KV。target 验证后，需要回滚 recurrent snapshot，并通过 catch-up 图重新生成 target-conditioned 的 MTP KV。

recurrent rollback 本身只是切换 snapshot plane，成本较低；真正昂贵的是后续 catch-up 图和 KV 写入。

## 8. 其他重要性能因素

### 8.1 没有真正的 backend batching

Python scheduler 可以一次调度多个序列或多个 prefill token，但 native runtime 内部仍然逐序列、逐 token 执行单 token graph。

当前没有真正实现：

- 批量 prefill GEMM；
- 多序列 decode batching；
- 一次图验证 K+1 个 target token；
- 一次 head GEMM 计算多行 hidden。

这是当前实现与正式 llama.cpp 推理性能差距的重要来源。

### 8.2 CPU_REPACK 没有接入 native 权重加载

GGML 编译了 `GGML_CPU_REPACK`，但 native loader 始终使用 backend 默认 buffer type，见 [gguf_loader.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/gguf_loader.cpp:88)。

正式 llama.cpp loader 会枚举 CPU extra buffer types，并为合适的量化权重选择 `CPU_REPACK`。当前 native loader 绕过了这一步，因此 Q4_0/Q6_K 的 CPU 重排优化很可能没有启用。

该因素对 CPU 可能有较大影响，需要 A/B 测试量化。

### 8.3 PagedKV 还不是高效 paged-attention

当前 KV cache 使用 F32。每个 token 都会：

- 在 CPU 生成完整物理索引；
- 上传 `read_slots`；
- 对完整上下文执行 `ggml_get_rows`；
- gather K/V；
- 执行普通 matmul 和 softmax。

当前没有 Flash Attention 或专用 paged-attention kernel。相关代码见 [paged_kv.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/paged_kv.cpp:455) 和 [graph.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/models/qwen35/graph.cpp:82)。

在当前 32～96 token 的短上下文测试中影响尚未最大，但上下文变长后会快速成为主要瓶颈。

### 8.4 placement 检查在热路径执行

每张图执行前都会遍历全部 graph nodes，检查 backend、解析 view source、构造字符串和审计结果，见 [graph_executor.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/graph_executor.cpp:393)。

这对调试和正确性验证有帮助，但正式性能路径应改成初始化检查或 debug 选项。

### 8.5 recurrent snapshot 状态较大且反复清零

Qwen3.5-2B 有 18 个 recurrent 层。估算如下：

- 单 snapshot plane、单序列约 19.3 MiB；
- MTP K=3 需要 4 个 plane，约 77.1 MiB/序列；
- 比不开 MTP 多约 57.8 MiB/序列。

这与测试中 MTP 多出的约 55～60 MiB RSS 基本一致。

序列创建和释放都会同步并清零所有 plane，见 [recurrent_state.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/recurrent_state.cpp:293)。

默认 `max_num_seqs=512` 对 native MTP 很危险，理论上 recurrent state 可能接近 39 GiB；当前 CLI 使用 `max_num_seqs=1` 时没有这个问题。

### 8.6 完成请求时清除完整 KV block

block size 强制为 256 的倍数。即使请求只使用了 96 token，释放时也会同步并清除完整 256-token block，以及所有 target/MTP KV tensor，见 [paged_kv.cpp](/home/kevin/kevin_prj/nano-vllm/csrc/runtime/paged_kv.cpp:495)。

短请求 benchmark 的最后一次 decode 时间会包含这部分开销。

### 8.7 native 没有 prefix cache

native backend 当前禁用了 prefix cache。多轮对话时，每轮都会重新 prefill 完整历史，实际聊天延迟会随历史长度增长。

这不影响本次单请求 benchmark，但会显著影响真实 CLI 对话体验。

## 9. 当前基本可以排除的因素

### Vulkan 计算偷偷回退 CPU

runtime 会将 Vulkan 图中的计算节点设置到 Vulkan backend，并在每张图上执行 placement assertion，因此当前没有证据表明存在大规模的静默 CPU fallback。

不过，所有小型输入处理、SET_ROWS、CPY 等操作也被强制放在 Vulkan 上，可能造成大量小 dispatch。这属于 Vulkan 调度效率问题，而不是 backend 误回退。

### MTP 接受率过低

当前接受率约 82%，属于较好的 speculative decoding 接受率。即使进一步提高接受率，也无法消除当前重复 target/head 执行的结构性成本。

### 末尾 token overrun

当前测试中：

- Vulkan 返回 191 个 speculative token，最终保留 189 个；
- CPU 返回 194 个，最终保留 189 个。

这带来约 1%～3% 的额外计算，只是小问题，不是 MTP 变慢的主要原因。

## 10. 性能统计口径问题

当前 benchmark 的 `verification_steps` 实际统计的是 MTP round/sequence 次数，并不是 target graph 数量，见 [runner.py](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/native/runner.py:199)。

后续性能验收建议增加以下计数器：

- `target_graph_calls`；
- `mtp_draft_graph_calls`；
- `mtp_catchup_graph_calls`；
- `lm_head_calls`；
- `target_tokens_per_graph`；
- `accepted_tokens_histogram`；
- `rejection_position_histogram`。

同时拆分以下耗时：

- graph build；
- placement audit；
- graph allocate；
- 输入上传；
- backend compute；
- hidden/token 读回；
- reset/synchronize；
- recurrent/KV 清理。

否则很难判断优化究竟减少了模型计算，还是只减少了 Python 调度时间。

## 11. 推荐优化顺序

建议按以下顺序推进：

1. 将 K+1 个 target verification token 合并到一个 target graph；
2. 将 K+1 个 target hidden 合并执行一次词表 head GEMM/argmax；
3. 复用计算图和 workspace，移除重复 reset/synchronize；
4. 将 placement audit 改为初始化检查或 debug 选项；
5. 避免无用 hidden 读回，尽量保持 hidden 在 backend 内；
6. 实现真正的批量 prefill 和多序列 decode；
7. CPU loader 接入 extra buffer type/CPU_REPACK；
8. 再优化 PagedKV、Flash Attention、状态懒分配和 prefix cache。

其中前两项最关键。只清理 Python 容器分配或调整线程数，无法解决当前约 50% 的结构性额外权重访问。
