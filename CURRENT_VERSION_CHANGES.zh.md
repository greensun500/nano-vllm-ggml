# nano-vLLM 当前版本修改说明：v3.1 MTP KV-only quick path

## 1. 版本定位

本文件只描述当前工作树版本。旧版本说明不在当前代码树中累积，历史内容由 Git 提交保存。

- 基线提交：`a27ba58`（v3.0 in-tree native runtime）
- 当前迭代：v3.1
- 固定 GGML：官方 llama.cpp `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- 模型：Qwen3.5-2B-Q4_0.gguf，内置单层 MTP
- 后端：CPU、Vulkan；不走 CUDA，不构造 `llama_context`
- 采样：greedy only

## 2. 本轮目标与结论

本轮优先删除 MTP 热路径里已经确认的冗余工作，并改善非 MTP 路径的同步/readback。实现与远程实测均已完成，但相对 v3.0 MTP-off 基线尚未达到 30% 提升，因此本版本不是性能优化终点。

结论如下：

1. MTP prefill 的额外维护成本已经从“每 token 一张完整 MTP 图”降为“每 sequence chunk 一张批量 KV-only 图”。
2. accepted path 的 MTP catch-up 从多张完整图变为一张 KV-only 图；立即拒绝时不再 catch-up。
3. 串行 target verification 在首次 mismatch 后停止，减少无效 target 图。
4. CPU/Vulkan 的非 MTP 性能约提升 0.1%～1.2%，证明 readback/sync 不是主瓶颈。
5. MTP decode 仍慢于 MTP-off：CPU tg128 -21.03%，Vulkan tg128 -28.57%。剩余主瓶颈是 K+1 次串行 target trunk/大词表 head。

下一轮必须实现单序列、多 token `TargetChunkGraph`，同时覆盖 prompt prefill 与 K+1 verification。

## 3. 代码修改

### 3.1 Batched MTP KV-only graph

新增批量状态维护图，输入为：

```text
tokens        I32 [N]
positions     I32 [4*N]
write_slots   I32 [N]
hidden_input  F32 [2048,N]
```

图中只执行：

```text
token embedding
  -> embedding/hidden RMSNorm + eh_proj
  -> MTP attention norm
  -> K/V projection
  -> K norm + IMRoPE
  -> contiguous
  -> SET_ROWS 到 MTP PagedKV
```

它明确不执行 Q/gate、KV gather、attention、output projection、FFN、output norm、词表 head 和 hidden readback。

完整 attention graph 与 KV-only graph 复用同一个 K/V projection、normalization、RoPE 和 store helper，避免布局或数值语义漂移。

### 3.2 Prefill MTP maintenance batching

普通 target prompt 仍按 token 运行，但运行过程中收集右移后的 target hidden：

```text
MTP hidden row 0 = 进入 chunk 前的 pending hidden
MTP hidden row i = target hidden[i-1]
```

chunk 结束后一次执行 KV-only graph。对 pp512，每个请求的 MTP maintenance graph 从约 512 张完整 MTP 图降为 1 张轻量图。

### 3.3 Catch-up 去重与批处理

本轮第一个 MTP draft 已经用完全相同的 `x_p + h_{p-1} + position + slot` 写入第 0 行，因此 post-verification catch-up 从 index 1 开始：

- `accepted == 0`：不执行 catch-up；
- `accepted > 0`：将 `1..accepted` 的 token、position、slot、target hidden 一次送入 KV-only graph。

### 3.4 Verification early-stop

串行 target verification 现在边计算边比较。首次 mismatch 位于 `index=a` 时，该图已经产生 correction token、`target_hidden[a]` 和 recurrent plane `K-a`，后续 target 图无需再执行。full accept 仍执行第 K 行得到额外 correction token。

### 3.5 Readback 与同步

- `execute_target()`/`execute_mtp()` 只在调用方消费 hidden 时才读回；
- 非 MTP prefill/decode 不再读取无用 hidden；
- 最后一个 draft 不再读取无消费者的 hidden；
- 删除同步 `graph_compute` 之后的重复 `synchronize()`；
- 删除每图入口的重复 scheduler reset，保留 RAII 出口 reset。

### 3.6 Benchmark 口径

CLI 新增：

```text
processed_tokens = prefill_tokens + decode_tokens
processed_tok_s  = processed_tokens / total_s
```

组合项使用 `processed_tok_s`，与 llama-bench 的 pp+tg 分子一致，不再误用只统计 completion 的 `generated_tok_s`。

## 4. 正确性证据

本地与远程均通过：

- C++ `graph_executor_placement`；
- 35 项 Python 单元测试；
- 4 项真实模型 CPU oracle；
- CPU/Vulkan 完整扩展构建；
- Vulkan MTP 实际推理烟雾测试。

真实模型 oracle 覆盖：

- full accept：`a=3 -> plane 0`；
- partial accept：`a=1 -> plane 2`；
- immediate reject：`a=0 -> plane 3`；
- 每种情况的下一轮 token，用于验证 rollback 和 target-conditioned MTP KV；
- MTP on/off 的 nano-vLLM scheduler 路径 greedy token 一致。

## 5. 远程基准口径

远程：`cix@172.16.64.219`，12 核 Cortex-A720/A520，Mali-G720-Immortalis。

公共参数：

```text
max_model_len=768
max_num_batched_tokens=768
max_num_seqs=1
num_kvcache_blocks=3
threads=8
temperature=0
warmup=1
repeat=3
MTP K=3
```

测试映射：

```text
pp512        = prompt_len 512, gen_len 1，读取 prefill_tok_s
tg128        = prompt_len 1, gen_len 129，读取 decode_tok_s
pp521+tg128  = prompt_len 521, gen_len 129，读取 processed_tok_s
```

`gen_len=129` 是因为第一个 completion 随 prefill 产生，随后正好保留 128 个 decode step。

## 6. 实测结果

表中变化均相对同一机器上的 v3.0 MTP-off 基线。

### CPU

| 版本/模式 | pp512 tok/s | 变化 | tg128 tok/s | 变化 | pp521+tg128 tok/s | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off 基线 | 18.669 | - | 15.431 | - | 17.640 | - |
| v3.1 MTP-off | 18.850 | +0.97% | 15.453 | +0.14% | 17.785 | +0.82% |
| v3.1 MTP-on | 18.792 | +0.66% | 12.185 | -21.03% | 16.811 | -4.70% |

### Vulkan

| 版本/模式 | pp512 tok/s | 变化 | tg128 tok/s | 变化 | pp521+tg128 tok/s | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off 基线 | 31.797 | - | 21.139 | - | 28.706 | - |
| v3.1 MTP-off | 32.173 | +1.18% | 21.343 | +0.97% | 28.919 | +0.74% |
| v3.1 MTP-on | 31.744 | -0.17% | 15.099 | -28.57% | 26.081 | -9.14% |

## 7. 本轮反思

本轮达成了预期的“清掉确定性冗余”，尤其是 MTP prefill 维护几乎不再增加时间；但没有达到 30%，原因不是接受率不足，而是 verification 的计算组织仍然错误：

```text
当前 K=3：3 张 draft 图 + 最多 4 张 target 图 + 1 张 KV-only catch-up 图
```

target 的 24 层 trunk 和约 398 MiB tied head 仍按 token 重复扫描。下一轮改为：

```text
3 张 draft 图 + 1 张 K+1 target chunk 图 + 至多 1 张 KV-only catch-up 图
```

同时普通 pp512 要从 512 张 target token graph 改为一张 target chunk graph。只有完成这项结构变化，才可能同时达到 pp、tg 与组合项的 30% 目标。
