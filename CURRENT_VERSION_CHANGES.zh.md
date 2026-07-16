# nano-vLLM 当前版本修改说明：v3.2 TargetChunkGraph

## 1. 版本定位

本文件只描述当前代码版本。旧版本说明不在工作树中累积，历史由 Git 提交保存。

- 基线提交：`fb5a028`（v3.1 MTP KV-only quick path）
- 当前迭代：v3.2
- GGML：官方 llama.cpp `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- 模型：Qwen3.5-2B-Q4_0.gguf，内置单层 MTP
- 后端：CPU、Vulkan；不使用 CUDA、`libllama` 或 `llama_context`
- 采样：greedy only

## 2. 本轮目标与结论

v3.2 将单序列的多个 target token 合并到一张 GGML graph，覆盖普通 prompt prefill 和 MTP verification。目标是减少重复 graph 调度，并让同一权重的多个 token projection 共享一次矩阵计算。

结果分成两部分：

1. 结构改造与正确性验证完成。CPU/Vulkan 的 greedy token、MTP full/partial/reject、rollback 后下一轮状态均通过真实模型 oracle。
2. 性能目标未完成。CPU prefill 提升 50.3%，组合项提升 33.8%；但 CPU/Vulkan MTP decode 分别比 v3.0 MTP-off 基线慢 24.2% 和 33.9%。

本轮揭示了一个重要事实：正确的批处理结构不等于当前硬件上一定更快。远程 ARM CPU 的原构建没有启用 dotprod/i8mm，也没有把 Q4/Q6 权重放入 CPU_REPACK；Mali-G720 对 `T<=8` 选择 DMMV 小批量 kernel。固定 `T=4` 还会在首次 mismatch 后继续计算，而 v3.1 平均只执行约 3.25～3.33 个 target 行。

因此 v3.2 是已验证但未达 30% 门槛的中间迭代，下一轮继续优化。

## 3. 代码修改

### 3.1 单序列多 token TargetChunkGraph

新增 `TargetChunkGraph`，输入为：

```text
tokens        I32 [T]
positions     I32 [4*T]
write_slots   I32 [T]
read_slots    I32 [C]
causal_mask   F32 [C,T]（T=1 时省略）
```

一次 graph 中执行完整 24 层 target trunk，并把输出模式分为：

- `Last`：只对最后一列 hidden 执行 tied vocab head，供普通 prefill/decode 使用；
- `All`：对全部 T 列执行 vocab head，供 MTP greedy verification 使用。

`Last` 避免在长 prompt 上物化约 `vocab_size × T` 的 logits。

### 3.2 多 token causal attention

6 个 full-attention 层一次投影 T 个 token 的 Q/K/V，批量写入 PagedKV，然后按 block table 展开的物理 slot gather 完整上下文。

mask 语义为：第 i 个 query 只能看到历史 KV 和当前 chunk 的 `0..i` 行。它同时满足：

- prompt chunk causal prefill；
- MTP verification 的 `[x_p,d1,...,dK]`；
- 非连续物理 KV slot 下的逻辑时序。

### 3.3 多 token Gated Delta Net

18 个 recurrent 层改为在一张图内按 token 递推：

- convolution 输入由旧的 3 行状态与本 chunk 的 T 行 QKV 拼接；
- delta state 在 graph 内逐 token 更新；
- 普通 prefill 只提交最新 canonical state；
- verification 按 newest-first 写入 K+1 个 snapshot plane。

verification 输入 index 与 snapshot plane 的关系为：

```text
index 0 -> plane K
index 1 -> plane K-1
...
index K -> plane 0
```

接受 a 个 draft 后选择 `plane K-a`，无需重算 target trunk。

### 3.4 普通 prefill

`run()` 不再逐 token 建 target graph，而是每个 scheduler sequence chunk 执行一次 `TargetChunkGraph(Last)`：

```text
T 个 token
  -> 一次 24-layer target chunk
  -> 批量写 target PagedKV
  -> 提交最新 recurrent state
  -> 仅最后一列 vocab head/argmax
```

CPU pp512 因此从约 512 张 target graph 降为 1 张，并获得本轮最明显的收益。

### 3.5 MTP prefill hidden 右移

开启 MTP 时，target chunk 返回全部 target hidden。MTP maintenance 输入保持严格右移：

```text
mtp_hidden[0] = 进入 chunk 前的 pending_hidden
mtp_hidden[i] = target_hidden[i-1]
```

随后一次执行 batched MTP KV-only graph，chunk 最后一行 target hidden 成为下一轮 `pending_hidden`。

### 3.6 一图 verification

K 个 draft 仍串行产生，target verification 从最多 K+1 张 token graph 改为一张：

```text
verification_inputs = [x_p, d1, ..., dK]
TargetChunkGraph(T=K+1, output=All, snapshots=K+1)
  -> K+1 个 target predictions
  -> K+1 个 target hidden
  -> K+1 组 recurrent snapshots
```

host 端比较 draft 与 target prediction，得到 accepted 数 a，选择 `plane K-a`，回滚未提交的尾部，并继续使用 v3.1 的 batched MTP KV catch-up。

### 3.7 兼容性

旧的单 token graph API 保留，新增 chunk graph 没有改变 Python scheduler 的 plan ABI、block table 所有权或 request 生命周期。

## 4. 正确性验证

本地验证：

- native CPU extension 构建通过；
- C++ `graph_executor_placement` 通过；
- 34 项 Python 测试通过，5 项按环境跳过；
- 4 项真实 Qwen3.5 CPU oracle 全部通过。

远程验证：

- CPU+Vulkan extension 构建通过；
- C++ 与 Python 测试通过；
- 4 项 CPU 真实模型 oracle 通过；
- Vulkan 专项 oracle 通过 fixed trace、full accept、partial accept、immediate reject、下一轮状态和 near-boundary 场景。

这些测试确认 v3.2 的下降来自执行成本与 kernel 选择，而不是 token 或状态语义错误。

## 5. 基准口径

远程：12 核 Cortex-A720/A520、Mali-G720-Immortalis。

公共参数：

```text
max_model_len=768
max_num_batched_tokens=768
max_num_seqs=1
num_kvcache_blocks=3
threads=8
device_index=0
temperature=0
warmup=1
repeat=3
MTP K=3
```

测试映射：

```text
pp512       = prompt_len 512, gen_len 1，读取 prefill_tok_s
tg128       = prompt_len 1, gen_len 129，读取 decode_tok_s
pp521+tg128 = prompt_len 521, gen_len 129，读取 processed_tok_s
```

## 6. 远程实测

变化均相对同机 v3.0 MTP-off 基线。30% 目标分别为：

- CPU：pp 24.270、tg 20.060、组合 22.932 tok/s；
- Vulkan：pp 41.337、tg 27.480、组合 37.317 tok/s。

### CPU

| 版本/模式 | pp512 tok/s | 相对基线 | tg128 tok/s | 相对基线 | pp521+tg128 tok/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off | 18.669 | - | 15.431 | - | 17.640 | - |
| v3.2 MTP-off | 28.065 | +50.32% | 15.380 | -0.33% | 23.599 | +33.78% |
| v3.2 MTP-on | 27.858 | +49.22% | 11.701 | -24.17% | 22.753 | +28.98% |

### Vulkan

| 版本/模式 | pp512 tok/s | 相对基线 | tg128 tok/s | 相对基线 | pp521+tg128 tok/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off | 31.797 | - | 21.139 | - | 28.706 | - |
| v3.2 MTP-off | 36.199 | +13.84% | 21.266 | +0.60% | 28.591 | -0.40% |
| v3.2 MTP-on | 35.821 | +12.65% | 13.978 | -33.88% | 25.998 | -9.43% |

K sweep 进一步确认批宽与 kernel 的相关性：K=1（target T=2）在 Vulkan 上优于 K=3（T=4），但仍未达到 tg128 的 30% 目标。

## 7. 本轮反思

### 7.1 已经成立的优化

- 长 prompt target batching 成立，CPU prefill/组合项显著提升；
- causal attention、multi-token GDN 与 snapshot rollback 语义成立；
- `Last` 模式成功避免长 prompt 的全量 logits；
- target verification 图数从 K+1 降到 1。

### 7.2 为什么 MTP decode 变慢

1. v3.1 可在 mismatch 后 early-stop，平均只执行约 3.25～3.33 个 target 行；v3.2 固定执行 4 行，多算约 20%～23%。
2. 原远程 ARM 构建的 `-mcpu=native` 没有启用 dotprod/i8mm；Q4_0/Q6_K 的 T=4 无法形成期望的多行高效矩阵内核。
3. nano loader 只使用默认 CPU buffer，`GGML_CPU_REPACK=ON` 并不代表模型权重实际进入 CPU_REPACK。
4. Vulkan 对 `T<=8` 固定选择 DMMV；Mali 无 int-dot 和 matrix-core，T=4 没有获得预期的权重复用。
5. 18 层 GDN 为 K+1 个候选状态分别执行 snapshot CPY。
6. K 个 MTP draft 仍是 K 张串行图，并且每张都扫描约 398 MiB tied Q6_K vocab head。

### 7.3 下一轮优先级

1. 修正 AArch64 构建自动探测，明确启用 dotprod+i8mm，并用运行时 feature query/反汇编验证。
2. CPU 按 tensor 使用 CPU_REPACK；保留 embedding 原布局，单独复制 tied output head 到 repack buffer。
3. Vulkan 实测 prompt chunk 和 verification T=2/3/4 的最优形状，不假设 CPU 最优形状适用于 Mali。
4. 引入 backend-aware K/shape 策略；必要时保留 early-stop 小图路径。
5. 将连续 draft 与 target verification 尽可能合并为单图，减少 graph build、同步和 host readback。
6. 避免物化所有 snapshot 副本，改为按 accepted index 提交或缩小复制范围。
7. 增加 graph-build、backend compute、readback、head、trunk、catch-up 分段计时，按实测继续迭代。
