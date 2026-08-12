# nano-vLLM 当前版本修改说明：v3.5 多 Executor 固定图复用

## 1. 版本定位

本文件只描述当前代码版本。旧版本说明由 Git 历史保存，不在工作树中累积。

- 基线版本：v3.4 Mali Q4_0 shape-aware kernel 路由
- 当前迭代：v3.5
- llama.cpp 官方基线：`91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- nano-vLLM 子模块补丁：`5ed3338`（Mali scalar Q4_0 使用 DMMV）
- 模型：Qwen3.5-2B-Q4_0.gguf，内置单层 MTP
- 后端：CPU、Vulkan；greedy only；不使用 CUDA、`libllama` 或 `llama_context`

v3.5 在 native Qwen3.5 runtime 中加入 persistent graph bucket。decode/MTP 高频路径首次执行时构建固定 shape GGML graph，并为每个 bucket 常驻独立 `GraphExecutor`；后续轮次只更新输入 tensor、mask 和 hidden，再重复 `compute()`。权重、PagedKV 和 recurrent state 仍然只有一份，bucket 只保存 graph metadata、scheduler allocation 和临时激活 buffer。

## 2. 代码逻辑

新增 `enable_graph_reuse`，默认开启；CLI 支持 `--no-graph-reuse` 做 A/B。第一版只在 CPU 且 `max_num_seqs=1` 时启用 graph reuse，多序列和 Vulkan 仍走 v3.4 fallback 路径。

首批固定图：

- MTP-off target decode：`TargetChunkGraph(T=1, Last)`
- MTP draft：`TokenGraph(T=1)`
- MTP target verification：`TargetChunkGraph(T=K+1, All, retain_hidden=true)`
- MTP catch-up：`MtpKvUpdateGraph(T=1..K)`

`n_kv` 使用 bucket 固定 shape：

```text
1..128      -> 128
129..256    -> 256
257..512    -> 512
513..1024   -> 1024
1025..2048  -> 2048
2049..4096  -> 4096
更长        -> max_model_len
```

真实 `read_slots` 写在前半段，padding 部分重复最后一个真实 slot；单 token 和 chunk attention 都上传 causal mask，屏蔽 padding KV。这样不会截断长上下文，只是用 bucket 容量换取 graph 复用命中。

缓存统计通过 `Qwen35Runtime.graph_reuse_stats()` 暴露，benchmark 输出：

```text
graph_cache_hits
graph_cache_misses
graph_cache_evictions
graph_cache_active_entries
```

## 3. 正确性验证

本地验证：

- `cmake --build build/native-cpu -j8` 通过；
- `ctest --test-dir build/native-cpu --output-on-failure` 通过；
- `PYTHONPATH=. python3 -m pytest -q tests`：34 passed、6 skipped；
- 真实 Qwen3.5 CPU oracle：4 passed；
- 真实 Qwen3.5 CPU long-context oracle：1 passed，覆盖 bucket 升档。

本地 quick A/B（CPU、threads=4、prompt 128、gen 64、warmup=1、repeat=1）：

| 模式 | graph reuse | cache hits/misses | decode tok/s | processed tok/s |
|---|---:|---:|---:|---:|
| MTP-off | off | 0/0 | 13.436 | 30.622 |
| MTP-off | on | 125/1 | 14.282 | 32.072 |
| MTP-on K=3 | off | 0/0 | 19.945 | 40.740 |
| MTP-on K=3 | on | 160/5 | 19.973 | 40.687 |

该 quick A/B 只用于证明机制和初步趋势，不替代远程正式 benchmark。MTP-off 已看到约 +6.3% decode 改善；MTP-on 当前基本持平，说明 MTP 的主要瓶颈仍在 draft head、hidden readback、verification/catch-up 分图和 host acceptance。

远程 tg128（Cortex-A720/A520 + Mali-G720，threads=8、prompt 1、gen 129、warmup=1、repeat=3）：

| 后端/模式 | graph cache | acceptance | decode tok/s | 结论 |
|---|---:|---:|---:|---|
| CPU MTP-off reuse on | 509/3 | - | 22.464 | 基本持平 |
| CPU MTP-off reuse off | 0/0 | - | 22.544 | A/B 基线 |
| CPU MTP-on K=3 reuse on | 744/14 | 75.91% | 23.961 | +1.74% |
| CPU MTP-on K=3 reuse off | 0/0 | 75.00% | 23.550 | A/B 基线 |
| Vulkan MTP-off | 0/0 | - | 21.341 | v3.5 fallback |
| Vulkan MTP-on K=3 | 0/0 | 75.00% | 18.029 | v3.5 fallback，MTP 仍慢 |

远程 Vulkan long-context oracle 在 graph reuse fallback 后仍显示 MTP-on 与 MTP-off greedy trace 不一致；因此 v3.5 不把 Vulkan MTP 速度作为 correctness-passed 验收结果。该问题与 CPU graph reuse 无关，需要下一轮单独定位 Vulkan MTP/off trace 差异。

## 4. 使用方式

默认开启 graph reuse：

```bash
PYTHONPATH=. python3 -m nanovllm.cli.bench "$MODEL" \
  --backend native_cpu \
  --tokenizer "$TOKENIZER" \
  --prompt-len 1 --gen-len 129 \
  --warmup 1 --repeat 3 --json
```

A/B 禁用：

```bash
PYTHONPATH=. python3 -m nanovllm.cli.bench "$MODEL" \
  --backend native_cpu \
  --tokenizer "$TOKENIZER" \
  --prompt-len 1 --gen-len 129 \
  --warmup 1 --repeat 3 \
  --no-graph-reuse --json
```

MTP 模式继续增加：

```bash
--enable-mtp --mtp-max-draft-tokens 3
```

## 5. 本轮反思

成立的结论：

1. CPU persistent graph bucket 能在 decode 中产生稳定 cache hit，并保持真实模型 token 正确。
2. bucket 不保存 KV cache，不复制权重；淘汰 bucket 不会丢上下文。
3. 第一版对普通 decode 更有效；MTP-on 的收益被其它结构性成本抵消。

剩余主要成本：

- MTP draft 仍是 K 张串行 graph；
- MTP draft 仍每步扫描 tied Q6_K vocab head；
- hidden/token readback 与 host acceptance 仍在；
- target verification 与 MTP catch-up 仍分图；
- prefill/chunk 暂未固定；
- CPU 权重尚未进入 CPU_REPACK。
- Vulkan padded-mask bucket 当前会改变真实 greedy trace，v3.5 暂时自动 fallback；下一轮需要单独定位 Vulkan T=1 masked softmax/padding 语义。
