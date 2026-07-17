# nano-vLLM 当前版本修改说明：v3.3 Arm/Mali 执行路径调优

## 1. 版本定位

本文件只描述当前代码版本。旧版本说明不在工作树中累积，历史由 Git 提交保存。

- 基线提交：`65f8b79`（v3.2 TargetChunkGraph）
- 当前迭代：v3.3
- GGML：官方 llama.cpp `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- 模型：Qwen3.5-2B-Q4_0.gguf，内置单层 MTP
- 后端：CPU、Vulkan；不使用 CUDA、`libllama` 或 `llama_context`
- 采样：greedy only

v3.3 修正 AArch64 构建未启用 dotprod/i8mm 的问题，并针对 Mali-G720 将长 target prefill 自动切成 64-token 图。CPU 的三项正式指标均比 v3.0 MTP-off 基线高 30% 以上；Vulkan prefill 和组合吞吐大幅提高，但默认的单 token MMVQ 路由仍使普通 decode 回退，已通过独立 A/B 定位，留给下一迭代按 shape 修正。

## 2. 本轮修改

### 2.1 Arm dotprod+i8mm 自动探测

`scripts/build_native_runtime.sh` 不再假设异构 Arm 上的 `-mcpu=native` 一定包含所有 GGML 需要的 ISA：

1. 仅在主机为 `aarch64/arm64`，且调用方没有显式传入 `GGML_NATIVE` 或 `GGML_CPU_ARM_ARCH` 时进入自动探测。
2. `/proc/cpuinfo` 的每个 `Features` 行都必须含有 `asimddp` 和 `i8mm`，避免只在部分 CPU 核支持时生成不可安全调度的指令。
3. 当前 C 编译器必须通过 `-march=armv8.6-a+dotprod+i8mm` 编译探针。
4. 成功后注入 `GGML_NATIVE=OFF` 和 `GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm`。
5. `NANOVLLM_NATIVE_ARM_ARCH=off` 可关闭自动选择；其他值可显式指定架构；命令行 CMake 参数始终优先。

远程构建已出现预期日志：

```text
nano-vLLM: enabling Arm dotprod+i8mm kernels (armv8.6-a+dotprod+i8mm)
```

all-CPU 检查中的 awk 循环变量使用 `field`，兼容远端 mawk，避免将内建函数名 `index` 当成变量而产生语法错误。

### 2.2 Vulkan shader compiler 显式入口

新增 `NANOVLLM_VULKAN_GLSLC`：设置后会校验文件可执行，并向 CMake 传递 `Vulkan_GLSLC_EXECUTABLE`。远程使用新版 glslc 后启动信息为：

```text
Mali-G720-Immortalis | int dot: 1 | matrix cores: KHR_coopmat
```

该信息证明设备能力和 shader 构建路径可用，不代表所有矩阵 shape 已自动选择最优 kernel。

### 2.3 `TargetChunkOutputMode::None`

TargetChunkGraph 现在有三种输出模式：

```text
None -> 0 个 prediction；按 retain_hidden 决定是否保留 normalized hidden
Last -> 只对最后一列执行 tied vocab head/argmax
All  -> 对全部 T 列执行 tied vocab head/argmax
```

- `None + retain_hidden=false` 只执行 target trunk 及其 KV/recurrent side effects，省去 output norm、约 248K 行 tied vocab head、argmax 和 token readback。
- `None + retain_hidden=true` 仍产生 MTP maintenance 所需的 normalized hidden，但跳过 vocab head、argmax 和 token readback。
- runtime 按 `None/Last/All` 推导 0/1/T 个 prediction，仅在输出存在时读取 `greedy_tokens`。

### 2.4 Mali-G720 的 64-token target prefill

在当前模型和 Mali-G720 上，T=64 的 target projection 比单张 T=512 图更快。因此 native runtime 在不改变 Python scheduler admission 的前提下，将一个长 sequence chunk 内部分片：

```text
Python scheduler 提交 T-token plan
  -> CPU / 非 Mali Vulkan：一张 TargetChunkGraph(T, Last)
  -> Mali Vulkan：按 64 token 循环
       非最终片：TargetChunkGraph(None)
       最终片：TargetChunkGraph(Last)
       每片重新展开 context read slots
       每片提交最新 recurrent state
       整段只返回最后一片的 greedy token
```

MTP 开启时，每片仍读取 target hidden、执行 batched MTP KV-only maintenance，并保持跨片右移关系：

```text
mtp_hidden[0] = 进入当前片前的 pending_hidden
mtp_hidden[i] = 当前片 target_hidden[i-1]
pending_hidden = 当前片最后一个 target hidden
```

因此分片没有改变 block table、slot mapping、PagedKV 所有权或 MTP pending-hidden 语义。

## 3. 正确性验证

本地：

- `bash -n scripts/build_native_runtime.sh` 与 `git diff --check` 通过；
- 34 项 Python 测试通过，5 项按环境跳过；
- C++ `graph_executor_placement` 通过；
- 4 项真实 Qwen3.5-2B CPU oracle 全部通过，覆盖 target trace、MTP 全接受/部分接受/立即拒绝、rollback 后下一轮、near-boundary 和完整 scheduler 路径。

远程：

- CPU+Vulkan extension 使用 Arm dotprod+i8mm 和指定 glslc 构建成功；
- build info 对应官方 GGML 完整 hash；
- 34 项 Python 测试通过，5 项跳过，4 项 CPU 真实模型 oracle 通过；
- Vulkan fixed trace 与 MTP K=3 full/partial/reject、rollback 后下一轮、near-boundary oracle 通过；
- 512-token 长提示分别在 MTP-off/on 下比较 scheduler 直接 64-token chunk 与 runtime 内部 64-token 分片，8 个输出 token 逐 token 一致。

长分片等价结果：

```text
MTP=False: [15, 15, 15, 15, 15, 15, 15, 15]
MTP=True:  [9419, 1814, 9419, 1814, 9419, 1814, 9419, 1814]
```

## 4. 正式基准口径

远程硬件为 12 核 Cortex-A720/A520 和 Mali-G720-Immortalis。公共参数：

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
```

测试映射：

```text
pp512       = prompt_len 512, gen_len 1，读取 prefill_tok_s
tg128       = prompt_len 1, gen_len 129，读取 decode_tok_s
pp521+tg128 = prompt_len 521, gen_len 129，读取 processed_tok_s
```

所有“相对基线”均相对同机 v3.0 MTP-off。MTP 的纯净收益另外用同版本 on/off 表示，二者不能混用。

## 5. CPU 正式结果

| 版本/模式 | pp512 tok/s | 相对基线 | tg128 tok/s | 相对基线 | pp521+tg128 tok/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off | 18.669 | - | 15.431 | - | 17.640 | - |
| v3.3 MTP-off | 68.737 | +268.18% | 20.141 | +30.52% | 36.042 | +104.32% |
| v3.3 MTP-on, K=3 | 68.197 | +265.29% | 21.038 | +36.34% | 39.053 | +121.39% |

同版本 MTP-on 相对 MTP-off：pp512 -0.79%，tg128 +4.46%，组合 +8.35%。CPU 的主要增益来自 v3.2 multi-token graph 与本轮 Arm ISA 修正的叠加，不能全部归因于 MTP。

## 6. Vulkan 正式结果

### 6.1 当前默认行为

| 版本/模式 | pp512 tok/s | 相对基线 | tg128 tok/s | 相对基线 | pp521+tg128 tok/s | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 MTP-off | 31.797 | - | 21.139 | - | 28.706 | - |
| v3.3 MTP-off | 72.227 | +127.15% | 17.994 | -14.88% | 40.144 | +39.85% |
| v3.3 MTP-on, K=3 | 69.224 | +117.70% | 17.887 | -15.38% | 42.384 | +47.65% |
| v3.3 MTP-on, K=2 | - | - | 19.187 | -9.23% | 41.566 | +44.80% |

同版本比较：K=3 相对默认 off 的 tg128 -0.59%、组合 +5.58%；K=2 的 tg128 +6.63%、组合 +3.54%。K=3 在短 prompt repeat=3 中的接受率为 75%，K=2 为 81.42%；两者在组合项的长 prompt 中均为 100%。接受率差异是吞吐的重要变量。

### 6.2 MTP-off 的 DMMV 诊断配置

设置 `GGML_VK_DISABLE_MMVQ=1` 后：

| 模式 | pp512 tok/s | tg128 tok/s | pp521+tg128 tok/s |
|---|---:|---:|---:|
| 默认 MTP-off | 72.227 | 17.994 | 40.144 |
| 全局 DMMV MTP-off | 72.262 | 21.287 | 42.991 |
| 变化 | +0.05% | +18.30% | +7.09% |

该环境变量不能作为最终默认：它虽然恢复 T=1 普通 decode，却同时使 MTP K=2 从 quick probe 的 21.533 降到 17.411 tok/s。下一版应只让 Mali/intdot/Q4_0 的 T=1 走 DMMV，T=2～4 继续走 MMVQ。

## 7. Vulkan A/B 诊断

以下为 warmup=1、repeat=1，目的是定位 kernel/submit，不与正式表混算。

### 7.1 MMVQ 路由

| 路由 | MTP-off tg128 | MTP K=2 tg128 | K=2 acceptance |
|---|---:|---:|---:|
| 默认 | 18.091 | 21.533 | 97.73% |
| 强制 MMVQ | 14.666 | 15.947 | 97.73% |
| 禁用 MMVQ | 21.285 | 17.411 | 97.73% |

结论：T=1 的 Q8_1 激活量化/额外 dispatch 无法摊薄，DMMV 更优；T=2～4 的 integer-dot MMVQ 更优，需要 shape-aware 混合路由。

### 7.2 `GGML_VK_MAX_NODES_PER_SUBMIT`

| max nodes | MTP-off tg128 | MTP K=2 tg128 |
|---:|---:|---:|
| 20 | 17.861 | 21.328 |
| 100 | 18.080 | 21.463 |
| 512 | 18.073 | 21.479 |
| 2048 | 18.154 | 21.424 |

100～2048 基本处于噪声带，20 反而退化。节点硬上限不是主因；官方 Vulkan 当前用上一张 graph 的 `last_total_flops / 40` 决定下一张 graph 的提交粒度，交替的小 draft/大 target 图仍值得按当前 graph FLOPs 继续优化。

## 8. 本轮反思

已经成立：

- Arm dotprod+i8mm 修正确实解除 CPU 量化 kernel 的主要限制；
- Mali T=64 prefill 策略将 Vulkan pp512 提高到 72 tok/s 以上；
- `None` 安全地消除了非最终 MTP-off 分片的 output norm/head/readback；
- 跨分片的 target KV、GDN recurrent state 和 MTP pending-hidden 语义已被真实模型等价测试证明。

尚未解决：

1. Vulkan 默认 T=1 错选 MMVQ，使普通 decode 比 v3.0 低约 15%。
2. K 必须同时考虑接受率、verification 宽度和 draft/target kernel，单次 K sweep 不能直接决定永久默认。
3. 每轮仍有 K 张串行 draft graph；每张扫描约 398 MiB tied Q6_K vocab head。
4. verification 仍读取 predictions 和全部 hidden，并在 host 计算 accepted；catch-up 是独立 graph。
5. 18 层 GDN 仍物化 K+1 份 snapshot；graph metadata/context 每轮重建。
6. CPU loader 尚未把 Q4 linear/tied output head 放入 CPU_REPACK。

下一迭代按可独立归因的顺序推进：Mali Q4 shape-aware kernel 路由、当前图 submit 策略、packed readback 与 catch-up 融图、Q4 MMVQ 多列 shader、CPU_REPACK、持久化 DraftChain/Round graph、snapshot 选择性提交和 draft-only 低精度 head。
