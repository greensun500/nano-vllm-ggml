# nano-vLLM 当前版本修改说明：v3.4 Mali Q4_0 shape-aware kernel 路由

## 1. 版本定位

本文件只描述当前代码版本。旧版本说明由 Git 历史保存，不在工作树中累积。

- 基线提交：`a7d4b24`（v3.3 Arm/Mali 执行路径调优）
- 当前迭代：v3.4
- llama.cpp 官方基线：`91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`
- nano-vLLM 子模块补丁：`5ed3338`（Mali scalar Q4_0 使用 DMMV）
- 模型：Qwen3.5-2B-Q4_0.gguf，内置单层 MTP
- 后端：CPU、Vulkan；greedy only；不使用 CUDA、`libllama` 或 `llama_context`

v3.4 只修改一个可独立归因的 Vulkan kernel 选择决策：Mali/int-dot/Q4_0 的单 token 矩阵使用 DMMV，多 token 小批量继续使用 MMVQ。正式测试中，MTP-off tg128 从 v3.3 默认的 17.994 提高到 21.336 tok/s（+18.58%），pp512 保持 72.270 tok/s，组合项提高到 43.006 tok/s。

该修复也消除了 v3.3 中“相对一个退化 MTP-off 路径”的假性 MTP 增益。以修复后的同版本 off 为基准，Vulkan K=2/K=3 的正式 decode 仍分别慢 9.56%/15.35%，说明下一阶段必须继续削减 draft head、graph 提交、readback、catch-up 和 snapshot 成本。

## 2. 问题定位

v3.3 的三路 quick A/B 已证明相反的 shape 偏好：

| 全局路由 | MTP-off tg128 | MTP K=2 tg128 |
|---|---:|---:|
| 默认 | 18.091 | 21.533 |
| 强制 MMVQ | 14.666 | 15.947 |
| 禁用 MMVQ | 21.285 | 17.411 |

官方 `ggml_vk_should_use_mmvq()` 对未知 Vulkan vendor 的默认值为 true，因此 Mali 的 T=1 也会量化 F32 activation 到 Q8_1，再执行 Q4_0×Q8_1 integer-dot MMVQ。单列时量化和额外 dispatch 无法摊薄；T=2～4 时，多列权重复用与 integer-dot 的收益则超过该开销。

全局设置 `GGML_VK_DISABLE_MMVQ=1` 不能成为正式方案，因为它恢复普通 decode 的同时会破坏 MTP verification 的多列路径。

## 3. 代码逻辑

llama.cpp 子模块新增 ARM Vulkan vendor ID `0x13b5`，并在 `ggml_vk_should_use_mmvq()` 的显式环境覆盖之后、通用 `n>1` 规则之前加入：

```text
if vendor == ARM/Mali
   and integer_dot_product
   and src0_type == Q4_0
   and RHS columns n == 1:
       use DMMV
```

最终路由为：

| 模式 | Mali Q4_0 T=1 | T=2～8 | T>8 |
|---|---|---|---|
| 默认 v3.4 | DMMV | MMVQ | 原矩阵路径 |
| `GGML_VK_FORCE_MMVQ=1` | MMVQ | MMVQ | 原矩阵路径 |
| `GGML_VK_DISABLE_MMVQ=1` | DMMV | DMMV | 原矩阵路径 |

环境变量判断仍位于特例之前，因此 FORCE/DISABLE 保留诊断覆盖能力，且 DISABLE 同时存在时仍按官方初始化顺序优先。补丁只限制已实测的 Q4_0，不改变 Q4_1、Q5、Q6_K、K-quant、NVIDIA、AMD、Intel、Qualcomm 或无 integer-dot 设备。

v3.3 的 Mali 64-token prefill 使用 T=64，不经过该小批量 helper；pp521 的最后一片为 T=9，也不受影响。MTP K=1/2/3 的 target verification 分别为 T=2/3/4，继续使用 MMVQ。

## 4. 正确性验证

本地/静态：

- llama.cpp 补丁 `git diff --check` 通过；
- 主仓库常规测试仍为 34 passed、5 skipped；
- 本地 C++ placement 与 4 项真实 CPU oracle 在 v3.3 收尾时通过。

远程 v3.4 独立 build：

- Arm dotprod+i8mm 探测、glslc、Vulkan int-dot、KHR cooperative matrix 和 C++ placement 均通过；
- runtime 报告官方 GGML 基线完整 hash；
- 34 项 Python 测试通过，5 项跳过；
- 4 项真实 Qwen3.5 CPU oracle 通过；
- 3 项真实 Vulkan oracle 通过 target fixed trace、MTP K=3 full/partial/reject、rollback 后下一轮和 near-boundary；
- MTP-off/on 的 512-token 外部 64 分块与 runtime 内部 64 分块仍逐 token 等价。

等价输出：

```text
MTP=False: [15, 15, 15, 15, 15, 15, 15, 15]
MTP=True:  [9419, 1814, 9419, 1814, 9419, 1814, 9419, 1814]
```

不同量化 kernel 不要求 logits bitwise 相同；验收使用数值路径的真实模型 greedy trace、rollback 状态和最终 token 序列。

## 5. 正式基准口径

远程硬件：12 核 Cortex-A720/A520、Mali-G720-Immortalis。公共参数：

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

指标映射：pp512 为 prompt 512/gen 1 的 `prefill_tok_s`；tg128 为 prompt 1/gen 129 的 `decode_tok_s`；组合为 prompt 521/gen 129 的 `processed_tok_s`。

## 6. CPU 正式结果

本轮补丁只编译进 Vulkan backend，CPU 数字用于每迭代全后端回归，不应归因于 shape policy。CPU 在本轮运行时的频率/负载状态优于 v3.3，因此同时列出同轮 on/off，后者才是可靠的 MTP 净增益。

| 模式 | pp512 tok/s | tg128 tok/s | pp521+tg128 tok/s |
|---|---:|---:|---:|
| v3.4 MTP-off | 76.448 | 22.160 | 40.102 |
| v3.4 MTP-on, K=3 | 75.594 | 23.278 | 43.495 |
| 同轮 MTP 变化 | -1.12% | +5.05% | +8.46% |

相对同机 v3.0 MTP-off，当前 CPU off 三项分别为 +309.49%、+43.61%、+127.34%；K=3 分别为 +304.92%、+50.86%、+146.57%。跨轮绝对变化含系统状态噪声。

## 7. Vulkan 正式结果

| 版本/模式 | pp512 tok/s | tg128 tok/s | pp521+tg128 tok/s |
|---|---:|---:|---:|
| v3.3 默认 MTP-off | 72.227 | 17.994 | 40.144 |
| v3.4 hybrid MTP-off | 72.270 | 21.336 | 43.006 |
| v3.4 hybrid MTP-on, K=3 | 69.184 | 18.062 | 42.429 |
| v3.4 hybrid MTP-on, K=2 | - | 19.296 | 41.674 |

v3.4 off 相对 v3.3 默认：pp512 +0.06%，tg128 +18.58%，组合 +7.13%。K=2/K=3 相对各自 v3.3 结果只有 +0.57%/+0.98% 的 decode 改善，符合“主要修复 T=1 target trunk，draft 大头仍在 Q6_K vocab head”的预期。

同版本比较：

| MTP 模式 | tg128 相对 off | 组合相对 off | 正式 acceptance |
|---|---:|---:|---:|
| K=2 | -9.56% | -3.10% | 短 prompt 81.42%，组合 100% |
| K=3 | -15.35% | -1.34% | 短 prompt 75%，组合 100% |

相对同机 v3.0 MTP-off，v3.4 off 为 pp +127.28%、tg +0.94%、组合 +49.82%。K=3 为 pp +117.58%、tg -14.55%、组合 +47.81%。

## 8. 本轮反思

成立的结论：

1. Mali Q4_0 的 kernel 选择确实必须依赖 RHS 列数，不能用全局 MMVQ 开关。
2. T=1 DMMV 与 T=2～4 MMVQ 可同时保留，prefill 不回退，普通 decode 恢复到 v3.0 以上。
3. quick probe 中 K=2 可达 21.612 tok/s，但 repeat=3 不同首 token 的接受率下降后只有 19.296；K 的选择必须结合接受率分布，不能依赖单样本。
4. 修复 off 后，当前 Vulkan MTP 的真实结构性负收益暴露出来，不能再用旧退化 off 作为参照。

剩余主要成本：

- K 张串行 draft graph 和每张约 398 MiB tied Q6_K head 扫描；
- target verification 与 MTP catch-up 分图；
- token/hidden 多次同步 readback，accepted 在 host 计算；
- GDN K+1 snapshot 物化；
- Vulkan graph submit 使用上一张 graph 的 FLOP 估计；
- CPU 权重尚未进入 CPU_REPACK。

下一迭代先以当前 graph FLOP 预扫描修正 Vulkan submit，并独立 sweep 提交 divisor；之后依次验证 Q4 MMVQ 多列 shader、verification/catch-up/packed readback 融合、CPU_REPACK、持久化 DraftChain/Round graph、snapshot 选择性提交和 draft-only 低精度 head。
