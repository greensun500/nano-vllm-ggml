# nano-vLLM Native Runtime 实测性能数据

> 最后整理：2026-08-21。本文只收录已在真实 `Qwen3.5-2B-Q4_0.gguf` 上完成的端到端运行；不是算子 micro-benchmark，也不把不同机器之间的绝对 tok/s 直接横向比较。

## 可用于项目介绍的结论

- **NVIDIA A100 Vulkan**：在单序列 Qwen3.5-2B-Q4_0 的实测中，启用 native Vulkan persistent graph reuse 后，短上下文 decode 从 `155.7` 提升至 `223.1 tok/s`（`+43%`）；再配合内置 MTP K=3 为 `288.4 tok/s`（`1.85x`）。2048-token prompt 下的对应最优 decode 为 `279.6 tok/s`。
- **Armv9.2 CPU**：在 Cortex-A720/A520 开发板的长生成中，MTP K=1 将 decode 从 `27.90` 提升至 `35.58 tok/s`（`+27.5%`），接受率 `100%`。
- **Mali-G720 Vulkan**：稳定默认路径（MTP-off、attention auto/Flash）已测得 `70.04` prefill / `23.12` decode tok/s。MTP K=3 的 decode 可到 `25.03 tok/s`，但其正确性保护下的整段 MTP prefill 很慢，端到端 generated 吞吐反而更低；因此 Mali 目前仍推荐 MTP-off。

下文中的“最优”仅指该表中同一硬件、同一工作负载下的已测配置，并不表示跨硬件或跨 GGML revision 的比较。

## 统一口径

除特别标注外，所有 nano-vLLM 结果均为：

- 模型：`Qwen3.5-2B-Q4_0.gguf`（约 1.94B 参数）、HuggingFace Qwen3.5-2B tokenizer；
- `batch_size=1`、`max_num_seqs=1`、temperature `0`（greedy）；
- `repeat=3`、`warmup=1`，报告 benchmark 工具的均值；
- `prefill_tok_s`：prompt 处理速度；`decode_tok_s`：生成阶段目标 token 速度；`generated_tok_s`：包含 prefill 的端到端生成吞吐；
- MTP 的 `acceptance` 是 accepted drafts / drafted tokens；最终 token 仍由 target verification 纠正，但较低接受率会直接影响吞吐。

## NVIDIA A100-SXM4-80GB：native Vulkan

**实测日期/代码**：2026-08-18，nano-vLLM `7ced3e6`（v3.92 前），vendored GGML `8e29a9e4`。  
**环境**：A100-SXM4-80GB、NVIDIA driver 570.124.06、Ubuntu 20.04、Python 3.11.15、Vulkan Loader/Headers 1.3.280、shaderc/glslc 2026.3、8 CPU threads、全模型 Vulkan offload。

`native_vulkan_graph_reuse` 是 native Vulkan 的显式 persistent graph 复用开关；普通 `enable_graph_reuse` 不等同于它。

| 工作负载 | 配置 | Prefill tok/s | Decode tok/s | Generated tok/s | MTP acceptance | Decode 相对基线 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pp128/tg128 | 基线，auto，MTP-off | 5448.7 | 155.7 | 152.4 | — | 1.00x |
| pp128/tg128 | native Vulkan graph reuse，auto，MTP-off | 5426.1 | 223.1 | 215.7 | — | 1.43x |
| pp128/tg128 | MTP K=3，auto | 4513.0 | 205.3 | 197.8 | 89.5% | 1.32x |
| pp128/tg128 | **MTP K=3 + native Vulkan graph reuse，auto** | 4327.7 | **288.4** | **272.3** | 89.5% | **1.85x** |
| pp2048/tg512 | 基线，auto，MTP-off | 12289.9 | 151.8 | 144.8 | — | 1.00x |
| pp2048/tg512 | native Vulkan graph reuse，auto，MTP-off | 12208.9 | 210.0 | 196.7 | — | 1.38x |
| pp2048/tg512 | MTP K=3，auto | 11143.5 | 229.8 | 212.6 | 89.4% | 1.51x |
| pp2048/tg512 | **MTP K=3 + native Vulkan graph reuse，auto** | 11469.7 | **279.6** | **255.2** | 89.4% | **1.84x** |

同次测试中，显式 FlashAttention 在该 2B、上下文不超过 2K 的场景没有额外收益：MTP K=3 + reuse 的短/长 decode 分别为 `288.4` / `274.1 tok/s`，略低于 auto 的 `288.4` / `279.6 tok/s`。这是工作负载结论，不应外推到更大模型或更长 context。

## Armv9.2 CPU：Cortex-A720/A520

**实测代码**：v3.92；真实 GGUF 远端 JSON。  
**环境**：Armv9.2 Cortex-A720/A520，8 个 A720 核绑定 `taskset -c 0,5-11`，GCC 12，`-march=armv9-a+dotprod+i8mm`，8 threads，32 KV blocks，`max_model_len=8092`。

| 工作负载 | 配置 | Prefill tok/s | Decode tok/s | Processed tok/s | Generated tok/s | MTP acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pp186/tg541 | math，MTP-off | 88.46 | 27.90 | 33.83 | 25.206 | — |
| pp186/tg541 | math，MTP K=1 | 87.60 | **35.58** | **41.96** | **31.269** | 100% |
| pp186/tg541 | math，MTP K=3 | 87.79 | 32.56 | 38.81 | 28.923 | 89.39% |
| pp512/tg128 | auto（CPU 实际为 math），MTP-off | 87.63 | 27.01 | 60.59 | 12.138 | — |
| pp512/tg128 | auto（CPU 实际为 math），MTP K=1 | 86.52 | 34.22 | 66.36 | 13.292 | 100% |
| pp1024/tg128 | auto/math，MTP-off | 85.36 | 24.41 | 66.91 | 7.441 | — |
| pp1024/tg128 | auto/math，MTP K=1 | 84.87 | **31.46** | **71.48** | **7.949** | 100% |

在 pp512/tg128 上，CPU graph reuse off/on 的 MTP-off decode 为 `26.89/27.01 tok/s`，MTP K=1 为 `34.27/34.22 tok/s`。命中率虽高，但对单序列 CPU decode 几乎无端到端收益；热路径主要不是 graph construction。

## Armv9.2 Mali-G720-Immortalis：native Vulkan（稳定路径）

**实测代码**：v3.92；真实 GGUF 远端 JSON。  
**环境**：与上节相同的 Arm 主机，Mali-G720-Immortalis，Vulkan `glslc-2025.2`，运行日志确认 `int dot=1`、`KHR_coopmat`，CPU 侧绑定 `taskset -c 0,5-11`。

| 工作负载 | 配置 | Prefill tok/s | Decode tok/s | Processed tok/s | Generated tok/s | MTP acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pp186/tg541 | auto（实际 Flash），MTP-off | **70.04** | 23.12 | **27.91** | **20.800** | — |
| pp186/tg541 | math，MTP-off | 69.72 | 22.70 | 27.43 | 20.443 | — |
| pp186/tg541 | auto（MTP graph 真实 probe 为 Flash），MTP K=3 | 13.58 | **25.03** | 20.58 | 15.339 | 100% |
| pp512/tg128 | math，MTP-off | 72.35 | 22.48 | 50.21 | 10.058 | — |
| pp512/tg128 | Flash，MTP-off | **72.73** | 22.96 | **50.83** | **10.181** | — |
| pp512/tg128 | Flash，MTP K=3 | 19.62 | **24.67** | 20.45 | 4.097 | 100% |
| pp1024/tg128 | auto/Flash，MTP-off | **72.40** | 22.46 | **58.13** | **6.465** | — |
| pp1024/tg128 | auto/Flash，MTP K=3 | 20.18 | **24.22** | 20.56 | 2.286 | 100% |

这里的关键取舍是：MTP K=3 能略提高纯 decode，但在目前的 Mali MTP prefill 正确性保护下，prefill 从约 `70` 降至约 `14–20 tok/s`，因而端到端 generated 吞吐低于 MTP-off。对交互式长 prompt，推荐公开示例保持 MTP-off；若只评估长续写，MTP K=3 可以作为明确标注的实验选项。

## v3.93 Mali MMQ 实验：已验证、不可宣传为默认结果

提交 `18a952a` 引入了一个默认关闭的底层 Mali MMQ tile/workgroup 实验，参考 CIX 的 Mali MMQ coverage 修复。它针对 **Mali cold whole-prompt prefill**，不改变 Python 的 prompt 分块策略；后续 decode、MTP draft 和 verification 继续使用安全路径。

以下是开发板同一模型、pp512/tg128、math、MTP K=3、`repeat=3/warmup=1` 的实际首轮 A/B：

| 路径 | Prefill tok/s | Decode tok/s | Processed tok/s | Generated tok/s | Drafted / accepted / verification | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认安全路径 | 19.42 | **22.33** | 19.93 | 3.993 | 288 / 288 / 96 | **100%** |
| `GGML_VK_ENABLE_MALI_MMQ_TUNE=1` 实验路径 | **159.61** | 19.92 | **66.68** | **13.358** | 315 / 285 / 105 | 90.48% |

这证明该 backend 级思路能显著改善 whole-prompt prefill kernel 吞吐（约 `8.2x`），但同时改变了 MTP draft 分布、降低接受率并拖慢 decode。因此它**不是默认功能**，也不应进入 GitHub 的“默认性能”图表；应只在“Experimental results”中展示，并附带上述 acceptance 和正确性资格化说明。

## 正确性与复现门槛

每一组 Arm/Mali 数据对应真实 GGUF 的 chat/smoke 或 greedy oracle。MTP 数据还须通过 full accept、partial reject/rollback、immediate reject 与长 context trace；不能只依据最终一段文本看起来正常就宣称正确。

发布或重新测试时，建议在结果旁固定记录：

- nano-vLLM commit、vendored GGML commit、模型文件校验值；
- GPU 名称、驱动、Vulkan loader、`glslc` 版本和 capability log；
- CPU `taskset`、线程数、编译 `-march`、governor/频率条件；
- 全部 benchmark 参数、warmup/repeat、attention 实现、MTP K、graph reuse 开关；
- MTP 的 drafted、accepted、verification steps 和 acceptance，而非只报 decode tok/s。

## 原始数据定位与宣传使用建议

- A100 结果来自 2026-08-18 的 16 个 native JSON 运行（`results/20260818-vulkan-perf/`）；详细环境与逐配置记录见内部测试报告 `nanovllm-ggml在a100的测试.md`。
- Arm/Mali v3.92 JSON 汇总来自远端 `/home/cix/nano-vllm/v37-results/`，完整矩阵及分析保留在内部研究记录 `10.nanovllmv3.9研究.md` 和 `8-mtp升级.md`。
- v3.93 MMQ 实验的 build、CTest、oracle 和 benchmark 日志保存在远端 `/home/cix/nano-vllm/v37-results/20260821-151334-mali-mmq/`。

用于 GitHub README 的图表应优先选取上面三张“稳定路径”表，页脚注明“single sequence, Qwen3.5-2B-Q4_0, greedy, repeat=3/warmup=1”。A100 数值来自 `7ced3e6`，Arm/Mali 数值来自 v3.92；在将它们标为 v3.93 release performance 前，必须按完全相同命令在 `18a952a` 上复跑。v3.93 MMQ 数据则必须标为 experimental，不能与默认路径混合。

## 当前默认性能策略

从 v3.93 起，native CLI/API 未显式指定 MTP、MTP K 或 native Vulkan graph reuse 时，只对已有实测覆盖的设备启用下列自动策略；其余硬件维持保守基线（MTP-off、Vulkan reuse-off）。任何显式参数都会覆盖自动选择。

| 识别的平台 | 默认 MTP | 默认 native Vulkan graph reuse | attention | 依据 |
| --- | --- | --- | --- | --- |
| Arm Cortex-A720 CPU（`aarch64` + `Cortex-A720`） | on，K=1 | 不适用 | auto（实际 math） | pp186/tg541 decode `27.90 -> 35.58 tok/s`，接受率 100% |
| Mali-G720 Vulkan | off | off | auto | MTP 的 pure decode 虽略快，但 whole-prompt prefill 回退会降低端到端 generated 吞吐 |
| NVIDIA A100 Vulkan | on，K=3 | on | auto | MTP K=3 + reuse 在已测 pp128/tg128 与 pp2048/tg512 分别达到 `1.85x`/`1.84x` decode |

使用 `--disable-mtp`、`--no-native-vulkan-graph-reuse` 或 `--native-performance-profile baseline` 可锁定基线；`--enable-mtp`、`--mtp-max-draft-tokens K` 和 `--native-vulkan-graph-reuse` 可按需覆盖。benchmark JSON 会记录最终解析的 profile 和开关。
