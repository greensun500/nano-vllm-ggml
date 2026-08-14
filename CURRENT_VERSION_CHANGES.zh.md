# nano-vLLM 当前版本修改说明：v3.92 Vulkan T>1 paged attention

## 1. 版本定位

本文件只描述当前代码版本；旧版本细节由 Git 历史保存。

- v3.5 基线：`71bd889 Add Qwen3.5 v3.5 graph reuse baseline`
- v3.6 功能提交：`12272ae Add native session scheduling and CUDA backend`
- v3.6 正确性修复：`cd6ced0 Restore Qwen3.5 chunk attention gate`
- v3.6 当前提交：`aabd733 v3.6: Add native MTP stage profiling`
- v3.7：`044e92e v3.7: add native runtime optimization controls`
- v3.71：`21474dd v3.71: remove fallback runtime overhead`
- v3.72：`b413650 v3.72: trim MTP draft hidden lifetime`
- v3.73：`d18b3d2 v3.73: enable CPU weight repacking`
- v3.74：`904c7d5 v3.74: probe Vulkan graph reuse safely`
- v3.75：`2e68c2a v3.75: keep embeddings out of CPU repack`
- v3.76：`ad52eff v3.76: disable unverified CPU repacking`
- v3.77：`5188da8 v3.77: profile native MTP stages`
- v3.78：`abe6de3 v3.78: record Arm MTP kernel profile`
- v3.79：`f24a212 v3.79: document long-context attention profile`
- v3.80：`fc160da v3.80: default non-MTP attention to auto`
- v3.81：`bcb7405 v3.81: record Mali auto attention benchmark`
- v3.82：`891692c v3.82: widen Mali Vulkan argmax`（vendored GGML `1bd647a`）
- v3.83 工作区：记录 v3.82 Arm Vulkan build、correctness 与统一 MTP benchmark 结果
- v3.84：在 Arm Vulkan 图执行器中实现实验性 Q6_K 词表 head 与 greedy argmax 融合
- v3.85：远端 profile 审计 v3.84 后默认关闭该实验融合，保留显式开关
- v3.86：新增显式 `paged` 的 Vulkan direct paged-attention 图路径；以
  `read_slots` 直读持久 KV，并在一个 kernel 内执行 online softmax/value
  accumulation，默认策略不变
- v3.86.1：恢复 v3.86 远程源码快照（只同步 source、不覆写 llama.cpp `.git`）
  的受审计 `5ed3338` revision 接受列表
- v3.87：修复 direct paged graph 的孤立 mask/padded-slot 正确性问题；仅对
  已通过 oracle 的 T=1 decode 启用，T>1 prefill/MTP verification 保持 math
- v3.88：把 verification 中 target-conditioned MTP KV 写入可选地附加到同一
  TargetChunkGraph，消除 accepted-prefix 的独立 KV-only graph；默认关闭
- v3.89：在 native Vulkan 的 MTP `auto` 路径按每张 draft/verification graph
  的真实 capability probe 选择 FlashAttention；其他 backend 仍为 math
- v3.90：native Vulkan 单序列 prefill 由 runtime 内部 chunk，避免 Python
  `max_num_batched_tokens` 边界反复结束 target graph 而改变 long-context trace
- v3.91：Mali Vulkan 的 MTP prefill 不再按 64 token 执行 target chunk；MTP
  maintenance 需要 target hidden，而非最终 chunk 的 hidden-output 图会改变后续
  target state。MTP prefill 改为一张完整 target graph，保持 MTP KV、recurrent
  snapshot 与 greedy trace 一致；普通 MTP-off prefill 仍保持 64-token chunk
- v3.92：显式 `native_attention_impl=paged` 解除 T=1 限制；Mali long-context
  MTP-off/on、`max_batch=64/768` 与 verification-KV fusion 的完整 greedy oracle
  已覆盖 multi-token prefill 和 verification
- 模型：Qwen3.5-2B-Q4_0 GGUF，24 层 target（6 attention + 18 recurrent）和 bundled 单层 MTP
- GGML：官方基线 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`；当前 v3.86 子模块 gitlink 为 `0caa416ded34e746f308ef75ea1d9cb24e50f552`。CMake 还精确接受远程 source snapshot 的 `5ed33380b4679533243ca45e172804d5ddfe59ec` 与仅导出 `GGML_TYPE_CPU_REPACK` 的受审计兼容提交 `62d87d7e76b584ffdec4763919dfd6833a8a2f3e`。
- Native 后端：`native_cpu`、`native_vulkan`、`native_cuda`；仍不创建 `llama_context`、不调用 `llama_decode`

v3.6 的主题不是改变 Qwen3.5 图结构，而是让 native runtime 更接近端侧可用形态：多轮对话不重复 prefill、调度不会无限压住 decode、GGML CUDA 可作为第三个 native 后端、MTP 的时间与常驻内存可以直接测量。`cd6ced0` 同时补回 attention 的 query gate，保证 Qwen3.5 attention 图与模型结构一致。

v3.7 在此基础上加入可独立 A/B 的图级优化：非 MTP 的 FlashAttention 由 runtime capability probe 选择；MTP 的 `auto` 固定 math，避免不同 token-shape 的 Flash 舍入差异降低 draft acceptance；GDN delta snapshot 改为可选连续写回；MTP prefill 的 hidden-to-KV maintenance 可进入同一张 target graph；CUDA Graph 仅在显式 CUDA Graph build variant 中使用。没有实现自定义 Q4_0 `lm_head + argmax` kernel，避免在缺乏目标 GPU profile 的情况下引入高风险的量化算子分叉。

v3.91 是正确性边界修复，不宣称 prefill 加速。Armv9.2/Mali-G720 的 521-token
long-context MTP probe（math、K=3）此前输出全 `255`；改为单 target graph 后恢复
与 MTP-off 相同的交替 `1814/9419` trace，`21/21` drafted token 被接受。该策略只在
Mali Vulkan 的 MTP prefill 生效；CPU、CUDA、MTP-off 和普通 decode 仍使用既有图形状。

v3.92 不改变 `auto` 默认策略：只有显式 `paged` 才会选择 direct-cache kernel。
CPU/CUDA 或不满足固定 Qwen3.5 F32/256-dim/GQA contract 的 Vulkan shape 仍报错；
已验证 T>1 的 token 语义，但性能是否优于 math/Flash 仍需以统一 wall-time benchmark 为准。

v3.71 不改变模型图、权重、缓存布局或 greedy token 语义，只去除两个已确认的运行时实现损耗：GGML `graph_compute` 成功后重复的 scheduler synchronize，以及 execution plan 校验中为 speculative tail 创建后即丢弃的完整 context slot 向量。

v3.72 保持 MTP draft 的计算和 host readback 内容不变，但让不需要 readback 的最后一张 draft graph 不再把 pre-LM-head hidden 声明为 GGML output，缩短 scheduler allocation 中该 activation 的不可复用生命周期。

v3.73 不改模型图、量化格式或 token 语义。此前 CPU 路径把所有 GGUF tensor 放入 default buffer，即使本机 GGML 已启用 `CPU_REPACK`，Q4_0/Q6_K 的专用 matmul 内核也无法被选中。现在只对运行时 ISA、量化类型和矩阵行数均满足 GGML 原生条件的二维 Q4_0/Q6_K 权重分配 repack buffer，其余 tensor 仍留在 default buffer；因此不会把不支持的 norm、bias、embedding 或量化布局放入会在上传时解引用空 traits 的 buffer。

v3.74 新增一个默认关闭的 Vulkan persistent graph reuse 安全探针。v3.5 曾记录 Mali 在 padded-mask bucket attention 下的不稳定行为，因此本版本不会改变 `native_vulkan` 默认执行链；只有显式 `--native-vulkan-graph-reuse` 或 `NANOVLLM_NATIVE_VULKAN_GRAPH_REUSE=1` 才让单序列 decode/MTP 复用 CPU 已验证的同一 bucket 机制。该开关用于远机 correctness oracle 和与 `--no-graph-reuse` 的交替 A/B，bench JSON 会记录它；多序列、prefill 和 MTP prefill fusion 保持 eager graph。

v3.75 修复 v3.73 的 Arm-only correctness bug。`token_embd.weight` 在该 GGUF 中是 Q6_K：它既是 tied output head 的 `MUL_MAT` 权重，也是 target/MTP embedding 的 `GET_ROWS` source。CPU_REPACK 只改变矩阵乘所需的物理 layout，`GET_ROWS` 不识别该 layout；Arm 远机在第一步 embedding 后 abort。加载器现在按 tensor 名称排除 `token_embd.weight` 和可选 `*.nextn.embed_tokens.weight`，令它们保持 default buffer；其余满足条件的 Q4_0 projection 和只作 lm_head 的 Q6_K 权重仍可 repack。

v3.76 根据 Arm 真机 oracle 收回 v3.73 的默认启用：即使 embedding 保持原始布局，Q4_0 `q4_0_4x8` repack 仍会改变 MTP partial-rollback trace（5 个 oracle 中 4 个通过，1 个生成序列少一个 token）。因此项目新增 `NANOVLLM_NATIVE_CPU_REPACK=OFF`（默认）；构建脚本同名环境变量可显式开启实验 build，但它不得用于正确性基线或性能结论。保留 loader 的混合 buffer 代码和 v3.75 embedding 排除，是为了后续比对上游修复后的 repack traits；默认运行时重新使用原始 GGUF layout，优先保证 exact greedy token。

v3.77 不改计算图和 token 语义。`bench --json` 读取 native runtime 已有的累计 MTP profile，并在排除 warmup 后导出 draft、target verification、MTP KV catch-up 三段的 wall time、建图时间和阶段 tok/s。远端 Mali-G720 K=1 A/B 已证明：打开 experimental Vulkan persistent graph reuse 后 cache hit 为 3227，但 decode 从 `18.58` 降至 `17.87 tok/s`；这与 profile 中 graph setup 仅占小部分的判断一致，开关保持默认关闭，后续优化聚焦 verification 的 target forward、词表 head 与 KV gather。新的完整 K=1 profile 为 draft `15.90s`、verification `72.06s`、KV catch-up `1.03s`，其中 verification setup 仅 `1.74s`；它占 decode wall 约 81%。同机 CPU K=3 试验性连续 snapshot + MTP prefill fusion 为 `32.62 tok/s`，相对 baseline `32.54 tok/s` 的差异在本轮重复测量噪声范围，故两个开关继续 opt-in。补充的 186-token Vulkan kernel log 显示一张 T=2 verification graph 为 `79.11ms`：Q6_K vocab `MUL_MAT_VEC` `15.15ms`、独立 `ARGMAX` `3.95ms`，而 13 次 `GET_ROWS` 合计 `0.25ms`。这给 fused head/argmax 明确的短上下文上限，也说明 direct paged attention 要以 8k 长上下文测量作为验收门槛。

v3.80 将 Python/CLI 的 `native_attention_impl` 默认值从 `math` 改为 `auto`。这不是对 MTP 的算法改动：C++ 选择器在 MTP 仍无条件返回 math，因而 draft 与 verification 的数值路径、acceptance 和现有基线命令（它们显式传 `math`）保持不变。仅在 MTP-off 且 backend/shape probe 确认 `FLASH_ATTN_EXT` 可用时才选择 Flash；不支持时 `auto` 自动回退 math。这个默认值变更经过本地 31 项单测与真实 Qwen3.5 CPU oracle（MTP off/on）5/5 验证；Armv9.2/Mali-G720 的统一参数远端 A/B（Q4_0、prompt 186、generate 541、repeat 3、warmup 1、`taskset -c 0,5-11`）也完成：显式 math 的 prefill/decode/processed/generated 为 `69.62/20.81/25.36/18.90 tok/s`，v3.80 auto 为 `69.93/21.24/25.85/19.26 tok/s`，decode 提升 `2.07%`。完整 stdout/stderr/JSON、build info（GGML `5ed33380`）与参数位于远端 `v37-results/20260814-014951-v380-auto/`。

v3.82 是 fused `lm_head + argmax` 前的独立、低风险归约阶段。Mali-G720 的 subgroup 为 16，而原 GGML `argmax.comp` 因此让每个 lane 扫描约 15500 个 Qwen3.5 vocab logits；Vulkan 后端现在仅对 Arm 设备创建 256-lane（受 `maxComputeWorkGroupInvocations` 限制）的同一 shader specialization。该 shader 仍以严格 `>` 比较、按低 index 优先处理相等值，语义与原 16-lane binary reduction 相同；非 Arm 保持上游 specialization。它不改变 Q6_K head 计算、输出 tensor 格式或 MTP acceptance。Armv9.2/Mali-G720 新建 `build/v382-vulkan` 已完成、CTest 通过、真实 Vulkan smoke 通过；统一 MTP K=1（math、186/541、repeat 3、warmup 1、`taskset -c 0,5-11`）的 drafted/accepted/verification 为 `810/810/810`、acceptance `100%`，decode 从 v3.76 baseline `18.58` 到 `19.92 tok/s`（`+7.24%`），verification `72.06 -> 68.05s`（`-5.57%`）。短 kernel profile 中 Q6_K head 仍为 T=1 `12.58ms`、T=2 `15.15ms`，而独立 `ARGMAX` 为约 `0.26ms`，相对 v3.77 profile 的 `3.95ms` 大幅缩短。完整 build/smoke/JSON/profile/build_info 位于远端 `v37-results/20260814-015954-v382/`。

v3.84 继续处理同一段 MTP verification 的词表输出，但不把 `MUL_MAT` 和 `ARGMAX` 仅做 API 层串联。对 Arm Vulkan、连续 Q6_K 权重、连续 F32 hidden、`T<=4` 的唯一连续 `MUL_MAT -> ARGMAX` 图边，执行器可改为两张专用 shader：第一张每个 16-row Q6_K workgroup 保持原 Q6_K 的 16-lane dot-product/reduction 顺序，只写一个 `{max_logit, vocab_id}` 候选；第二张以 256 lanes 对候选做稳定归约并直接写 I32 token。Qwen3.5 的约 248k 词表因此从每 token 约 0.95 MiB F32 logits 写入，缩减为约 0.12 MiB 的 transient candidate buffer，且不再让 graph allocator 为 logits 安排存活区间，也不再独立扫描完整 logits。相等分数显式选更小 id，与旧 `argmax.comp` 语义一致。这里不能调用 GGML 通用的 `can_fuse(MUL_MAT, ARGMAX)`：该 helper 为逐元素算子要求相邻节点 shape 相同，会把合法的 `[vocab,T] -> [T]` 归约错误拒绝；v3.84.2 改为验证连续节点、唯一直接 producer edge 和全部量化/布局边界。远端 Armv9.2/Mali-G720 的新建目录 `v37-results/20260814-022314-v384-fused-head/` 已完成 v3.84.2 build、CTest、Vulkan smoke，融合开/关短生成 stdout 完全一致；full MTP K=1 统一 A/B 的 acceptance 都是 `810/810/810`。但首次 A/B 仅因融合条件未命中而得到噪声级 `19.925 -> 19.968 tok/s`；修正 eligibility 后 Vulkan profile 证实融合命中且 T=2 head 从原生约 `15.17ms` 回归到 `26.36ms`。原因是当前 first-pass 让一个 16-lane subgroup 串行处理 16 行，丢失 Mali 原 Q6_K matvec 的 row-level parallelism。故 v3.85 起只在 `GGML_VK_ENABLE_Q6_K_LM_HEAD_ARGMAX_FUSION=1` 时实验性启用，默认继续走已验证更快的 `MUL_MAT + ARGMAX`；`GGML_VK_DISABLE_Q6_K_LM_HEAD_ARGMAX_FUSION=1` 仍可用于对照。v3.85 的远端增量 build/CTest/build_info 和默认短 profile 通过，且 profile 确认默认不再出现 `Q6_K_LM_HEAD_ARGMAX`。该路径不改变 Python 请求/调度、MTP acceptance、CPU/CUDA 或非 Arm Vulkan；不满足条件的图也会走原路径。

v3.86 把 direct paged-attention 作为单独、默认关闭的实验阶段，而不是删除
`GET_ROWS` 后仍调用普通 attention。新的 GGML `PAGED_ATTN` 节点接收连续 F32
query、persistent F32 key/value cache 和 I32 `read_slots`。Vulkan shader 针对
Qwen3.5-2B 的 `head_dim=256`、8 query heads / 2 KV heads，按 16 个逻辑
slot 分块直接解引用 physical slot。一个 256-lane workgroup 对应一个
`(query token, query head)`：16 个 16-lane dot-product group 生成当前 tile
score，再以 online `(max, sum, value)` 归约累积。因此不再为每层构造
`[key_width,n_kv]` / `[value_width,n_kv]` 的 gather，也不物化
`[n_kv,T,heads]` score/probability。chunk 的第 `t` 行只读取
`0..n_kv-n_tokens+t`，等价于原 `[n_kv,T]` causal mask；`SET_ROWS` 仍是
K/V 输入的 producer，保证当前 token 的新 K/V 对自身可见。

接口为 `native_attention_impl="paged"`、CLI
`--native-attention-impl paged`。runtime 对每个实际 `(T,n_kv)` shape probe
新 op，并拒绝 CPU/CUDA 或不满足固定 F32/256-dim/GQA contract 的请求；
`math`、`flash`、`auto` 的选择逻辑不变，MTP 的 `auto` 仍为 math。该实现
尚未作为性能结论或默认值：远端验收必须同时证明 real-model greedy 输出、
MTP acceptance 和长上下文 kernel/wall-time 均不回退；否则保持实验开关并继续
调优 tile、workgroup 或 split-K。本机 CPU build、CTest 和 32 项 Python
配置/runner/stage 测试已经通过，只证明默认路径与 ABI 未被新增 GGML op 破坏，
不能替代 Mali Vulkan 验证。

v3.86.1 只修复 fresh remote build 的 revision gate：v3.86 子模块在本地的
gitlink 为 `0caa416`，但受操作约束，远程同步不能写入子模块 `.git`，其
metadata 仍报告已受审计的 `5ed3338`。CMake 现在同时接受新的 v3.86 gitlink、
`5ed3338` 的远程 source snapshot 和既有 CPU_REPACK compatibility revision；
这不会放宽到任意 revision，也不会改变 shader、图或默认运行路径。

v3.88 专注于 MTP verification 的图边界，而不改变 target、MTP draft、LM head
或 acceptance 规则。legacy 路径在 host 比较 accepted 数量后，才把
`inputs[1:a+1]` 与 `target_hidden[0:a]` 送入独立 `MtpKvUpdateGraph`。新开关
`native_mtp_verification_kv_fusion=True` 让 verification `TargetChunkGraph(T=K+1)`
直接接收同一输入的 `[1,T)` token/position/slot 后缀，并以
`target_hidden[0:T-1]` 在图内执行 MTP merge/norm/KV projection/`SET_ROWS`。
它会预写未接受尾部的 MTP KV 行；这些行不在 `next_position` 的可见 context 内，
且下一轮 draft 在读之前必定覆写，因此不需要额外清零。`pending_hidden` 和
K+1 greedy prediction 仍需要 readback，recurrent rollback 也不变。

该开关要求 native backend + MTP，Python `Config`、`NativeRunner`、C++ binding、
chat/bench CLI 和 JSON 都显式记录它，默认 `False`。远端 Armv9.2/Mali-G720
real-model short MTP oracle（full accept、partial accept、immediate reject、release、
fusion A/B）5/5 通过；CPU 同一 oracle 5/5 通过。统一 Vulkan K=1 math A/B
（prompt 186、generate 541、repeat 3、warmup 1、8 cores）两组均为
`drafted/accepted/verification=810/810/810`、acceptance `100%`：legacy 的
draft/verification/KV 为 `12.377/68.134/0.784s`，融合后为
`12.311/68.303/0s`，decode `19.885 -> 20.057 tok/s`（`+0.87%`），generated
`18.084 -> 18.226 tok/s`。收益很小且仍需更多 context/K 值验证，故不改变默认。
完整 build、oracle、失败的历史 long-MTP baseline 与 JSON 位于远端
`v37-results/20260814-041222-v388-mtp-verification-kv-fusion/`。

v3.89 不改变 MTP 的 speculative accept/rollback、KV/state 布局或 graph 边界，
只调整 `native_attention_impl="auto"` 的选择器。此前它为避免历史版本的
shape-dependent Flash rounding，在所有 MTP backend 无条件选择 math；现在仅当
primary backend 是 Vulkan，且当前 T=1 draft 或 T=K+1 verification 的真实
`FLASH_ATTN_EXT` capability probe 成功时选择 Flash。CPU、CUDA、probe 不支持的
Vulkan shape 与显式 `math/flash/paged` 请求不变。真实模型 oracle 同时比较
`math`、`flash` 和新的 `auto`：Armv9.2/Mali-G720 5/5 通过 full accept、partial
accept、immediate reject、release 和 optimization trace，证明该目标环境的 greedy
trace 与 acceptance 没有变化。

统一 Vulkan K=1 基准（Qwen3.5-2B-Q4_0、prompt 186、generate 541、repeat 3、
warmup 1、8 cores）下，math 的 draft/verification/KV 为
`12.377/68.134/0.784s`、decode `19.885 tok/s`；v3.89 auto 为
`12.191/56.656/0.765s`、decode `23.219 tok/s`（`+16.76%`），两组
`drafted/accepted/verification=810/810/810`、acceptance `100%`。单独打开
batched recurrent snapshots 仍是噪声级变化（`19.904 tok/s`），故不改变其默认值。
完整 build、math/flash/auto oracle 和 JSON 位于远端
`v37-results/20260814-125732-v389-q6k-argmax-fix/`。

v3.90 修复了一个 native Vulkan long prefill 的实现边界：Python scheduler 在
`max_num_batched_tokens=64` 时把单条 521-token prompt 切成多个 plan，而 native
Mali runtime 又在每个 plan 内以 64-token target graph 执行；每个外部 fragment
都会被当成 `Last` output mode，结果与一次性 plan 的 greedy trace 不一致。现在
单 sequence native Vulkan prefill 始终交给 runtime 的既有 64-token内部 chunk，
并只放宽 Vulkan prefill 的 ABI plan-size 校验；多 sequence、CPU/CUDA、decode 和
非 native scheduler 上限不变。远端 Arm/Mali math long probe 的 64-token scheduler
配置已从全 `255` trace 恢复为与 768-token 配置相同的 `1814/9419` trace。完整
MTP long-context trace 仍有独立历史差异，未在本阶段掩盖或声称解决。

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
| `nanovllm/config.py`、native runner、chat/bench CLI | 暴露 `native_attention_impl`、`native_batched_recurrent_snapshots`、`native_mtp_prefill_fusion`、`native_mtp_verification_kv_fusion`；v3.80 起 attention 默认为受 capability probe 保护的 `auto`，其余默认关闭，CLI/JSON 会记录运行时 A/B 配置。 |
| native tests | 增加精确 GQA FlashAttention layout/F16 causal-mask oracle、连续 GDN snapshot CPY plane-order 测试、Python 配置/ABI 传递测试，以及 opt-in real-model MTP trace oracle。 |

关键边界：Flash 路径仍先 `SET_ROWS` 到 PagedKV、再 `GET_ROWS` gather，所以它不是新的 paged-attention kernel；只是在 gather 后把 QK、softmax、PV 的中间分数/概率留在 backend kernel 内。v3.88 的 verification KV fusion 不需要事先知道 accepted 数量：只预写 `[1,T)` 后缀，未接受行不会被后续 context 索引且会被覆盖；但它仍保留 target hidden/prediction readback 和 host rollback。Mali 实测显示 Flash 在正确性 oracle 中可用，但 draft `T=1` 与 verification `T=K+1` 的舍入差异会把 K=1 acceptance 从 100% 降至 50%，并让 decode 吞吐退化；因此 MTP 下 `auto` 保守选择 math，显式 `flash` 只作实验。`lm_head + argmax` 的全词表量化 matmul 仍是第二阶段课题；当前没有为了“融合”而添加不成熟的自定义 Q4 kernel。

### 2.7 v3.71 fallback 生命周期与 plan 校验优化

| 文件 | v3.71 改动 |
| --- | --- |
| `csrc/runtime/graph_executor.{h,cpp}` | 新增只用于“成功同步 compute 之后”的 reset；保留通用 `reset()` 的同步和异常路径保护。 |
| `csrc/runtime/qwen35_runtime.cpp` | fallback target、MTP draft、MTP KV catch-up 的 RAII guard 仅在成功 compute 后走无重复同步的 reset；构图或执行异常仍走同步 reset。 |
| `csrc/runtime/paged_kv.{h,cpp}` | 提供无分配 `validate_logical_range()`；`physical_indices()` 继续保持原有展开语义。 |
| `tests/native/test_graph_executor.cpp` | 覆盖成功 compute 后 reset、立即重新 allocate/compute 的 scheduler 生命周期。 |

GGML 当前 `ggml_backend_sched_graph_compute()` 已按同步语义完成 backend work，因此 fallback graph 在作用域退出时再次调用 `ggml_backend_sched_synchronize()` 是额外等待。v3.71 只在该同步成功返回后跳过第二次等待；若 allocate、上传或 compute 抛异常，guard 仍使用原始同步 reset，避免在未知 backend 状态下重用 scheduler。plan 校验仍验证 block table 的最大逻辑范围、block ID 合法性与同 sequence block 唯一性，只是不再物化本轮尚未参与计算的 context slot 列表。

### 2.8 v3.72 MTP hidden 输出活跃区间

| 文件 | v3.72 改动 |
| --- | --- |
| `csrc/models/qwen35/graph.{h,cpp}` | `TokenGraph` 构建显式接收 `retain_hidden`；只有调用方读回 hidden 时才设置 GGML output flag。 |
| `csrc/runtime/qwen35_runtime.cpp` | MTP persistent graph key 纳入 `retain_hidden`，前序 draft 与最后一轮 draft 分别缓存；fallback 路径传入相同语义。 |

MTP K draft 中，除最后一个 draft 外都要把 hidden 传给下一轮；最后一个 draft 仅需要 greedy token。此前两类图共享同一“hidden 输出”构造，导致最后一轮不必要地固定 2048 个 F32 activation。v3.72 仍以 greedy token 为 graph root，故 LM head/argmax 和 cache/state 副作用完全相同；只改变 allocator 是否将 hidden 视为 host-visible output。

### 2.9 v3.73 CPU 混合权重 buffer / repack

| 文件 | v3.73 改动 |
| --- | --- |
| `third_party/llama.cpp/ggml/include/ggml-cpu.h` | 将已存在的 `ggml_backend_cpu_repack_buffer_type()` 作为 CPU backend 公共能力声明，供嵌入式 runtime 安全选择 buffer type。 |
| `csrc/runtime/gguf_loader.{h,cpp}` | 加载时检测 CPU backend 的 repack 支持；先在一个 `CPU_REPACK` buffer 中安置合格矩阵，再由 GGML 为剩余 descriptor 分配紧凑 default buffer；记录全部 buffer 的所有权与 resident bytes。 |

候选选择与 GGML 当前原生实现一致的必要条件为：tensor 是二维；Q4_0 在 x86 AVX2 时行数为 8 的倍数、或 Arm NEON+dotprod/i8mm 时行数为 4 的倍数；Q6_K 仅在 Arm NEON+dotprod/i8mm 且行数为 8 的倍数时进入 repack。Qwen3.5-2B-Q4_0 的投影矩阵在 x86 本地验证实际进入 `q4_0_8x8`；Arm 还会覆盖 tied Q6_K vocab head。GGML 的 repack buffer 上传接口要求 `offset=0` 且完整 tensor，故只有这类矩阵在模型加载期临时整块读取；其他 tensor 保持原 16 MiB 有界分块上传。该峰值只发生一次，不进入 decode 热路径。

### 2.10 v3.74 Vulkan graph reuse 安全探针

| 文件 | v3.74 改动 |
| --- | --- |
| `nanovllm/config.py`、native runner、Python/C++ binding | 新增 `native_vulkan_graph_reuse` / `enable_vulkan_graph_reuse`，默认 `false` 且仅允许 `native_vulkan`。 |
| `csrc/runtime/qwen35_runtime.cpp` | 显式开关开启时允许单 sequence Vulkan 命中已有 persistent graph bucket；仍复用原 key、LRU、输入更新和 placement audit。 |
| chat/bench CLI | 新增 `--native-vulkan-graph-reuse` 和环境变量；bench 结果记录开关，便于交替 A/B。 |

历史 profile 中 Vulkan MTP graph setup/placement/allocation 约占总 wall 的 1%–2%，因此即使通过也预期为小收益。更重要的是先证明带 padded causal mask 的 bucket 在当前 Mali/GGML revision 中保持 greedy token 一致；若 oracle 失败，开关保持关闭并将问题留给 direct paged attention/fused verification 图，而不将未验证路径变成默认。

### 2.11 v3.75 CPU_REPACK embedding correctness 修复

| 文件 | v3.75 改动 |
| --- | --- |
| `csrc/runtime/gguf_loader.cpp` | CPU repack eligibility 接收 tensor name；显式排除 global 与 MTP token embedding，避免 `GET_ROWS` 读取重排后的物理 layout。 |

本地 x86 真实 GGUF oracle（5/5）继续通过，但该问题的触发与验证都以 Arm 的 Q6_K token embedding 为准。远端结果目录保留了修复前的 abort 日志；修复后必须重新 clean build 并通过 Arm CPU oracle 才能进行性能基准。

### 2.12 v3.76 CPU_REPACK 默认关闭

| 文件 | v3.76 改动 |
| --- | --- |
| `CMakeLists.txt`、`scripts/build_native_runtime.sh` | 新增 `NANOVLLM_NATIVE_CPU_REPACK`，默认 `OFF` 并传给 GGML 的 `GGML_CPU_REPACK`；仅显式环境变量/`-D` 可开启实验重排。CMake 同时把唯一的本地 `CPU_REPACK` header 兼容提交列为允许 revision，避免固定版本校验与子模块指针脱节。 |

远机验证结果说明，v3.75 只修复了 embedding abort，不能证明所有 Q4_0 projection 的 Arm repack 数值等价。v3.76 的判断是正确性优先的止损，不把 x86 prefill A/B 当作 Arm/MTP 可用性证据。后续若升级 GGML repack 实现，必须至少重跑 CPU oracle 的 full-accept、partial-accept/rollback、release/shutdown 三类路径，才可讨论默认打开。

### 2.13 v3.77 MTP 分阶段 benchmark profile

| 文件 | v3.77 改动 |
| --- | --- |
| `nanovllm/cli/bench.py` | 在 warmup 后读取 `NativeRunner.mtp_profile_stats()`；JSON/人类可读输出分别给出 draft、verification、KV catch-up 的累计 wall/setup 时间和 token 吞吐。 |

该字段包含 native 同步 compute、host/backend tensor 传输与建图，不能与 Python 侧 scheduler timer 混淆。它专用于判断 MTP 是否因低 acceptance、verification graph 或 KV catch-up 变慢；MTP-off 时字段为零。此改动也避免根据 graph cache hit 数推断收益：Mali K=1 的 graph reuse A/B 已显示高命中仍会退化。

远端结果目录还保留 `GGML_VK_PERF_LOGGER=1` 的短 MTP kernel log。它不是端到端吞吐结论：每次 backend graph 的计时各自输出；但同一张 target verification 内，Q6_K head + argmax 已占约四分之一 GPU kernel 时间，说明自定义 head epilogue 有合理收益空间。反之，186-token 时 `GET_ROWS` 极小；direct paged kernel 的设计应直接读物理 slot/page table、执行 online softmax，并在 1k/4k/8k 三个 context 证明避免 gather 的收益。

4096-token 的独占 MTP K=1 profile 已完成这一验证：math draft/verification 为 `219.6 / 463.4ms`，其中 verification 的 13 次 `GET_ROWS` 为 `5.25ms`，但分离的 F32 attention score/value matmul 约 `62.7 / 134.5ms`；只消除 gather 不能成为完整方案。显式 Flash 的同一单步探针使 verification 降至 `333.5ms`、decode 从 `1.45` 到 `1.73 tok/s`，且该步 acceptance 为 100%。这证明 online attention 融合方向有效；但历史完整长 decode 已观察到 Flash 的 shape-dependent MTP acceptance 回退，所以 `auto` 在 MTP 下仍强制 math，Flash 保持显式实验路径，不能以单步样本改默认。

### 2.14 v3.80 non-MTP attention 默认值

| 文件 | v3.80 改动 |
| --- | --- |
| `nanovllm/config.py`、`nanovllm/backends/native/runner.py`、`nanovllm/cli/bench.py` | 默认 attention implementation 改为 `auto`；旧配置缺字段时也按 `auto` 处理。 |
| `tests/test_native_runner.py` | 更新 ABI 传递测试，固定新默认值。 |

`auto` 的语义是安全选择而不是强制 Flash：MTP 仍走 math；普通 target graph 只有 capability probe 通过才走 Flash，否则回退 math。因此用户要求的复现实验仍应显式设置 `NANOVLLM_NATIVE_ATTENTION_IMPL=math`，而默认 chat/bench 能在已验证 accelerator 上获得 non-MTP attention 融合收益。

### 2.15 v3.82 Mali argmax workgroup

| 文件 | v3.82 改动 |
| --- | --- |
| `third_party/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Arm Vulkan 创建 `argmax_f32` pipeline 时把 local size specialization 从 `subgroup_size` 提升为 `min(256, maxComputeWorkGroupInvocations 的 2 次幂)`；其他 vendor 不变。 |

这是一个单独的 `ARGMAX` 调度优化，不宣称已经完成 fused Q6_K head：后续融合仍需要令 matvec 直接产生局部 `(logit, token_id)`，再只归约局部候选，才能避免全量 logits 写回。v3.82 的完整 MTP acceptance 和 profile 已证明更宽的精确归约可保留；下一阶段可以基于 256-lane reduction 写专用 Q6_K partial-max，而不再猜测 MMVQ（`GGML_VK_FORCE_MMVQ=1` 的短 K=1 probe 反而使 decode `17.08 -> 13.45 tok/s`）。

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

v3.87 在 Armv9.2 + Mali-G720（`taskset -c 0,5-11`、同一 GGUF/参数）重新完成 Vulkan build/CTest、短 chat、以及 521-token context 的跨 chunk/MTP oracle；paged T=1 的 token trace 和 acceptance 通过。统一 A/B 的 MTP-off math/paged 为 `prefill 69.76/69.75`、`decode 22.70/22.17`、`processed 27.44/26.87`、`generated 20.45/20.02 tok/s`，direct paged 当前约慢 `2.3%`。MTP K=1 math/paged 为 `decode 19.914/19.889 tok/s`，二者均为 `drafted/accepted/verification=810/810/810`、acceptance `100%`；paged 只改变 draft（`12.36 -> 12.51s`），而 T=2 verification 保持 math（`68.04 -> 68.00s`），没有端到端收益。结论是：保留正确性修复和显式实验开关，但不进入 `auto` 或默认策略；下一步应以 Mali profile 重新设计单-token kernel（例如 split-K/更多并行 workgroup），而不是简单扩大 tile。完整 build、oracle、失败的 T>1/tile-32 试验和四组 JSON 在远端 `v37-results/20260814-030521-v386-paged-attn/`。

llama.cpp 参考数据的 decode 长度与 nano-vLLM 不同，适合作为量级对比，不作为 v3.5/v3.6 回归判断。v3.7 已在 Armv9.2 Cortex-A720/A520 + Mali-G720-Immortalis 远机完成真实 GGUF oracle 和统一长 benchmark；GCC 12 使用兼容拼写 `armv9-a+dotprod+i8mm`，Vulkan 日志确认 `int dot=1, matrix cores=KHR_coopmat`。相对 v3.6，v3.7 CPU MTP-off / K=3 decode 为 `27.75 / 32.93 tok/s`，Vulkan Flash MTP-off / math MTP K=1 为 `21.24 / 18.59 tok/s`。MTP 的 `auto` 固定 math 已避免 Mali Flash 的 shape-dependent rounding 将 K=1 acceptance 从 100% 降至 50%、decode 降至 `16.08 tok/s` 的回退。完整 benchmark、K sweep 和 profile 记录在开发笔记 `8-mtp升级.md`。

v3.73 本地 x86 A/B（Qwen3.5-2B-Q4_0、CPU 8 threads、math、prompt 186、generate 128、warmup/repeat=1）在关闭/开启 `GGML_CPU_REPACK` 时，prefill 为 `165.9 / 394.3 tok/s`；decode 为 `24.25 / 24.21 tok/s`，后者在该平台不覆盖 Q6_K head，视为测量噪声范围。该结果只证明加载路径与 Q4_0 prefill 内核生效。Armv9.2 oracle 已证明当前 `q4_0_4x8` repack 不保持本项目 exact MTP trace，故 v3.76 默认关闭它，不可将这项 x86 数据外推为实际可用收益。

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

可通过 `enable_session_cache=False`、`max_retained_sessions` 和 `max_consecutive_prefill_rounds` 调整会话保留与公平策略。图优化 A/B 使用 `--native-attention-impl {math,auto,flash,paged}`、`--native-batched-recurrent-snapshots`、`--native-mtp-prefill-fusion`、`--native-mtp-verification-kv-fusion`；CUDA Graph 的 A/B 则使用独立 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 构建。benchmark 继续使用 `--enable-mtp --mtp-max-draft-tokens K`；性能比较必须同时记录 GGML commit、shader compiler、Mali capability 日志、warmup/repeat 和模型上下文长度。

## 6. 当前边界与下一步

1. Native 不支持跨请求 prefix cache；hybrid recurrent state 不能像纯 attention KV 一样廉价地共享。
2. Native 不支持 preemption；容量不足时 session LRU 释放 idle state，active request 仍沿用原有的显式错误/调度路径。
3. graph reuse 默认只覆盖 CPU 单 sequence 高频 decode/MTP；native CUDA 需使用 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 的独立构建且尚未在 NVIDIA 实机验收。Vulkan 的 bucket reuse 仅有 v3.74 默认关闭的实验开关，需先通过 Mali oracle。
4. Native 路径使用 HF tokenizer；当前 MTP 热路径并不调用 Python tokenizer，验证耗时主要在 native graph 和 LM head。
5. v3.76 默认关闭当前 GGML CPU_REPACK，等待上游/target oracle 证明数值等价。v3.92 的 direct paged kernel 保持显式，但已通过 T>1 long-context/MTP oracle；v3.88 verification-KV fusion 同样保持 opt-in，直到更多 context/K 值的 Vulkan oracle 和收益通过。后续高风险第二阶段是 Q4_0/Q6_K lm_head + argmax 融合、split-K paged attention 和自适应 K 策略。
6. v3.77 的 stage profile 是上述融合与自适应 K 的验收指标；Vulkan graph reuse 仍是默认关闭的 correctness 探针，当前 Mali 数据显示不应启用。
