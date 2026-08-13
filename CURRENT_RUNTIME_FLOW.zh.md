# nano-vLLM 当前执行流程：v3.83

## 1. 架构边界

nanovllm Python 层负责 request、scheduler、sequence、block table、slot mapping、PagedKV 分配、greedy speculative accept/rollback、EOS/max_tokens、会话保留和统计。

in-tree C++ runtime 负责 GGUF 加载、Qwen3.5 target/MTP GGML graph、CPU/Vulkan/CUDA backend、6 层 attention PagedKV、18 层 GDN recurrent state 及 K+1 snapshot planes。

项目只编译 llama.cpp 的 GGML 子目录；上游基线为官方 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`，其上带有 nano-vLLM 的 Mali shape-policy 子模块提交 `5ed3338`。项目不链接 `libllama`、不创建 `llama_context`；GGML 提供 tensor、graph、CPU/Vulkan/CUDA backend 和底层算子，调度与缓存所有权仍属于 nano-vLLM。

## 2. 构建阶段

```text
scripts/build_native_runtime.sh
  -> 检查显式 GGML_NATIVE/GGML_CPU_ARM_ARCH 参数
  -> aarch64 auto 模式：检查所有 CPU 的 asimddp+i8mm
  -> 编译器探测 armv8.6-a+dotprod+i8mm
  -> 成功时注入 GGML_NATIVE=OFF + GGML_CPU_ARM_ARCH
  -> 可选校验 NANOVLLM_VULKAN_GLSLC
  -> 选择 GGML Vulkan 或 CUDA（两者互斥）
  -> CMake 编译 in-tree GGML + nanovllm._C
```

显式 CMake 参数优先于自动参数。feature 或编译探针失败时安全回退到 GGML 默认构建。

## 3. 初始化

```text
LLM(config)
  -> NativeRunner
  -> nanovllm._C.Qwen35Runtime
  -> validate Qwen3.5/Q4_0/MTP GGUF contract
  -> create [CPU]、[Vulkan, CPU] or [CUDA, CPU] backends
  -> load model tensors（默认全部使用 GGUF 原始 layout；仅实验 CPU_REPACK build 才将合格 MUL_MAT 权重进入重排 buffer）
  -> allocate target/MTP PagedKV
  -> allocate recurrent canonical/snapshot planes
  -> create fallback GGML graph scheduler
  -> lazy persistent graph bucket cache
```

accelerator 模式把模型 compute node 固定到 Vulkan 或 CUDA，并在 graph allocation 后执行 placement audit。runtime 读取 `BackendDeviceInfo`；设备 name/description 含 `Mali` 时，普通长 prefill 的 target chunk limit 设为 64。默认 `enable_graph_reuse` 让 CPU 单序列 decode/MTP lazy 创建 persistent graph bucket；CUDA 只有 extension 由 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 编译时才允许 GGML CUDA Graph capture/replay。Vulkan 默认仍走 fallback scheduler，但 v3.74 可用默认关闭的 `native_vulkan_graph_reuse` 对其稳定 bucket 做显式 oracle/A/B。多序列和 prefill/chunk 仍走 fallback scheduler。native 默认 `max_num_seqs=1`、`max_num_batched_tokens=2048`，避免端侧启动时先为大量 recurrent slot 分配状态。

## 4. Python 执行计划

Scheduler 每步生成：

```text
mode
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

NativeRunner 将 Python sequence ID 映射到 native sequence slot。C++ 根据 block table 展开物理 read/write slots，并校验 Python slot mapping 与同图重复写入。校验 speculative tail 时只检查所需逻辑范围及其 block-table 前缀，不再提前展开一个随后丢弃的完整 context slot 向量；实际执行时仍在对应 target/MTP graph 前展开 read slots。

fallback graph 的生命周期为 `build -> allocate -> synchronous compute -> scheduler reset -> ggml metadata arena reset`。GGML 的 `graph_compute` 已完成同步，因此成功路径的 scheduler reset 不重复发起 synchronize；任何构图、上传或计算异常仍走带 synchronize 的通用 reset，防止未完成 backend work 被错误复用。

MTP draft 的 hidden 只在还要继续 draft 时才标为 graph output 并读回 host；最后一轮只以 greedy token 为输出。persistent graph cache 将这两个内存生命周期作为不同 key，避免“需要 hidden”的 graph 被误用于最后一轮，或反向复用。

v3.6 在 native chat 中增加长期 session。首轮完整 prompt 创建 `Sequence(WAITING)`，完成后 status 变为 `PARKED`；下一轮将格式化后的新 user turn 追加到同一 sequence，保留原 block table、native sequence slot、target/MTP PagedKV、recurrent state 和 `pending_hidden`，只 prefill 新增 token。`/reset`、`/system`、显式 `close()` 或 LRU 淘汰会走 `release_blocks()`，统一清理这些 native 持久状态。

Scheduler 仍是低开销 FIFO：有 waiting request 时优先 prefill；但 waiting 与 running 同时存在并连续 prefill 达到 `max_consecutive_prefill_rounds`（默认 4）时，强制插入一轮 decode，避免持续新请求使已有生成饿死。session cache 仅复用同一条对话；它不是跨请求 prefix cache。

## 5. TargetChunkGraph

单 sequence 的 T 个 token 共用一张 graph：

```text
tokens[T] + positions[4T]
  -> token embedding
  -> 24-layer target trunk
       18 multi-token recurrent GDN layers
       6 causal full-attention layers
  -> 按输出模式决定 output norm/head/argmax
```

full-attention 层先批量写 K/V，再按 `read_slots[C]` gather，使用 `[C,T]` causal mask防止 query 看到未来 token。`native_attention_impl` 默认 `auto`：非 MTP 会按真实 shape probe `FLASH_ATTN_EXT`，成功后把 QK -> softmax -> PV 留在 GGML backend kernel 内，失败则回退 math；显式 `flash` 在不支持时报告清晰错误。MTP 的 `auto` 固定 math：draft 的 `T=1` 与 verification 的 `T=K+1` 图可能因 Flash 的 shape-dependent rounding 降低 greedy acceptance，显式 `flash` 仍保留用于 A/B。attention 的 weighted value 在 `attn_output` 前还必须乘 `sigmoid(query_gate)`；这是 Qwen3.5 query gate，v3.6 已恢复。GDN 层在 graph 内逐 token 更新 convolution/delta state；普通路径只提交最新 state，MTP verification 同时生成 newest-first K+1 snapshot planes。`native_batched_recurrent_snapshots=True` 时 delta 的 K 个连续 plane 用一次 copy 写回，CUDA 可进一步匹配上游 GDN cache-write fusion。

输出模式：

- `None`：不产生 greedy token；`retain_hidden=false` 时连 output norm/head 也不构建；
- `Last`：只计算最后一列 greedy token；
- `All`：计算 T 列 prediction，供 MTP verification。

readback 数分别为 0、1、T；hidden 只在 `retain_hidden=true` 时读取。

## 6. MTP-off 流程

CPU 或非 Mali Vulkan：

```text
scheduled T tokens
  -> TargetChunkGraph(T, Last)
  -> target KV batch write
  -> recurrent canonical state commit
  -> one greedy token
```

Mali Vulkan：

```text
chunk[0:64]   -> TargetChunkGraph(None) -> commit KV/recurrent
chunk[64:128] -> TargetChunkGraph(None) -> commit KV/recurrent
...
final chunk   -> TargetChunkGraph(Last) -> commit KV/recurrent -> one token
```

每片都按片尾位置重新展开 context read slots，并使用上一片的 active snapshot plane。decode 的 T=1 小于分片上限，仍是一张 `Last` graph。

## 7. MTP prefill / fallback

开启 MTP 时，legacy 路径会把每个 target chunk 的 normalized hidden 读回。runtime 维护跨 chunk 的 `pending_hidden`，构造严格右移配对：

```text
mtp_hidden[0] = old pending_hidden
mtp_hidden[i] = target_hidden[i-1]
```

然后执行 MTP KV-only graph：

```text
[tokens, shifted_hidden, positions, write_slots]
  -> MTP merge/norm
  -> K/V projection + K norm + IMRoPE
  -> batch SET_ROWS into MTP PagedKV
```

它不执行 MTP attention、FFN、head 或 token readback。当前片最后一个 target hidden 成为下一片的 `pending_hidden`。因此 Mali 非最终片虽然使用 `None`，仍会保留 hidden 和 output norm，只跳过 tied vocab head/argmax。

`native_mtp_prefill_fusion=True` 时，target graph 内部直接构造同一右移 hidden，继续执行 MTP merge/norm/KV projection/`SET_ROWS`。host 只读回最后一列 `[E,1]` 作为下一 chunk 的 `pending_hidden`，不再上传 `[E,T]` shifted hidden 或单独执行 prefill KV-only graph；为了保持 graph 语义简单，该路径当前不复用旧 persistent bucket。verification 的 accepted 数量必须由 host 先比较 draft/target token 才能确定，仍保留独立的 catch-up KV-only graph。

## 8. MTP speculative decode

设 K=3，当前 pending token 为 `x_p`，pending hidden 为 `h_{p-1}`。

### 8.1 Draft

```text
MTP(x_p, h_{p-1}, p)   -> d1, m0
MTP(d1,  m0,      p+1) -> d2, m1
MTP(d2,  m1,      p+2) -> d3
```

三步有真实自回归依赖，当前仍执行三张串行 draft graph。每步包含 MTP layer、tied vocab head 和 greedy argmax；单序列 CPU 时 draft graph 使用 fixed `n_kv` bucket 复用同一个 graph/executor。

### 8.2 Target verification

```text
inputs = [x_p, d1, d2, d3]
TargetChunkGraph(T=4, All, snapshots=4)
  -> predictions t1,t2,t3,t4
  -> hidden h_p,h_p+1,h_p+2,h_p+3
  -> recurrent planes 3,2,1,0
```

64-token prefill policy只作用于普通 `run()` 的长 prefill；T=4 verification 始终是一张 `All` graph。

host 顺序比较 `t1==d1`、`t2==d2`、`t3==d3`，首个 mismatch 前的相等数为 accepted `a`。每轮返回 a 个 accepted token 加一个 correction/bonus token `t[a]`。

### 8.3 Rollback/commit

```text
recurrent canonical <- snapshot plane K-a
pending_hidden       <- target_hidden[a]
next_position        <- p+a+1
```

target attention KV 的未提交尾部无需清零：下一轮只 gather committed context，未来写入会覆盖尾部。

### 8.4 MTP KV catch-up

draft index 0 已写入真实 `x_p + h_{p-1}`。若 `a>0`，将 index `1..a` 一次补齐：

```text
tokens        = inputs[1:a+1]
hidden_inputs = target_hidden[0:a]
positions     = p+1 ... p+a
write_slots   = committed slots
```

`a=0` 时不执行 catch-up。该 graph 仍与 verification 分离，但单序列 CPU 时 `T=1..K` 的 KV-only graph 会进入 persistent cache。

## 9. 图数、同步与 readback

K=3 常规轮次为：

```text
3 × serial MTP draft graph
1 × TargetChunkGraph(T=4, All)
0/1 × MTP KV-only catch-up graph
```

readback 规则：

- 非最终 MTP-off Mali prefill 片：无 token/hidden readback，无 output norm/head；
- legacy 非最终 MTP-on prefill 片：读取完整 hidden，执行 KV-only maintenance，不读 token；fusion 路径只读取最后一列 hidden；
- 最终 prefill 片：读取一个 token；legacy MTP 还读取完整 hidden，fusion MTP 只读取最后一列；
- MTP draft：读取 draft token，非末步还读取下一步 hidden；
- verification：读取 K+1 predictions 与全部 target hidden；
- MTP KV-only：无 host 输出。

`GraphExecutor::compute()` 使用同步 GGML scheduler API。fallback 路径仍在 graph guard 退出时 reset transient allocation，并按轮重建 graph metadata。

v3.5 引入、v3.7 保留的 persistent graph 路径按 key 常驻独立 executor：

```text
graph_kind + T + n_kv_bucket + output_mode + snapshot_count
  + sequence_slot + input_plane + retain_hidden
```

entry 首次 miss 时构图、accelerator placement、scheduler allocation；命中时只更新 token、position、read/write slots、mask 和 hidden input，然后重复 `compute()`。bucket 只保存 graph metadata、scheduler allocation 和 transient activation buffer，不保存 KV cache、不复制权重。

`n_kv_bucket` 默认使用 `128/256/512/1024/2048/4096/max_model_len`。真实 KV slot 放在前缀，padding slot 重复最后一个真实 slot，并通过 causal mask 屏蔽；上下文不会被截断。

## 10. Release

request release 会校验 sequence ID，清理 target/MTP KV block、recurrent planes、pending hidden/position，并释放 Python-to-native slot mapping。v3.6 的 parked session 被 LRU 淘汰或显式关闭时也走同一条 release 路径。

## 11. 当前 backend 策略边界

- Mali 的 64-token 值来自当前 Qwen3.5-2B/Mali-G720 实测，不应直接泛化到其他模型或 GPU。
- v3.4 起默认将 Mali/int-dot/Q4_0 的 T=1 路由到 DMMV，T=2～8 继续使用 MMVQ。显式 FORCE/DISABLE 环境变量仍可覆盖默认策略；其他 vendor、量化类型和无 int-dot 设备保持上游逻辑。
- graph reuse 默认只保证 CPU 单序列 decode/MTP；CUDA 需使用 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 构建的 extension。Vulkan 的 persistent bucket 只在显式 `native_vulkan_graph_reuse=true` 时实验性启用，因历史 Mali padded-mask 风险必须先经 oracle 验证。
- `NANOVLLM_NATIVE_CPU_REPACK` 默认关闭：Arm real-model MTP oracle 已发现当前 Q4_0 `q4_0_4x8` 重排会改变 partial rollback trace。实验 build 开启时，加载器仍只将满足 ISA/shape 条件、且不被 `GET_ROWS` 读取的二维 Q4_0/Q6_K 投影放入该 buffer；`token_embd.weight` 和可选 MTP embedding 始终保持 default buffer。Vulkan/CUDA 不走此路径。
- MTP 的串行 draft、tied vocab head 扫描、host acceptance/readback 和多份 GDN snapshot 仍是主要成本。v3.77 bench 会将 native draft / target verification / KV catch-up 的累计 wall 与 setup time 分开导出；Mali K=1 实测为 `15.90 / 72.06 / 1.03s`，故 verification 而非图 setup（`1.74s`）是主导成本。186-token 的 T=2 verification kernel 内，Q6_K head + argmax 为 `15.15 + 3.95ms` / `79.11ms`，是 fused head 的第一目标；KV gather 仅 `0.25ms`。v3.82 将 Mali `ARGMAX` workgroup 从 16 lane 提升到 256 lane，保持低 index tie-break；真实 K=1 acceptance 保持 `100%`，短 profile 将 ARGMAX 降至约 `0.26ms`，统一 decode 为 `19.92 tok/s`（旧基线 `18.58`）。它减少独立 reduction 的串行扫描，但仍保留全量 logits，因此不是最终 fused head。4096-token math verification 中 GET_ROWS 增至 `5.25ms`，但 score/value matmul 仍约 `62.7 / 134.5ms`；因此 direct paged attention 必须做 page-table 直读和 online softmax/value accumulation，而非仅消除 gather。显式 Flash 单步把该 verification 从 `463.4` 降至 `333.5ms`，但 MTP auto 仍固定 math 以避免历史的 shape-dependent acceptance 回退。v3.80 把这一已验收的融合用于 non-MTP 默认值：同一 Mali 长 decode 的 auto 为 `21.24 tok/s`，显式 math 为 `20.81 tok/s`（+2.07%）；MTP 不共享这项默认变更。固定 high-performance Vulkan build 的长 decode 中，K=1/2/3 都没有超过 MTP-off；K=1 的小 verification 窗口最慢。Mali K=1 的 experimental persistent-graph A/B 还显示高 cache hit 不能抵消 padded bucket 与动态输入开销，故默认继续 eager。
- CUDA 已接入 build、backend discovery、placement audit、native runner 与 opt-in CUDA Graph，但尚未在 NVIDIA 实机完成 Qwen3.5 模型正确性/性能验收。

## 12. 构建与基准命令结构

```bash
# 自动 Arm ISA 探测，并指定 Vulkan shader compiler
NANOVLLM_NATIVE_VULKAN=ON \
NANOVLLM_VULKAN_GLSLC=/path/to/glslc \
scripts/build_native_runtime.sh

# CUDA 与 Vulkan 不能在同一个 extension 中同时开启
NANOVLLM_NATIVE_CUDA=ON scripts/build_native_runtime.sh

# 禁用自动 Arm ISA 选择
NANOVLLM_NATIVE_ARM_ARCH=off scripts/build_native_runtime.sh

# 显式指定 Arm ISA
NANOVLLM_NATIVE_ARM_ARCH=armv8.6-a+dotprod+i8mm \
scripts/build_native_runtime.sh
```

基准映射：

```bash
# pp512
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 512 --gen-len 1 --warmup 1 --repeat 3 --json

# tg128
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 1 --gen-len 129 --warmup 1 --repeat 3 --json

# pp521+tg128
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 521 --gen-len 129 --warmup 1 --repeat 3 --json
```

MTP 模式增加 `--enable-mtp --mtp-max-draft-tokens K`。graph reuse 默认开启；A/B 时增加 `--no-graph-reuse`。v3.7 图级 A/B 参数是 `--native-attention-impl {math,auto,flash}`、`--native-batched-recurrent-snapshots`、`--native-mtp-prefill-fusion`；CUDA Graph 用 `NANOVLLM_NATIVE_CUDA_GRAPHS=ON` 的单独构建做 A/B。pp 读取 `prefill_tok_s`，tg 读取 `decode_tok_s`，组合项读取 `processed_tok_s`，同时记录 `graph_cache_hits/misses/evictions/active_entries`。native runtime 还可读取 `memory_stats()` 和 `mtp_profile_stats()`：后者分开统计 draft、target verification、KV catch-up 及其 graph setup 时间。
