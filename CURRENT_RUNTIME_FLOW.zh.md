# nano-vLLM 当前执行流程：v3.3

## 1. 架构边界

nano-vLLM Python 层负责 request、scheduler、sequence、block table、slot mapping、PagedKV 分配、greedy speculative accept/rollback、EOS/max_tokens 和统计。

in-tree C++ runtime 负责 GGUF 加载、Qwen3.5 target/MTP GGML graph、CPU/Vulkan backend、6 层 attention PagedKV、18 层 GDN recurrent state 及 K+1 snapshot planes。

项目只编译官方 llama.cpp 的 GGML 子目录；不链接 `libllama`、不创建 `llama_context`、不使用 CUDA。llama.cpp 提供 GGML tensor、graph、CPU/Vulkan backend 和底层算子，调度与缓存所有权仍属于 nano-vLLM。

## 2. 构建阶段

```text
scripts/build_native_runtime.sh
  -> 检查显式 GGML_NATIVE/GGML_CPU_ARM_ARCH 参数
  -> aarch64 auto 模式：检查所有 CPU 的 asimddp+i8mm
  -> 编译器探测 armv8.6-a+dotprod+i8mm
  -> 成功时注入 GGML_NATIVE=OFF + GGML_CPU_ARM_ARCH
  -> 可选校验 NANOVLLM_VULKAN_GLSLC
  -> CMake 编译 in-tree GGML + nanovllm._C
```

显式 CMake 参数优先于自动参数。feature 或编译探针失败时安全回退到 GGML 默认构建。

## 3. 初始化

```text
LLM(config)
  -> NativeRunner
  -> nanovllm._C.Qwen35Runtime
  -> validate Qwen3.5/Q4_0/MTP GGUF contract
  -> create CPU or [Vulkan, CPU] backends
  -> load model tensors
  -> allocate target/MTP PagedKV
  -> allocate recurrent canonical/snapshot planes
  -> create GGML graph scheduler
```

Vulkan 模式把模型 compute node 固定到 Vulkan，并在 graph allocation 后执行 placement audit。runtime 读取 `BackendDeviceInfo`；设备 name/description 含 `Mali` 时，普通长 prefill 的 target chunk limit 设为 64。

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

NativeRunner 将 Python sequence ID 映射到 native sequence slot。C++ 根据 block table 展开物理 read/write slots，并校验 Python slot mapping 与同图重复写入。

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

full-attention 层先批量写 K/V，再按 `read_slots[C]` gather，使用 `[C,T]` causal mask防止 query 看到未来 token。GDN 层在 graph 内逐 token 更新 convolution/delta state；普通路径只提交最新 state，MTP verification 同时生成 newest-first K+1 snapshot planes。

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

开启 MTP 时，每个 target chunk 都返回各列 normalized hidden。runtime 维护跨 chunk 的 `pending_hidden`，构造严格右移配对：

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

## 8. MTP speculative decode

设 K=3，当前 pending token 为 `x_p`，pending hidden 为 `h_{p-1}`。

### 8.1 Draft

```text
MTP(x_p, h_{p-1}, p)   -> d1, m0
MTP(d1,  m0,      p+1) -> d2, m1
MTP(d2,  m1,      p+2) -> d3
```

三步有真实自回归依赖，v3.3 仍执行三张串行 draft graph。每步包含 MTP layer、tied vocab head 和 greedy argmax。

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

`a=0` 时不执行 catch-up。该 graph 在 v3.3 仍与 verification 分离。

## 9. 图数、同步与 readback

K=3 常规轮次为：

```text
3 × serial MTP draft graph
1 × TargetChunkGraph(T=4, All)
0/1 × MTP KV-only catch-up graph
```

readback 规则：

- 非最终 MTP-off Mali prefill 片：无 token/hidden readback，无 output norm/head；
- 非最终 MTP-on prefill 片：读取 hidden，执行 KV-only maintenance，不读 token；
- 最终 prefill 片：读取一个 token；MTP-on 还读取 hidden；
- MTP draft：读取 draft token，非末步还读取下一步 hidden；
- verification：读取 K+1 predictions 与全部 target hidden；
- MTP KV-only：无 host 输出。

`GraphExecutor::compute()` 使用同步 GGML scheduler API，graph guard 退出时 reset transient allocation。当前每轮仍会重建 graph metadata/context，尚未做 persistent graph bucket。

## 10. Release

request release 会校验 sequence ID，清理 target/MTP KV block、recurrent planes、pending hidden/position，并释放 Python-to-native slot mapping。

## 11. 当前 backend 策略边界

- Mali 的 64-token 值来自当前 Qwen3.5-2B/Mali-G720 实测，不应直接泛化到其他模型或 GPU。
- 当前官方 Vulkan 默认把 Mali Q4_0 的 T=1 和 T=2～4 都路由到 MMVQ。实测表明 T=1 应走 DMMV，T=2～4 应保留 MMVQ；v3.3 尚未自动混合，下一版修正。
- CPU 权重当前仍在 default buffer，尚未进入 CPU_REPACK。
- MTP 的串行 draft、tied Q6_K head 扫描、host acceptance/readback 和多份 GDN snapshot 仍是主要成本。

## 12. 构建与基准命令结构

```bash
# 自动 Arm ISA 探测，并指定 Vulkan shader compiler
NANOVLLM_NATIVE_VULKAN=ON \
NANOVLLM_VULKAN_GLSLC=/path/to/glslc \
scripts/build_native_runtime.sh

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

MTP 模式增加 `--enable-mtp --mtp-max-draft-tokens K`。pp 读取 `prefill_tok_s`，tg 读取 `decode_tok_s`，组合项读取 `processed_tok_s`。
