# nano-vLLM 当前执行流程：v3.2

## 1. 架构边界

nano-vLLM Python 层负责 request、scheduler、sequence、block table、slot mapping、PagedKV 分配、greedy speculative accept/rollback、EOS/max_tokens 和统计。

in-tree C++ runtime 负责 GGUF 加载、Qwen3.5 target/MTP GGML graph、CPU/Vulkan backend、6 层 attention PagedKV、18 层 GDN recurrent state 及 K+1 snapshot planes。

项目只编译官方 llama.cpp 的 GGML 子目录；不链接 `libllama`、不创建 `llama_context`、不使用 CUDA。

## 2. 初始化

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

Vulkan 模式把模型 compute node 固定到 Vulkan，并在 graph allocation 后执行 placement audit。CPU backend 只作为 scheduler 辅助 backend 存在。

## 3. Python 执行计划

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

## 4. TargetChunkGraph

单 sequence 的 T 个 token 共用一张 graph：

```text
tokens[T] + positions[4T]
  -> token embedding
  -> 24-layer target trunk
       18 multi-token recurrent GDN layers
       6 causal full-attention layers
  -> output norm
  -> Last 或 All vocab head/argmax
```

full-attention 层先批量写 K/V，再按 `read_slots[C]` gather，使用 `[C,T]` causal mask 防止 query 看到未来 token。

GDN 层在 graph 内逐 token 更新 convolution/delta state。普通路径只提交最新状态；MTP verification 同时生成 newest-first K+1 snapshot planes。

输出模式：

- `Last`：只计算最后一列 greedy token；
- `All`：计算每一列 greedy prediction，并返回全部 target hidden。

## 5. MTP-off 流程

每个 scheduler sequence chunk 执行一次 `TargetChunkGraph(Last)`：

```text
scheduled T tokens
  -> batch target trunk
  -> target KV batch write
  -> recurrent canonical state commit
  -> last-column vocab head
  -> one greedy token
```

prompt 长度不超过 `max_num_batched_tokens` 时，整个 prompt 只需一张 target graph。decode 时 T=1，语义等价于原 token graph。

## 6. MTP prefill / fallback

开启 MTP 时，target chunk 还返回每一列 hidden。runtime 维护跨 chunk 的 `pending_hidden`，并构造右移配对：

```text
mtp_hidden[0] = old pending_hidden
mtp_hidden[i] = target_hidden[i-1]
```

随后执行一张 MTP KV-only graph：

```text
[tokens, shifted_hidden, positions, write_slots]
  -> MTP merge/norm
  -> K/V projection + K norm + IMRoPE
  -> batch SET_ROWS into MTP PagedKV
```

它不执行 MTP attention、FFN、head 或 hidden readback。最后一列 target hidden 成为新的 `pending_hidden`。

## 7. MTP speculative decode

设最大 draft 数为 K=3，当前输入为 `x_p`，已有 target hidden 为 `h_{p-1}`。

### 7.1 Draft

```text
MTP(x_p, h_{p-1}, p)     -> d1, m0
MTP(d1,  m0,      p+1)   -> d2, m1
MTP(d2,  m1,      p+2)   -> d3
```

三步有真实自回归依赖，v3.2 仍执行三张串行 draft graph。每步包含 MTP layer、tied vocab head 和 greedy argmax。

### 7.2 Batch target verification

```text
inputs = [x_p, d1, d2, d3]
TargetChunkGraph(T=4, output=All, snapshots=4)
  -> predictions t1,t2,t3,t4
  -> hidden h_p,h_p+1,h_p+2,h_p+3
  -> recurrent planes 3,2,1,0
```

host 按顺序比较：

```text
t1 == d1
t2 == d2
t3 == d3
```

第一个 mismatch 前的严格相等个数为 accepted `a`。每轮返回 `a+1` 个 token：a 个 accepted target prediction，以及一个 correction/bonus token `t[a]`。

### 7.3 Rollback/commit

接受 a 个 draft 后：

```text
recurrent canonical <- snapshot plane K-a
pending_hidden       <- target_hidden[a]
next_position        <- p+a+1
```

target attention KV 的未提交尾部不需要清零：下一轮 gather 只读取 committed context，未来 write slot 会覆盖尾部。

### 7.4 MTP KV catch-up

draft 0 已以真实 `x_p + h_{p-1}` 写入 MTP KV。若 `a>0`，只补齐 index `1..a`：

```text
tokens        = inputs[1:a+1]
hidden_inputs = target_hidden[0:a]
positions     = p+1 ... p+a
write_slots   = corresponding committed slots
```

这些行一次送入 batched KV-only graph。`a=0` 时不执行 catch-up。

## 8. 图数与主要成本

MTP K=3 的常规轮次为：

```text
3 × serial MTP draft graph
1 × TargetChunkGraph(T=4, All)
0/1 × MTP KV-only catch-up graph
```

相较 v3.1，verification graph 数从最多 4 降为 1，但 v3.2 固定计算全部 4 行，失去了 mismatch early-stop。实际成本还包括：

- 每个 draft 的大词表 tied head；
- target T=4 在 CPU/Vulkan 上对应的实际量化 kernel；
- 18 层 GDN 的 4 组 snapshot 写入；
- graph build/allocation、backend compute 和 host readback。

## 9. 同步与 readback

`GraphExecutor::compute()` 使用同步 GGML scheduler API，RAII guard 在退出时 reset transient allocation。

readback 规则：

- MTP-off：只读取最后 greedy token；
- MTP draft：读取 draft token，非末步读取下一步所需 hidden；
- verification：读取 K+1 predictions 与 target hidden；
- MTP KV-only：无 host 输出。

## 10. Release

request release 会校验 sequence ID，清理 target/MTP KV block、recurrent planes 和 pending hidden/position，并释放 Python-to-native slot mapping。

## 11. 基准命令结构

```bash
# pp512
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 512 --gen-len 1 --warmup 1 --repeat 3 --json

# tg128（prefill 产生第一个 completion，再执行 128 个 decode step）
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 1 --gen-len 129 --warmup 1 --repeat 3 --json

# pp521+tg128
python3 -m nanovllm.cli.bench "$MODEL" ... \
  --prompt-len 521 --gen-len 129 --warmup 1 --repeat 3 --json
```

MTP 模式增加：

```text
--enable-mtp --mtp-max-draft-tokens 3
```

pp 读取 `prefill_tok_s`，tg 读取 `decode_tok_s`，组合项读取 `processed_tok_s`。
