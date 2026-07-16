# nano-vLLM 当前执行流程：v3.1

## 1. 架构边界

当前版本由 nano-vLLM 负责：

- Python request/scheduler/sequence 生命周期；
- block table、slot mapping 与 PagedKV 分配；
- speculative decoding 的 draft、verification、accept、rollback 调度；
- greedy 输出保留、EOS/max_tokens 处理和统计。

in-tree native C++ runtime 负责：

- GGUF 元数据与 tensor 加载；
- Qwen3.5 target/MTP GGML graph 构建；
- CPU/Vulkan backend、scheduler 和 tensor buffer；
- 6 层 attention PagedKV；
- 18 层 Gated Delta Net recurrent state 与 K+1 snapshot planes。

项目只编译官方 llama.cpp 的 GGML 子目录。运行时不链接 `libllama`，不创建 `llama_context`，也不使用 CUDA。

## 2. 初始化

```text
LLM(config)
  -> NativeRunner
  -> nanovllm._C.Qwen35Runtime
  -> validate GGUF qwen35/Q4_0/MTP contract
  -> create CPU or [Vulkan,CPU] backend list
  -> load model tensors into backend buffers
  -> allocate target KV + MTP KV
  -> allocate recurrent canonical/snapshot planes
  -> create GGML graph scheduler
```

Vulkan 模式将所有真实 compute node（包括 CPY/SET_ROWS）固定到 Vulkan，并在 graph allocation 后执行严格 placement audit。CPU 作为 Vulkan scheduler 的辅助 backend 存在，但不承载模型 compute node。

## 3. Python 到 native 的执行计划

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

NativeRunner 将外部 sequence ID 映射到有限 native sequence slot，再调用 C++ `run()` 或 `run_mtp()`。C++ 会重新根据 block table 展开物理 slot，校验 Python 传入的 mapping，并拒绝同一 graph 内重复写 slot。

## 4. 非 MTP / 普通 target 流程

当前 `run()` 对每个 sequence、每个 token 执行：

```text
token + IMRoPE position + target slot/context
  -> 24-layer target graph
       18 recurrent GDN layers
       6 full-attention layers with PagedKV
  -> output norm
  -> 仅最后一个 scheduled token 执行 tied vocab head + argmax
  -> update recurrent plane 0 and target KV
```

MTP 关闭时不再下载中间 hidden，只读取最后 greedy token。当前限制是 target graph 仍为单 token，因此 pp512 会执行 512 张 target graph；这是下一版本的首要改造点。

## 5. MTP prefill / target fallback

MTP 打开时，target token graph 还需要返回每个 token 的 target hidden。runtime 在 host 端构造 MTP hidden 的右移配对：

```text
token[0] <-> 进入 chunk 前的 pending_hidden
token[i] <-> target_hidden[i-1]
```

target chunk 的所有 token 完成后，一次执行 batched MTP KV-only graph：

```text
[tokens, shifted hidden, positions, write slots]
  -> MTP merge
  -> attention norm
  -> K/V projection + K norm/IMRoPE
  -> batch SET_ROWS into MTP PagedKV
```

这条路径不执行 MTP attention/FFN/head。chunk 最后一个 target hidden 成为下一轮 `pending_hidden`。

## 6. MTP speculative decode

设最大 draft 数 `K=3`，当前位置输入为 `x_p`，进入轮次前 target hidden 为 `h_{p-1}`。

### 6.1 Draft

```text
draft 0: MTP(x_p, h_{p-1}, position p) -> d1, mtp_hidden_0
draft 1: MTP(d1, mtp_hidden_0, p+1)    -> d2, mtp_hidden_1
draft 2: MTP(d2, mtp_hidden_1, p+2)    -> d3
```

三步有自回归依赖，当前仍是三张串行图。最后一步 hidden 没有消费者，因此不再读回。第 0 步写入的 MTP KV 已经是 target-conditioned，因为它使用真实的 `x_p + h_{p-1}`。

### 6.2 Target verification

verification inputs：

```text
[x_p, d1, d2, d3]
```

当前 target 仍逐行执行，但每得到一行 target prediction 就立即比较：

- mismatch at `a < K`：停止，不再执行尾部 target 图；
- full accept `a=K`：执行第 K 行，取得额外 correction token。

每个 target 行按 newest-first 写入 recurrent snapshot plane：

```text
input index 0 -> plane K
input index 1 -> plane K-1
...
input index K -> plane 0
```

接受 `a` 个 draft 后选择 `plane K-a`。target KV 的未提交尾部不会被下轮 context gather，未来写入会覆盖，因此不需要物理清零或复制回滚。

### 6.3 Accept 与 correction

greedy acceptance 只能使用严格相等：

```text
target_prediction[i] == draft[i]
```

每轮返回 `a+1` 个 token：前 a 个为接受的 draft 对应 target prediction，最后一个为 target correction/bonus token。Python Scheduler 只保留未超过 EOS/max_tokens 的前缀。

### 6.4 MTP KV catch-up

第 0 行已由 draft 0 正确写入，不再重复。若 `a>0`，将以下行一次送入 KV-only graph：

```text
tokens          = verification_inputs[1:a+1]
hidden_inputs   = target_hidden[0:a]
positions       = p+1 ... p+a
write_slots     = 对应 PagedKV slots
```

这样 committed MTP KV 全部是 target-conditioned。若 `a=0`，本轮没有 catch-up graph。

最后：

```text
pending_hidden = target_hidden[a]
next_position  = p + a + 1
```

## 7. 同步与输出

`GraphExecutor::compute()` 当前使用同步 GGML scheduler API。v3.1 删除了其后的重复显式 synchronize，并移除了每图入口的重复 reset。RAII guard 在退出时统一 reset scheduler transient allocation。

Host readback 规则：

- target/MTP greedy graph：读取 I32 token；
- MTP draft 非末步：读取下一 draft 所需 hidden；
- target verification：读取 target hidden，用于 pending state/catch-up；
- MTP KV-only：无 host 输出；
- 非 MTP target：不读取 hidden。

## 8. Cache/state release

当前 request release 仍执行：

```text
validate sequence IDs
clear released target/MTP KV blocks
clear recurrent slot planes
reset pending_hidden/position
release Python-to-native slot mapping
```

物理清零不是 cache reuse 正确性的必要条件，但本版本仍保留它作为确定性诊断和数据卫生策略；后续可增加 performance/secure-clear 配置。

## 9. 当前性能结构

K=3、平均接受 a 时，每轮图数约为：

```text
3 draft
+ (a+1) target verification（early-stop 后）
+ 0 或 1 KV-only catch-up
```

v3.1 已消除原来的 `K+1` 固定 target 尾部和 `a+1` 完整 MTP catch-up，但 target trunk/head 仍逐 token 扫描。下一版本将新增单序列 `TargetChunkGraph`：

```text
prefill: T 张 target token graph -> 1 张 target chunk graph
verify:  a+1 张 target graph      -> 1 张 K+1 target chunk graph
```

chunk graph 必须实现多 token causal attention、GDN multi-token state、newest-first snapshots，以及 Last/All 两种 head 输出模式。

## 10. 基准流程

严格 tg128 需要 128 次 decode step，因此 CLI 使用 `gen_len=129`：

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

MTP 模式增加：

```text
--enable-mtp --mtp-max-draft-tokens 3
```

pp 读取 `prefill_tok_s`，tg 读取 `decode_tok_s`，组合项读取 `processed_tok_s`。
