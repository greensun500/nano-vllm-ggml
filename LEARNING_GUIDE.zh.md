# nano-vLLM 项目学习目录与执行流程

这份文档面向第一次系统阅读 nano-vLLM 的学习者。建议按“外层 API -> Engine 调度 -> ModelRunner 执行 -> 模型与算子实现”的顺序阅读，这样可以先建立主流程，再逐步下钻到性能优化细节。

## 1. 项目定位

`nano-vllm` 是一个轻量版离线推理框架，API 模仿 vLLM，核心目标是用较少代码实现：

- 批量请求调度
- prefill / decode 两阶段推理
- KV cache block 管理
- prefix cache 复用
- tensor parallel
- FlashAttention
- CUDA graph 加速 decode

推荐入口文件：

- [`example.py`](example.py)：最小使用示例。
- [`nanovllm/llm.py`](nanovllm/llm.py)：公开的 `LLM` 类入口。
- [`nanovllm/engine/llm_engine.py`](nanovllm/engine/llm_engine.py)：推理主循环。

## 2. 总体执行流程

```text
用户调用 example.py
  ↓
LLM(path, ...)
  ↓
LLMEngine 初始化
  ↓
加载 Config / tokenizer / ModelRunner / Scheduler
  ↓
generate(prompts, sampling_params)
  ↓
每个 prompt 转成 Sequence，加入 Scheduler.waiting
  ↓
循环 step()
  ↓
Scheduler.schedule()
  ├─ prefill：处理 prompt token，分配 KV cache block，尝试 prefix cache
  └─ decode：每个 running sequence 生成 1 个 token
  ↓
ModelRunner.run()
  ↓
prepare_prefill / prepare_decode 构造 input_ids、positions、block_tables、slot_mapping
  ↓
Qwen3ForCausalLM forward
  ↓
Attention 写入/读取 KV cache，调用 FlashAttention
  ↓
Sampler 从 logits 采样 token
  ↓
Scheduler.postprocess()
  ├─ 更新 cache hash
  ├─ append 新 token
  ├─ 判断 EOS / max_tokens
  └─ finished 后释放 block
  ↓
tokenizer.decode()
  ↓
返回 [{"text": ..., "token_ids": ...}]
```

## 3. 核心模块目录

### 外层 API

- [`nanovllm/__init__.py`](nanovllm/__init__.py)：暴露 `LLM` 和 `SamplingParams`。
- [`nanovllm/llm.py`](nanovllm/llm.py)：`LLM` 只是继承 `LLMEngine`，真正逻辑都在 engine。
- [`nanovllm/sampling_params.py`](nanovllm/sampling_params.py)：控制 `temperature`、`max_tokens`、`ignore_eos`。
- [`nanovllm/config.py`](nanovllm/config.py)：模型路径、batch 限制、KV cache block size、tensor parallel 等全局配置。

### Engine 层

- [`nanovllm/engine/llm_engine.py`](nanovllm/engine/llm_engine.py)：总控入口，负责初始化、添加请求、循环推理、输出解码。
- [`nanovllm/engine/sequence.py`](nanovllm/engine/sequence.py)：每个请求的状态对象，记录 prompt、生成 token、block table、调度进度。
- [`nanovllm/engine/scheduler.py`](nanovllm/engine/scheduler.py)：核心调度器，维护 `waiting` / `running` 队列，决定本轮 prefill 还是 decode。
- [`nanovllm/engine/block_manager.py`](nanovllm/engine/block_manager.py)：KV cache block 分配、释放、引用计数、prefix hash 匹配。
- [`nanovllm/engine/model_runner.py`](nanovllm/engine/model_runner.py)：真正执行模型，负责 tensor parallel 进程、权重加载、KV cache 分配、CUDA graph、采样。

### 模型层

- [`nanovllm/models/qwen3.py`](nanovllm/models/qwen3.py)：Qwen3 模型结构，包括 embedding、decoder layer、attention、MLP、lm head。
- [`nanovllm/layers/linear.py`](nanovllm/layers/linear.py)：tensor parallel 线性层，包含 column parallel、row parallel、QKV 合并投影。
- [`nanovllm/layers/attention.py`](nanovllm/layers/attention.py)：KV cache 写入、prefill/decode attention 调用。
- [`nanovllm/layers/embed_head.py`](nanovllm/layers/embed_head.py)：词表并行 embedding 和 lm head。
- [`nanovllm/layers/sampler.py`](nanovllm/layers/sampler.py)：temperature sampling。
- [`nanovllm/layers/layernorm.py`](nanovllm/layers/layernorm.py)、[`nanovllm/layers/rotary_embedding.py`](nanovllm/layers/rotary_embedding.py)、[`nanovllm/layers/activation.py`](nanovllm/layers/activation.py)：基础算子。

### 工具层

- [`nanovllm/utils/loader.py`](nanovllm/utils/loader.py)：从 safetensors 加载权重，并处理 QKV / gate-up 这种合并权重。
- [`nanovllm/utils/context.py`](nanovllm/utils/context.py)：保存当前 forward 所需的全局上下文，例如 prefill/decode 标记、block table、slot mapping。

## 4. 推荐学习顺序

1. 先看 [`example.py`](example.py)，弄清楚外部 API 怎么用。
2. 看 [`LLMEngine.generate()`](nanovllm/engine/llm_engine.py) 和 `step()`，掌握主循环。
3. 看 [`Sequence`](nanovllm/engine/sequence.py)、[`Scheduler`](nanovllm/engine/scheduler.py)、[`BlockManager`](nanovllm/engine/block_manager.py)，理解请求状态、批处理和 KV cache 管理。
4. 看 [`ModelRunner.prepare_prefill()`](nanovllm/engine/model_runner.py) / `prepare_decode()`，理解调度结果如何变成 GPU 张量。
5. 看 [`Attention.forward()`](nanovllm/layers/attention.py)，重点理解 prefill、decode、prefix cache 如何接入 FlashAttention。
6. 看 [`Qwen3ForCausalLM`](nanovllm/models/qwen3.py) 和并行线性层，理解模型结构和 tensor parallel。
7. 最后看 CUDA graph、sampler、loader，这些属于性能和工程细节。

## 5. 重点概念索引

- `Sequence`：一个请求的生命周期。
- `block_table`：逻辑 token 序列到物理 KV cache block 的映射。
- `slot_mapping`：当前 token 的 KV 应该写到 cache 的哪个物理位置。
- `prefill`：处理 prompt，可以一次处理多个 token。
- `decode`：每个序列每轮生成一个 token。
- `prefix cache`：相同 prefix 的完整 block 可以复用。
- `preempt`：KV cache 不够时，把 running 请求退回 waiting，释放其 block。
- `tensor parallel`：不同 GPU rank 持有部分权重，必要时 all-reduce / gather。
- `CUDA graph`：decode 小 batch 场景下复用捕获好的计算图。

## 6. 源码阅读主线

### 6.1 初始化阶段

`LLMEngine.__init__()` 是初始化主入口：

1. 从传入参数构造 `Config`，并读取 Hugging Face config。
2. 设置 `Sequence.block_size`，保证请求对象和 KV cache block size 一致。
3. 如果 `tensor_parallel_size > 1`，启动额外 `ModelRunner` 子进程。
4. 在 rank 0 创建主 `ModelRunner`。
5. 加载 tokenizer，并把 EOS token id 写入 config。
6. 创建 `Scheduler`。

### 6.2 请求进入系统

`LLMEngine.generate()` 会把所有 prompt 加入调度器：

- 字符串 prompt 会通过 tokenizer 编码成 token id。
- token id list 会直接作为 prompt。
- 每个请求被包装成 `Sequence`。
- `Sequence` 初始状态是 `WAITING`，进入 `Scheduler.waiting` 队列。

### 6.3 调度阶段

`Scheduler.schedule()` 先尝试 prefill，再尝试 decode：

- prefill 优先处理 waiting 队列中的新请求。
- 如果请求已有完整 block 命中 prefix cache，则复用已有 KV block。
- 如果 batch token 数超过 `max_num_batched_tokens`，可能发生 chunked prefill。
- 当 prompt prefill 完成后，请求进入 `RUNNING` 队列。
- decode 阶段每个 running 请求只调度 1 个 token。
- 如果 KV cache 空间不够，会通过 `preempt()` 把请求退回 waiting。

### 6.4 KV cache block 管理

`BlockManager` 负责物理 block 的生命周期：

- `allocate()`：为序列建立 `block_table`。
- `can_allocate()`：检查可用 block 数量，并统计 prefix cache 命中的完整 block 数。
- `may_append()`：decode 时如果跨入新 block，就追加一个物理 block。
- `hash_blocks()`：prefill 后给完整 block 计算 hash，用于后续 prefix cache。
- `deallocate()`：请求结束或被抢占时释放 block，减少引用计数。

### 6.5 模型执行阶段

`ModelRunner.run()` 根据调度类型走两条路径：

- `prepare_prefill()`：构造变长序列 attention 所需的 `cu_seqlens_q`、`cu_seqlens_k`、`slot_mapping` 和可选 `block_tables`。
- `prepare_decode()`：每个序列只输入最后一个 token，并传入完整上下文长度和 block table。
- `run_model()`：prefill 通常直接 eager 执行；decode 在条件满足时复用 CUDA graph。
- `Sampler`：rank 0 根据 logits 和 temperature 采样下一个 token。

### 6.6 Attention 与 KV cache

`Attention.forward()` 是理解推理性能的关键：

- 先用 Triton kernel 把当前 K/V 写入预分配的 KV cache。
- prefill 阶段调用 `flash_attn_varlen_func()`。
- 如果有 prefix cache，prefill 会通过 `block_table` 读取已有 cache。
- decode 阶段调用 `flash_attn_with_kvcache()`，直接基于历史 KV cache 计算当前 token。

### 6.7 输出回写

`Scheduler.postprocess()` 负责把本轮模型输出写回序列：

- 对刚完成的完整 block 计算 hash。
- 更新 `num_cached_tokens`。
- 如果 chunked prefill 还没结束，只更新进度，不采样输出。
- prefill 完成或 decode 阶段会 append 新 token。
- 命中 EOS 或达到 `max_tokens` 后标记 `FINISHED` 并释放 block。

## 7. 学习时建议重点画出的数据结构

阅读时可以单独画三张图：

1. `Sequence` 状态变化图：`WAITING -> RUNNING -> FINISHED`，以及 preempt 回到 `WAITING`。
2. KV cache 映射图：`Sequence.token_ids -> block_table -> physical KV cache slot`。
3. prefill/decode 张量图：`input_ids`、`positions`、`slot_mapping`、`block_tables` 在两阶段中的差异。
