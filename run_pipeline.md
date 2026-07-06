下面按当前这版代码，从 GGUF 加载到最终 decode 输出，把完整链路串起来。
1. 入口
你现在一般会从 CLI 或示例进入：
PYTHONPATH=. python -m nanovllm.cli.chat \
  --backend llamacpp_vulkan \
  --gguf-model /home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf \
  --library-path /home/cix/nano-vlm/llama.cpp/build_nanovllm_vulkan/bin/libnanollama_backend.so
CLI 做三件事：
拼 chat prompt，比如 Qwen 风格的：
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
创建 LLM(...)
每轮调用：
llm.generate([prompt], SamplingParams(...))
CLI 入口在 [chat.py (line 148)](/home/kevin/kevin_prj/nano-vllm/nanovllm/cli/chat.py:148)。
2. Config 初始化
LLM 继承 LLMEngine，真正初始化在：
[llm_engine.py (line 13)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/llm_engine.py:13)
对于 llama.cpp 后端，关键配置是：
backend="llamacpp_cpu" 或 "llamacpp_vulkan"
model_format="gguf"
gguf_model="/path/to/model.gguf"
tokenizer_backend="llamacpp"
device_config={
    "library_path": ".../libnanollama_backend.so",
    "n_threads": 8,
    "n_threads_batch": 8,
    "n_gpu_layers": -1,
    "n_ubatch": 512,
}
这里 tokenizer_backend="llamacpp" 很重要：表示分词和反分词走 GGUF 内置 tokenizer，避免 HF tokenizer 和 GGUF tokenizer 不一致。
3. 创建 llama.cpp 后端
LLMEngine.__init__() 里会调用：
self.model_runner = create_backend(config)
如果是 llamacpp_cpu 或 llamacpp_vulkan，会创建：
[LlamaCppRunner (line 47)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/llamacpp/runner.py:47)
它会加载动态库：
libnanollama_backend.so
然后通过 ctypes 调 C ABI：
nano_llama_backend_create(...)
C++ 侧入口在：
[nano-vllm-backend.cpp (line 37)](/home/kevin/kevin_prj/llama.cpp/src/nano-vllm-backend.cpp:37)
这里完成：
llama_backend_init()
llama_model_load_from_file(params->model_path, mparams)
从 GGUF 加载模型结构、权重、tokenizer metadata
创建 llama_context
设置 CPU/Vulkan 参数：
mparams.n_gpu_layers = params->n_gpu_layers;
cparams.n_ctx = params->n_ctx;
cparams.n_batch = params->n_batch;
cparams.n_ubatch = params->n_ubatch;
cparams.n_seq_max = params->n_seq_max;
cparams.offload_kqv = params->offload_kqv;
cparams.kv_unified = true;
Vulkan 时，n_gpu_layers=-1 会尽量把层 offload 到 Mali GPU；KV cache 也会在 Vulkan 设备上创建。
4. 分词
当你输入一句话后，CLI 先拼完整 prompt，然后：
llm.generate([prompt], sampling_params)
进入：
[LLMEngine.generate() (line 87)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/llm_engine.py:87)
每个 prompt 会调用：
self.add_request(prompt, sp)
如果 tokenizer 后端是 llama.cpp：
prompt = self.model_runner.call("tokenize", prompt)
Python 侧：
[LlamaCppRunner.tokenize() (line 141)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/llamacpp/runner.py:141)
C++ 侧：
nano_llama_backend_tokenize(...)
  -> llama_tokenize(...)
这样 prompt string 变成 token id list。
5. Sequence 创建
分词后的 token 会包装成一个 Sequence：
[sequence.py (line 14)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/sequence.py:14)
它保存：
seq_id
token_ids
last_token
num_prompt_tokens
num_cached_tokens
num_scheduled_tokens
block_table
temperature
max_tokens
ignore_eos
一个 Sequence 就是一条请求的生命周期对象。
然后请求进入：
Scheduler.waiting
6. 主循环
generate() 会不断执行：
while not self.is_finished():
    output, num_tokens = self.step()
核心在：
[LLMEngine.step() (line 73)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/llm_engine.py:73)
一轮 step 做：
Scheduler.schedule()
  -> build_execution_plan()
  -> backend.run(plan)
  -> Scheduler.postprocess()
  -> collect finished outputs
7. Scheduler：先 prefill，再 decode
调度器在：
[scheduler.py (line 25)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/scheduler.py:25)
它有两个队列：
waiting: 等待 prefill 的请求
running: 已经 prefill 完，正在 decode 的请求
调度规则是：
如果 waiting 里有请求，优先做 prefill。
如果没有 prefill，就对 running 里的请求做 decode。
decode 每个 seq 每轮只生成 1 个 token。
KV block 不够时，会 preempt，把 running 请求退回 waiting。
8. BlockManager：nano-vLLM 统一管理 KV block
Block 管理在：
[block_manager.py (line 20)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/block_manager.py:20)
它维护：
free_block_ids
used_block_ids
hash_to_block_id
Block.ref_count
Block.hash
Block.token_ids
每个 block 对应一段 KV cache 空间：
block_id * block_size + offset
当前 block size 默认是 256。
也就是说 nano-vLLM 的逻辑 KV 地址是：
cell_idx = block_id * block_size + token_offset_in_block
这个 cell_idx 后面会传给 llama.cpp，让 llama.cpp 按 nano-vLLM 指定的位置写 KV。
9. Prefix Cache
prefill 前，BlockManager.can_allocate(seq) 会检查完整 block 的 hash：
hash(prompt_block_0)
hash(prompt_block_0 + prompt_block_1)
...
如果发现已有相同 prefix block：
seq.block_table.append(existing_block_id)
block.ref_count += 1
seq.num_cached_tokens = num_cached_blocks * block_size
这表示：
这部分 prefix 的 KV 已经存在，不用重新 prefill。
注意：prefix cache 只复用完整 block，不复用半个 block。
10. 构造统一执行计划
调度器决定本轮执行哪些 seq 后，LLMEngine.step() 调用：
plan = build_execution_plan(seqs, is_prefill, block_size)
定义在：
[base.py (line 31)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/base.py:31)
这个 BackendExecutionPlan 是 CUDA 和 llama.cpp 后端共用的统一执行描述。
它包含：
mode                  # prefill 或 decode
input_ids             # 本轮要送进模型的 token
positions             # 每个 token 的 position
seq_ids               # 每个请求的 seq_id
scheduled_token_counts
block_tables          # 每个 seq 的逻辑 block -> 物理 block 映射
slot_mapping          # 每个输入 token 要写入哪个 KV cell
context_lens          # 每个 seq 当前上下文长度
num_cached_tokens     # prefix cache 命中的 token 数
temperatures
11. Prefill plan 长什么样
假设 prompt 有 600 token，block size 256，命中 1 个 prefix block。
那么：
num_cached_tokens = 256
scheduled_token_counts = 344
input_ids = prompt[256:600]
positions = [256, 257, ..., 599]
block_table = [cached_block, new_block_1, new_block_2]
slot_mapping = [
  block_1*256 + 0,
  block_1*256 + 1,
  ...
  block_2*256 + ...
]
也就是说，已经缓存的 0-255 不再送进模型；只计算 256-599，并把新 KV 写到 nano 指定的 cell。
12. Decode plan 长什么样
decode 阶段，每个 running seq 每轮只喂最后一个 token：
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
也就是说：
本轮 token 的 KV 写入最后一个 block 的最后位置。
decode 的 attention 会通过 seq_id 和 llama.cpp KV metadata 看到这个 seq 历史上所有 cell。
13. Python llama.cpp runner 转 C plan
LlamaCppRunner.run(plan) 会把 Python plan 转成 C struct：
[LlamaCppRunner.run() (line 168)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/llamacpp/runner.py:168)
C struct 是：
nano_llama_kv_plan
字段包括：
is_prefill
n_tokens
n_seqs
block_size
block_table_cols
tokens
positions
seq_ids
scheduled_token_counts
slot_mapping
block_tables
context_lens
num_cached_tokens
然后调用：
nano_llama_backend_run(...)
14. C++ shim 构造 llama_batch
C++ 入口：
[nano-vllm-backend.cpp (line 140)](/home/kevin/kevin_prj/llama.cpp/src/nano-vllm-backend.cpp:140)
它把 nano plan 转成 llama.cpp 的 llama_batch：
batch.token[i] = plan->tokens[i];
batch.pos[i] = plan->positions[i];
batch.n_seq_id[i] = 1;
batch.seq_id[i][0] = token_seq_ids[i];
batch.logits[i] = 0;
然后每个 seq 的最后一个 scheduled token 标记要输出 logits：
batch.logits[offset + count - 1] = 1;
prefill 时，一个 seq 可能有很多 token，但只取最后一个 token 的 logits 来采样下一 token。
decode 时，每个 seq 本来就只有 1 个 token，所以每个 token 都要 logits。
15. 设置外部 KV plan
在调用 llama.cpp 前，shim 会设置 thread-local plan：
nano_llama_set_external_plan(plan);
llama_decode(handle->ctx, batch);
nano_llama_clear_external_plan();
这个非常关键。
它让 llama.cpp 在内部 decode 时，不使用原本的 KV slot 分配逻辑，而是走我们插进去的外部 KV 逻辑。
16. llama.cpp decode 被 hook
我们改了 llama.cpp 的 llama_context::decode()。
当发现当前有 external plan 时，不走默认：
memory->init_batch(...)
而是走：
nano_llama_init_external_memory(...)
这个函数在：
[nano-vllm-ext.cpp (line 55)](/home/kevin/kevin_prj/llama.cpp/src/nano-vllm-ext.cpp:55)
它做两件关键事：
对 prefix cache 命中的 cell，给 llama.cpp KV metadata 补上当前 seq_id。
对本轮要计算的 token，用 slot_mapping 指定 KV 写入 cell。
17. Prefix cache 在 llama.cpp 内部怎么接上
如果是 prefill，并且有 num_cached_tokens > 0，会执行：
nano_llama_attach_cached_prefix(...)
它根据：
block_tables
block_size
num_cached_tokens
还原每个 cached token 的 cell：
cell_idx = block_id * block_size + pos % block_size
然后调用：
kv->external_mark_cell(cell_idx, pos, seq_id)
实现在：
[llama-kv-cache.cpp (line 1184)](/home/kevin/kevin_prj/llama.cpp/src/llama-kv-cache.cpp:1184)
作用是：
这个 cell 已经有 KV 数据了，现在把当前 seq_id 也挂到这个 cell 上。
所以 prefix cache 不是复制 KV，而是共享同一批 cell，并增加 seq 归属。
18. 本轮 token 的 KV 写入
对于本轮真正要计算的 token，nano_llama_init_external_memory() 会把 slot_mapping 转成 llama.cpp 的 slot_info：
sinfo.idxs[0].push_back(plan->slot_mapping[token_offset + i]);
这告诉 llama.cpp：
第 i 个 token 的 K/V 写入这个 cell_idx。
所以 KV 物理位置由 nano-vLLM 决定，而不是 llama.cpp 自己找空位。
19. llama.cpp 真正执行模型
之后就回到 llama.cpp 原生计算图：
embedding
  -> transformer layers
  -> attention
  -> mlp
  -> norm
  -> lm_head
CPU 后端时，算子跑 CPU/Kleidiai。
Vulkan 后端时，权重、KV、计算图尽量 offload 到 Mali Vulkan backend。你日志里看到的：
using device Vulkan0 (Mali-G720-Immortalis)
offloaded 37/37 layers to GPU
Vulkan0 KV buffer
就是这里生效。
20. Attention 如何读历史 KV
llama.cpp attention 仍然按它自己的 KV cache metadata 来读历史。
我们做的事情是提前保证：
需要读的历史 token cell 都已经被标记为属于当前 seq_id
包括：
当前 seq 之前 decode 写过的 token
当前 prefill 新写入的 token
prefix cache 复用的旧 cell
因此 llama.cpp 在 attention mask / KV lookup 时，能按 seq_id 找到这条请求的上下文。
21. 取 logits
llama_decode() 返回后，shim 调：
float * logits = llama_get_logits(handle->ctx);
然后把每个 seq 对应的 logits row 拷贝回 Python：
memcpy(logits_out + row * vocab_size, src, vocab_size * sizeof(float));
也就是说，llama.cpp 后端返回的是 logits，不是在 C++ 里采样。
22. Python 统一采样
回到 Python：
[LlamaCppRunner._sample() (line 224)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/llamacpp/runner.py:224)
当前采样逻辑是 temperature sampling：
scaled = logits / temperature
probs = softmax(scaled)
token = argmax(probs / exponential_noise)
所以 llama.cpp 后端只负责执行模型，采样仍然在 nano-vLLM 层做。
这和“统一架构”的目标是一致的。
23. postprocess：更新 Sequence 和 Block
采样得到 token_ids 后，回到：
[Scheduler.postprocess() (line 73)](/home/kevin/kevin_prj/nano-vllm/nanovllm/engine/scheduler.py:73)
它做：
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
如果 prefill 还没处理完整个 prompt，例如 chunked prefill：
continue
否则 append 新 token：
seq.append_token(token_id)
然后判断是否结束：
token_id == eos
或
num_completion_tokens == max_tokens
如果结束：
seq.status = FINISHED
block_ids = block_manager.deallocate(seq)
block_releases.append((block_ids, seq.seq_id))
running.remove(seq)
24. KV block 释放同步到 llama.cpp
LLMEngine.step() 前后都会调用：
flush_backend_releases()
它把 Scheduler 里释放的 block 通知后端：
self.model_runner.call("release_blocks", block_ids, [seq_id])
Python 侧：
LlamaCppRunner.release_blocks(...)
C++ 侧：
nano_llama_backend_release_blocks(...)
它遍历：
block_id
  -> block_id * block_size + offset
然后调用：
kv->external_rm_cell_seq(cell_idx, seq_id)
意思是：
这个 seq 不再引用这些 KV cell。
如果某个 cell 还被 prefix cache 的其他 seq 引用，不会立刻破坏数据；只是移除当前 seq 的归属。
25. detokenize
当所有请求结束后，generate() 收集每个 seq 的 completion token ids：
outputs[seq_id] = seq.completion_token_ids
如果 tokenizer 后端是 llama.cpp：
self.model_runner.call("detokenize", token_ids)
Python 侧：
[LlamaCppRunner.detokenize() (line 152)](/home/kevin/kevin_prj/nano-vllm/nanovllm/backends/llamacpp/runner.py:152)
C++ 侧：
nano_llama_backend_detokenize(...)
  -> llama_detokenize(...)
最后返回：
[
  {
    "text": "...",
    "token_ids": [...]
  }
]
整体流程图
CLI / example
  ↓
LLM(...)
  ↓
Config
  ↓
create_backend()
  ↓
LlamaCppRunner
  ↓
ctypes load libnanollama_backend.so
  ↓
nano_llama_backend_create()
  ↓
llama_model_load_from_file(GGUF)
  ↓
llama_init_from_model()
  ↓
tokenize(prompt)
  ↓
Sequence
  ↓
Scheduler.waiting
  ↓
generate loop
  ↓
Scheduler.schedule()
  ├─ prefill: 分配 block / prefix cache / chunked prefill
  └─ decode: 每个 running seq 追加 1 token
  ↓
build_execution_plan()
  ↓
LlamaCppRunner.run(plan)
  ↓
nano_llama_backend_run(plan)
  ↓
构造 llama_batch
  ↓
设置 external KV plan
  ↓
llama_decode()
  ↓
nano_llama_init_external_memory()
  ├─ cached prefix cell 挂 seq_id
  └─ slot_mapping 指定本轮 KV 写入 cell
  ↓
llama.cpp CPU/Vulkan 执行 transformer
  ↓
llama_get_logits()
  ↓
Python temperature sampling
  ↓
Scheduler.postprocess()
  ├─ hash 完整 block
  ├─ append token
  ├─ 判断 eos / max_tokens
  └─ 释放 block 并通知 llama.cpp
  ↓
detokenize()
  ↓
返回 text
一句话总结
现在这版的核心是：nano-vLLM 仍然负责请求生命周期、调度、paged KV block、prefix cache、preemption、采样和输出；llama.cpp 负责 GGUF 模型加载、tokenizer、CPU/Vulkan 算子执行。两边通过 BackendExecutionPlan + external KV plan 接起来，KV 物理 cell 由 nano-vLLM 统一规划，再让 llama.cpp 按指定 cell 写入和读取。