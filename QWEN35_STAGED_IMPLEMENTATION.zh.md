# Qwen3.5 分阶段实现记录

## 固定目标

- 模型：`Qwen3.5-2B-Q4_0.gguf`，SHA256 为 `d73547a2371be95c17fecb37321e7437ff937444ad87cbf2f9bc43849d5fe8d7`。
- 模型范围：纯文本；GGUF 内含 24 层主模型和 1 层内置 MTP。
- 硬件路径：CPU 和 Vulkan，不使用 CUDA。
- llama.cpp 基线：官方提交 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`。
- 解码策略：第一版仅支持 greedy。
- 验收原则：优先保证框架正确跑通，性能和内存不是阶段验收门槛。

## 阶段划分

1. Qwen3.5-2B 使用 llama.cpp 原生 hybrid memory 正常推理。
2. 接入 nano-vLLM paged KV，暂不做 prefix cache 和 preemption。
3. 接入 GGUF 内置 MTP，完成 greedy speculative decoding。

每个阶段单独反思并提交，后一个阶段从前一个阶段的 Git 提交继续。

## 阶段 1：正常推理

状态：完成。

实现边界：

- 新增独立、窄接口的 `libnanollama_backend.so`，不修改 llama.cpp 核心内存实现。
- 主模型上下文使用官方 `llama_memory_hybrid`，同时覆盖 full attention KV 与 recurrent state。
- nano-vLLM 请求 ID 映射到有限、可复用的 llama.cpp sequence slot；请求完成后删除完整原生序列状态。
- llama.cpp 原生 KV 模式关闭 nano-vLLM prefix cache，避免软件层跳过 prompt 而原生上下文中没有对应 KV。
- 从词表读取完整 EOG 集合，而不是只读取单个 EOS。
- llama.cpp 路径拒绝非零 temperature，只执行 logits argmax。
- 默认 Qwen3.5 模板使用关闭思考的 generation prompt：`<think>\n\n</think>\n\n`。

验证结果：

- 本机 x86 CPU：连续两次请求成功，slot 清理和复用成功。
- CIX CPU：连续两次请求成功。
- CIX Vulkan：识别 `Mali-G720-Immortalis`，日志显示 26/26 层 offload，连续两次请求成功。
- CPU 与 Vulkan 对同一 prompt 生成完全一致的 greedy token。
- CIX 双序列 batch：分别生成 `red` 和 `blue`，未发生 logits 行或序列状态串线。
- 本机与 CIX 的 4 项阶段单元测试全部通过。

### 阶段 1 反思

最初的外部 paged KV 接口只适用于普通 attention KV cache，不能直接覆盖 Qwen3.5 的 hybrid memory。阶段 1 先使用官方原生 memory 是必要的隔离：它先证明模型、量化文件、CPU/Vulkan 图、tokenizer、调度和停止条件均正确，阶段 2 的问题域因此只剩 KV 所有权与地址映射。

实现中最容易遗漏的不是模型结构，而是 sequence ID 生命周期。nano-vLLM 的请求 ID 单调递增，llama.cpp recurrent state 使用固定数量的 slot；直接透传会让第二批请求越界或复用错误状态。显式映射、完整清理、成功后再归还 slot，已经通过连续请求和双序列 batch 验证。

阶段 1 仍有意保留以下限制：没有 paged KV、没有 prefix cache、没有 preemption、没有启用 MTP 推测、llama.cpp 路径只接受 greedy。GGUF 中的 MTP 张量在普通上下文中被加载，但不会参与主模型生成图。

阶段 2 开始前的设计门槛：必须分别定义 full-attention KV page 与 recurrent state slot 的所有权、分配、释放和序列映射；不能把 hybrid memory 强制转换成普通 `llama_kv_cache`。
