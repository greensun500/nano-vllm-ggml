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

## 阶段 2：Hybrid paged KV

状态：完成。

实现边界：

- nano-vLLM `BlockManager` 继续负责物理 page 的分配，`slot_mapping` 是 attention KV 的唯一写入地址来源。
- llama.cpp 在每次 `llama_decode` 期间绑定只读 paged plan；调用结束立即解绑，不保留悬空请求指针。
- Qwen3.5 hybrid memory 保留官方 recurrent ubatch 切分和 state 准备流程，只替换 full-attention cache 的 slot 准备流程。
- 外部 slot 会检查范围、目标 cell 为空、ubatch token 数一致；冲突直接使 decode 失败。
- ubatch 可能重排多序列 token，因此使用 `(native sequence slot, position)` 查找物理 slot，不依赖 ubatch 的线性偏移。
- 请求完成时删除该序列的全部 attention metadata 和 recurrent state；当前阶段不支持局部回收、prefix cache 或 preemption。

验证结果：

- 软件测试构造出 block table `[3, 1]`，对应物理 slot `768...1023, 256`。
- 本机 CPU、CIX CPU、CIX Vulkan 均在碎片上下文和干净上下文生成相同 greedy token `14556`。
- 将 prefill batch 限制为 256 后，257-token prompt 跨两个非连续 page、两次 `llama_decode`，三种环境仍生成相同 token。
- CIX CPU/Vulkan 增量构建成功，6 项阶段测试通过。

### 阶段 2 反思

Qwen3.5 的 paged KV 不能等价成“把整个 hybrid memory 分页”。Full-attention 层保存每个历史 token 的 K/V，适合 page；linear-attention 层只保存每个序列的递归状态，强行按 token page 会复制大量无意义状态并破坏更新语义。阶段 2 因此只分页 attention KV，recurrent state 继续按 sequence slot 管理。

官方 `llama_kv_cache::slot_info` 已支持非连续 cell，阶段 2 不需要改 attention 计算图或搬运 KV 数据。真正需要处理的是 hybrid ubatch 重排：直接按原始 plan 偏移取 `slot_mapping` 在多序列或 chunked prefill 时可能写错位置，使用 sequence slot 和 position 做确定性查找规避了这个问题。

当前释放粒度仍是完整序列，而不是单独 page。这与“无 prefix cache、无 preemption”的阶段边界一致；序列生命周期内 page 独占，结束时完整清理即可。当前 attention 计算仍可能扫描到最高已用 cell 的 padding 范围，性能不是本阶段验收项。

阶段 3 的关键新增约束是拒绝 draft token 后的回滚：attention page metadata、target KV 内容和 recurrent state 必须回到最后接受位置。不能只回收 attention page 而忽略 recurrent state；应优先复用官方 MTP 上下文与 `n_rs_seq` 回滚快照语义。
