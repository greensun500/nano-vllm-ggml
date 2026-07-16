# nano-vLLM 原生 CPU/Vulkan Runtime 实施记录

## 1. 架构边界

当前实现不是把 llama.cpp 当完整推理器动态加载，而是让 nano-vLLM 自己拥有模型执行和推理状态：

```text
nano-vLLM Python
  Scheduler / BlockManager / Sequence / Sampling
        |
        | BackendExecutionPlan（token、position、block table、slot mapping）
        v
nanovllm._C.Qwen35Runtime
  GGUF 权重映射 / Qwen3.5 graph / Paged KV / recurrent state / MTP
        |
        v
静态编入同一个 Python 扩展的 GGML CPU 或 Vulkan backend/量化算子
```

原生路径不调用或链接以下 llama.cpp 高层接口：

- `llama_model`
- `llama_context`
- `llama_decode`
- `llama_memory_*`
- `common_speculative`

允许复用的是官方 llama.cpp 中的 `ggml/` 源码、GGUF 数据格式、CPU/Vulkan backend、量化类型和底层算子。仓库通过 `third_party/llama.cpp` 固定官方提交 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`；根 CMake 只执行其 `ggml/` 子目录，并把结果静态编入 `nanovllm._C`。旧 `LlamaCppRunner` 只保留为迁移期 oracle，不是 `native_cpu`/`native_vulkan` 的依赖。

## 2. 已实现范围

当前原生 runtime 针对严格的 `Qwen3.5-2B-Q4_0.gguf` 合同：

```text
architecture=qwen35
block_count=25
target_layers=24
bundled_mtp_layers=1
vocabulary_size=248320
required_tensors=335
full_attention_layers=[3,7,11,15,19,23]
recurrent_layers=18
```

实现内容包括：

- 不经过 llama loader 的 GGUF metadata/tensor 解析和持久权重上传。
- 24 层 target graph：6 层 full attention、18 层 Gated Delta Net、IMRoPE 位置 `[p,p,p,0]`。
- nano-vLLM BlockManager 分配逻辑对应的 F32 Paged KV；物理 slot 由 Python `block_table` 和 `slot_mapping` 决定。
- 每个 native sequence slot 独立的卷积状态和 delta state。
- GGUF 内置 `blk.24` MTP：greedy draft、target verify、接收前缀、Paged KV/recurrent snapshot rollback、MTP cache catch-up。
- CPU 和 Vulkan 共用同一套自有模型 graph；差别只在 GGML backend/device。
- 第一版只接受 `temperature=0`，暂不支持 prefix cache 和 preemption。
- `max_model_len` 是单序列逻辑上限，`num_kvcache_blocks` 是所有序列共享的物理容量；增加物理 page 不会改写逻辑上下文长度。

当前实现按 token 构图并执行，目标首先是正确性和完整控制面，不把速度和内存作为本阶段验收门槛。

## 3. 三阶段实施与反思

### 阶段一：Qwen3.5-2B target greedy

完成了严格 GGUF 合同检查、335 个 tensor 加载、24 层混合 target graph 和 greedy argmax。主层数必须解释为 `25 = 24 target + 1 MTP`，不能把 `blk.24` 混入 target trunk。Qwen3.5 也不能按普通 Transformer 实现：18 个 recurrent 层与 6 个 attention 层必须同时存在。

反思：仅能加载模型或运行一个 Q4_0 matmul 不等于支持模型；阶段门槛必须是完整 target graph 的连续 token oracle。Q4_0 不同合法 CPU kernel 在近似并列 logits 上可能选择不同 argmax，因此回归 trace 选用跨普通/repack kernel 都有稳定 margin 的 prompt，同时保留 logits 级诊断能力。

### 阶段二：Paged KV 与 recurrent state

full-attention K/V 被写入 nano-vLLM 分配的物理 page slot；读取上下文通过当前序列的 block table 映射。18 层 GDN 状态按 native sequence slot 持久保存，序列释放时同时清理 KV block 和 recurrent slot。

反思：Python 负责逻辑 page 的分配、复用和释放，C++ runtime 负责实际 backend buffer 及读写算子；两边必须用 `slot_mapping`、`context_lens` 和 `num_cached_tokens` 做双向一致性检查。GGML graph scheduler 只应拥有瞬时 graph buffer，不能暗中接管这些持久状态。

### 阶段三：内置 MTP greedy

每个 decode step 最多生成 `K=mtp_max_draft_tokens` 个 draft，用 target 逐个验证并返回“已接收 draft + 一个 target token”。target recurrent state 保存 `K+1` 个平面；若接收数为 `a`，恢复平面 `K-a`。Paged KV 中未接收的 target/MTP slot 被清理，MTP 再追赶到新的已提交边界。

若距离 `max_model_len` 已不足一个完整的 `K` token 验证窗口，runner 自动退化为一次普通 target greedy，并同步推进 MTP cache；开启投机不会让原本合法的边界 decode 失败。

反思：投机解码的正确性不是“输出看起来合理”，而是对 full accept、partial accept 和 immediate reject 三条路径都验证下一轮 token，并在 release 后复用同一 sequence slot/block 检查不存在旧状态泄漏。MTP 是执行优化，关闭或开启时 target greedy 语义必须相同。

## 4. 已有验证证据

本机 CPU 已验证：

- F32 `mul_mat` 最大误差 `0`；Q4_0 相对 host reference 最大绝对误差小于 `2e-3`。
- GGUF v3、42 项 metadata、335 tensors；权重数据和 resident buffer 均为 `1,225,778,432` bytes。
- target 稳定 oracle：`Hello` 的 token `[9419]` 连续得到 `[11,353,1044]`。
- MTP K=3 的 full/partial/zero-accept、下一轮 rollback、block/sequence release 和 shutdown 均通过真实模型测试。
- 经过完整 `LLM -> Scheduler -> BlockManager -> NativeRunner -> Qwen35Runtime` 路径，MTP 开关均可正常生成；CLI benchmark 也已端到端运行。
- `ldd nanovllm/_C*.so` 不包含 `libllama`、`libggml` 或外部 `libnanollama_backend.so`；`nm -D` 无 `llama_*` 符号。

CIX Vulkan 实机已验证：

- 在 `Mali-G720-Immortalis` 上用 `NANOVLLM_NATIVE_VULKAN=ON` 构建通过；`build_info()` 报告 `vulkan=True`、`uses_llama_context=False`、GGML commit 为 `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`。
- `ctest` 的 `graph_executor_placement` 通过，运行时会对所有非纯 metadata-view 计算节点执行 Vulkan placement 审计。
- Vulkan F32/Q4_0 smoke 误差均为 `0`；335 个 GGUF tensor、`1,225,778,432` bytes 权重 resident buffer 均位于 `Vulkan0`。
- CPU 与 Vulkan 均通过 target 稳定 trace、MTP full/partial/reject、rollback、release/reuse 和 near-boundary fallback。
- 完整 `LLM -> Scheduler -> BlockManager -> NativeRunner` 路径在 CPU/Vulkan、MTP 开/关四种组合下输出一致：`[0,353,1044,264,5286,314]`。
- `native_vulkan` CLI benchmark smoke 通过；示例结果为 `generated_tok_s=2.46`、`decode_tok_s=5.25`、`mtp_drafted_tokens=9`、`mtp_accepted_tokens=1`、`vulkan_compiled=true`。性能不是当前验收标准，仅作为可运行证据。

## 5. 构建和运行

初始化固定的 GGML 源码：

```bash
git submodule update --init --recursive
```

CPU 构建与聊天：

```bash
./scripts/build_native_runtime.sh

PYTHONPATH=. python3 -m nanovllm.cli.chat \
  /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_cpu \
  --tokenizer /path/to/Qwen3.5-2B \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --num-kvcache-blocks 1 \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

CPU + Vulkan 构建使用同一个扩展：

```bash
NANOVLLM_NATIVE_VULKAN=ON ./scripts/build_native_runtime.sh

PYTHONPATH=. python3 -m nanovllm.cli.chat \
  /path/to/Qwen3.5-2B-Q4_0.gguf \
  --backend native_vulkan \
  --tokenizer /path/to/Qwen3.5-2B \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --num-kvcache-blocks 1 \
  --enable-mtp \
  --mtp-max-draft-tokens 3 \
  --temperature 0
```

Vulkan 构建机需要 Vulkan headers、loader 和 `glslc`。`native_vulkan` 不接受 `--library-path`、`--gpu-layers` 或 llama context 参数。

真实 CPU oracle：

```bash
NANOVLLM_TEST_QWEN35_GGUF=/path/to/Qwen3.5-2B-Q4_0.gguf \
NANOVLLM_TEST_QWEN35_TOKENIZER=/path/to/Qwen3.5-2B \
PYTHONPATH=. python3 -m unittest tests.test_native_qwen35_cpu_oracle -v
```

## 6. 当前状态

原生 Qwen3.5-2B + PagedKV + greedy MTP 的 CPU/Vulkan 框架已经跑通。当前仍是 correctness-first 版本：单序列优先，状态使用 F32，性能和内存还没有作为验收目标优化。
