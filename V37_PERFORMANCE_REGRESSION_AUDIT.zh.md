# v3.7 本次修改内容汇总

## 1. 范围

- 对比基线：`556368f Docs: update current runtime for v3.6`。
- 当前工作区相对该提交修改了 23 个已跟踪文件。
- 本文只记录实际代码、接口、构建、测试和文档修改。
- 未包含未跟踪文件：`examples/speculative_sampling_toy.py`、`get_rows`。

## 2. 新增配置与 Python/C++ 接口

修改文件：

- `nanovllm/config.py`
- `nanovllm/backends/native/runner.py`
- `csrc/python/qwen35_runtime_binding.cpp`
- `csrc/runtime/qwen35_runtime.h`

新增配置：

| 配置 | 默认值 | 传入 native runtime 的参数 |
| --- | --- | --- |
| `native_attention_impl` | `"math"` | `attention_impl` |
| `native_batched_recurrent_snapshots` | `False` | `enable_batched_recurrent_snapshots` |
| `native_mtp_prefill_fusion` | `False` | `enable_mtp_prefill_fusion` |

接口行为：

- `native_attention_impl` 允许 `math`、`auto`、`flash`。
- Python binding 将字符串转换为 `Qwen35AttentionImplementation` 枚举。
- 两个布尔值写入 `Qwen35RuntimeOptions`。
- `native_mtp_prefill_fusion=True` 要求 native backend 且 `enable_mtp=True`。

## 3. Attention 图

修改文件：

- `csrc/models/qwen35/graph.h`
- `csrc/models/qwen35/graph.cpp`
- `csrc/runtime/qwen35_runtime.cpp`

新增 `qwen35::AttentionImplementation` 枚举：

```text
Math
Flash
```

### Math 路径

保留原有 GGML 算子序列：

```text
Q x K^T
-> scale + causal mask + softmax
-> probabilities x V
-> query gate
-> output projection
```

### Flash 路径

新增 `ggml_flash_attn_ext()` 分支：

- 在单 token attention 和 target chunk attention 中均可使用。
- 使用 `ggml_flash_attn_ext_set_prec(..., GGML_PREC_F32)`。
- Flash 输出 reshape 为后续 `attn_output` 所需的二维形状。
- Flash 路径创建 `GGML_TYPE_F16` causal mask；Math 路径创建 `GGML_TYPE_F32` causal mask。

### Runtime 选择

`select_attention_implementation()` 的行为：

| 输入配置 | 运行时选择 |
| --- | --- |
| `math` | `Math` |
| `flash` | 检查 backend 和实际 shape；支持时选 `Flash`，否则报错 |
| `auto`，非 MTP | 检查 backend 和实际 shape；支持时选 `Flash`，否则选 `Math` |
| `auto`，MTP | `Math` |

本次没有新增自定义 FlashAttention kernel 或 paged-attention kernel。K/V 仍按原有流程写入 PagedKV，并通过 `GET_ROWS` 取出用于 attention 计算。

## 4. Recurrent delta snapshot 连续写回

修改文件：

- `csrc/runtime/recurrent_state.h`
- `csrc/runtime/recurrent_state.cpp`
- `csrc/runtime/qwen35_runtime.cpp`
- `csrc/models/qwen35/graph.cpp`

新增：

```cpp
RecurrentStateCache::view_delta_snapshots(...)
```

该函数返回相邻 delta snapshot plane 的零拷贝 GGML view，逻辑形状为：

```text
[state_elements, 1, plane_count, 1]
```

`build_chunk_recurrent()` 新增两种 delta state 写回方式：

| 开关 | 写回方式 |
| --- | --- |
| `native_batched_recurrent_snapshots=False` | 原有的逐 snapshot plane `ggml_cpy` |
| `native_batched_recurrent_snapshots=True` | 从 GDN 输出尾部建立 packed view，以一次 `ggml_cpy` 写入连续 snapshot view |

这项修改只处理 delta state snapshot 的 destination 与 copy 节点；conv state 的写回流程未替换。

## 5. MTP prefill graph fusion

修改文件：

- `csrc/models/qwen35/graph.h`
- `csrc/models/qwen35/graph.cpp`
- `csrc/runtime/qwen35_runtime.cpp`

新增 `append_mtp_prefill_kv_update()`。

当调用方传入 MTP cache view 时，target chunk graph 在 target hidden 之后追加：

```text
previous hidden + current target hidden
-> MTP hidden 右移配对
-> MTP merge / norm / KV projection
-> SET_ROWS 写入 MTP K/V cache
```

启用条件：

```text
plan.is_prefill == true
enable_mtp == true
enable_mtp_prefill_fusion == true
```

启用后的 runtime 行为：

- `execute_target_chunk()` 向 target graph 传入 MTP attention cache view。
- target graph 完成 MTP prefill KV 更新。
- runtime 只读回最后一列 hidden，写入 `sequence.pending_hidden`。
- 此路径不使用原有 persistent graph bucket。

开关关闭时保留已有的 target hidden 读回和独立 MTP KV-only graph 路径。MTP verification 的 draft/target 比较和 catch-up KV-only graph 没有改为该 fusion 路径。

## 6. CUDA Graph 构建选项

修改文件：

- `CMakeLists.txt`
- `scripts/build_native_runtime.sh`
- `csrc/native_module.cpp`
- `csrc/runtime/qwen35_runtime.cpp`

新增 CMake 选项：

```text
NANOVLLM_NATIVE_CUDA_GRAPHS=OFF
```

构建行为：

- 只有 `NANOVLLM_NATIVE_CUDA=ON` 时可设置为 `ON`。
- CMake 和构建脚本都会拒绝“CUDA Graph 开启但 CUDA 未开启”的组合。
- 选项传递为 GGML 的 `GGML_CUDA_GRAPHS`。
- `build_info()` 新增 `cuda_graphs` 字段。
- CUDA Graph 构建可用时，runtime 允许对应的 graph capture/replay 路径。

## 7. CLI、benchmark 与 build 信息

修改文件：

- `nanovllm/cli/bench.py`
- `nanovllm/cli/chat.py`
- `csrc/native_module.cpp`

新增 CLI 参数：

```text
--native-attention-impl {math,auto,flash}
--native-batched-recurrent-snapshots
--native-mtp-prefill-fusion
```

新增环境变量：

```text
NANOVLLM_NATIVE_ATTENTION_IMPL
NANOVLLM_NATIVE_BATCHED_RECURRENT_SNAPSHOTS
NANOVLLM_NATIVE_MTP_PREFILL_FUSION
```

`BenchResult` 和 benchmark JSON 新增字段：

```text
native_attention_impl
native_batched_recurrent_snapshots
native_mtp_prefill_fusion
cuda_compiled
cuda_graphs_compiled
```

## 8. 测试与文档

测试修改文件：

- `tests/native/test_graph_executor.cpp`
- `tests/test_native_backend_config.py`
- `tests/test_native_qwen35_cpu_oracle.py`
- `tests/test_native_runner.py`
- `tests/test_native_runtime.py`

测试覆盖新增或扩展：

1. `FLASH_ATTN_EXT` 的 Qwen3.5 Q/K/V layout、GQA 输出和 F16 causal mask。
2. 连续 delta snapshot `ggml_cpy` 的 plane 顺序与数据写回。
3. 三个新增 native 配置的默认值、合法值和约束。
4. Python runner 到 native binding 的参数传递。
5. native Qwen3.5 CPU oracle 的 opt-in MTP 路径。
6. CUDA / CUDA Graph build info 字段读取。

已更新文档：

- `README.md`
- `CURRENT_VERSION_CHANGES.zh.md`
- `CURRENT_RUNTIME_FLOW.zh.md`
- `CURRENT_CODE_DIFF_FROM_ORIGINAL.zh.md`

## 9. 未直接修改的模块

相对 `556368f`，本次没有直接修改：

```text
nanovllm/scheduler.py
nanovllm/block_manager.py
nanovllm/sequence.py
```

因此本次 diff 不包含 scheduler 调度策略、BlockManager 物理 block 分配规则或 Sequence 数据结构字段的直接改动。
