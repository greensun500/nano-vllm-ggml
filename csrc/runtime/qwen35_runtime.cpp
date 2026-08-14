#include "runtime/qwen35_runtime.h"

#include "models/qwen35/graph.h"
#include "models/qwen35/ops.h"
#include "models/qwen35/weights.h"
#include "runtime/backend.h"
#include "runtime/gguf_loader.h"
#include "runtime/graph_executor.h"
#include "runtime/paged_kv.h"
#include "runtime/recurrent_state.h"

#include "ggml-backend.h"
#include "ggml.h"

#ifndef NANOVLLM_NATIVE_HAS_CUDA_GRAPHS
#define NANOVLLM_NATIVE_HAS_CUDA_GRAPHS 0
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanovllm::native {
namespace {

constexpr std::size_t kGraphMetadataBytes = 32U * 1024U * 1024U;
constexpr std::size_t kGraphNodeCapacity = 4096;
// Mali-G720 is substantially faster at the model's projection shapes with 64
// prompt columns than with one 512-column graph.  Keep this backend policy in
// the native runtime so callers do not have to distort scheduler admission.
constexpr std::size_t kVulkanTargetPrefillChunk = 64;

[[noreturn]] void fail(const std::string & detail) {
    throw std::runtime_error("native Qwen3.5 runtime: " + detail);
}

// Keep graph-input allocation failures actionable.  ggml_backend_tensor_set()
// otherwise aborts without identifying which input was omitted from the graph
// allocator's dependency closure.
void require_allocated_input(const ggml_tensor * tensor, const char * label) {
    if (tensor == nullptr) {
        fail(std::string("graph input '") + label + "' is null");
    }
    const ggml_backend_buffer_t buffer =
        tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr || tensor->data == nullptr) {
        fail(std::string("graph input '") + label +
             "' was not allocated; graph construction must retain it as a dependency");
    }
}

std::size_t checked_positive(std::size_t value, const char * label) {
    if (value == 0) {
        fail(std::string(label) + " must be positive");
    }
    return value;
}

std::size_t checked_product(std::size_t left, std::size_t right, const char * label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        fail(std::string(label) + " overflows size_t");
    }
    return left * right;
}

std::uint64_t checked_sum(
    std::initializer_list<std::uint64_t> values,
    const char * label) {
    std::uint64_t result = 0;
    for (const std::uint64_t value : values) {
        if (value > std::numeric_limits<std::uint64_t>::max() - result) {
            fail(std::string(label) + " overflows uint64_t");
        }
        result += value;
    }
    return result;
}

std::uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point started) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
}

Qwen35RuntimeOptions validate_options(Qwen35RuntimeOptions options) {
    if (options.model_path.empty()) {
        fail("model_path cannot be empty");
    }
    (void) parse_backend_kind(options.backend);
    checked_positive(options.max_model_len, "max_model_len");
    checked_positive(options.max_num_batched_tokens, "max_num_batched_tokens");
    checked_positive(options.max_num_seqs, "max_num_seqs");
    checked_positive(options.block_size, "block_size");
    checked_positive(options.num_blocks, "num_blocks");
    if (options.cpu_threads <= 0) {
        fail("cpu_threads must be positive");
    }
    const std::size_t slots =
        checked_product(options.block_size, options.num_blocks, "Paged-KV slot count");
    if (options.max_model_len > slots) {
        fail("max_model_len exceeds block_size * num_blocks");
    }
    if (options.max_num_batched_tokens < options.max_num_seqs) {
        fail("max_num_batched_tokens must be at least max_num_seqs");
    }
    if (options.enable_mtp && options.mtp_max_draft_tokens == 0) {
        fail("MTP requires mtp_max_draft_tokens > 0");
    }
    if (!options.enable_mtp) {
        options.mtp_max_draft_tokens = 0;
    }
    return options;
}

BackendList create_backend_list(const Qwen35RuntimeOptions & options) {
    const BackendKind kind = parse_backend_kind(options.backend);
    if (kind == BackendKind::Cpu) {
        if (options.device_index != 0) {
            fail("CPU runtime only supports device_index=0");
        }
        return BackendList::cpu_only(options.cpu_threads);
    }
    if (kind == BackendKind::Vulkan) {
        return BackendList::vulkan_with_cpu(options.cpu_threads, options.device_index);
    }
    return BackendList::cuda_with_cpu(options.cpu_threads, options.device_index);
}

class GraphContext {
public:
    GraphContext() {
        ggml_init_params params{};
        params.mem_size = kGraphMetadataBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        value_ = ggml_init(params);
        if (value_ == nullptr) {
            fail("could not allocate transient GGML graph metadata");
        }
    }

    ~GraphContext() {
        if (value_ != nullptr) {
            ggml_free(value_);
        }
    }

    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;
    ggml_context * get() const noexcept { return value_; }

    // Rewind the no-alloc metadata arena after the scheduler has been reset.
    // Tensor descriptors from the previous graph must not survive this call;
    // callers rebuild a fresh graph against the same arena on the next
    // execution.  Keeping the arena alive avoids a 32 MiB allocator
    // teardown/recreation for every draft, verification, and catch-up graph.
    void reset() {
        if (value_ == nullptr) {
            fail("cannot reset a null graph metadata context");
        }
        ggml_reset(value_);
    }

private:
    ggml_context * value_ = nullptr;
};

class ExecutorResetGuard {
public:
    explicit ExecutorResetGuard(GraphExecutor & executor) : executor_(&executor) {}
    ~ExecutorResetGuard() {
        if (executor_ != nullptr) {
            try {
                if (compute_completed_) {
                    executor_->reset_after_synchronous_compute();
                } else {
                    executor_->reset();
                }
            } catch (...) {
                // Destructors cannot report a second error. The original
                // execution exception remains the actionable failure.
            }
        }
    }
    ExecutorResetGuard(const ExecutorResetGuard &) = delete;
    ExecutorResetGuard & operator=(const ExecutorResetGuard &) = delete;
    void mark_synchronous_compute_complete() noexcept { compute_completed_ = true; }

private:
    GraphExecutor * executor_;
    bool compute_completed_ = false;
};

struct SequenceState {
    bool live = false;
    std::size_t next_position = 0;
    std::vector<float> pending_hidden;
};

struct PreparedSequence {
    std::size_t row = 0;
    std::size_t token_offset = 0;
    std::size_t token_count = 0;
    std::size_t sequence_slot = 0;
    std::size_t start_position = 0;
    std::vector<std::int32_t> block_table;
};

struct TokenResult {
    std::int32_t token = -1;
    std::vector<float> hidden;
};

struct TargetChunkResult {
    std::vector<std::int32_t> predictions;
    std::vector<float> hidden;
    std::vector<float> mtp_last_hidden;
};

enum class PersistentGraphKind : std::uint8_t {
    TargetChunk,
    MtpDraft,
    MtpKvUpdate,
};

struct PersistentGraphKey {
    PersistentGraphKind kind = PersistentGraphKind::TargetChunk;
    std::size_t n_tokens = 0;
    std::size_t n_kv_bucket = 0;
    std::size_t snapshot_count = 0;
    std::size_t sequence_slot = 0;
    std::size_t input_plane = 0;
    std::uint8_t output_mode = 0;
    std::uint8_t attention_implementation = 0;
    bool retain_hidden = false;
    bool emit_greedy = false;
    bool mtp_verification_kv_fusion = false;

    bool operator==(const PersistentGraphKey & other) const noexcept {
        return kind == other.kind &&
               n_tokens == other.n_tokens &&
               n_kv_bucket == other.n_kv_bucket &&
               snapshot_count == other.snapshot_count &&
               sequence_slot == other.sequence_slot &&
               input_plane == other.input_plane &&
               output_mode == other.output_mode &&
               attention_implementation == other.attention_implementation &&
               retain_hidden == other.retain_hidden &&
               emit_greedy == other.emit_greedy &&
               mtp_verification_kv_fusion == other.mtp_verification_kv_fusion;
    }
};

struct PersistentGraphEntry {
    explicit PersistentGraphEntry(PersistentGraphKey supplied_key)
        : key(supplied_key) {}

    PersistentGraphKey key;
    GraphContext context;
    std::unique_ptr<GraphExecutor> executor;
    qwen35::TargetChunkGraph target_chunk;
    qwen35::TokenGraph token;
    qwen35::MtpKvUpdateGraph mtp_kv_update;
    std::uint64_t last_used = 0;
};

struct FlashAttentionProbeResult {
    std::size_t n_tokens = 0;
    std::size_t n_kv = 0;
    bool use_causal_mask = false;
    bool supported = false;
};

std::vector<std::int32_t> slice(
    const std::vector<std::int32_t> & source,
    std::size_t offset,
    std::size_t count) {
    if (offset > source.size() || count > source.size() - offset) {
        fail("internal plan slice is outside its source vector");
    }
    return std::vector<std::int32_t>(
        source.begin() + static_cast<std::ptrdiff_t>(offset),
        source.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

}  // namespace

struct Qwen35Runtime::Impl {
    explicit Impl(Qwen35RuntimeOptions supplied_options)
        : options(validate_options(std::move(supplied_options))),
          backends(create_backend_list(options)) {
        primary_backend = &backends.at(0);//调度优先级由位置决定
        model_storage = std::make_unique<GgufWeights>(options.model_path);//1. 初始化Ggufweight加载器
        model_storage->load(primary_backend->get());//2. 将GGUF权重加载到后端的持久缓冲区中
        model = std::make_unique<qwen35::Qwen35Weights>(*model_storage);    //3.已经上传到ggml缓存区的权重绑定到qwen35::Qwen35Weights中，方便后续使用

        const qwen35::Config & config = model->config();    //4. 获取模型配置参数,主要是模型结构参数，包括attention head数，embedding长度，context长度等以及具体的层和结构
        if (options.max_model_len > config.context_length) {
            fail("max_model_len exceeds the model context length");
        }
        if (options.enable_mtp && config.nextn_predict_layers != 1) {
            fail("MTP requested, but the GGUF does not contain exactly one bundled MTP layer");
        }

        paged_kv = std::make_unique<PagedKvCache>(  //5. 初始化PagedKvCache，主要是用于存储KV缓存
            primary_backend->get(),
            config,
            options.block_size,
            options.num_blocks,
            options.enable_mtp);
        RecurrentStateOptions state_options;    //6. 开始初始化recurrent state cache的参数
        state_options.max_sequence_slots = options.max_num_seqs;    //此处是总共能同时并发多少个请求
        state_options.max_draft_tokens = options.mtp_max_draft_tokens;  //每个序列最多允许的草稿token数
        state_options.include_mtp_recurrent = false;    //mtp层不需要额外存储
        recurrent = std::make_unique<RecurrentStateCache>(  //7. 初始化RecurrentStateCache，主要是用于存储模型的recurrent state
            config, primary_backend->get(), state_options);
        executor = std::make_unique<GraphExecutor>( //8. 初始化GraphExecutor，主要是用于执行图计算
            backends, kGraphNodeCapacity, false, true);

        sequences.resize(options.max_num_seqs);//
        for (SequenceState & sequence : sequences) {
            sequence.pending_hidden.assign(config.embedding_length, 0.0f);  //初始化每个序列的pending_hidden为0，长度为embedding_length
        }
    }

    Qwen35RuntimeOptions options;
    BackendList backends;
    Backend * primary_backend = nullptr;
    std::unique_ptr<GgufWeights> model_storage;
    std::unique_ptr<qwen35::Qwen35Weights> model;
    std::unique_ptr<PagedKvCache> paged_kv;
    std::unique_ptr<RecurrentStateCache> recurrent;
    std::unique_ptr<GraphExecutor> executor;
    GraphContext graph_context;
    std::vector<SequenceState> sequences;
    std::vector<std::unique_ptr<PersistentGraphEntry>> graph_cache;
    std::vector<FlashAttentionProbeResult> flash_attention_probe_cache;
    Qwen35GraphReuseStats graph_stats;
    Qwen35MtpProfileStats mtp_profile;
    std::uint64_t graph_use_clock = 0;

    bool graph_reuse_available() const noexcept {
        if (!options.enable_graph_reuse ||
            options.max_num_seqs != 1 ||
            primary_backend == nullptr) {
            return false;
        }
        if (primary_backend->kind() == BackendKind::Cpu ||
            (primary_backend->kind() == BackendKind::Vulkan &&
             options.enable_vulkan_graph_reuse)) {
            return true;
        }
        // GGML captures CUDA work only after the graph's tensor metadata and
        // addresses are stable. Persistent entries provide that stability;
        // all other backends retain their existing eager graph lifetime.
        return primary_backend->kind() == BackendKind::Cuda &&
               NANOVLLM_NATIVE_HAS_CUDA_GRAPHS != 0;
    }

    bool mtp_prefill_fusion_available() const noexcept {
        return options.enable_mtp && options.enable_mtp_prefill_fusion &&
               primary_backend != nullptr;
    }

    bool mtp_verification_kv_fusion_available() const noexcept {
        return options.enable_mtp && options.enable_mtp_verification_kv_fusion &&
               paged_kv != nullptr && paged_kv->has_mtp_layer();
    }

    bool flash_attention_supported(
        std::size_t n_tokens,
        std::size_t n_kv,
        bool use_causal_mask) {
        if (primary_backend == nullptr) {
            return false;
        }
        if (n_tokens == 0 || n_kv == 0) {
            return false;
        }
        for (const FlashAttentionProbeResult & cached : flash_attention_probe_cache) {
            if (cached.n_tokens == n_tokens && cached.n_kv == n_kv &&
                cached.use_causal_mask == use_causal_mask) {
                return cached.supported;
            }
        }

        // Probe the exact [D, token, head] / [D, KV, KV-head] layout that
        // the graph builder passes to FLASH_ATTN_EXT. This avoids guessing
        // from backend kind alone: a driver may reject a shape, precision, or
        // device feature even though it exposes the operation in general.
        ggml_init_params params{};
        params.mem_size = 32U * 1024U;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * probe = ggml_init(params);
        if (probe == nullptr) {
            return false;
        }
        const qwen35::Config & config = model->config();
        const std::int64_t key_width =
            static_cast<std::int64_t>(config.attention_key_length);
        const std::int64_t value_width =
            static_cast<std::int64_t>(config.attention_value_length);
        const std::int64_t query_heads =
            static_cast<std::int64_t>(config.attention_head_count);
        const std::int64_t kv_heads =
            static_cast<std::int64_t>(config.attention_head_count_kv);
        const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
        const std::int64_t kv_count = static_cast<std::int64_t>(n_kv);

        ggml_tensor * query = ggml_new_tensor_3d(
            probe, GGML_TYPE_F32, key_width, query_heads, token_count);
        ggml_tensor * key = ggml_new_tensor_3d(
            probe, GGML_TYPE_F32, key_width, kv_heads, kv_count);
        ggml_tensor * value = ggml_new_tensor_3d(
            probe, GGML_TYPE_F32, value_width, kv_heads, kv_count);
        query = ggml_permute(probe, query, 0, 2, 1, 3);
        key = ggml_permute(probe, key, 0, 2, 1, 3);
        value = ggml_permute(probe, value, 0, 2, 1, 3);
        ggml_tensor * mask = use_causal_mask
            ? ggml_new_tensor_2d(probe, GGML_TYPE_F16, kv_count, token_count)
            : nullptr;
        ggml_tensor * attention = ggml_flash_attn_ext(
            probe,
            query,
            key,
            value,
            mask,
            1.0f / std::sqrt(static_cast<float>(key_width)),
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
        const bool supported = ggml_backend_supports_op(
            primary_backend->get(), attention);
        ggml_free(probe);
        constexpr std::size_t kMaxFlashAttentionProbeEntries = 16;
        if (flash_attention_probe_cache.size() == kMaxFlashAttentionProbeEntries) {
            flash_attention_probe_cache.erase(flash_attention_probe_cache.begin());
        }
        flash_attention_probe_cache.push_back({
            n_tokens,
            n_kv,
            use_causal_mask,
            supported,
        });
        return supported;
    }

    bool paged_attention_supported(std::size_t n_tokens, std::size_t n_kv) {
        if (primary_backend == nullptr ||
            primary_backend->kind() != BackendKind::Vulkan ||
            n_tokens == 0 || n_kv < n_tokens) {
            return false;
        }
        ggml_init_params params{};
        params.mem_size = 16U * 1024U;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * probe = ggml_init(params);
        if (probe == nullptr) {
            return false;
        }
        const qwen35::Config & config = model->config();
        const std::int64_t head_dim = config.attention_key_length;
        const std::int64_t query_heads = config.attention_head_count;
        const std::int64_t kv_heads = config.attention_head_count_kv;
        const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
        const std::int64_t kv_count = static_cast<std::int64_t>(n_kv);
        ggml_tensor * query = ggml_new_tensor_3d(
            probe, GGML_TYPE_F32, head_dim, query_heads, token_count);
        ggml_tensor * key_cache = ggml_new_tensor_2d(
            probe, GGML_TYPE_F32, head_dim * kv_heads, kv_count);
        ggml_tensor * value_cache = ggml_new_tensor_2d(
            probe, GGML_TYPE_F32, head_dim * kv_heads, kv_count);
        ggml_tensor * slots = ggml_new_tensor_1d(probe, GGML_TYPE_I32, kv_count);
        ggml_tensor * context_len = ggml_new_tensor_1d(probe, GGML_TYPE_I32, 1);
        ggml_tensor * attention = ggml_paged_attn(
            probe,
            query,
            key_cache,
            value_cache,
            slots,
            context_len,
            1.0f / std::sqrt(static_cast<float>(head_dim)));
        const bool supported = ggml_backend_supports_op(
            primary_backend->get(), attention);
        ggml_free(probe);
        return supported;
    }

    qwen35::AttentionImplementation select_attention_implementation(
        std::size_t n_tokens,
        std::size_t n_kv,
        bool use_causal_mask) {
        switch (options.attention_implementation) {
            case Qwen35AttentionImplementation::Math:
                return qwen35::AttentionImplementation::Math;
            case Qwen35AttentionImplementation::Flash:
                if (!flash_attention_supported(n_tokens, n_kv, use_causal_mask)) {
                    fail(
                        "attention_impl='flash' is unsupported for the requested backend "
                        "or attention shape");
                }
                return qwen35::AttentionImplementation::Flash;
            case Qwen35AttentionImplementation::Paged:
                // The direct kernel is currently validated for decode only.
                // Keep multi-token prefill and MTP verification on the exact
                // math path until the tiled T>1 kernel has a cross-chunk
                // oracle.  This is intentional fallback, not a capability
                // probe failure: explicit paged still accelerates the
                // latency-critical T=1 decode steps.
                if (n_tokens != 1) {
                    return qwen35::AttentionImplementation::Math;
                }
                if (!paged_attention_supported(n_tokens, n_kv)) {
                    fail(
                        "attention_impl='paged' is supported only by the Vulkan "
                        "direct-cache kernel for the requested Qwen3.5 attention shape");
                }
                return qwen35::AttentionImplementation::Paged;
            case Qwen35AttentionImplementation::Auto:
                // MTP must use the same attention implementation for both
                // T=1 draft and T=K+1 verification.  The Qwen3.5-2B Mali
                // oracle covers their exact greedy trace, so accept the
                // backend's Flash capability for both shapes instead of
                // leaving the dominant verification path on math attention.
                if (options.enable_mtp) {
                    if (primary_backend != nullptr &&
                        primary_backend->kind() == BackendKind::Vulkan &&
                        flash_attention_supported(n_tokens, n_kv, use_causal_mask)) {
                        return qwen35::AttentionImplementation::Flash;
                    }
                    return qwen35::AttentionImplementation::Math;
                }
                if (primary_backend == nullptr ||
                    primary_backend->kind() == BackendKind::Cpu) {
                    return qwen35::AttentionImplementation::Math;
                }
                return flash_attention_supported(n_tokens, n_kv, use_causal_mask)
                    ? qwen35::AttentionImplementation::Flash
                    : qwen35::AttentionImplementation::Math;
        }
        return qwen35::AttentionImplementation::Math;
    }

    std::size_t kv_bucket_for(std::size_t actual_n_kv) const {
        if (actual_n_kv == 0) {
            fail("cannot choose a graph bucket for an empty KV context");
        }
        constexpr std::array<std::size_t, 6> fixed_buckets{
            128, 256, 512, 1024, 2048, 4096};
        for (const std::size_t bucket : fixed_buckets) {
            if (actual_n_kv <= bucket) {
                return std::min(bucket, options.max_model_len);
            }
        }
        return options.max_model_len;
    }

    std::vector<std::int32_t> padded_read_slots(
        const std::vector<std::int32_t> & read_slots,
        std::size_t n_kv_bucket) const {
        if (read_slots.empty()) {
            fail("cannot pad an empty attention context");
        }
        if (read_slots.size() > n_kv_bucket) {
            fail("actual attention context exceeds its graph bucket");
        }
        std::vector<std::int32_t> padded = read_slots;
        padded.resize(n_kv_bucket, read_slots.back());
        return padded;
    }

    std::vector<float> make_single_token_mask(
        std::size_t actual_n_kv,
        std::size_t n_kv_bucket) const {
        if (actual_n_kv == 0 || actual_n_kv > n_kv_bucket) {
            fail("single-token mask shape is outside its bucket");
        }
        std::vector<float> mask(n_kv_bucket, 0.0f);
        std::fill(
            mask.begin() + static_cast<std::ptrdiff_t>(actual_n_kv),
            mask.end(),
            -std::numeric_limits<float>::infinity());
        return mask;
    }

    std::vector<float> make_chunk_mask(
        std::size_t actual_n_kv,
        std::size_t n_kv_bucket,
        std::size_t n_tokens) const {
        if (n_tokens == 0 || actual_n_kv < n_tokens || actual_n_kv > n_kv_bucket) {
            fail("chunk mask shape is outside its bucket");
        }
        const std::size_t prefix_length = actual_n_kv - n_tokens;
        std::vector<float> mask(
            checked_product(n_kv_bucket, n_tokens, "persistent graph causal mask"),
            -std::numeric_limits<float>::infinity());
        for (std::size_t token_index = 0; token_index < n_tokens; ++token_index) {
            const std::size_t visible = prefix_length + token_index;
            float * row = mask.data() + token_index * n_kv_bucket;
            std::fill(row, row + visible + 1, 0.0f);
        }
        return mask;
    }

    void upload_causal_mask(
        ggml_tensor * mask_tensor,
        const std::vector<float> & values) const {
        if (mask_tensor == nullptr) {
            fail("cannot upload a null causal mask tensor");
        }
        if (ggml_nelements(mask_tensor) != static_cast<std::int64_t>(values.size())) {
            fail("causal mask payload does not match its graph input shape");
        }
        if (mask_tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(
                mask_tensor, values.data(), 0, values.size() * sizeof(values.front()));
            return;
        }
        if (mask_tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> converted(values.size());
            ggml_fp32_to_fp16_row(
                values.data(), converted.data(), static_cast<std::int64_t>(values.size()));
            ggml_backend_tensor_set(
                mask_tensor,
                converted.data(),
                0,
                converted.size() * sizeof(converted.front()));
            return;
        }
        fail("causal mask tensor must be F32 or F16");
    }

    PersistentGraphEntry * find_graph_entry(const PersistentGraphKey & key) {
        for (const std::unique_ptr<PersistentGraphEntry> & entry : graph_cache) {
            if (entry->key == key) {
                return entry.get();
            }
        }
        return nullptr;
    }

    void evict_smaller_buckets(const PersistentGraphKey & key) {
        graph_cache.erase(
            std::remove_if(
                graph_cache.begin(),
                graph_cache.end(),
                [&](const std::unique_ptr<PersistentGraphEntry> & entry) {
                    if (entry->key.kind != key.kind ||
                        entry->key.sequence_slot != key.sequence_slot ||
                        entry->key.n_tokens != key.n_tokens ||
                        entry->key.snapshot_count != key.snapshot_count ||
                        entry->key.output_mode != key.output_mode ||
                        entry->key.attention_implementation != key.attention_implementation ||
                        entry->key.retain_hidden != key.retain_hidden ||
                        entry->key.emit_greedy != key.emit_greedy ||
                        entry->key.n_kv_bucket >= key.n_kv_bucket) {
                        return false;
                    }
                    ++graph_stats.evictions;
                    return true;
                }),
            graph_cache.end());
    }

    void evict_lru_if_needed() {
        constexpr std::size_t kMaxPersistentGraphEntries = 12;
        while (graph_cache.size() > kMaxPersistentGraphEntries) {
            auto victim = std::min_element(
                graph_cache.begin(),
                graph_cache.end(),
                [](const std::unique_ptr<PersistentGraphEntry> & left,
                   const std::unique_ptr<PersistentGraphEntry> & right) {
                    return left->last_used < right->last_used;
                });
            if (victim == graph_cache.end()) {
                return;
            }
            graph_cache.erase(victim);
            ++graph_stats.evictions;
        }
    }

    std::vector<PreparedSequence> validate_plan(
        const Qwen35ExecutionPlan & plan,
        bool require_decode,
        std::size_t future_tokens) const {
        if (plan.n_tokens <= 0 || plan.n_seqs <= 0) {
            fail("execution plan must contain at least one token and sequence");
        }
        const std::size_t n_tokens = static_cast<std::size_t>(plan.n_tokens);
        const std::size_t n_seqs = static_cast<std::size_t>(plan.n_seqs);
        const bool allow_vulkan_prefill_over_batch =
            plan.is_prefill && primary_backend != nullptr &&
            primary_backend->kind() == BackendKind::Vulkan;
        if (n_tokens > options.max_num_batched_tokens &&
            !allow_vulkan_prefill_over_batch) {
            fail("execution plan exceeds max_num_batched_tokens");
        }
        if (n_seqs > options.max_num_seqs) {
            fail("execution plan exceeds max_num_seqs");
        }
        if (plan.block_size <= 0 ||
            static_cast<std::size_t>(plan.block_size) != options.block_size) {
            fail("execution-plan block_size differs from the runtime");
        }
        if (plan.block_table_cols <= 0) {
            fail("execution plan requires a non-empty block table");
        }
        const std::size_t columns = static_cast<std::size_t>(plan.block_table_cols);
        const std::size_t table_values = checked_product(n_seqs, columns, "block table");
        const auto require_size = [](const std::vector<std::int32_t> & values,
                                     std::size_t expected,
                                     const char * label) {
            if (values.size() != expected) {
                fail(std::string(label) + " has " + std::to_string(values.size()) +
                     " entries, expected " + std::to_string(expected));
            }
        };
        require_size(plan.tokens, n_tokens, "tokens");
        require_size(plan.positions, n_tokens, "positions");
        require_size(plan.slot_mapping, n_tokens, "slot_mapping");
        require_size(plan.seq_ids, n_seqs, "seq_ids");
        require_size(plan.scheduled_token_counts, n_seqs, "scheduled_token_counts");
        require_size(plan.block_tables, table_values, "block_tables");
        require_size(plan.context_lens, n_seqs, "context_lens");
        require_size(plan.num_cached_tokens, n_seqs, "num_cached_tokens");
        if (require_decode && plan.is_prefill) {
            fail("MTP execution requires a decode plan");
        }

        std::unordered_set<std::int32_t> seen_sequence_ids;
        std::size_t token_offset = 0;
        std::vector<PreparedSequence> prepared;
        prepared.reserve(n_seqs);
        for (std::size_t row = 0; row < n_seqs; ++row) {
            const std::int32_t raw_slot = plan.seq_ids[row];
            if (raw_slot < 0 || static_cast<std::size_t>(raw_slot) >= sequences.size()) {
                fail("sequence slot is outside the configured range");
            }
            if (!seen_sequence_ids.insert(raw_slot).second) {
                fail("execution plan repeats a sequence slot");
            }
            const std::int32_t raw_count = plan.scheduled_token_counts[row];
            if (raw_count <= 0) {
                fail("scheduled_token_counts entries must be positive");
            }
            const std::size_t count = static_cast<std::size_t>(raw_count);
            if (require_decode && count != 1) {
                fail("MTP decode requires one pending token per sequence");
            }
            if (token_offset > n_tokens || count > n_tokens - token_offset) {
                fail("scheduled_token_counts exceed n_tokens");
            }

            const SequenceState & sequence = sequences[static_cast<std::size_t>(raw_slot)];
            const std::size_t expected_start = sequence.live ? sequence.next_position : 0;
            if (plan.num_cached_tokens[row] < 0 ||
                static_cast<std::size_t>(plan.num_cached_tokens[row]) != expected_start) {
                fail("num_cached_tokens differs from native sequence state for row " +
                     std::to_string(row));
            }
            if (!sequence.live && expected_start != 0) {
                fail("internal new-sequence position is not zero");
            }
            if (expected_start > options.max_model_len ||
                count + future_tokens > options.max_model_len - expected_start) {
                fail("scheduled token range exceeds max_model_len");
            }
            for (std::size_t index = 0; index < count; ++index) {
                const std::int32_t position = plan.positions[token_offset + index];
                if (position < 0 ||
                    static_cast<std::size_t>(position) != expected_start + index) {
                    fail("token positions are not contiguous with native sequence state");
                }
                const std::int32_t token = plan.tokens[token_offset + index];
                if (token < 0 ||
                    static_cast<std::uint64_t>(token) >= model->config().vocabulary_size) {
                    fail("token ID is outside the Qwen3.5 vocabulary");
                }
            }
            if (plan.context_lens[row] < 0 ||
                static_cast<std::size_t>(plan.context_lens[row]) != expected_start + count) {
                fail("context_lens must equal num_cached_tokens + scheduled_token_count");
            }

            PreparedSequence item;
            item.row = row;
            item.token_offset = token_offset;
            item.token_count = count;
            item.sequence_slot = static_cast<std::size_t>(raw_slot);
            item.start_position = expected_start;
            item.block_table = slice(plan.block_tables, row * columns, columns);

            const std::vector<std::int32_t> expected_slots = paged_kv->physical_indices(
                item.block_table, expected_start, count);
            const std::vector<std::int32_t> supplied_slots =
                slice(plan.slot_mapping, token_offset, count);
            if (expected_slots != supplied_slots) {
                fail("slot_mapping differs from block_tables for sequence row " +
                     std::to_string(row));
            }
            // Also validates all blocks needed by the speculative tail.
            paged_kv->validate_logical_range(
                item.block_table, 0, expected_start + count + future_tokens);
            prepared.push_back(std::move(item));
            token_offset += count;
        }
        if (token_offset != n_tokens) {
            fail("scheduled_token_counts do not sum to n_tokens");
        }
        paged_kv->validate_write_indices(plan.slot_mapping);
        return prepared;
    }

    void initialize_new_sequences(const std::vector<PreparedSequence> & prepared) {
        for (const PreparedSequence & item : prepared) {
            SequenceState & sequence = sequences[item.sequence_slot];
            if (sequence.live) {
                continue;
            }
            recurrent->initialize_slot(item.sequence_slot);
            std::fill(sequence.pending_hidden.begin(), sequence.pending_hidden.end(), 0.0f);
            sequence.next_position = 0;
            sequence.live = true;
        }
    }

    qwen35::TargetPersistentView make_target_persistent(
        ggml_context * ctx,
        std::size_t sequence_slot,
        std::size_t input_plane,
        std::size_t output_plane) const {
        qwen35::TargetPersistentView persistent;
        persistent.recurrent.reserve(recurrent->recurrent_layer_count());
        for (const std::uint32_t layer : recurrent->recurrent_layers()) {
            qwen35::RecurrentStateView state;
            state.convolution = recurrent->view_conv(
                ctx, layer, sequence_slot, input_plane);
            state.delta = recurrent->view_delta(ctx, layer, sequence_slot, input_plane);
            state.convolution_output = recurrent->view_conv(
                ctx, layer, sequence_slot, output_plane);
            state.delta_output = recurrent->view_delta(
                ctx, layer, sequence_slot, output_plane);
            persistent.recurrent.push_back(state);
        }
        persistent.attention.reserve(paged_kv->target_layers().size());
        for (const PagedKvLayer & layer : paged_kv->target_layers()) {
            persistent.attention.push_back({layer.key, layer.value});
        }
        return persistent;
    }

    qwen35::TargetChunkPersistentView make_target_chunk_persistent(
        ggml_context * ctx,
        std::size_t sequence_slot,
        std::size_t input_plane,
        std::size_t snapshot_count) const {
        if (snapshot_count == 0 || snapshot_count > recurrent->snapshot_planes()) {
            fail("target chunk snapshot count is outside the recurrent cache");
        }
        qwen35::TargetChunkPersistentView persistent;
        persistent.recurrent.reserve(recurrent->recurrent_layer_count());
        for (const std::uint32_t layer : recurrent->recurrent_layers()) {   //按照有recurrent的层来一层一层找state
            qwen35::RecurrentChunkStateView state;
            state.convolution = recurrent->view_conv(
                ctx, layer, sequence_slot, input_plane);    //找到了ssm-conv的位置
            state.delta = recurrent->view_delta(ctx, layer, sequence_slot, input_plane);    //找到了ssm-delta的位置
            state.convolution_outputs.reserve(snapshot_count);
            if (options.enable_batched_recurrent_snapshots) {
                state.delta_snapshots_output = recurrent->view_delta_snapshots(
                    ctx, layer, sequence_slot, 0, snapshot_count);
            } else {
                state.delta_outputs.reserve(snapshot_count);    //按照实际需要写回的palane数量来预留
            }
            for (std::size_t plane = 0; plane < snapshot_count; ++plane) {
                state.convolution_outputs.push_back(recurrent->view_conv(
                    ctx, layer, sequence_slot, plane)); //怎么存储新state的位置
                if (!options.enable_batched_recurrent_snapshots) {
                    state.delta_outputs.push_back(recurrent->view_delta(
                        ctx, layer, sequence_slot, plane));
                }
            }
            persistent.recurrent.push_back(std::move(state));
        }
        persistent.attention.reserve(paged_kv->target_layers().size());
        for (const PagedKvLayer & layer : paged_kv->target_layers()) {
            persistent.attention.push_back({layer.key, layer.value});   //kvcache是保存在每一个layer里面的，所以直接找到对应layer就能找到kvcache位置了
        }
        return persistent;
    }

    void place_primary_compute_nodes(GraphExecutor & graph_executor, ggml_cgraph * graph) {
        if (primary_backend->kind() == BackendKind::Cpu) {
            return;
        }
        graph_executor.set_all_compute_nodes_backend(graph, primary_backend->kind());
    }

    void assert_compute_placement(
        const GraphExecutor & graph_executor,
        ggml_cgraph * graph) const {
        graph_executor.assert_all_compute_nodes_on_backend(
            graph, primary_backend->kind());
    }

    PersistentGraphEntry & touch_or_create_entry(const PersistentGraphKey & key) {
        ++graph_use_clock;
        if (PersistentGraphEntry * existing = find_graph_entry(key)) {
            ++graph_stats.hits;
            existing->last_used = graph_use_clock;
            return *existing;
        }
        ++graph_stats.misses;
        evict_smaller_buckets(key);
        auto entry = std::make_unique<PersistentGraphEntry>(key);
        entry->executor = std::make_unique<GraphExecutor>(
            backends, kGraphNodeCapacity, false, true);
        entry->last_used = graph_use_clock;
        PersistentGraphEntry & result = *entry;
        graph_cache.push_back(std::move(entry));
        evict_lru_if_needed();
        return result;
    }

    PersistentGraphEntry & target_chunk_entry(
        std::size_t sequence_slot,
        std::size_t input_plane,
        std::size_t snapshot_count,
        std::size_t n_tokens,
        std::size_t n_kv_bucket,
        qwen35::TargetChunkOutputMode output_mode,
        bool retain_hidden,
        qwen35::AttentionImplementation attention_implementation,
        bool fuse_mtp_verification_kv) {
        PersistentGraphKey key;
        key.kind = PersistentGraphKind::TargetChunk;
        key.n_tokens = n_tokens;
        key.n_kv_bucket = n_kv_bucket;
        key.snapshot_count = snapshot_count;
        key.sequence_slot = sequence_slot;
        key.input_plane = input_plane;
        key.output_mode = static_cast<std::uint8_t>(output_mode);
        key.attention_implementation =
            static_cast<std::uint8_t>(attention_implementation);
        key.retain_hidden = retain_hidden;
        key.mtp_verification_kv_fusion = fuse_mtp_verification_kv;

        PersistentGraphEntry & entry = touch_or_create_entry(key);
        if (entry.target_chunk.graph == nullptr) {
            qwen35::TargetChunkPersistentView persistent = make_target_chunk_persistent(
                entry.context.get(), sequence_slot, input_plane, snapshot_count);
            qwen35::AttentionCacheView mtp_verification_cache;
            const qwen35::AttentionCacheView * mtp_verification_cache_view = nullptr;
            if (fuse_mtp_verification_kv) {
                const PagedKvLayer & layer = paged_kv->mtp_layer();
                mtp_verification_cache = {layer.key, layer.value};
                mtp_verification_cache_view = &mtp_verification_cache;
            }
            entry.target_chunk = qwen35::build_target_chunk_graph(
                entry.context.get(),
                *model,
                persistent,
                n_tokens,
                n_kv_bucket,
                snapshot_count,
                output_mode,
                retain_hidden,
                true,
                attention_implementation,
                nullptr,
                mtp_verification_cache_view);
            place_primary_compute_nodes(*entry.executor, entry.target_chunk.graph);
            entry.executor->allocate(entry.target_chunk.graph);
            assert_compute_placement(*entry.executor, entry.target_chunk.graph);
        }
        return entry;
    }

    PersistentGraphEntry & mtp_draft_entry(
        std::size_t n_kv_bucket,
        bool emit_greedy,
        bool retain_hidden,
        qwen35::AttentionImplementation attention_implementation) {
        PersistentGraphKey key;
        key.kind = PersistentGraphKind::MtpDraft;
        key.n_tokens = 1;
        key.n_kv_bucket = n_kv_bucket;
        key.attention_implementation =
            static_cast<std::uint8_t>(attention_implementation);
        key.emit_greedy = emit_greedy;
        key.retain_hidden = retain_hidden;

        PersistentGraphEntry & entry = touch_or_create_entry(key);
        if (entry.token.graph == nullptr) {
            const PagedKvLayer & layer = paged_kv->mtp_layer();
            qwen35::AttentionCacheView cache{layer.key, layer.value};
            entry.token = qwen35::build_mtp_token_graph(
                entry.context.get(),
                *model,
                cache,
                n_kv_bucket,
                emit_greedy,
                retain_hidden,
                true,
                attention_implementation);
            place_primary_compute_nodes(*entry.executor, entry.token.graph);
            entry.executor->allocate(entry.token.graph);
            assert_compute_placement(*entry.executor, entry.token.graph);
        }
        return entry;
    }

    PersistentGraphEntry & mtp_kv_update_entry(std::size_t n_tokens) {
        PersistentGraphKey key;
        key.kind = PersistentGraphKind::MtpKvUpdate;
        key.n_tokens = n_tokens;

        PersistentGraphEntry & entry = touch_or_create_entry(key);
        if (entry.mtp_kv_update.graph == nullptr) {
            const PagedKvLayer & layer = paged_kv->mtp_layer();
            qwen35::AttentionCacheView cache{layer.key, layer.value};
            entry.mtp_kv_update = qwen35::build_mtp_kv_update_graph(
                entry.context.get(), *model, cache, n_tokens);
            place_primary_compute_nodes(*entry.executor, entry.mtp_kv_update.graph);
            entry.executor->allocate(entry.mtp_kv_update.graph);
            assert_compute_placement(*entry.executor, entry.mtp_kv_update.graph);
        }
        return entry;
    }

    void upload_common_inputs(
        const qwen35::TokenGraph & token_graph,
        std::int32_t token,
        std::size_t position,
        std::int32_t write_slot,
        const std::vector<std::int32_t> & read_slots,
        std::size_t actual_context_len) const {
        if (position > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            fail("position exceeds the I32 IMRoPE ABI");
        }
        if (actual_context_len == 0 || actual_context_len > read_slots.size() ||
            actual_context_len > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            fail("actual paged-attention context length is outside its slot input range");
        }
        const std::int32_t position_i32 = static_cast<std::int32_t>(position);
        const std::int32_t context_len_i32 = static_cast<std::int32_t>(actual_context_len);
        const std::array<std::int32_t, 4> positions{
            position_i32, position_i32, position_i32, 0};
        require_allocated_input(token_graph.token, "token");
        require_allocated_input(token_graph.positions, "positions");
        require_allocated_input(token_graph.write_slot, "write_slot");
        require_allocated_input(token_graph.read_slots, "read_slots");
        if (token_graph.context_len != nullptr) {
            require_allocated_input(token_graph.context_len, "context_len");
        }
        ggml_backend_tensor_set(token_graph.token, &token, 0, sizeof(token));
        ggml_backend_tensor_set(
            token_graph.positions, positions.data(), 0, sizeof(positions));
        ggml_backend_tensor_set(
            token_graph.write_slot, &write_slot, 0, sizeof(write_slot));
        ggml_backend_tensor_set(
            token_graph.read_slots,
            read_slots.data(),
            0,
            read_slots.size() * sizeof(std::int32_t));
        if (token_graph.context_len != nullptr) {
            ggml_backend_tensor_set(
                token_graph.context_len, &context_len_i32, 0, sizeof(context_len_i32));
        }
    }

    TokenResult execute_target(
        std::size_t sequence_slot,
        std::int32_t token,
        std::size_t position,
        std::int32_t write_slot,
        const std::vector<std::int32_t> & read_slots,
        std::size_t input_plane,
        std::size_t output_plane,
        bool emit_greedy,
        bool read_hidden) {
        paged_kv->validate_read_indices(read_slots);
        paged_kv->validate_write_indices({write_slot});
        if (read_slots.empty() || read_slots.back() != write_slot) {
            fail("target attention context must end at the current write slot");
        }
        graph_context.reset();
        ExecutorResetGuard reset_guard(*executor);
        qwen35::TargetPersistentView persistent = make_target_persistent(
            graph_context.get(), sequence_slot, input_plane, output_plane);
        const qwen35::AttentionImplementation attention_implementation =
            select_attention_implementation(1, read_slots.size(), false);
        qwen35::TokenGraph token_graph = qwen35::build_target_token_graph(
            graph_context.get(),
            *model,
            persistent,
            read_slots.size(),
            emit_greedy,
            read_hidden,
            false,
            attention_implementation);
        place_primary_compute_nodes(*executor, token_graph.graph);
        executor->allocate(token_graph.graph);
        assert_compute_placement(*executor, token_graph.graph);
        upload_common_inputs(
            token_graph, token, position, write_slot, read_slots, read_slots.size());
        executor->compute(token_graph.graph);
        reset_guard.mark_synchronous_compute_complete();

        TokenResult result;
        if (read_hidden) {
            result.hidden.resize(model->config().embedding_length);
            ggml_backend_tensor_get(
                token_graph.hidden,
                result.hidden.data(),
                0,
                result.hidden.size() * sizeof(float));
        }
        if (emit_greedy) {
            ggml_backend_tensor_get(
                token_graph.greedy_token, &result.token, 0, sizeof(result.token));
        }
        return result;
    }

    TargetChunkResult execute_target_chunk(
        std::size_t sequence_slot,      // 当前 chunk 所属的序列
        const std::vector<std::int32_t> & tokens,   //当前进行推理的token
        std::size_t start_position,
        const std::vector<std::int32_t> & write_slots,
        const std::vector<std::int32_t> & read_slots,
        std::size_t input_plane,
        std::size_t snapshot_count,
        qwen35::TargetChunkOutputMode output_mode,
        bool retain_hidden,
        bool allow_graph_reuse,
        bool profile_mtp_verification,
        const std::vector<float> * mtp_prefill_previous_hidden = nullptr,
        bool fuse_mtp_verification_kv = false) {
        // 这是 native target model 执行连续 token chunk 的主路径。
        // 上层调度器已经决定了本轮要计算哪些 token，以及这些 token 的 K/V
        // 应该写到哪些物理 slot；这里负责把这些执行计划转成 GGML graph 的输入，
        // 执行 graph，并按需把预测结果或 hidden 读回到 host。
        if (tokens.empty()) {
            fail("target chunk requires at least one token");
        }
        const bool fuse_mtp_prefill = mtp_prefill_previous_hidden != nullptr;
        if (fuse_mtp_prefill && fuse_mtp_verification_kv) {
            fail("target chunk cannot fuse MTP prefill and verification KV updates together");
        }
        if (fuse_mtp_prefill) {
            if (!options.enable_mtp || !paged_kv->has_mtp_layer()) {
                fail("MTP prefill fusion was requested on an MTP-disabled runtime");
            }
            if (mtp_prefill_previous_hidden->size() !=
                model->config().embedding_length) {
                fail("MTP prefill fusion previous hidden has the wrong width");
            }
        }
        if (fuse_mtp_verification_kv && tokens.size() <= 1) {
            fail("MTP verification KV fusion requires at least two target tokens");
        }
        // 当前 chunk 里的每个 token 都必须对应一个 PagedKV 物理写入位置。
        // graph 会为每个 token 产生一行 K 和一行 V。
        if (write_slots.size() != tokens.size()) {
            fail("target chunk token and write-slot counts differ");
        }
        // 当前 chunk 覆盖的绝对 position 范围是：
        //   [start_position, start_position + tokens.size()).
        // 这个范围不能超过模型配置的最大上下文长度。
        if (start_position > options.max_model_len ||
            tokens.size() > options.max_model_len - start_position) {
            fail("target chunk exceeds max_model_len");
        }
        // read_slots 表示当前 chunk 最后一个 token 能看到的完整 attention 上下文。
        // 对单 sequence 且无空洞的情况，它的逻辑长度应该等于：
        //   历史 prefix 长度 + 当前 chunk 长度。
        if (read_slots.size() != start_position + tokens.size()) {
            fail("target chunk attention context has the wrong length");
        }
        // snapshot_count 控制 graph 要物化多少份 recurrent state 输出 plane。
        // 普通 target 执行只需要 1 份；MTP verification 可能需要多个候选 state。
        if (snapshot_count == 0 || snapshot_count > tokens.size() ||
            snapshot_count > recurrent->snapshot_planes()) {
            fail("target chunk snapshot count is outside the supported range");
        }
        paged_kv->validate_read_indices(read_slots);
        paged_kv->validate_write_indices(write_slots);
        // graph 会先把当前 token 的 K/V 写入 PagedKV，然后再 gather 完整上下文做 attention。
        // 因此当前 chunk 的 write_slots 必须是 read_slots 的后缀，例如：
        //   [历史 slots..., 当前 write slots...]
        if (!std::equal(
                write_slots.begin(),
                write_slots.end(),
                read_slots.end() - static_cast<std::ptrdiff_t>(write_slots.size()))) {
            fail("target chunk attention context must end with its write slots");
        }

        // 构造当前 chunk 的绝对 token position。是整个序列的位置，包括prefill+decode，记做T
        // Qwen3.5 graph 消费的是 IMRoPE 展开后的 position，所以 [T] 会变成 [4 * T]。
        std::vector<std::int32_t> positions;
        positions.reserve(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            const std::size_t position = start_position + index;    //第n个token的绝对位置
            if (position > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
                fail("position exceeds the I32 IMRoPE ABI");
            }
            positions.push_back(static_cast<std::int32_t>(position));
        }
        const std::vector<std::int32_t> expanded_positions =    //为了满足IMRoPE的要求，需要将position展开为4倍长度，其实前三维度是一样的，第四维度是0（空间尺度，文本模型设置为0）
            qwen35::ops::expand_text_positions(positions);
        std::vector<std::int32_t> mtp_verification_tokens;
        std::vector<std::int32_t> mtp_verification_positions;
        std::vector<std::int32_t> mtp_verification_write_slots;
        if (fuse_mtp_verification_kv) {
            mtp_verification_tokens.assign(tokens.begin() + 1, tokens.end());
            mtp_verification_positions.assign(positions.begin() + 1, positions.end());
            mtp_verification_positions =
                qwen35::ops::expand_text_positions(mtp_verification_positions);
            mtp_verification_write_slots.assign(write_slots.begin() + 1, write_slots.end());
        }

        // 快路径：对于形状稳定的场景复用 persistent graph，主要服务 decode。
        // 缓存 graph 使用 KV bucket 固定形状，因此 read_slots 可能会被 padding，
        // 再通过 causal mask 屏蔽 padding 出来的无效位置或未来位置。
        if (allow_graph_reuse && !fuse_mtp_prefill && graph_reuse_available())   //使用graph reuse
        {
            const std::size_t n_kv_bucket = kv_bucket_for(read_slots.size());
            const qwen35::AttentionImplementation attention_implementation =
                select_attention_implementation(tokens.size(), n_kv_bucket, true);
            PersistentGraphEntry & entry = target_chunk_entry(
                sequence_slot,
                input_plane,
                snapshot_count,
                tokens.size(),
                n_kv_bucket,
                output_mode,
                retain_hidden,
                attention_implementation,
                fuse_mtp_verification_kv);
            qwen35::TargetChunkGraph & graph = entry.target_chunk;
            const std::vector<std::int32_t> padded_slots =
                padded_read_slots(read_slots, n_kv_bucket);
            // 把本次执行的动态输入上传到已经分配好的 GGML input tensor。
            // graph 结构和临时 buffer 都会复用，只有 token/position/slot/mask 的值变化。
            require_allocated_input(graph.tokens, "target_chunk.tokens");
            require_allocated_input(graph.positions, "target_chunk.positions");
            require_allocated_input(graph.write_slots, "target_chunk.write_slots");
            require_allocated_input(graph.read_slots, "target_chunk.read_slots");
            if (graph.context_len != nullptr) {
                require_allocated_input(graph.context_len, "target_chunk.context_len");
            }
            if (fuse_mtp_verification_kv) {
                require_allocated_input(
                    graph.mtp_verification_tokens,
                    "target_chunk.mtp_verification_tokens");
                require_allocated_input(
                    graph.mtp_verification_positions,
                    "target_chunk.mtp_verification_positions");
                require_allocated_input(
                    graph.mtp_verification_write_slots,
                    "target_chunk.mtp_verification_write_slots");
                ggml_backend_tensor_set(
                    graph.mtp_verification_tokens,
                    mtp_verification_tokens.data(),
                    0,
                    mtp_verification_tokens.size() * sizeof(mtp_verification_tokens.front()));
                ggml_backend_tensor_set(
                    graph.mtp_verification_positions,
                    mtp_verification_positions.data(),
                    0,
                    mtp_verification_positions.size() * sizeof(mtp_verification_positions.front()));
                ggml_backend_tensor_set(
                    graph.mtp_verification_write_slots,
                    mtp_verification_write_slots.data(),
                    0,
                    mtp_verification_write_slots.size() * sizeof(mtp_verification_write_slots.front()));
            }
            const std::int32_t context_len_i32 = static_cast<std::int32_t>(read_slots.size());
            ggml_backend_tensor_set(
                graph.tokens, tokens.data(), 0, tokens.size() * sizeof(tokens.front()));
            ggml_backend_tensor_set(
                graph.positions,
                expanded_positions.data(),
                0,
                expanded_positions.size() * sizeof(expanded_positions.front()));
            ggml_backend_tensor_set(
                graph.write_slots,
                write_slots.data(),
                0,
                write_slots.size() * sizeof(write_slots.front()));
            ggml_backend_tensor_set(
                graph.read_slots,
                padded_slots.data(),
                0,
                padded_slots.size() * sizeof(padded_slots.front()));
            if (graph.context_len != nullptr) {
                ggml_backend_tensor_set(
                    graph.context_len, &context_len_i32, 0, sizeof(context_len_i32));
            }
            if (attention_implementation != qwen35::AttentionImplementation::Paged) {
                if (graph.causal_mask == nullptr) {
                    fail("persistent target chunk graph omitted its causal mask");
                }
                // mask 的宽度是 n_kv_bucket。
                // 它允许真实 prefix 和当前 token 可见的 causal 范围，
                // 同时屏蔽 bucket padding 产生的无效位置和 chunk 内的未来位置。
                const std::vector<float> mask =
                    make_chunk_mask(read_slots.size(), n_kv_bucket, tokens.size());
                upload_causal_mask(graph.causal_mask, mask);
            }
            // 执行已经构建好的 graph。
            // graph 内部 full-attention 层会用 write_slots 写 PagedKV，
            // 再用 read_slots 读取完整上下文；recurrent 层会从 input_plane
            // 读取旧 state，并写出新的 snapshot state。
            entry.executor->compute(graph.graph);

            TargetChunkResult result;
            std::size_t prediction_count = 0;
            // output_mode 控制 graph 需要物化多少个 greedy token：
            // None：不输出 LM head 结果；Last：只输出最后一个预测；All：输出 T 个预测。
            if (output_mode == qwen35::TargetChunkOutputMode::All) {
                prediction_count = tokens.size();
            } else if (output_mode == qwen35::TargetChunkOutputMode::Last) {
                prediction_count = 1;
            }
            if (prediction_count > 0) {
                if (graph.greedy_tokens == nullptr) {
                    fail("persistent target chunk graph omitted requested greedy output");
                }
                // 把 argmax 结果从 backend memory 拷回 host 侧 std::vector。
                result.predictions.resize(prediction_count, -1);
                ggml_backend_tensor_get(
                    graph.greedy_tokens,
                    result.predictions.data(),
                    0,
                    result.predictions.size() * sizeof(result.predictions.front()));
            }
            if (retain_hidden) {
                result.hidden.resize(checked_product(
                    tokens.size(),
                    model->config().embedding_length,
                    "target chunk hidden output"));
                // hidden 主要给 MTP 维护流程使用。
                // 普通非 MTP target 执行通常不需要把 hidden 读回 host。
                ggml_backend_tensor_get(
                    graph.hidden,
                    result.hidden.data(),
                    0,
                    result.hidden.size() * sizeof(result.hidden.front()));
            }
            return result;
        }

        // 慢路径 / 通用路径：按当前 chunk 的精确形状临时构建 graph。
        // prefill chunk、禁用 graph reuse 或无法复用时通常走这里。
        graph_context.reset();  // 释放上次 graph 的临时 buffer，避免占用过多内存
        ExecutorResetGuard reset_guard(*executor);  //使用RAII对象，确保作用域结束后可以释放buffer
        // 创建指向持久化缓存的 tensor view。这些 view 指向：
        //   - 所有 full-attention 层的 PagedKV tensor。
        //   - 当前 sequence_slot 和 input_plane 对应的 recurrent conv/delta state。
        //   - 用于接收新 recurrent state 的输出 plane。
        const auto graph_setup_started = std::chrono::steady_clock::now();
        qwen35::TargetChunkPersistentView persistent = make_target_chunk_persistent(
            graph_context.get(), sequence_slot, input_plane, snapshot_count);
        const qwen35::AttentionImplementation attention_implementation =
            select_attention_implementation(
                tokens.size(), read_slots.size(), tokens.size() > 1);
        qwen35::AttentionCacheView mtp_prefill_cache;
        const qwen35::AttentionCacheView * mtp_prefill_cache_view = nullptr;
        qwen35::AttentionCacheView mtp_verification_cache;
        const qwen35::AttentionCacheView * mtp_verification_cache_view = nullptr;
        if (fuse_mtp_prefill) {
            const PagedKvLayer & mtp_layer = paged_kv->mtp_layer();
            mtp_prefill_cache = {mtp_layer.key, mtp_layer.value};
            mtp_prefill_cache_view = &mtp_prefill_cache;
        }
        if (fuse_mtp_verification_kv) {
            const PagedKvLayer & mtp_layer = paged_kv->mtp_layer();
            mtp_verification_cache = {mtp_layer.key, mtp_layer.value};
            mtp_verification_cache_view = &mtp_verification_cache;
        }
        // 手写构建 T 个 token 的 Qwen3.5 forward graph：
        // embedding -> 24 decoder layers -> optional output_norm/lm_head.
        qwen35::TargetChunkGraph graph = qwen35::build_target_chunk_graph(
            graph_context.get(),
            *model,
            persistent,
            tokens.size(),
            read_slots.size(),
            snapshot_count,
            output_mode,
            retain_hidden,
            false,
            attention_implementation,
            mtp_prefill_cache_view,
            mtp_verification_cache_view);

        //计算图建立完毕，开始进行实际执行
        // 让 GGML scheduler 为 graph 做节点放置和临时 tensor 分配。
        // 权重、PagedKV、recurrent state 属于持久化存储，前面初始化时已经有 backend storage。
        place_primary_compute_nodes(*executor, graph.graph);
        executor->allocate(graph.graph);    //为graph分配临时buffer
        assert_compute_placement(*executor, graph.graph);
        if (profile_mtp_verification) {
            mtp_profile.verification_graph_setup_elapsed_ns +=
                elapsed_nanoseconds(graph_setup_started);
        }
        // 上传由 Python/native 调度层准备好的运行时输入。
        require_allocated_input(graph.tokens, "target_chunk.tokens");
        require_allocated_input(graph.positions, "target_chunk.positions");
        require_allocated_input(graph.write_slots, "target_chunk.write_slots");
        require_allocated_input(graph.read_slots, "target_chunk.read_slots");
        if (graph.context_len != nullptr) {
            require_allocated_input(graph.context_len, "target_chunk.context_len");
        }
        if (fuse_mtp_verification_kv) {
            require_allocated_input(
                graph.mtp_verification_tokens,
                "target_chunk.mtp_verification_tokens");
            require_allocated_input(
                graph.mtp_verification_positions,
                "target_chunk.mtp_verification_positions");
            require_allocated_input(
                graph.mtp_verification_write_slots,
                "target_chunk.mtp_verification_write_slots");
            ggml_backend_tensor_set(
                graph.mtp_verification_tokens,
                mtp_verification_tokens.data(),
                0,
                mtp_verification_tokens.size() * sizeof(mtp_verification_tokens.front()));
            ggml_backend_tensor_set(
                graph.mtp_verification_positions,
                mtp_verification_positions.data(),
                0,
                mtp_verification_positions.size() * sizeof(mtp_verification_positions.front()));
            ggml_backend_tensor_set(
                graph.mtp_verification_write_slots,
                mtp_verification_write_slots.data(),
                0,
                mtp_verification_write_slots.size() * sizeof(mtp_verification_write_slots.front()));
        }
        const std::int32_t context_len_i32 = static_cast<std::int32_t>(read_slots.size());
        ggml_backend_tensor_set(
            graph.tokens, tokens.data(), 0, tokens.size() * sizeof(tokens.front()));
        ggml_backend_tensor_set(
            graph.positions,
            expanded_positions.data(),
            0,
            expanded_positions.size() * sizeof(expanded_positions.front()));
        ggml_backend_tensor_set(
            graph.write_slots,
            write_slots.data(),
            0,
            write_slots.size() * sizeof(write_slots.front()));
        ggml_backend_tensor_set(
            graph.read_slots,
            read_slots.data(),
            0,
            read_slots.size() * sizeof(read_slots.front()));
        if (graph.context_len != nullptr) {
            ggml_backend_tensor_set(
                graph.context_len, &context_len_i32, 0, sizeof(context_len_i32));
        }
        if (graph.causal_mask != nullptr) {
            // 多 token chunk 需要 causal mask，因为 read_slots 包含的是直到
            // chunk 最后一个 token 为止的完整上下文。
            // 第 i 行 query 可以看到 prefix + 当前 chunk 的 [0, i]，
            // 但不能看到同一个 chunk 中更靠后的 token。
            const std::size_t context_length = read_slots.size();
            const std::size_t prefix_length = context_length - tokens.size();
            std::vector<float> mask(checked_product(
                context_length, tokens.size(), "target chunk causal mask"));
            const float masked = -std::numeric_limits<float>::infinity();
            for (std::size_t token_index = 0;
                 token_index < tokens.size();
                 ++token_index) {
                const std::size_t visible = prefix_length + token_index;
                float * row = mask.data() + token_index * context_length;
                std::fill(row, row + context_length, masked);
                std::fill(row, row + visible + 1, 0.0f);
            }
            upload_causal_mask(graph.causal_mask, mask);
        }
        // 执行 graph。执行过程中会产生两个重要副作用：
        //   - full-attention 层通过 ggml_set_rows 把 K/V 写入 PagedKV。
        //   - recurrent 层通过 ggml_cpy 写回 conv/delta state snapshot。
        executor->compute(graph.graph);
        reset_guard.mark_synchronous_compute_complete();

        //执行采样
        TargetChunkResult result;
        std::size_t prediction_count = 0;
        // 根据 output_mode 决定需要读回多少个预测 token。
        if (output_mode == qwen35::TargetChunkOutputMode::All) {
            prediction_count = tokens.size();
        } else if (output_mode == qwen35::TargetChunkOutputMode::Last) {
            prediction_count = 1;
        }
        if (prediction_count > 0) {
            if (graph.greedy_tokens == nullptr) {
                fail("target chunk omitted requested greedy output");
            }
            // 把 greedy argmax token id 从 backend memory 拷回 std::vector。
            result.predictions.resize(prediction_count, -1);
            ggml_backend_tensor_get(
                graph.greedy_tokens,
                result.predictions.data(),
                0,
                result.predictions.size() * sizeof(result.predictions.front()));
        }
        if (retain_hidden) {
            result.hidden.resize(checked_product(
                tokens.size(),
                model->config().embedding_length,
                "target chunk hidden output"));
            // hidden 的形状是 [embedding_length, tokens.size()]。
            // 只有调用方需要时才读回，主要用于 MTP 后续 graph。
            ggml_backend_tensor_get(
                graph.hidden,
                result.hidden.data(),
                0,
                result.hidden.size() * sizeof(result.hidden.front()));
        }
        if (fuse_mtp_prefill) {
            if (graph.mtp_last_hidden == nullptr) {
                fail("target chunk omitted fused MTP last hidden output");
            }
            result.mtp_last_hidden.resize(model->config().embedding_length);
            ggml_backend_tensor_get(
                graph.mtp_last_hidden,
                result.mtp_last_hidden.data(),
                0,
                result.mtp_last_hidden.size() * sizeof(result.mtp_last_hidden.front()));
        }
        return result;
    }

    TokenResult execute_mtp(
        std::int32_t token,
        std::size_t position,
        std::int32_t write_slot,
        const std::vector<std::int32_t> & read_slots,
        const std::vector<float> & hidden_input,
        bool emit_greedy,
        bool read_hidden,
        bool allow_graph_reuse) {
        if (!options.enable_mtp || !paged_kv->has_mtp_layer()) {
            fail("MTP execution was requested on an MTP-disabled runtime");
        }
        if (hidden_input.size() != model->config().embedding_length) {
            fail("MTP hidden input has the wrong width");
        }
        paged_kv->validate_read_indices(read_slots);
        paged_kv->validate_write_indices({write_slot});
        if (read_slots.empty() || read_slots.back() != write_slot) {
            fail("MTP attention context must end at the current write slot");
        }
        if (allow_graph_reuse && graph_reuse_available()) {
            const std::size_t n_kv_bucket = kv_bucket_for(read_slots.size());
            const qwen35::AttentionImplementation attention_implementation =
                select_attention_implementation(1, n_kv_bucket, true);
            PersistentGraphEntry & entry = mtp_draft_entry(
                n_kv_bucket, emit_greedy, read_hidden, attention_implementation);
            qwen35::TokenGraph & token_graph = entry.token;
            const std::vector<std::int32_t> padded_slots =
                padded_read_slots(read_slots, n_kv_bucket);
            upload_common_inputs(
                token_graph, token, position, write_slot, padded_slots, read_slots.size());
            if (attention_implementation != qwen35::AttentionImplementation::Paged) {
                if (token_graph.causal_mask == nullptr) {
                    fail("persistent MTP draft graph omitted its causal mask");
                }
                const std::vector<float> mask =
                    make_single_token_mask(read_slots.size(), n_kv_bucket);
                upload_causal_mask(token_graph.causal_mask, mask);
            }
            ggml_backend_tensor_set(
                token_graph.hidden_input,
                hidden_input.data(),
                0,
                hidden_input.size() * sizeof(float));
            entry.executor->compute(token_graph.graph);

            TokenResult result;
            if (read_hidden) {
                result.hidden.resize(model->config().embedding_length);
                ggml_backend_tensor_get(
                    token_graph.hidden,
                    result.hidden.data(),
                    0,
                    result.hidden.size() * sizeof(float));
            }
            if (emit_greedy) {
                ggml_backend_tensor_get(
                    token_graph.greedy_token, &result.token, 0, sizeof(result.token));
            }
            return result;
        }
        graph_context.reset();
        ExecutorResetGuard reset_guard(*executor);
        const auto graph_setup_started = std::chrono::steady_clock::now();
        const PagedKvLayer & layer = paged_kv->mtp_layer();
        qwen35::AttentionCacheView cache{layer.key, layer.value};
        const qwen35::AttentionImplementation attention_implementation =
            select_attention_implementation(1, read_slots.size(), false);
        qwen35::TokenGraph token_graph = qwen35::build_mtp_token_graph(
            graph_context.get(),
            *model,
            cache,
            read_slots.size(),
            emit_greedy,
            read_hidden,
            false,
            attention_implementation);
        place_primary_compute_nodes(*executor, token_graph.graph);
        executor->allocate(token_graph.graph);
        assert_compute_placement(*executor, token_graph.graph);
        mtp_profile.draft_graph_setup_elapsed_ns += elapsed_nanoseconds(graph_setup_started);
        upload_common_inputs(
            token_graph, token, position, write_slot, read_slots, read_slots.size());
        ggml_backend_tensor_set(
            token_graph.hidden_input,
            hidden_input.data(),
            0,
            hidden_input.size() * sizeof(float));
        executor->compute(token_graph.graph);
        reset_guard.mark_synchronous_compute_complete();

        TokenResult result;
        if (read_hidden) {
            result.hidden.resize(model->config().embedding_length);
            ggml_backend_tensor_get(
                token_graph.hidden,
                result.hidden.data(),
                0,
                result.hidden.size() * sizeof(float));
        }
        if (emit_greedy) {
            ggml_backend_tensor_get(
                token_graph.greedy_token, &result.token, 0, sizeof(result.token));
        }
        return result;
    }

    void execute_mtp_kv_update(
        const std::vector<std::int32_t> & tokens,
        const std::vector<std::size_t> & positions,
        const std::vector<std::int32_t> & write_slots,
        const std::vector<float> & hidden_inputs,
        bool allow_graph_reuse) {
        if (!options.enable_mtp || !paged_kv->has_mtp_layer()) {
            fail("MTP KV update was requested on an MTP-disabled runtime");
        }
        if (tokens.empty()) {
            fail("MTP KV update requires at least one token");
        }
        if (positions.size() != tokens.size() || write_slots.size() != tokens.size()) {
            fail("MTP KV update input row counts differ");
        }
        const std::size_t hidden_elements = checked_product(
            tokens.size(), model->config().embedding_length, "MTP KV hidden input");
        if (hidden_inputs.size() != hidden_elements) {
            fail("MTP KV update hidden input has the wrong shape");
        }
        paged_kv->validate_write_indices(write_slots);

        std::vector<std::int32_t> positions_i32;
        positions_i32.reserve(positions.size());
        for (const std::size_t position : positions) {
            if (position > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
                fail("position exceeds the I32 IMRoPE ABI");
            }
            positions_i32.push_back(static_cast<std::int32_t>(position));
        }
        const std::vector<std::int32_t> expanded_positions =
            qwen35::ops::expand_text_positions(positions_i32);

        if (allow_graph_reuse && graph_reuse_available()) {
            PersistentGraphEntry & entry = mtp_kv_update_entry(tokens.size());
            qwen35::MtpKvUpdateGraph & graph = entry.mtp_kv_update;
            ggml_backend_tensor_set(
                graph.tokens, tokens.data(), 0, tokens.size() * sizeof(tokens.front()));
            ggml_backend_tensor_set(
                graph.positions,
                expanded_positions.data(),
                0,
                expanded_positions.size() * sizeof(expanded_positions.front()));
            ggml_backend_tensor_set(
                graph.write_slots,
                write_slots.data(),
                0,
                write_slots.size() * sizeof(write_slots.front()));
            ggml_backend_tensor_set(
                graph.hidden_input,
                hidden_inputs.data(),
                0,
                hidden_inputs.size() * sizeof(hidden_inputs.front()));
            entry.executor->compute(graph.graph);
            return;
        }

        graph_context.reset();
        ExecutorResetGuard reset_guard(*executor);
        const auto graph_setup_started = std::chrono::steady_clock::now();
        const PagedKvLayer & layer = paged_kv->mtp_layer();
        qwen35::AttentionCacheView cache{layer.key, layer.value};
        qwen35::MtpKvUpdateGraph graph = qwen35::build_mtp_kv_update_graph(
            graph_context.get(), *model, cache, tokens.size());
        place_primary_compute_nodes(*executor, graph.graph);
        executor->allocate(graph.graph);
        assert_compute_placement(*executor, graph.graph);
        mtp_profile.kv_update_graph_setup_elapsed_ns += elapsed_nanoseconds(graph_setup_started);
        ggml_backend_tensor_set(
            graph.tokens, tokens.data(), 0, tokens.size() * sizeof(tokens.front()));
        ggml_backend_tensor_set(
            graph.positions,
            expanded_positions.data(),
            0,
            expanded_positions.size() * sizeof(expanded_positions.front()));
        ggml_backend_tensor_set(
            graph.write_slots,
            write_slots.data(),
            0,
            write_slots.size() * sizeof(write_slots.front()));
        ggml_backend_tensor_set(
            graph.hidden_input,
            hidden_inputs.data(),
            0,
            hidden_inputs.size() * sizeof(hidden_inputs.front()));
        executor->compute(graph.graph);
        reset_guard.mark_synchronous_compute_complete();
    }

    std::vector<std::int32_t> run(const Qwen35ExecutionPlan & plan) {
        const std::vector<PreparedSequence> prepared = validate_plan(plan, false, 0);//校验输入的plan是否合法，确认没问题并把python的参数转化成PreparedSequence的结构体
        initialize_new_sequences(prepared);
        std::vector<std::int32_t> outputs;
        outputs.reserve(prepared.size());
        for (const PreparedSequence & item : prepared) {    //遍历每一个seqs，如果只是单请求就只做一遍
            SequenceState & sequence = sequences[item.sequence_slot];
            const std::vector<std::int32_t> target_tokens = slice(
                plan.tokens, item.token_offset, item.token_count);//取出当前计算需要的token（可以是多个）

// 从整轮 plan.slot_mapping 里，取出当前 sequence 本轮输入 token 对应的写入位置。
// write_slots 的长度 = item.token_count。
// 每个元素都是一个 PagedKV 的 physical_slot，表示对应 token 的 K/V 要写到哪里。
            const std::vector<std::int32_t> write_slots = slice(
                plan.slot_mapping, item.token_offset, item.token_count);//取出这些token实际要写入那些物理slot
            const BackendDeviceInfo & device = primary_backend->device_info();
// 从整轮 plan.slot_mapping 里，取出当前 sequence 本轮输入 token 对应的写入位置。
// write_slots 的长度 = item.token_count。
// 每个元素都是一个 PagedKV 的 physical_slot，表示对应 token 的 K/V 要写到哪里。            
            const bool is_mali_vulkan =
                primary_backend->kind() == BackendKind::Vulkan &&
                (device.name.find("Mali") != std::string::npos ||
                 device.description.find("Mali") != std::string::npos);
// 决定当前 sequence 的 prefill/decode 是否需要切 chunk。
// 如果是 Mali Vulkan，则每个 target chunk 最多跑 kVulkanTargetPrefillChunk 个 token，当前代码里是 64。
// 如果不是 Mali Vulkan，则整个 item.token_count 一次性跑完。                 
            const std::size_t chunk_limit = is_mali_vulkan &&
                    !(options.enable_mtp && plan.is_prefill)
                ? kVulkanTargetPrefillChunk
                : item.token_count;
            const bool fuse_mtp_prefill =
                plan.is_prefill && mtp_prefill_fusion_available();

// 保存当前 sequence 最后一个 chunk 产生的 greedy prediction。
// 初始化为 -1，后面只有 final chunk 会真正写入预测 token。
            std::int32_t final_prediction = -1;

// 遍历当前 sequence 本轮要执行的 token。
// 如果不切 chunk，循环只执行一次。
// 如果是 Mali Vulkan 且 token_count 很大，会按 chunk_limit 分多次执行。
            for (std::size_t offset = 0; offset < item.token_count;) {

                    // 当前 chunk 实际要处理多少 token。
                    // 不能超过 chunk_limit，也不能超过剩余 token 数。
                const std::size_t chunk_count = std::min(
                    chunk_limit, item.token_count - offset);

    // 当前 chunk 的起始逻辑 position。
    // item.start_position 是这个 sequence 本轮开始前已经缓存到的位置。
    // offset 是当前 chunk 在本轮 token 中的偏移。
                const std::size_t chunk_position = item.start_position + offset;

        // 判断当前 chunk 是否是这个 sequence 本轮的最后一个 chunk。
    // 只有最后一个 chunk 需要输出 greedy token。
    // 前面的 chunk 只负责更新 KV cache / recurrent state。
                const bool is_final = offset + chunk_count == item.token_count;
        // 从当前 sequence 的 target_tokens 中切出当前 chunk 的输入 token。
    // chunk_tokens 长度 = chunk_count。
                const std::vector<std::int32_t> chunk_tokens = slice(
                    target_tokens, offset, chunk_count);

        // 从当前 sequence 的 write_slots 中切出当前 chunk 的写入 physical slots。
    // chunk_write_slots[i] 对应 chunk_tokens[i] 的 K/V 写入位置。
                const std::vector<std::int32_t> chunk_write_slots = slice(
                    write_slots, offset, chunk_count);

        // 生成当前 chunk 做 attention 时需要读取的全部 physical slots。
    // chunk_position + chunk_count 表示当前 chunk 执行完成后的上下文长度。
    // context 包含从 position 0 到当前 chunk 末尾的所有 token 对应的 physical_slot。
    //
    // 例如当前 chunk 是 position [64, 127]，
    // 那么 context 会包含 position [0, 127] 对应的所有 physical slots。
                const std::vector<std::int32_t> context =
                    paged_kv->context_indices(
                        item.block_table, chunk_position + chunk_count);
        // 获取当前 sequence 的 recurrent state 输入 snapshot plane。
    // 普通 run 通常是 plane 0。
    // 如果之前发生过 MTP rollback，则 active_snapshot_plane 可能指向被选中的状态版本。
    //
    // 后续 execute_target_chunk 会从这个 plane 读取 conv_state / delta_state。
                const std::size_t input_plane =
                    recurrent->active_snapshot_plane(item.sequence_slot);
                TargetChunkResult target = execute_target_chunk(            //普通prefill或者decode进入graph的如来
                    item.sequence_slot,
                    chunk_tokens,   //输入token
                    chunk_position, //输入token的起始位置
                    chunk_write_slots,  //输入token对应的slot
                    context,    //attention要读取的slots
                    input_plane,    //输入token对应的snapshot plane
                    1,
                    is_final ? qwen35::TargetChunkOutputMode::Last
                             : qwen35::TargetChunkOutputMode::None,
                    options.enable_mtp && !fuse_mtp_prefill,
                    !plan.is_prefill,
                    false,
                    fuse_mtp_prefill ? &sequence.pending_hidden : nullptr);
                recurrent->select_latest(item.sequence_slot);
                if (is_final) {
                    if (target.predictions.size() != 1 ||
                        target.predictions.front() < 0) {
                        fail("final target chunk did not produce one greedy token");
                    }
                    final_prediction = target.predictions.front();
                } else if (!target.predictions.empty()) {
                    fail("non-final target chunk unexpectedly produced a greedy token");
                }

                if (options.enable_mtp) {
                    const std::size_t embedding = model->config().embedding_length;
                    if (fuse_mtp_prefill) {
                        if (target.mtp_last_hidden.size() != embedding) {
                            fail("target chunk returned the wrong fused MTP hidden shape");
                        }
                        sequence.pending_hidden = std::move(target.mtp_last_hidden);
                        offset += chunk_count;
                        continue;
                    }
                    const std::size_t hidden_elements = checked_product(
                        chunk_count, embedding, "MTP prefill hidden input");
                    if (target.hidden.size() != hidden_elements) {
                        fail("target chunk returned the wrong hidden-output shape");
                    }
                    std::vector<std::size_t> mtp_positions(chunk_count);
                    std::iota(
                        mtp_positions.begin(),
                        mtp_positions.end(),
                        chunk_position);
                    std::vector<float> mtp_hidden_inputs;
                    mtp_hidden_inputs.reserve(hidden_elements);
                    mtp_hidden_inputs.insert(
                        mtp_hidden_inputs.end(),    //上一次的hidden输入
                        sequence.pending_hidden.begin(),
                        sequence.pending_hidden.end());
                    if (chunk_count > 1) {
                        mtp_hidden_inputs.insert(
                            mtp_hidden_inputs.end(),
                            target.hidden.begin(),
                            target.hidden.end() - static_cast<std::ptrdiff_t>(embedding));
                    }
                    execute_mtp_kv_update(
                        chunk_tokens,
                        mtp_positions,
                        chunk_write_slots,
                        mtp_hidden_inputs,
                        false);
                    sequence.pending_hidden.assign(
                        target.hidden.end() - static_cast<std::ptrdiff_t>(embedding),
                        target.hidden.end());
                }
                offset += chunk_count;
            }
            sequence.next_position = item.start_position + item.token_count;
            outputs.push_back(final_prediction);
        }
        return outputs;
    }

    Qwen35MtpResult run_mtp(
        const Qwen35ExecutionPlan & plan,
        std::size_t token_capacity) {
        if (!options.enable_mtp) {
            fail("run_mtp() requires enable_mtp=true");
        }
        const std::size_t draft_count = options.mtp_max_draft_tokens;
        if (token_capacity < draft_count + 1) {
            fail("MTP token capacity is smaller than draft_count + 1");
        }
        const std::vector<PreparedSequence> prepared =
            validate_plan(plan, true, draft_count);
        initialize_new_sequences(prepared);

        Qwen35MtpResult result;
        result.token_ids.assign(prepared.size() * token_capacity, -1);
        result.output_counts.resize(prepared.size(), 0);
        result.draft_counts.resize(prepared.size(), static_cast<std::int32_t>(draft_count));

        for (std::size_t output_row = 0; output_row < prepared.size(); ++output_row) {
            const PreparedSequence & item = prepared[output_row];
            SequenceState & sequence = sequences[item.sequence_slot];
            const std::size_t position = item.start_position;
            const std::int32_t pending_token = plan.tokens[item.token_offset];
            const std::vector<float> old_pending_hidden = sequence.pending_hidden;

            std::vector<std::int32_t> drafts;
            drafts.reserve(draft_count);
            std::vector<float> draft_hidden = old_pending_hidden;
            std::int32_t draft_input = pending_token;
            for (std::size_t index = 0; index < draft_count; ++index) {
                const std::size_t draft_position = position + index;
                const std::int32_t slot = paged_kv->physical_indices(
                    item.block_table, draft_position, 1).front();
                const std::vector<std::int32_t> context =
                    paged_kv->context_indices(item.block_table, draft_position + 1);
                const auto started = std::chrono::steady_clock::now();
                TokenResult drafted = execute_mtp(
                    draft_input,
                    draft_position,
                    slot,
                    context,
                    draft_hidden,
                    true,
                    index + 1 < draft_count,
                    true);
                ++mtp_profile.draft_calls;
                ++mtp_profile.draft_tokens;
                mtp_profile.draft_elapsed_ns += elapsed_nanoseconds(started);
                if (drafted.token < 0) {
                    fail("MTP graph did not produce a greedy draft token");
                }
                drafts.push_back(drafted.token);
                draft_input = drafted.token;
                draft_hidden = std::move(drafted.hidden);
            }

            std::vector<std::int32_t> verification_inputs;
            verification_inputs.reserve(draft_count + 1);
            verification_inputs.push_back(pending_token);
            verification_inputs.insert(
                verification_inputs.end(), drafts.begin(), drafts.end());
            const std::size_t verification_count = draft_count + 1;
            const std::vector<std::int32_t> verification_slots =
                paged_kv->physical_indices(
                    item.block_table, position, verification_count);
            const std::vector<std::int32_t> verification_context =
                paged_kv->context_indices(
                    item.block_table, position + verification_count);
            const std::size_t input_plane =
                recurrent->active_snapshot_plane(item.sequence_slot);
            const bool fuse_mtp_verification_kv =
                mtp_verification_kv_fusion_available();
            const auto verification_started = std::chrono::steady_clock::now();
            TargetChunkResult target = execute_target_chunk(
                item.sequence_slot,
                verification_inputs,
                position,
                verification_slots,
                verification_context,
                input_plane,
                verification_count,
                qwen35::TargetChunkOutputMode::All,
                true,
                true,
                true,
                nullptr,
                fuse_mtp_verification_kv);
            ++mtp_profile.verification_calls;
            mtp_profile.verification_tokens += verification_count;
            mtp_profile.verification_elapsed_ns += elapsed_nanoseconds(verification_started);
            if (target.predictions.size() != verification_count ||
                std::any_of(
                    target.predictions.begin(),
                    target.predictions.end(),
                    [](std::int32_t token) { return token < 0; })) {
                fail("target verification chunk produced invalid greedy tokens");
            }
            const std::size_t embedding = model->config().embedding_length;
            if (target.hidden.size() != checked_product(
                    verification_count,
                    embedding,
                    "target verification hidden output")) {
                fail("target verification chunk returned the wrong hidden-output shape");
            }

            std::size_t accepted = 0;
            while (accepted < draft_count &&
                   target.predictions[accepted] == drafts[accepted]) {
                ++accepted;
            }
            recurrent->select_mtp_rollback(
                item.sequence_slot, draft_count, accepted);

            // The first draft graph already wrote x_p paired with h_{p-1}.
            // The fused verification graph additionally writes every later
            // target-conditioned candidate row.  Rows beyond the accepted
            // prefix are not reachable from sequence.next_position and are
            // overwritten before a later MTP draft can read them.  Without
            // that fusion, retain the legacy accepted-prefix-only maintenance
            // graph so the two modes remain directly comparable.
            if (accepted > 0 && !fuse_mtp_verification_kv) {
                std::vector<std::int32_t> catchup_tokens(
                    verification_inputs.begin() + 1,
                    verification_inputs.begin() + accepted + 1);
                std::vector<std::size_t> catchup_positions;
                catchup_positions.reserve(accepted);
                std::vector<float> catchup_hidden;
                catchup_hidden.reserve(checked_product(
                    accepted,
                    embedding,
                    "MTP catch-up hidden input"));
                for (std::size_t index = 0; index < accepted; ++index) {
                    catchup_positions.push_back(position + index + 1);
                    catchup_hidden.insert(
                        catchup_hidden.end(),
                        target.hidden.begin() + static_cast<std::ptrdiff_t>(index * embedding),
                        target.hidden.begin() +
                            static_cast<std::ptrdiff_t>((index + 1) * embedding));
                }
                const std::vector<std::int32_t> catchup_slots =
                    paged_kv->physical_indices(
                        item.block_table, position + 1, accepted);
                const auto kv_update_started = std::chrono::steady_clock::now();
                execute_mtp_kv_update(
                    catchup_tokens,
                    catchup_positions,
                    catchup_slots,
                    catchup_hidden,
                    true);
                ++mtp_profile.kv_update_calls;
                mtp_profile.kv_update_tokens += accepted;
                mtp_profile.kv_update_elapsed_ns += elapsed_nanoseconds(kv_update_started);
            }
            sequence.pending_hidden.assign(
                target.hidden.begin() + static_cast<std::ptrdiff_t>(accepted * embedding),
                target.hidden.begin() +
                    static_cast<std::ptrdiff_t>((accepted + 1) * embedding));
            sequence.next_position = position + accepted + 1;

            const std::size_t count = accepted + 1;
            result.output_counts[output_row] = static_cast<std::int32_t>(count);
            for (std::size_t index = 0; index < count; ++index) {
                result.token_ids[output_row * token_capacity + index] =
                    target.predictions[index];
            }
        }
        return result;
    }

    void release_blocks(
        const std::vector<std::int32_t> & block_ids,
        const std::vector<std::int32_t> & sequence_ids,
        std::size_t supplied_block_size) {
        if (supplied_block_size != options.block_size) {
            fail("release block_size differs from the runtime");
        }
        if (sequence_ids.empty()) {
            fail("release requires at least one sequence ID");
        }
        std::unordered_set<std::int32_t> seen;
        for (const std::int32_t raw_sequence : sequence_ids) {
            if (raw_sequence < 0 ||
                static_cast<std::size_t>(raw_sequence) >= sequences.size()) {
                fail("released sequence ID is outside the configured range");
            }
            if (!seen.insert(raw_sequence).second) {
                fail("release repeats a sequence ID");
            }
            if (!sequences[static_cast<std::size_t>(raw_sequence)].live) {
                fail("released sequence ID is not live");
            }
        }
        // Prefix sharing is disabled for this first native runtime. Clearing
        // once after validating every sequence makes block reuse deterministic.
        if (!block_ids.empty()) {
            paged_kv->clear_blocks(block_ids);
        }
        for (const std::int32_t raw_sequence : sequence_ids) {
            const std::size_t sequence_slot = static_cast<std::size_t>(raw_sequence);
            recurrent->release_slot(sequence_slot);
            SequenceState & sequence = sequences[sequence_slot];
            sequence.live = false;
            sequence.next_position = 0;
            std::fill(sequence.pending_hidden.begin(), sequence.pending_hidden.end(), 0.0f);
        }
    }

    Qwen35GraphReuseStats graph_reuse_stats() const {
        Qwen35GraphReuseStats result = graph_stats;
        result.active_entries = graph_cache.size();
        return result;
    }

    Qwen35MtpProfileStats mtp_profile_stats() const {
        return mtp_profile;
    }

    Qwen35MemoryStats memory_stats() const {
        Qwen35MemoryStats result;
        result.weights_bytes = static_cast<std::uint64_t>(
            model_storage == nullptr ? 0 : model_storage->resident_buffer_bytes());
        result.paged_kv_bytes = static_cast<std::uint64_t>(
            paged_kv == nullptr ? 0 : paged_kv->resident_bytes());
        result.recurrent_state_bytes = static_cast<std::uint64_t>(
            recurrent == nullptr ? 0 : recurrent->resident_bytes());
        result.graph_cache_entries = static_cast<std::uint64_t>(graph_cache.size());
        const std::uint64_t graph_contexts = result.graph_cache_entries + 1;
        result.graph_metadata_bytes = graph_contexts *
            static_cast<std::uint64_t>(kGraphMetadataBytes);
        result.known_persistent_bytes = checked_sum(
            {result.weights_bytes,
             result.paged_kv_bytes,
             result.recurrent_state_bytes,
             result.graph_metadata_bytes},
            "runtime memory stats");
        return result;
    }
};

Qwen35Runtime::Qwen35Runtime(Qwen35RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

Qwen35Runtime::~Qwen35Runtime() = default;

std::vector<std::int32_t> Qwen35Runtime::run(const Qwen35ExecutionPlan & plan) {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    return impl_->run(plan);
}

Qwen35MtpResult Qwen35Runtime::run_mtp(
    const Qwen35ExecutionPlan & plan,
    std::size_t token_capacity) {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    return impl_->run_mtp(plan, token_capacity);
}

void Qwen35Runtime::release_blocks(
    const std::vector<std::int32_t> & block_ids,
    const std::vector<std::int32_t> & sequence_ids,
    std::size_t block_size) {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    impl_->release_blocks(block_ids, sequence_ids, block_size);
}

Qwen35GraphReuseStats Qwen35Runtime::graph_reuse_stats() const {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    return impl_->graph_reuse_stats();
}

Qwen35MtpProfileStats Qwen35Runtime::mtp_profile_stats() const {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    return impl_->mtp_profile_stats();
}

Qwen35MemoryStats Qwen35Runtime::memory_stats() const {
    if (impl_ == nullptr) {
        fail("runtime has been shut down");
    }
    return impl_->memory_stats();
}

void Qwen35Runtime::shutdown() {
    impl_.reset();
}

bool Qwen35Runtime::is_shutdown() const noexcept {
    return impl_ == nullptr;
}

}  // namespace nanovllm::native
