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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
    return BackendList::vulkan_with_cpu(options.cpu_threads, options.device_index);
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

private:
    ggml_context * value_ = nullptr;
};

class ExecutorResetGuard {
public:
    explicit ExecutorResetGuard(GraphExecutor & executor) : executor_(&executor) {}
    ~ExecutorResetGuard() {
        if (executor_ != nullptr) {
            try {
                executor_->reset();
            } catch (...) {
                // Destructors cannot report a second error. The original
                // execution exception remains the actionable failure.
            }
        }
    }
    ExecutorResetGuard(const ExecutorResetGuard &) = delete;
    ExecutorResetGuard & operator=(const ExecutorResetGuard &) = delete;

private:
    GraphExecutor * executor_;
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
        primary_backend = &backends.at(0);
        model_storage = std::make_unique<GgufWeights>(options.model_path);
        model_storage->load(primary_backend->get());
        model = std::make_unique<qwen35::Qwen35Weights>(*model_storage);

        const qwen35::Config & config = model->config();
        if (options.max_model_len > config.context_length) {
            fail("max_model_len exceeds the model context length");
        }
        if (options.enable_mtp && config.nextn_predict_layers != 1) {
            fail("MTP requested, but the GGUF does not contain exactly one bundled MTP layer");
        }

        paged_kv = std::make_unique<PagedKvCache>(
            primary_backend->get(),
            config,
            options.block_size,
            options.num_blocks,
            options.enable_mtp);
        RecurrentStateOptions state_options;
        state_options.max_sequence_slots = options.max_num_seqs;
        state_options.max_draft_tokens = options.mtp_max_draft_tokens;
        state_options.include_mtp_recurrent = false;
        recurrent = std::make_unique<RecurrentStateCache>(
            config, primary_backend->get(), state_options);
        executor = std::make_unique<GraphExecutor>(
            backends, kGraphNodeCapacity, false, true);

        sequences.resize(options.max_num_seqs);
        for (SequenceState & sequence : sequences) {
            sequence.pending_hidden.assign(config.embedding_length, 0.0f);
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
    std::vector<SequenceState> sequences;

    std::vector<PreparedSequence> validate_plan(
        const Qwen35ExecutionPlan & plan,
        bool require_decode,
        std::size_t future_tokens) const {
        if (plan.n_tokens <= 0 || plan.n_seqs <= 0) {
            fail("execution plan must contain at least one token and sequence");
        }
        const std::size_t n_tokens = static_cast<std::size_t>(plan.n_tokens);
        const std::size_t n_seqs = static_cast<std::size_t>(plan.n_seqs);
        if (n_tokens > options.max_num_batched_tokens) {
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
            (void) paged_kv->context_indices(
                item.block_table, expected_start + count + future_tokens);
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
        for (const std::uint32_t layer : recurrent->recurrent_layers()) {
            qwen35::RecurrentChunkStateView state;
            state.convolution = recurrent->view_conv(
                ctx, layer, sequence_slot, input_plane);
            state.delta = recurrent->view_delta(ctx, layer, sequence_slot, input_plane);
            state.convolution_outputs.reserve(snapshot_count);
            state.delta_outputs.reserve(snapshot_count);
            for (std::size_t plane = 0; plane < snapshot_count; ++plane) {
                state.convolution_outputs.push_back(recurrent->view_conv(
                    ctx, layer, sequence_slot, plane));
                state.delta_outputs.push_back(recurrent->view_delta(
                    ctx, layer, sequence_slot, plane));
            }
            persistent.recurrent.push_back(std::move(state));
        }
        persistent.attention.reserve(paged_kv->target_layers().size());
        for (const PagedKvLayer & layer : paged_kv->target_layers()) {
            persistent.attention.push_back({layer.key, layer.value});
        }
        return persistent;
    }

    void place_vulkan_compute_nodes(ggml_cgraph * graph) {
        if (primary_backend->kind() != BackendKind::Vulkan) {
            return;
        }
        executor->set_all_compute_nodes_backend(graph, BackendKind::Vulkan);
    }

    void assert_compute_placement(ggml_cgraph * graph) const {
        executor->assert_all_compute_nodes_on_backend(
            graph, primary_backend->kind());
    }

    void upload_common_inputs(
        const qwen35::TokenGraph & token_graph,
        std::int32_t token,
        std::size_t position,
        std::int32_t write_slot,
        const std::vector<std::int32_t> & read_slots) const {
        if (position > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            fail("position exceeds the I32 IMRoPE ABI");
        }
        const std::int32_t position_i32 = static_cast<std::int32_t>(position);
        const std::array<std::int32_t, 4> positions{
            position_i32, position_i32, position_i32, 0};
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
        GraphContext graph_context;
        ExecutorResetGuard reset_guard(*executor);
        qwen35::TargetPersistentView persistent = make_target_persistent(
            graph_context.get(), sequence_slot, input_plane, output_plane);
        qwen35::TokenGraph token_graph = qwen35::build_target_token_graph(
            graph_context.get(), *model, persistent, read_slots.size(), emit_greedy);
        place_vulkan_compute_nodes(token_graph.graph);
        executor->allocate(token_graph.graph);
        assert_compute_placement(token_graph.graph);
        upload_common_inputs(token_graph, token, position, write_slot, read_slots);
        executor->compute(token_graph.graph);

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
        std::size_t sequence_slot,
        const std::vector<std::int32_t> & tokens,
        std::size_t start_position,
        const std::vector<std::int32_t> & write_slots,
        const std::vector<std::int32_t> & read_slots,
        std::size_t input_plane,
        std::size_t snapshot_count,
        qwen35::TargetChunkOutputMode output_mode,
        bool retain_hidden) {
        if (tokens.empty()) {
            fail("target chunk requires at least one token");
        }
        if (write_slots.size() != tokens.size()) {
            fail("target chunk token and write-slot counts differ");
        }
        if (start_position > options.max_model_len ||
            tokens.size() > options.max_model_len - start_position) {
            fail("target chunk exceeds max_model_len");
        }
        if (read_slots.size() != start_position + tokens.size()) {
            fail("target chunk attention context has the wrong length");
        }
        if (snapshot_count == 0 || snapshot_count > tokens.size() ||
            snapshot_count > recurrent->snapshot_planes()) {
            fail("target chunk snapshot count is outside the supported range");
        }
        paged_kv->validate_read_indices(read_slots);
        paged_kv->validate_write_indices(write_slots);
        if (!std::equal(
                write_slots.begin(),
                write_slots.end(),
                read_slots.end() - static_cast<std::ptrdiff_t>(write_slots.size()))) {
            fail("target chunk attention context must end with its write slots");
        }

        std::vector<std::int32_t> positions;
        positions.reserve(tokens.size());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            const std::size_t position = start_position + index;
            if (position > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
                fail("position exceeds the I32 IMRoPE ABI");
            }
            positions.push_back(static_cast<std::int32_t>(position));
        }
        const std::vector<std::int32_t> expanded_positions =
            qwen35::ops::expand_text_positions(positions);

        GraphContext graph_context;
        ExecutorResetGuard reset_guard(*executor);
        qwen35::TargetChunkPersistentView persistent = make_target_chunk_persistent(
            graph_context.get(), sequence_slot, input_plane, snapshot_count);
        qwen35::TargetChunkGraph graph = qwen35::build_target_chunk_graph(
            graph_context.get(),
            *model,
            persistent,
            tokens.size(),
            read_slots.size(),
            snapshot_count,
            output_mode,
            retain_hidden);
        place_vulkan_compute_nodes(graph.graph);
        executor->allocate(graph.graph);
        assert_compute_placement(graph.graph);
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
        if (graph.causal_mask != nullptr) {
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
            ggml_backend_tensor_set(
                graph.causal_mask,
                mask.data(),
                0,
                mask.size() * sizeof(mask.front()));
        }
        executor->compute(graph.graph);

        TargetChunkResult result;
        std::size_t prediction_count = 0;
        if (output_mode == qwen35::TargetChunkOutputMode::All) {
            prediction_count = tokens.size();
        } else if (output_mode == qwen35::TargetChunkOutputMode::Last) {
            prediction_count = 1;
        }
        if (prediction_count > 0) {
            if (graph.greedy_tokens == nullptr) {
                fail("target chunk omitted requested greedy output");
            }
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
            ggml_backend_tensor_get(
                graph.hidden,
                result.hidden.data(),
                0,
                result.hidden.size() * sizeof(result.hidden.front()));
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
        bool read_hidden) {
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
        GraphContext graph_context;
        ExecutorResetGuard reset_guard(*executor);
        const PagedKvLayer & layer = paged_kv->mtp_layer();
        qwen35::AttentionCacheView cache{layer.key, layer.value};
        qwen35::TokenGraph token_graph = qwen35::build_mtp_token_graph(
            graph_context.get(), *model, cache, read_slots.size(), emit_greedy);
        place_vulkan_compute_nodes(token_graph.graph);
        executor->allocate(token_graph.graph);
        assert_compute_placement(token_graph.graph);
        upload_common_inputs(token_graph, token, position, write_slot, read_slots);
        ggml_backend_tensor_set(
            token_graph.hidden_input,
            hidden_input.data(),
            0,
            hidden_input.size() * sizeof(float));
        executor->compute(token_graph.graph);

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
        const std::vector<float> & hidden_inputs) {
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

        GraphContext graph_context;
        ExecutorResetGuard reset_guard(*executor);
        const PagedKvLayer & layer = paged_kv->mtp_layer();
        qwen35::AttentionCacheView cache{layer.key, layer.value};
        qwen35::MtpKvUpdateGraph graph = qwen35::build_mtp_kv_update_graph(
            graph_context.get(), *model, cache, tokens.size());
        place_vulkan_compute_nodes(graph.graph);
        executor->allocate(graph.graph);
        assert_compute_placement(graph.graph);
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
    }

    std::vector<std::int32_t> run(const Qwen35ExecutionPlan & plan) {
        const std::vector<PreparedSequence> prepared = validate_plan(plan, false, 0);
        initialize_new_sequences(prepared);
        std::vector<std::int32_t> outputs;
        outputs.reserve(prepared.size());
        for (const PreparedSequence & item : prepared) {
            SequenceState & sequence = sequences[item.sequence_slot];
            const std::vector<std::int32_t> target_tokens = slice(
                plan.tokens, item.token_offset, item.token_count);
            const std::vector<std::int32_t> write_slots = slice(
                plan.slot_mapping, item.token_offset, item.token_count);
            const BackendDeviceInfo & device = primary_backend->device_info();
            const bool is_mali_vulkan =
                primary_backend->kind() == BackendKind::Vulkan &&
                (device.name.find("Mali") != std::string::npos ||
                 device.description.find("Mali") != std::string::npos);
            const std::size_t chunk_limit = is_mali_vulkan
                ? kVulkanTargetPrefillChunk
                : item.token_count;
            std::int32_t final_prediction = -1;
            for (std::size_t offset = 0; offset < item.token_count;) {
                const std::size_t chunk_count = std::min(
                    chunk_limit, item.token_count - offset);
                const std::size_t chunk_position = item.start_position + offset;
                const bool is_final = offset + chunk_count == item.token_count;
                const std::vector<std::int32_t> chunk_tokens = slice(
                    target_tokens, offset, chunk_count);
                const std::vector<std::int32_t> chunk_write_slots = slice(
                    write_slots, offset, chunk_count);
                const std::vector<std::int32_t> context =
                    paged_kv->context_indices(
                        item.block_table, chunk_position + chunk_count);
                const std::size_t input_plane =
                    recurrent->active_snapshot_plane(item.sequence_slot);
                TargetChunkResult target = execute_target_chunk(
                    item.sequence_slot,
                    chunk_tokens,
                    chunk_position,
                    chunk_write_slots,
                    context,
                    input_plane,
                    1,
                    is_final ? qwen35::TargetChunkOutputMode::Last
                             : qwen35::TargetChunkOutputMode::None,
                    options.enable_mtp);
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
                        mtp_hidden_inputs.end(),
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
                        mtp_hidden_inputs);
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
                TokenResult drafted = execute_mtp(
                    draft_input,
                    draft_position,
                    slot,
                    context,
                    draft_hidden,
                    true,
                    index + 1 < draft_count);
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
            TargetChunkResult target = execute_target_chunk(
                item.sequence_slot,
                verification_inputs,
                position,
                verification_slots,
                verification_context,
                input_plane,
                verification_count,
                qwen35::TargetChunkOutputMode::All,
                true);
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
            // Replace only later accepted draft-conditioned rows, batching the
            // target-conditioned K/V projections without running dead MTP
            // attention, FFN, output norm, or hidden readback work.
            if (accepted > 0) {
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
                execute_mtp_kv_update(
                    catchup_tokens,
                    catchup_positions,
                    catchup_slots,
                    catchup_hidden);
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

void Qwen35Runtime::shutdown() {
    impl_.reset();
}

bool Qwen35Runtime::is_shutdown() const noexcept {
    return impl_ == nullptr;
}

}  // namespace nanovllm::native
