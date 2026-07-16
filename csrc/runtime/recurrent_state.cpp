#include "runtime/recurrent_state.h"

#include "runtime/backend.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

namespace nanovllm::native {
namespace {

constexpr std::size_t kExpectedTargetRecurrentLayers = 18;

// Independent compile-time verification of the K-a mapping, including both
// boundary cases and an interior accepted prefix.
static_assert(mtp_rollback_snapshot_plane_unchecked(4, 0) == 4);
static_assert(mtp_rollback_snapshot_plane_unchecked(4, 1) == 3);
static_assert(mtp_rollback_snapshot_plane_unchecked(4, 3) == 1);
static_assert(mtp_rollback_snapshot_plane_unchecked(4, 4) == 0);

std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char * description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw RecurrentStateError(std::string(description) + " size overflows size_t");
    }
    return left * right;
}

void require_context(ggml_context * context, const char * operation) {
    if (context == nullptr) {
        throw RecurrentStateError(std::string(operation) + ": graph context is null");
    }
}

std::size_t index_count(const ggml_tensor * indices, const char * operation) {
    if (indices == nullptr) {
        throw RecurrentStateError(std::string(operation) + ": row-index tensor is null");
    }
    if (indices->type != GGML_TYPE_I32) {
        throw RecurrentStateError(std::string(operation) + ": row indices must be I32");
    }
    if (!ggml_is_vector(indices)) {
        throw RecurrentStateError(std::string(operation) + ": row indices must be one-dimensional");
    }
    const std::int64_t count = ggml_nelements(indices);
    if (count <= 0) {
        throw RecurrentStateError(std::string(operation) + ": row indices cannot be empty");
    }
    return static_cast<std::size_t>(count);
}

std::string layer_label(const char * kind, std::uint32_t model_layer) {
    return std::string("recurrent_") + kind + "_l" + std::to_string(model_layer);
}

}  // namespace

RecurrentStateError::RecurrentStateError(const std::string & message)
    : std::runtime_error("native recurrent state: " + message) {}

RecurrentStateCache::RecurrentStateCache(
    const qwen35::Config & model_config,
    ggml_backend_t backend,
    const RecurrentStateOptions & options)
    : model_config_(model_config), options_(options), backend_(backend) {
    if (backend_ == nullptr) {
        throw RecurrentStateError("backend is null");
    }
    if (options_.max_sequence_slots == 0) {
        throw RecurrentStateError("max_sequence_slots must be greater than zero");
    }
    if (options_.max_draft_tokens == std::numeric_limits<std::size_t>::max()) {
        throw RecurrentStateError("max_draft_tokens is too large");
    }
    if (model_config_.main_layers + model_config_.nextn_predict_layers !=
        model_config_.block_count) {
        throw RecurrentStateError(
            "model layer counts are inconsistent: main + MTP does not equal block_count");
    }
    if (model_config_.ssm_conv_kernel < 2) {
        throw RecurrentStateError("ssm_conv_kernel must be at least two");
    }
    if (model_config_.ssm_state_size == 0 || model_config_.ssm_group_count == 0 ||
        model_config_.ssm_time_step_rank == 0 || model_config_.ssm_inner_size == 0) {
        throw RecurrentStateError("SSM dimensions must be non-zero");
    }
    if (model_config_.ssm_inner_size % model_config_.ssm_time_step_rank != 0) {
        throw RecurrentStateError("ssm_inner_size must be divisible by ssm_time_step_rank");
    }

    delta_head_size_ =
        model_config_.ssm_inner_size / model_config_.ssm_time_step_rank;
    delta_head_count_ = model_config_.ssm_time_step_rank;
    if (delta_head_size_ != model_config_.ssm_state_size) {
        throw RecurrentStateError(
            "Qwen3.5 GDN requires inner_size / time_step_rank == state_size");
    }

    for (std::uint32_t layer = 0; layer < model_config_.main_layers; ++layer) {
        if (model_config_.is_recurrent_layer(layer)) {
            recurrent_layers_.push_back(layer);
        }
    }
    if (recurrent_layers_.size() != kExpectedTargetRecurrentLayers) {
        std::ostringstream message;
        message << "Qwen3.5-2B target stack has " << recurrent_layers_.size()
                << " recurrent layers, expected " << kExpectedTargetRecurrentLayers;
        throw RecurrentStateError(message.str());
    }

    // qwen35::required_tensor_specs() classifies every MTP layer as a dense
    // full-attention block.  Confirm that none can be mistaken for a target
    // recurrent layer, and reject an incompatible explicit request.
    for (std::uint32_t layer = model_config_.main_layers;
         layer < model_config_.block_count;
         ++layer) {
        if (!model_config_.is_mtp_layer(layer) || model_config_.is_recurrent_layer(layer)) {
            throw RecurrentStateError(
                "model configuration does not classify MTP blocks as full attention");
        }
    }
    if (options_.include_mtp_recurrent) {
        throw RecurrentStateError(
            "MTP recurrent state was requested, but the supported Qwen3.5 model "
            "declares blk.24 as a full-attention block");
    }

    snapshot_planes_ = options_.max_draft_tokens + 1;
    total_rows_ = checked_multiply(
        options_.max_sequence_slots, snapshot_planes_, "persistent state row");
    if (total_rows_ > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw RecurrentStateError("persistent state row count exceeds GGML I32 indexing");
    }

    conv_kernel_size_ = model_config_.ssm_conv_kernel;
    const std::size_t key_channels = checked_multiply(
        model_config_.ssm_group_count,
        model_config_.ssm_state_size,
        "SSM key channels");
    conv_channels_ = model_config_.ssm_inner_size + checked_multiply(
        2, key_channels, "SSM key/value channels");
    conv_elements_ = checked_multiply(
        conv_kernel_size_ - 1, conv_channels_, "convolution state");

    const std::size_t delta_plane = checked_multiply(
        delta_head_size_, delta_head_size_, "delta head state");
    delta_elements_ = checked_multiply(
        delta_plane, delta_head_count_, "delta recurrent state");
    const std::size_t expected_delta_elements = checked_multiply(
        model_config_.ssm_state_size,
        model_config_.ssm_inner_size,
        "Qwen3.5 delta recurrent state");
    if (delta_elements_ != expected_delta_elements) {
        throw RecurrentStateError("derived GDN state size differs from state_size * inner_size");
    }

    slot_initialized_.assign(options_.max_sequence_slots, 0);
    active_snapshot_planes_.assign(options_.max_sequence_slots, 0);

    const std::size_t tensor_count = checked_multiply(
        recurrent_layers_.size(), 2, "recurrent tensor descriptor");
    const std::size_t metadata_bytes = checked_multiply(
        tensor_count + 1, ggml_tensor_overhead(), "recurrent tensor metadata");
    ggml_init_params params{};
    params.mem_size = metadata_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    tensor_context_ = ggml_init(params);
    if (tensor_context_ == nullptr) {
        throw RecurrentStateError("failed to create the persistent tensor context");
    }

    try {
        conv_states_.reserve(recurrent_layers_.size());
        delta_states_.reserve(recurrent_layers_.size());
        for (const std::uint32_t layer : recurrent_layers_) {
            ggml_tensor * conv = ggml_new_tensor_2d(
                tensor_context_,
                GGML_TYPE_F32,
                static_cast<std::int64_t>(conv_elements_),
                static_cast<std::int64_t>(total_rows_));
            ggml_tensor * delta = ggml_new_tensor_2d(
                tensor_context_,
                GGML_TYPE_F32,
                static_cast<std::int64_t>(delta_elements_),
                static_cast<std::int64_t>(total_rows_));
            if (conv == nullptr || delta == nullptr) {
                throw RecurrentStateError("failed to create persistent tensor descriptors");
            }
            ggml_set_name(conv, layer_label("conv", layer).c_str());
            ggml_set_name(delta, layer_label("delta", layer).c_str());
            conv_states_.push_back(conv);
            delta_states_.push_back(delta);
        }

        const ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend_);
        if (buffer_type == nullptr) {
            throw RecurrentStateError("backend returned a null default buffer type");
        }
        assert_buffer_type_compatible(backend_, buffer_type, "recurrent state");
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(tensor_context_, buffer_type);
        if (buffer_ == nullptr) {
            throw RecurrentStateError("backend recurrent-state buffer allocation failed");
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        assert_buffer_on_backend(buffer_, backend_, "recurrent state");
        for (std::size_t index = 0; index < recurrent_layers_.size(); ++index) {
            assert_tensor_on_backend(conv_states_[index], backend_, "recurrent convolution state");
            assert_tensor_on_backend(delta_states_[index], backend_, "recurrent delta state");
        }
        ggml_backend_buffer_clear(buffer_, 0);
        ggml_backend_synchronize(backend_);
    } catch (...) {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        ggml_free(tensor_context_);
        tensor_context_ = nullptr;
        throw;
    }
}

RecurrentStateCache::~RecurrentStateCache() {
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
    }
    if (tensor_context_ != nullptr) {
        ggml_free(tensor_context_);
    }
}

bool RecurrentStateCache::has_recurrent_layer(std::uint32_t model_layer) const noexcept {
    return std::find(recurrent_layers_.begin(), recurrent_layers_.end(), model_layer) !=
           recurrent_layers_.end();
}

std::size_t RecurrentStateCache::layer_position(std::uint32_t model_layer) const {
    const auto it =
        std::find(recurrent_layers_.begin(), recurrent_layers_.end(), model_layer);
    if (it == recurrent_layers_.end()) {
        throw RecurrentStateError(
            "model layer " + std::to_string(model_layer) + " is not recurrent");
    }
    return static_cast<std::size_t>(std::distance(recurrent_layers_.begin(), it));
}

ggml_tensor * RecurrentStateCache::conv_storage(std::uint32_t model_layer) const {
    return conv_states_.at(layer_position(model_layer));
}

ggml_tensor * RecurrentStateCache::delta_storage(std::uint32_t model_layer) const {
    return delta_states_.at(layer_position(model_layer));
}

std::size_t RecurrentStateCache::resident_bytes() const noexcept {
    return buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(buffer_);
}

void RecurrentStateCache::require_slot(std::size_t sequence_slot) const {
    if (sequence_slot >= options_.max_sequence_slots) {
        std::ostringstream message;
        message << "sequence slot " << sequence_slot << " is out of range [0, "
                << options_.max_sequence_slots << ')';
        throw RecurrentStateError(message.str());
    }
}

void RecurrentStateCache::require_active_slot(std::size_t sequence_slot) const {
    require_slot(sequence_slot);
    if (slot_initialized_[sequence_slot] == 0) {
        throw RecurrentStateError(
            "sequence slot " + std::to_string(sequence_slot) + " is not initialized");
    }
}

void RecurrentStateCache::require_plane(std::size_t snapshot_plane) const {
    if (snapshot_plane >= snapshot_planes_) {
        std::ostringstream message;
        message << "snapshot plane " << snapshot_plane << " is out of range [0, "
                << snapshot_planes_ << ')';
        throw RecurrentStateError(message.str());
    }
}

void RecurrentStateCache::zero_slot_data(std::size_t sequence_slot) {
    require_slot(sequence_slot);
    ggml_backend_synchronize(backend_);
    for (std::size_t layer = 0; layer < recurrent_layers_.size(); ++layer) {
        for (ggml_tensor * tensor : {conv_states_[layer], delta_states_[layer]}) {
            const std::size_t offset =
                static_cast<std::size_t>(row_index(sequence_slot, 0)) * tensor->nb[1];
            const std::size_t bytes = checked_multiply(
                snapshot_planes_, tensor->nb[1], "slot clear");
            if (offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset) {
                throw RecurrentStateError("slot clear range exceeds persistent tensor storage");
            }
            ggml_backend_tensor_memset(tensor, 0, offset, bytes);
        }
    }
    ggml_backend_synchronize(backend_);
}

void RecurrentStateCache::initialize_slot(std::size_t sequence_slot) {
    require_slot(sequence_slot);
    if (slot_initialized_[sequence_slot] != 0) {
        throw RecurrentStateError(
            "sequence slot " + std::to_string(sequence_slot) + " is already initialized");
    }
    zero_slot_data(sequence_slot);
    active_snapshot_planes_[sequence_slot] = 0;
    slot_initialized_[sequence_slot] = 1;
}

void RecurrentStateCache::clear_slot(std::size_t sequence_slot) {
    require_active_slot(sequence_slot);
    zero_slot_data(sequence_slot);
    active_snapshot_planes_[sequence_slot] = 0;
}

void RecurrentStateCache::release_slot(std::size_t sequence_slot) {
    require_active_slot(sequence_slot);
    zero_slot_data(sequence_slot);
    active_snapshot_planes_[sequence_slot] = 0;
    slot_initialized_[sequence_slot] = 0;
}

void RecurrentStateCache::clear_all() {
    ggml_backend_synchronize(backend_);
    ggml_backend_buffer_clear(buffer_, 0);
    ggml_backend_synchronize(backend_);
    std::fill(slot_initialized_.begin(), slot_initialized_.end(), 0);
    std::fill(active_snapshot_planes_.begin(), active_snapshot_planes_.end(), 0);
}

bool RecurrentStateCache::slot_initialized(std::size_t sequence_slot) const {
    require_slot(sequence_slot);
    return slot_initialized_[sequence_slot] != 0;
}

std::size_t RecurrentStateCache::rollback_snapshot_plane(
    std::size_t draft_count,
    std::size_t accepted_drafts) {
    if (accepted_drafts > draft_count) {
        throw RecurrentStateError("accepted draft count cannot exceed draft count");
    }
    return mtp_rollback_snapshot_plane_unchecked(draft_count, accepted_drafts);
}

std::size_t RecurrentStateCache::select_mtp_rollback(
    std::size_t sequence_slot,
    std::size_t draft_count,
    std::size_t accepted_drafts) {
    require_active_slot(sequence_slot);
    if (draft_count > options_.max_draft_tokens) {
        throw RecurrentStateError("verification draft count exceeds configured maximum");
    }
    const std::size_t plane = rollback_snapshot_plane(draft_count, accepted_drafts);
    require_plane(plane);
    active_snapshot_planes_[sequence_slot] = plane;
    return plane;
}

void RecurrentStateCache::select_latest(std::size_t sequence_slot) {
    require_active_slot(sequence_slot);
    active_snapshot_planes_[sequence_slot] = 0;
}

std::size_t RecurrentStateCache::active_snapshot_plane(
    std::size_t sequence_slot) const {
    require_active_slot(sequence_slot);
    return active_snapshot_planes_[sequence_slot];
}

std::int32_t RecurrentStateCache::row_index(
    std::size_t sequence_slot,
    std::size_t snapshot_plane) const {
    require_slot(sequence_slot);
    require_plane(snapshot_plane);
    const std::size_t row = sequence_slot * snapshot_planes_ + snapshot_plane;
    if (row > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw RecurrentStateError("persistent state row id exceeds I32");
    }
    return static_cast<std::int32_t>(row);
}

std::vector<std::int32_t> RecurrentStateCache::active_row_indices(
    const std::vector<std::size_t> & sequence_slots) const {
    if (sequence_slots.empty()) {
        throw RecurrentStateError("sequence slot list cannot be empty");
    }
    std::vector<std::int32_t> result;
    result.reserve(sequence_slots.size());
    for (const std::size_t slot : sequence_slots) {
        result.push_back(row_index(slot, active_snapshot_plane(slot)));
    }
    return result;
}

std::vector<std::int32_t> RecurrentStateCache::plane_row_indices(
    const std::vector<std::size_t> & sequence_slots,
    std::size_t snapshot_plane) const {
    if (sequence_slots.empty()) {
        throw RecurrentStateError("sequence slot list cannot be empty");
    }
    require_plane(snapshot_plane);
    std::vector<std::int32_t> result;
    result.reserve(sequence_slots.size());
    for (const std::size_t slot : sequence_slots) {
        require_active_slot(slot);
        result.push_back(row_index(slot, snapshot_plane));
    }
    return result;
}

std::vector<std::int32_t> RecurrentStateCache::snapshot_row_indices(
    const std::vector<std::size_t> & sequence_slots,
    std::size_t plane_count) const {
    if (sequence_slots.empty()) {
        throw RecurrentStateError("sequence slot list cannot be empty");
    }
    if (plane_count == 0 || plane_count > snapshot_planes_) {
        throw RecurrentStateError("snapshot plane count is out of configured range");
    }
    std::vector<std::int32_t> result;
    result.reserve(checked_multiply(
        sequence_slots.size(), plane_count, "snapshot row-index"));
    for (std::size_t plane = 0; plane < plane_count; ++plane) {
        for (const std::size_t slot : sequence_slots) {
            require_active_slot(slot);
            result.push_back(row_index(slot, plane));
        }
    }
    return result;
}

ggml_tensor * RecurrentStateCache::view_conv(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    std::size_t sequence_slot,
    std::size_t snapshot_plane) const {
    require_context(graph_context, "view conv state");
    const std::int32_t row = row_index(sequence_slot, snapshot_plane);
    ggml_tensor * storage = conv_storage(model_layer);
    const std::size_t offset = static_cast<std::size_t>(row) * storage->nb[1];
    ggml_tensor * result = ggml_view_3d(
        graph_context,
        storage,
        static_cast<std::int64_t>(conv_kernel_size_ - 1),
        static_cast<std::int64_t>(conv_channels_),
        1,
        ggml_row_size(GGML_TYPE_F32, static_cast<std::int64_t>(conv_kernel_size_ - 1)),
        storage->nb[1],
        offset);
    ggml_set_name(result, "recurrent_conv_view");
    return result;
}

ggml_tensor * RecurrentStateCache::view_delta(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    std::size_t sequence_slot,
    std::size_t snapshot_plane) const {
    require_context(graph_context, "view delta state");
    const std::int32_t row = row_index(sequence_slot, snapshot_plane);
    ggml_tensor * storage = delta_storage(model_layer);
    const std::size_t offset = static_cast<std::size_t>(row) * storage->nb[1];
    const std::size_t head_row = ggml_row_size(
        GGML_TYPE_F32, static_cast<std::int64_t>(delta_head_size_));
    ggml_tensor * result = ggml_view_4d(
        graph_context,
        storage,
        static_cast<std::int64_t>(delta_head_size_),
        static_cast<std::int64_t>(delta_head_size_),
        static_cast<std::int64_t>(delta_head_count_),
        1,
        head_row,
        checked_multiply(head_row, delta_head_size_, "delta view head stride"),
        storage->nb[1],
        offset);
    ggml_set_name(result, "recurrent_delta_view");
    return result;
}

ggml_tensor * RecurrentStateCache::get_rows(
    ggml_context * graph_context,
    ggml_tensor * storage,
    ggml_tensor * row_indices,
    const char * label) const {
    require_context(graph_context, label);
    (void) index_count(row_indices, label);
    ggml_tensor * result = ggml_get_rows(graph_context, storage, row_indices);
    ggml_set_name(result, label);
    return result;
}

ggml_tensor * RecurrentStateCache::get_conv(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    ggml_tensor * row_indices) const {
    const std::size_t count = index_count(row_indices, "get conv state");
    ggml_tensor * rows = get_rows(
        graph_context, conv_storage(model_layer), row_indices, "recurrent_conv_get_rows");
    ggml_tensor * result = ggml_reshape_3d(
        graph_context,
        rows,
        static_cast<std::int64_t>(conv_kernel_size_ - 1),
        static_cast<std::int64_t>(conv_channels_),
        static_cast<std::int64_t>(count));
    ggml_set_name(result, "recurrent_conv_get");
    return result;
}

ggml_tensor * RecurrentStateCache::get_delta(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    ggml_tensor * row_indices) const {
    const std::size_t count = index_count(row_indices, "get delta state");
    ggml_tensor * rows = get_rows(
        graph_context, delta_storage(model_layer), row_indices, "recurrent_delta_get_rows");
    ggml_tensor * result = ggml_reshape_4d(
        graph_context,
        rows,
        static_cast<std::int64_t>(delta_head_size_),
        static_cast<std::int64_t>(delta_head_size_),
        static_cast<std::int64_t>(delta_head_count_),
        static_cast<std::int64_t>(count));
    ggml_set_name(result, "recurrent_delta_get");
    return result;
}

ggml_tensor * RecurrentStateCache::set_rows(
    ggml_context * graph_context,
    ggml_tensor * storage,
    ggml_tensor * source,
    ggml_tensor * row_indices,
    std::size_t row_elements,
    const char * label) const {
    require_context(graph_context, label);
    const std::size_t count = index_count(row_indices, label);
    if (source == nullptr) {
        throw RecurrentStateError(std::string(label) + ": source tensor is null");
    }
    if (source->type != GGML_TYPE_F32) {
        throw RecurrentStateError(std::string(label) + ": source tensor must be F32");
    }
    const std::size_t expected = checked_multiply(row_elements, count, label);
    if (static_cast<std::size_t>(ggml_nelements(source)) != expected) {
        std::ostringstream message;
        message << label << ": source has " << ggml_nelements(source)
                << " elements, expected " << expected;
        throw RecurrentStateError(message.str());
    }
    if (!ggml_is_contiguous(source)) {
        source = ggml_cont(graph_context, source);
    }
    ggml_tensor * rows = ggml_reshape_2d(
        graph_context,
        source,
        static_cast<std::int64_t>(row_elements),
        static_cast<std::int64_t>(count));
    ggml_tensor * result = ggml_set_rows(graph_context, storage, rows, row_indices);
    ggml_set_name(result, label);
    return result;
}

ggml_tensor * RecurrentStateCache::set_conv(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    ggml_tensor * source,
    ggml_tensor * row_indices) const {
    return set_rows(
        graph_context,
        conv_storage(model_layer),
        source,
        row_indices,
        conv_elements_,
        "recurrent_conv_set");
}

ggml_tensor * RecurrentStateCache::set_delta(
    ggml_context * graph_context,
    std::uint32_t model_layer,
    ggml_tensor * source,
    ggml_tensor * row_indices) const {
    return set_rows(
        graph_context,
        delta_storage(model_layer),
        source,
        row_indices,
        delta_elements_,
        "recurrent_delta_set");
}

}  // namespace nanovllm::native
