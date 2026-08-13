#pragma once

#include "runtime/qwen35_model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nanovllm::native {

class RecurrentStateError : public std::runtime_error {
public:
    explicit RecurrentStateError(const std::string & message);
};

// GGML's gated-delta-net output stores snapshots newest first.  A target
// verification pass over the pending token plus K draft tokens therefore
// selects snapshot K-a after accepting a draft prefix of length a:
//
//   plane 0 = state after the last draft token
//   plane K = state after the pending/target token
//
// This unchecked constexpr form exists so the mapping can be independently
// compile-time tested.  Runtime callers should use
// RecurrentStateCache::rollback_snapshot_plane(), which validates a <= K.
constexpr std::size_t mtp_rollback_snapshot_plane_unchecked(
    std::size_t draft_count,
    std::size_t accepted_drafts) noexcept {
    return draft_count - accepted_drafts;
}

struct RecurrentStateOptions {
    // nano-vLLM assigns one stable native slot to every live Python sequence.
    std::size_t max_sequence_slots = 0;

    // K draft tokens require K+1 newest-first state snapshots because target
    // verification also consumes the pending token.
    std::size_t max_draft_tokens = 0;

    // This is deliberately an explicit request rather than an architecture
    // guess.  The currently supported Qwen3.5-2B manifest declares blk.24 as
    // full attention, so requesting an MTP recurrent bank is rejected.  The
    // option keeps that model-dependent decision visible for a future GGUF
    // contract that actually marks an MTP block recurrent.
    bool include_mtp_recurrent = false;
};

// Persistent convolution and gated-delta state for Qwen3.5 recurrent layers.
//
// This component owns only GGML descriptors and one backend buffer.  It does
// not own the Backend passed to the constructor; the backend must outlive the
// cache.  Both CPU and Vulkan default buffer types are supported.  No
// llama_model, llama_context, or llama memory implementation is used.
//
// Each recurrent layer has two F32 tensors:
//
//   conv  [conv_elements,  max_sequence_slots * snapshot_planes]
//   delta [delta_elements, max_sequence_slots * snapshot_planes]
//
// Rows for one sequence slot are contiguous and ordered by newest-first
// snapshot plane.  snapshot_planes is max_draft_tokens + 1.
class RecurrentStateCache {
public:
    RecurrentStateCache(
        const qwen35::Config & model_config,
        ggml_backend_t backend,
        const RecurrentStateOptions & options);
    ~RecurrentStateCache();

    RecurrentStateCache(const RecurrentStateCache &) = delete;
    RecurrentStateCache & operator=(const RecurrentStateCache &) = delete;
    RecurrentStateCache(RecurrentStateCache &&) = delete;
    RecurrentStateCache & operator=(RecurrentStateCache &&) = delete;

    const qwen35::Config & model_config() const noexcept { return model_config_; }
    const RecurrentStateOptions & options() const noexcept { return options_; }
    const std::vector<std::uint32_t> & recurrent_layers() const noexcept {
        return recurrent_layers_;
    }

    std::size_t recurrent_layer_count() const noexcept { return recurrent_layers_.size(); }
    bool has_recurrent_layer(std::uint32_t model_layer) const noexcept;

    std::size_t max_sequence_slots() const noexcept { return options_.max_sequence_slots; }
    std::size_t max_draft_tokens() const noexcept { return options_.max_draft_tokens; }
    std::size_t snapshot_planes() const noexcept { return snapshot_planes_; }

    std::size_t conv_kernel_size() const noexcept { return conv_kernel_size_; }
    std::size_t conv_channels() const noexcept { return conv_channels_; }
    std::size_t conv_elements() const noexcept { return conv_elements_; }

    std::size_t delta_head_size() const noexcept { return delta_head_size_; }
    std::size_t delta_head_count() const noexcept { return delta_head_count_; }
    std::size_t delta_elements() const noexcept { return delta_elements_; }

    ggml_backend_t backend() const noexcept { return backend_; }
    ggml_backend_buffer_t buffer() const noexcept { return buffer_; }
    std::size_t resident_bytes() const noexcept;

    ggml_tensor * conv_storage(std::uint32_t model_layer) const;
    ggml_tensor * delta_storage(std::uint32_t model_layer) const;

    // Slot lifecycle.  initialize_slot() requires an unused slot, zeroes every
    // snapshot plane, and selects plane 0.  clear_slot() preserves ownership;
    // release_slot() zeroes the data and makes the slot reusable.  clear_all()
    // releases every slot.
    void initialize_slot(std::size_t sequence_slot);
    void clear_slot(std::size_t sequence_slot);
    void release_slot(std::size_t sequence_slot);
    void clear_all();
    bool slot_initialized(std::size_t sequence_slot) const;

    // Select the newest-first snapshot which represents a verification result.
    // draft_count may be smaller than the configured maximum.  The returned
    // plane and the per-slot active-plane metadata both use K-a.
    static std::size_t rollback_snapshot_plane(
        std::size_t draft_count,
        std::size_t accepted_drafts);
    std::size_t select_mtp_rollback(
        std::size_t sequence_slot,
        std::size_t draft_count,
        std::size_t accepted_drafts);
    void select_latest(std::size_t sequence_slot);
    std::size_t active_snapshot_plane(std::size_t sequence_slot) const;

    // Persistent row ids for GGML get_rows/set_rows.  Snapshot index vectors
    // are plane-major, matching a contiguous source shaped
    // [state_elements, n_sequences, n_planes].
    std::int32_t row_index(
        std::size_t sequence_slot,
        std::size_t snapshot_plane) const;
    std::vector<std::int32_t> active_row_indices(
        const std::vector<std::size_t> & sequence_slots) const;
    std::vector<std::int32_t> plane_row_indices(
        const std::vector<std::size_t> & sequence_slots,
        std::size_t snapshot_plane) const;
    std::vector<std::int32_t> snapshot_row_indices(
        const std::vector<std::size_t> & sequence_slots,
        std::size_t plane_count) const;

    // Zero-copy scalar-slot views into the persistent buffer.  These tensors
    // are graph-context descriptors only; their storage remains owned here.
    ggml_tensor * view_conv(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        std::size_t sequence_slot,
        std::size_t snapshot_plane) const;
    ggml_tensor * view_delta(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        std::size_t sequence_slot,
        std::size_t snapshot_plane) const;

    // Zero-copy view of adjacent snapshot planes for one sequence. The result
    // is [state_elements, 1, plane_count, 1], matching the packed GDN snapshot
    // tail and CUDA's optional direct-cache-write fusion contract.
    ggml_tensor * view_delta_snapshots(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        std::size_t sequence_slot,
        std::size_t first_snapshot_plane,
        std::size_t plane_count) const;

    // Batched graph reads. row_indices is a 1D I32 tensor containing ids from
    // one of the helpers above.  Results are shaped for ggml_ssm_conv and
    // ggml_gated_delta_net respectively:
    //   conv  [kernel-1, channels, n_sequences]
    //   delta [head_size, head_size, heads, n_sequences]
    ggml_tensor * get_conv(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        ggml_tensor * row_indices) const;
    ggml_tensor * get_delta(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        ggml_tensor * row_indices) const;

    // Batched graph writes. source must be contiguous F32 and contain exactly
    // state_elements * row_count values.  A snapshot source shaped
    // [state_elements, n_sequences, n_planes] can be passed directly together
    // with snapshot_row_indices().  The returned SET_ROWS node must be expanded
    // into the graph by the caller.
    ggml_tensor * set_conv(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        ggml_tensor * source,
        ggml_tensor * row_indices) const;
    ggml_tensor * set_delta(
        ggml_context * graph_context,
        std::uint32_t model_layer,
        ggml_tensor * source,
        ggml_tensor * row_indices) const;

private:
    std::size_t layer_position(std::uint32_t model_layer) const;
    void require_slot(std::size_t sequence_slot) const;
    void require_active_slot(std::size_t sequence_slot) const;
    void require_plane(std::size_t snapshot_plane) const;
    void zero_slot_data(std::size_t sequence_slot);

    ggml_tensor * get_rows(
        ggml_context * graph_context,
        ggml_tensor * storage,
        ggml_tensor * row_indices,
        const char * label) const;
    ggml_tensor * set_rows(
        ggml_context * graph_context,
        ggml_tensor * storage,
        ggml_tensor * source,
        ggml_tensor * row_indices,
        std::size_t row_elements,
        const char * label) const;

    qwen35::Config model_config_;
    RecurrentStateOptions options_;
    ggml_backend_t backend_ = nullptr;  // non-owning
    ggml_context * tensor_context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    std::vector<std::uint32_t> recurrent_layers_;
    std::vector<ggml_tensor *> conv_states_;
    std::vector<ggml_tensor *> delta_states_;

    std::size_t snapshot_planes_ = 0;
    std::size_t total_rows_ = 0;
    std::size_t conv_kernel_size_ = 0;
    std::size_t conv_channels_ = 0;
    std::size_t conv_elements_ = 0;
    std::size_t delta_head_size_ = 0;
    std::size_t delta_head_count_ = 0;
    std::size_t delta_elements_ = 0;

    std::vector<std::uint8_t> slot_initialized_;
    std::vector<std::size_t> active_snapshot_planes_;
};

}  // namespace nanovllm::native
