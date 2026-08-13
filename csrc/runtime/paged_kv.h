#pragma once

#include "runtime/qwen35_model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanovllm::native {

class PagedKvCacheError : public std::runtime_error {
public:
    explicit PagedKvCacheError(const std::string & message);
};

enum class PagedKvDomain {
    Target,
    Mtp,
};

// Non-owning descriptors for tensors whose storage belongs to PagedKvCache's
// persistent backend buffer.  They must not be treated as scheduler-allocated
// graph temporaries and become invalid when their cache is destroyed.
struct PagedKvLayer {
    PagedKvDomain domain = PagedKvDomain::Target;
    std::uint32_t model_layer = 0;
    ggml_tensor * key = nullptr;    // [key_width, physical_slots], F32
    ggml_tensor * value = nullptr;  // [value_width, physical_slots], F32
};

// Graph nodes returned by write_and_gather().
//
// key_store/value_store are views of persistent cache tensors with SET_ROWS
// side effects.  key/value are scheduler-owned, transient F32 GET_ROWS
// results.  Reading through the SET_ROWS results establishes the graph edge
// that makes each gather observe the current graph's cache writes.
struct PagedKvGraphIo {
    ggml_tensor * key_store = nullptr;
    ggml_tensor * value_store = nullptr;
    ggml_tensor * key = nullptr;    // [key_width, read_indices]
    ggml_tensor * value = nullptr;  // [value_width, read_indices]
};

// A nano-vLLM-owned persistent Paged-KV allocation for the Qwen3.5 target and
// optional bundled MTP layer.  It uses only GGML tensors/backend buffers; no
// llama_model, llama_context, or llama KV-cache object is involved.
//
// The supplied backend is non-owning and must outlive this object.  CPU and
// Vulkan are both supported by allocating from that backend's default buffer
// type.  Cache data is deliberately F32 in the first correctness-oriented
// implementation; graph gathers are therefore F32 on both backends as well.
class PagedKvCache {
public:
    PagedKvCache(
        ggml_backend_t backend,
        const qwen35::Config & model_config,
        std::size_t block_size,
        std::size_t num_blocks,
        bool enable_mtp);
    ~PagedKvCache();

    PagedKvCache(const PagedKvCache &) = delete;
    PagedKvCache & operator=(const PagedKvCache &) = delete;
    PagedKvCache(PagedKvCache &&) = delete;
    PagedKvCache & operator=(PagedKvCache &&) = delete;

    ggml_backend_t backend() const noexcept { return backend_; }
    ggml_backend_buffer_t buffer() const noexcept { return buffer_; }
    ggml_backend_buffer_type_t buffer_type() const noexcept;
    std::size_t resident_bytes() const noexcept;

    std::size_t block_size() const noexcept { return block_size_; }
    std::size_t num_blocks() const noexcept { return num_blocks_; }
    std::size_t physical_slot_count() const noexcept { return physical_slots_; }
    std::size_t max_sequence_length() const noexcept { return max_sequence_length_; }
    std::int64_t key_width() const noexcept { return key_width_; }
    std::int64_t value_width() const noexcept { return value_width_; }

    const std::vector<PagedKvLayer> & target_layers() const noexcept {
        return target_layers_;
    }
    bool has_mtp_layer() const noexcept { return mtp_layer_.has_value(); }
    const PagedKvLayer & target_layer(std::uint32_t model_layer) const;
    const PagedKvLayer & mtp_layer() const;
    const PagedKvLayer & layer(PagedKvDomain domain, std::uint32_t model_layer) const;

    // Maps one allocator block/offset pair to the flat physical slot ABI used
    // by nano-vLLM's execution plan.
    std::int32_t physical_slot(std::int32_t block_id, std::size_t block_offset) const;

    // Expands logical token positions through one sequence's block table.  A
    // block table may contain padded entries after the requested logical end;
    // required entries must be in range and unique within the sequence.
    void validate_logical_range(
        const std::vector<std::int32_t> & block_table,
        std::size_t logical_start,
        std::size_t token_count) const;
    std::vector<std::int32_t> physical_indices(
        const std::vector<std::int32_t> & block_table,
        std::size_t logical_start,
        std::size_t token_count) const;
    std::vector<std::int32_t> context_indices(
        const std::vector<std::int32_t> & block_table,
        std::size_t context_length) const;

    // Read gathers may intentionally repeat a slot (for example when two
    // sequences share prefix-cache blocks), but every index must be in range.
    void validate_read_indices(const std::vector<std::int32_t> & indices) const;

    // SET_ROWS has undefined behavior for overlapping destination rows.  The
    // runtime must call this once on the concatenated slot_mapping of a whole
    // graph batch, so collisions across sequences are caught too.  A slot may
    // be written again by a later graph (needed after rollback/reuse).
    void validate_write_indices(const std::vector<std::int32_t> & indices) const;

    // Validates nano-vLLM's supplied slot_mapping against the authoritative
    // block table expansion, including the no-duplicate-write requirement.
    void validate_write_mapping(
        const std::vector<std::int32_t> & block_table,
        std::size_t logical_start,
        const std::vector<std::int32_t> & supplied_indices) const;

    // Adds persistent-cache writes and transient gathers to an existing graph
    // context.  current_key/current_value must be contiguous F32 tensors with
    // shapes [head_dim, kv_heads, tokens] (or equivalent 2D flattened rows).
    // write_indices and read_indices are transient I32 graph inputs.  Their
    // host values must be checked with validate_write_indices() and
    // validate_read_indices(), respectively, before upload.
    PagedKvGraphIo write_and_gather(
        ggml_context * graph_ctx,
        const PagedKvLayer & cache_layer,
        ggml_tensor * current_key,
        ggml_tensor * current_value,
        ggml_tensor * write_indices,
        ggml_tensor * read_indices) const;

    // Read-only graph path.  The returned tensors are transient graph tensors;
    // the cache layer remains persistent and externally allocated.
    PagedKvGraphIo gather(
        ggml_context * graph_ctx,
        const PagedKvLayer & cache_layer,
        ggml_tensor * read_indices) const;

    // Explicit lifecycle operations used when the Python block manager frees
    // physical blocks.  Reuse does not require clearing for correctness, but
    // zeroing is available for deterministic diagnostics and data hygiene.
    void clear_blocks(const std::vector<std::int32_t> & block_ids);
    void clear();

private:
    const PagedKvLayer & require_owned_layer(const PagedKvLayer & layer) const;
    void validate_block_table_prefix(
        const std::vector<std::int32_t> & block_table,
        std::size_t required_blocks) const;
    void validate_physical_index(std::int32_t index) const;
    ggml_tensor * flatten_current(
        ggml_context * graph_ctx,
        ggml_tensor * current,
        std::int64_t width,
        std::int64_t token_count,
        const char * label) const;
    void release() noexcept;

    ggml_backend_t backend_ = nullptr;  // non-owning; must outlive this cache
    ggml_context * tensor_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    std::size_t block_size_ = 0;
    std::size_t num_blocks_ = 0;
    std::size_t physical_slots_ = 0;
    std::size_t max_sequence_length_ = 0;
    std::int64_t key_width_ = 0;
    std::int64_t value_width_ = 0;

    std::vector<PagedKvLayer> target_layers_;
    std::optional<PagedKvLayer> mtp_layer_;
};

}  // namespace nanovllm::native
