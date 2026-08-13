#include "runtime/paged_kv.h"

#include "runtime/backend.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace nanovllm::native {
namespace {

std::string cache_error_message(const std::string & detail) {
    return "Qwen3.5 Paged-KV cache: " + detail;
}

std::string layer_label(PagedKvDomain domain, std::uint32_t model_layer) {
    return std::string(domain == PagedKvDomain::Target ? "target" : "mtp") +
           " layer " + std::to_string(model_layer);
}

void require_context(ggml_context * context, const char * operation) {
    if (context == nullptr) {
        throw PagedKvCacheError(cache_error_message(std::string(operation) + ": graph context is null"));
    }
}

void require_tensor(const ggml_tensor * tensor, const char * label) {
    if (tensor == nullptr) {
        throw PagedKvCacheError(cache_error_message(std::string(label) + " tensor is null"));
    }
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char * label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw PagedKvCacheError(cache_error_message(std::string(label) + " overflows size_t"));
    }
    return lhs * rhs;
}

std::int64_t checked_dimension_product(
    std::uint32_t lhs,
    std::uint32_t rhs,
    const char * label) {
    const std::uint64_t value = static_cast<std::uint64_t>(lhs) * rhs;
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw PagedKvCacheError(cache_error_message(std::string(label) + " is invalid"));
    }
    return static_cast<std::int64_t>(value);
}

void set_name(ggml_tensor * tensor, const std::string & name) {
    if (tensor != nullptr) {
        ggml_set_name(tensor, name.c_str());
    }
}

}  // namespace

PagedKvCacheError::PagedKvCacheError(const std::string & message) : std::runtime_error(message) {}

PagedKvCache::PagedKvCache(//初始化pagedkv缓存
    ggml_backend_t backend,
    const qwen35::Config & model_config,
    std::size_t block_size,
    std::size_t num_blocks,
    bool enable_mtp)
    : backend_(backend),
      block_size_(block_size),
      num_blocks_(num_blocks),
      max_sequence_length_(model_config.context_length),
      key_width_(checked_dimension_product(
          model_config.attention_key_length,
          model_config.attention_head_count_kv,
          "key width")),
      value_width_(checked_dimension_product(
          model_config.attention_value_length,
          model_config.attention_head_count_kv,
          "value width")) {
    if (backend_ == nullptr) {
        throw PagedKvCacheError(cache_error_message("backend handle is null"));
    }
    if (block_size_ == 0) {
        throw PagedKvCacheError(cache_error_message("block_size must be positive"));
    }
    if (num_blocks_ == 0) {
        throw PagedKvCacheError(cache_error_message("num_blocks must be positive"));
    }
    if (max_sequence_length_ == 0) {
        throw PagedKvCacheError(cache_error_message("model context length must be positive"));
    }

    physical_slots_ = checked_product(num_blocks_, block_size_, "physical slot count");
    // GGML GET_ROWS consumes I32 indices on Vulkan, so the flat slot ABI must
    // remain representable by a non-negative int32 value.
    if (physical_slots_ > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1) {
        throw PagedKvCacheError(cache_error_message("physical slot count exceeds the I32 graph-index ABI"));
    }

    std::vector<std::uint32_t> target_layer_ids;
    for (std::uint32_t layer = 0; layer < model_config.main_layers; ++layer) {
        if (model_config.is_full_attention_layer(layer)) {//把attention的层都放到target_layer_ids中
            target_layer_ids.push_back(layer);
        }
    }
    if (target_layer_ids.size() != 6) {//这个是针对Qwen3.5模型的，Qwen3.5模型有6个full attention层,如果后续拓展其他模型，这里应该不需要
        std::ostringstream message;
        message << "expected exactly 6 target full-attention layers, got "
                << target_layer_ids.size();
        throw PagedKvCacheError(cache_error_message(message.str()));
    }

    if (enable_mtp && model_config.nextn_predict_layers != 1) {
        throw PagedKvCacheError(cache_error_message(
            "the first native MTP cache requires exactly one bundled prediction layer"));
    }

    const std::size_t tensor_count = (target_layer_ids.size() + (enable_mtp ? 1 : 0)) * 2;//kv两种cache，所以是两倍
    const std::size_t metadata_bytes = checked_product(
        tensor_count, ggml_tensor_overhead(), "persistent tensor metadata size");//计算metadata所需的字节数
    ggml_init_params params{        //ggml的初始化参数
        /* .mem_size = */ metadata_bytes,   //总容量
        /* .mem_buffer = */ nullptr,    //如果为nullptr，ggml会自己分配内存
        /* .no_alloc = */ true, //ggml会为metadata(tensor shape\type访问很快，先分配内存，tensor的内存后面才会进行分配，
    };
    tensor_ctx_ = ggml_init(params);    //创建空的ggml上下文（只是能放下12个tensor的metadata，tensor的内存还没有分配）
    if (tensor_ctx_ == nullptr) {
        throw PagedKvCacheError(cache_error_message("could not allocate persistent tensor metadata"));
    }

    try {
        target_layers_.reserve(target_layer_ids.size());
        for (const std::uint32_t model_layer : target_layer_ids) {
            PagedKvLayer layer;
            layer.domain = PagedKvDomain::Target;
            layer.model_layer = model_layer;
            layer.key = ggml_new_tensor_2d( //只是写入了tensor的metadata，tensor的data部分还只是指针
                tensor_ctx_,
                GGML_TYPE_F32,
                key_width_,
                static_cast<std::int64_t>(physical_slots_));
            layer.value = ggml_new_tensor_2d(
                tensor_ctx_,
                GGML_TYPE_F32,
                value_width_,
                static_cast<std::int64_t>(physical_slots_));
            set_name(layer.key, "qwen35.target.k_cache.layer." + std::to_string(model_layer));
            set_name(layer.value, "qwen35.target.v_cache.layer." + std::to_string(model_layer));
            target_layers_.push_back(layer);
        }

        if (enable_mtp) {
            const std::uint32_t model_layer = model_config.main_layers;
            if (!model_config.is_mtp_layer(model_layer)) {
                throw PagedKvCacheError(cache_error_message("derived MTP layer is outside the model"));
            }
            PagedKvLayer layer;
            layer.domain = PagedKvDomain::Mtp;
            layer.model_layer = model_layer;
            layer.key = ggml_new_tensor_2d(
                tensor_ctx_,
                GGML_TYPE_F32,
                key_width_,
                static_cast<std::int64_t>(physical_slots_));
            layer.value = ggml_new_tensor_2d(
                tensor_ctx_,
                GGML_TYPE_F32,
                value_width_,
                static_cast<std::int64_t>(physical_slots_));
            set_name(layer.key, "qwen35.mtp.k_cache.layer." + std::to_string(model_layer));
            set_name(layer.value, "qwen35.mtp.v_cache.layer." + std::to_string(model_layer));
            mtp_layer_ = layer;
        }

        const ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend_);
        if (buffer_type == nullptr) {
            throw PagedKvCacheError(cache_error_message("backend has no default buffer type"));
        }
        assert_buffer_type_compatible(backend_, buffer_type, "Paged-KV allocation");
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(tensor_ctx_, buffer_type);//遍历所有tensor，确认所有tensor需要多大的空间，然后分配连续的空间，并把data的指针赋值过来
        if (buffer_ == nullptr) {
            throw PagedKvCacheError(cache_error_message("persistent backend-buffer allocation failed"));
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        assert_buffer_on_backend(buffer_, backend_, "Paged-KV buffer");
        for (const PagedKvLayer & layer : target_layers_) {
            assert_tensor_on_backend(layer.key, backend_, "target key cache");
            assert_tensor_on_backend(layer.value, backend_, "target value cache");
        }
        if (mtp_layer_) {
            assert_tensor_on_backend(mtp_layer_->key, backend_, "MTP key cache");
            assert_tensor_on_backend(mtp_layer_->value, backend_, "MTP value cache");
        }
        ggml_backend_buffer_clear(buffer_, 0);
        ggml_backend_synchronize(backend_);
    } catch (...) {
        release();
        throw;
    }
}

PagedKvCache::~PagedKvCache() {
    release();
}

void PagedKvCache::release() noexcept {
    if (buffer_ != nullptr) {
        // Runtime member ordering must keep backend_ alive until this point.
        if (backend_ != nullptr) {
            ggml_backend_synchronize(backend_);
        }
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (tensor_ctx_ != nullptr) {
        ggml_free(tensor_ctx_);
        tensor_ctx_ = nullptr;
    }
}

ggml_backend_buffer_type_t PagedKvCache::buffer_type() const noexcept {
    return buffer_ == nullptr ? nullptr : ggml_backend_buffer_get_type(buffer_);
}

std::size_t PagedKvCache::resident_bytes() const noexcept {
    return buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(buffer_);
}

const PagedKvLayer & PagedKvCache::target_layer(std::uint32_t model_layer) const {
    const auto it = std::find_if(
        target_layers_.begin(), target_layers_.end(), [&](const PagedKvLayer & layer) {
            return layer.model_layer == model_layer;
        });
    if (it == target_layers_.end()) {
        throw PagedKvCacheError(cache_error_message(
            "model layer " + std::to_string(model_layer) +
            " is not a target full-attention cache layer"));
    }
    return *it;
}

const PagedKvLayer & PagedKvCache::mtp_layer() const {
    if (!mtp_layer_) {
        throw PagedKvCacheError(cache_error_message("MTP cache is disabled"));
    }
    return *mtp_layer_;
}

const PagedKvLayer & PagedKvCache::layer(
    PagedKvDomain domain,
    std::uint32_t model_layer) const {
    if (domain == PagedKvDomain::Target) {
        return target_layer(model_layer);
    }
    const PagedKvLayer & result = mtp_layer();
    if (result.model_layer != model_layer) {
        throw PagedKvCacheError(cache_error_message(
            "MTP cache belongs to model layer " + std::to_string(result.model_layer) +
            ", not " + std::to_string(model_layer)));
    }
    return result;
}

const PagedKvLayer & PagedKvCache::require_owned_layer(const PagedKvLayer & layer) const {
    const PagedKvLayer & owned = this->layer(layer.domain, layer.model_layer);
    if (owned.key != layer.key || owned.value != layer.value) {
        throw PagedKvCacheError(cache_error_message(
            layer_label(layer.domain, layer.model_layer) +
            " descriptor does not belong to this cache"));
    }
    return owned;
}

std::int32_t PagedKvCache::physical_slot(
    std::int32_t block_id,
    std::size_t block_offset) const {
    if (block_id < 0 || static_cast<std::size_t>(block_id) >= num_blocks_) {
        std::ostringstream message;
        message << "block id " << block_id << " is outside [0, " << num_blocks_ << ')';
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
    if (block_offset >= block_size_) {
        std::ostringstream message;
        message << "block offset " << block_offset << " is outside [0, " << block_size_ << ')';
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
    const std::size_t slot = static_cast<std::size_t>(block_id) * block_size_ + block_offset;
    if (slot >= physical_slots_ ||
        slot > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw PagedKvCacheError(cache_error_message("derived physical slot is not representable"));
    }
    return static_cast<std::int32_t>(slot);
}

void PagedKvCache::validate_block_table_prefix(
    const std::vector<std::int32_t> & block_table,
    std::size_t required_blocks) const {
    if (required_blocks > block_table.size()) {
        std::ostringstream message;
        message << "block table has " << block_table.size() << " entries, but "
                << required_blocks << " are required";
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
    std::unordered_set<std::int32_t> seen;
    seen.reserve(required_blocks);
    for (std::size_t index = 0; index < required_blocks; ++index) {
        const std::int32_t block_id = block_table[index];
        if (block_id < 0 || static_cast<std::size_t>(block_id) >= num_blocks_) {
            std::ostringstream message;
            message << "block table entry " << index << " has invalid block id " << block_id;
            throw PagedKvCacheError(cache_error_message(message.str()));
        }
        if (!seen.insert(block_id).second) {
            std::ostringstream message;
            message << "block table maps multiple logical blocks to physical block " << block_id;
            throw PagedKvCacheError(cache_error_message(message.str()));
        }
    }
}

void PagedKvCache::validate_logical_range(
    const std::vector<std::int32_t> & block_table,
    std::size_t logical_start,
    std::size_t token_count) const {
    if (logical_start > max_sequence_length_) {
        throw PagedKvCacheError(cache_error_message("logical_start exceeds model context length"));
    }
    if (token_count > max_sequence_length_ - logical_start) {
        throw PagedKvCacheError(cache_error_message("logical token range exceeds model context length"));
    }
    const std::size_t logical_end = logical_start + token_count;
    const std::size_t required_blocks =
        logical_end == 0 ? 0 : 1 + (logical_end - 1) / block_size_;
    validate_block_table_prefix(block_table, required_blocks);
}

std::vector<std::int32_t> PagedKvCache::physical_indices(
    const std::vector<std::int32_t> & block_table,
    std::size_t logical_start,
    std::size_t token_count) const {
    validate_logical_range(block_table, logical_start, token_count);
    const std::size_t logical_end = logical_start + token_count;

    std::vector<std::int32_t> result;
    result.reserve(token_count);
    for (std::size_t position = logical_start; position < logical_end; ++position) {
        const std::size_t logical_block = position / block_size_;
        result.push_back(physical_slot(block_table[logical_block], position % block_size_));
    }
    return result;
}

std::vector<std::int32_t> PagedKvCache::context_indices(
    const std::vector<std::int32_t> & block_table,
    std::size_t context_length) const {
    return physical_indices(block_table, 0, context_length);
}

void PagedKvCache::validate_physical_index(std::int32_t index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= physical_slots_) {
        std::ostringstream message;
        message << "physical slot " << index << " is outside [0, " << physical_slots_ << ')';
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
}

void PagedKvCache::validate_read_indices(
    const std::vector<std::int32_t> & indices) const {
    for (const std::int32_t index : indices) {
        validate_physical_index(index);
    }
}

void PagedKvCache::validate_write_indices(
    const std::vector<std::int32_t> & indices) const {
    validate_read_indices(indices);
    std::unordered_set<std::int32_t> seen;
    seen.reserve(indices.size());
    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
        const std::int32_t index = indices[offset];
        if (!seen.insert(index).second) {
            std::ostringstream message;
            message << "write slot " << index << " is repeated in one graph batch"
                    << " (second occurrence at offset " << offset << ')';
            throw PagedKvCacheError(cache_error_message(message.str()));
        }
    }
}

void PagedKvCache::validate_write_mapping(
    const std::vector<std::int32_t> & block_table,
    std::size_t logical_start,
    const std::vector<std::int32_t> & supplied_indices) const {
    const std::vector<std::int32_t> expected =
        physical_indices(block_table, logical_start, supplied_indices.size());
    validate_write_indices(supplied_indices);
    if (expected != supplied_indices) {
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), supplied_indices.begin());
        const std::size_t offset = static_cast<std::size_t>(mismatch.first - expected.begin());
        std::ostringstream message;
        message << "slot_mapping differs from the block table at token offset " << offset
                << ": got " << supplied_indices[offset] << ", expected " << expected[offset];
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
}

ggml_tensor * PagedKvCache::flatten_current(
    ggml_context * graph_ctx,
    ggml_tensor * current,
    std::int64_t width,
    std::int64_t token_count,
    const char * label) const {
    require_tensor(current, label);
    if (current->type != GGML_TYPE_F32) {
        throw PagedKvCacheError(cache_error_message(
            std::string(label) + " must be F32 for the first native cache format"));
    }
    const bool flattened_2d =
        current->ne[0] == width && current->ne[1] == token_count &&
        current->ne[2] == 1 && current->ne[3] == 1;
    const bool head_shaped_3d =
        current->ne[0] * current->ne[1] == width &&
        current->ne[2] == token_count && current->ne[3] == 1;
    if (!flattened_2d && !head_shaped_3d) {
        std::ostringstream message;
        message << label << " shape must flatten to [" << width << ", " << token_count << ']';
        throw PagedKvCacheError(cache_error_message(message.str()));
    }
    if (!ggml_is_contiguous(current)) {
        current = ggml_cont(graph_ctx, current);
    }
    return ggml_reshape_2d(graph_ctx, current, width, token_count);
}

PagedKvGraphIo PagedKvCache::write_and_gather(
    ggml_context * graph_ctx,
    const PagedKvLayer & cache_layer,
    ggml_tensor * current_key,
    ggml_tensor * current_value,
    ggml_tensor * write_indices,
    ggml_tensor * read_indices) const {
    require_context(graph_ctx, "write_and_gather");
    const PagedKvLayer & owned = require_owned_layer(cache_layer);
    require_tensor(write_indices, "write indices");
    require_tensor(read_indices, "read indices");
    if (write_indices->type != GGML_TYPE_I32 || ggml_n_dims(write_indices) != 1) {
        throw PagedKvCacheError(cache_error_message("write indices must be a one-dimensional I32 tensor"));
    }
    if (read_indices->type != GGML_TYPE_I32 || ggml_n_dims(read_indices) != 1) {
        throw PagedKvCacheError(cache_error_message("read indices must be a one-dimensional I32 tensor"));
    }
    if (read_indices->ne[0] <= 0) {
        throw PagedKvCacheError(cache_error_message("cache gather requires at least one index"));
    }
    const std::int64_t token_count = write_indices->ne[0];
    if (token_count <= 0) {
        throw PagedKvCacheError(cache_error_message("cache write requires at least one token"));
    }
    ggml_tensor * key_rows =
        flatten_current(graph_ctx, current_key, key_width_, token_count, "current key");
    ggml_tensor * value_rows =
        flatten_current(graph_ctx, current_value, value_width_, token_count, "current value");

    PagedKvGraphIo result;
    result.key_store = ggml_set_rows(graph_ctx, owned.key, key_rows, write_indices);
    result.value_store = ggml_set_rows(graph_ctx, owned.value, value_rows, write_indices);
    set_name(result.key_store, "qwen35.k_cache_store.layer." + std::to_string(owned.model_layer));
    set_name(result.value_store, "qwen35.v_cache_store.layer." + std::to_string(owned.model_layer));

    // Gather from the SET_ROWS views, rather than the original persistent
    // leaves, so GGML's graph dependency makes the read observe this write.
    result.key = ggml_get_rows(graph_ctx, result.key_store, read_indices);
    result.value = ggml_get_rows(graph_ctx, result.value_store, read_indices);
    set_name(result.key, "qwen35.k_cache_gather.layer." + std::to_string(owned.model_layer));
    set_name(result.value, "qwen35.v_cache_gather.layer." + std::to_string(owned.model_layer));
    return result;
}

PagedKvGraphIo PagedKvCache::gather(
    ggml_context * graph_ctx,
    const PagedKvLayer & cache_layer,
    ggml_tensor * read_indices) const {
    require_context(graph_ctx, "gather");
    const PagedKvLayer & owned = require_owned_layer(cache_layer);
    require_tensor(read_indices, "read indices");
    if (read_indices->type != GGML_TYPE_I32 || ggml_n_dims(read_indices) != 1) {
        throw PagedKvCacheError(cache_error_message("read indices must be a one-dimensional I32 tensor"));
    }
    if (read_indices->ne[0] <= 0) {
        throw PagedKvCacheError(cache_error_message("cache gather requires at least one index"));
    }
    PagedKvGraphIo result;
    result.key = ggml_get_rows(graph_ctx, owned.key, read_indices);
    result.value = ggml_get_rows(graph_ctx, owned.value, read_indices);
    set_name(result.key, "qwen35.k_cache_gather.layer." + std::to_string(owned.model_layer));
    set_name(result.value, "qwen35.v_cache_gather.layer." + std::to_string(owned.model_layer));
    return result;
}

void PagedKvCache::clear_blocks(const std::vector<std::int32_t> & block_ids) {
    if (buffer_ == nullptr) {
        throw PagedKvCacheError(cache_error_message("cache buffer is not allocated"));
    }
    std::unordered_set<std::int32_t> seen;
    seen.reserve(block_ids.size());
    for (const std::int32_t block_id : block_ids) {
        if (block_id < 0 || static_cast<std::size_t>(block_id) >= num_blocks_) {
            throw PagedKvCacheError(cache_error_message(
                "cannot clear invalid block id " + std::to_string(block_id)));
        }
        if (!seen.insert(block_id).second) {
            throw PagedKvCacheError(cache_error_message(
                "clear block list repeats block id " + std::to_string(block_id)));
        }
    }
    if (block_ids.empty()) {
        return;
    }

    ggml_backend_synchronize(backend_);
    auto clear_tensor_blocks = [&](ggml_tensor * tensor) {
        const std::size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        const std::size_t block_bytes = checked_product(block_size_, row_bytes, "block byte size");
        for (const std::int32_t block_id : block_ids) {
            const std::size_t offset =
                checked_product(static_cast<std::size_t>(block_id), block_bytes, "block byte offset");
            ggml_backend_tensor_memset(tensor, 0, offset, block_bytes);
        }
    };
    for (const PagedKvLayer & layer : target_layers_) {
        clear_tensor_blocks(layer.key);
        clear_tensor_blocks(layer.value);
    }
    if (mtp_layer_) {
        clear_tensor_blocks(mtp_layer_->key);
        clear_tensor_blocks(mtp_layer_->value);
    }
    ggml_backend_synchronize(backend_);
}

void PagedKvCache::clear() {
    if (buffer_ == nullptr) {
        throw PagedKvCacheError(cache_error_message("cache buffer is not allocated"));
    }
    ggml_backend_synchronize(backend_);
    ggml_backend_buffer_clear(buffer_, 0);
    ggml_backend_synchronize(backend_);
}

}  // namespace nanovllm::native
