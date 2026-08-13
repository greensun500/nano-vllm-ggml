#include "runtime/gguf_loader.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ggml-cpu.h"

namespace nanovllm::native {
namespace {

std::runtime_error load_error(const std::string & path, const std::string & detail) {
    return std::runtime_error("GGUF load failed for '" + path + "': " + detail);
}

bool is_cpu_repack_candidate(ggml_backend_t backend, const ggml_tensor * tensor) {
    if (!ggml_backend_is_cpu(backend) || ggml_n_dims(tensor) != 2) {
        return false;
    }

    const int64_t rows = tensor->ne[1];
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            return (ggml_cpu_has_avx2() && rows % 8 == 0) ||
                   (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8() && rows % 4 == 0) ||
                   (ggml_cpu_has_neon() && ggml_cpu_has_dotprod() && rows % 4 == 0);
        case GGML_TYPE_Q6_K:
            return (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8() && rows % 8 == 0) ||
                   (ggml_cpu_has_neon() && ggml_cpu_has_dotprod() && rows % 8 == 0);
        default:
            return false;
    }
}

size_t padded_allocation_size(
    ggml_backend_buffer_type_t buffer_type,
    const ggml_tensor * tensor) {
    const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
    const size_t size = ggml_backend_buft_get_alloc_size(buffer_type, tensor);
    if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("GGML tensor allocation size overflow");
    }
    return ((size + alignment - 1) / alignment) * alignment;
}

void free_buffers(std::vector<ggml_backend_buffer_t> * buffers) noexcept {
    for (ggml_backend_buffer_t buffer : *buffers) {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
    buffers->clear();
}

}  // namespace

GgufWeights::GgufWeights(std::string path) : path_(std::move(path)) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &tensor_ctx_;
    gguf_ = gguf_init_from_file(path_.c_str(), params); //初始化GGUF上下文，读取GGUF文件的元数据和tensor描述符
    if (gguf_ == nullptr || tensor_ctx_ == nullptr) {
        if (tensor_ctx_ != nullptr) {
            ggml_free(tensor_ctx_);
            tensor_ctx_ = nullptr;
        }
        throw load_error(path_, "could not parse metadata and tensor descriptors");
    }

    try {
        const int64_t count = gguf_get_n_tensors(gguf_);    //获取GGUF文件中tensor的数量
        tensors_.reserve(static_cast<size_t>(count));   //预留空间，避免频繁的内存分配
        for (int64_t index = 0; index < count; ++index) {   //遍历每个tensor
            const char * name = gguf_get_tensor_name(gguf_, index);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx_, name);  //根据tensor的名字获取对应的ggml_tensor对象
            if (tensor == nullptr) {
                throw load_error(path_, "GGML descriptor is missing tensor '" + std::string(name) + "'");
            }
            const auto [it, inserted] = tensors_.emplace(name, tensor);
            if (!inserted) {
                throw load_error(path_, "duplicate tensor name '" + std::string(name) + "'");
            }
            (void) it;
            const size_t size = gguf_get_tensor_size(gguf_, index);
            if (size != ggml_nbytes(tensor)) {
                throw load_error(path_, "byte-size mismatch for tensor '" + std::string(name) + "'");
            }
            if (file_tensor_bytes_ > std::numeric_limits<size_t>::max() - size) {
                throw load_error(path_, "tensor byte count overflow");
            }
            file_tensor_bytes_ += size;
        }
    } catch (...) {
        ggml_free(tensor_ctx_);
        tensor_ctx_ = nullptr;
        gguf_free(gguf_);
        gguf_ = nullptr;
        throw;
    }
}

GgufWeights::~GgufWeights() {
    // Tensor data belongs to the backend buffer; free it before the descriptor
    // context and before GGUF metadata.
    free_buffers(&owned_buffers_);
    buffer_ = nullptr;
    if (tensor_ctx_ != nullptr) {
        ggml_free(tensor_ctx_);
    }
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
    }
}

void GgufWeights::load(ggml_backend_t backend, size_t chunk_bytes) {
    if (backend == nullptr) {
        throw std::invalid_argument("GgufWeights::load requires a non-null GGML backend");
    }
    if (buffer_ != nullptr) {
        throw std::logic_error("GgufWeights::load may only be called once");
    }
    if (chunk_bytes == 0) {
        throw std::invalid_argument("GGUF upload chunk size must be positive");
    }

    const ggml_backend_buffer_type_t default_buft = ggml_backend_get_default_buffer_type(backend);
    const ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    const bool use_cpu_repack =
        repack_buft != nullptr && ggml_backend_supports_buft(backend, repack_buft);

    try {
        size_t repack_bytes = 0;
        if (use_cpu_repack) {
            for (const auto & [name, tensor] : tensors_) {
                (void) name;
                if (!is_cpu_repack_candidate(backend, tensor)) {
                    continue;
                }
                const size_t size = padded_allocation_size(repack_buft, tensor);
                if (repack_bytes > std::numeric_limits<size_t>::max() - size) {
                    throw load_error(path_, "CPU repack weight-buffer size overflow");
                }
                repack_bytes += size;
            }
        }

        if (repack_bytes != 0) {
            ggml_backend_buffer_t repack_buffer =
                ggml_backend_buft_alloc_buffer(repack_buft, repack_bytes);
            if (repack_buffer == nullptr) {
                throw load_error(path_, "CPU repack weight-buffer allocation failed");
            }
            owned_buffers_.push_back(repack_buffer);
            ggml_backend_buffer_set_usage(repack_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            size_t offset = 0;
            for (const auto & [name, tensor] : tensors_) {
                (void) name;
                if (!is_cpu_repack_candidate(backend, tensor)) {
                    continue;
                }
                const size_t size = padded_allocation_size(repack_buft, tensor);
                const ggml_status status = ggml_backend_tensor_alloc(
                    repack_buffer,
                    tensor,
                    static_cast<char *>(ggml_backend_buffer_get_base(repack_buffer)) + offset);
                if (status != GGML_STATUS_SUCCESS) {
                    throw load_error(path_, "CPU repack tensor allocation failed");
                }
                offset += size;
            }
        }

        // GGML skips descriptors that already own data, so this allocates one
        // compact default buffer for every non-repacked tensor.
        ggml_backend_buffer_t default_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(tensor_ctx_, default_buft);
        if (default_buffer != nullptr) {
            owned_buffers_.push_back(default_buffer);
            ggml_backend_buffer_set_usage(default_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        if (owned_buffers_.empty()) {
            throw load_error(path_, "backend weight-buffer allocation failed");
        }
        buffer_ = default_buffer != nullptr ? default_buffer : owned_buffers_.front();
    } catch (...) {
        free_buffers(&owned_buffers_);
        buffer_ = nullptr;
        throw;
    }

    try {//此时缓冲区已经分配好了，接下来将GGUF文件中的tensor数据上传到后端缓冲区中
        std::ifstream file(path_, std::ios::binary);
        if (!file) {
            throw load_error(path_, "could not reopen file for tensor upload");
        }
        std::vector<char> staging(std::min(chunk_bytes, std::max<size_t>(file_tensor_bytes_, 1)));
        const uint64_t data_offset = gguf_get_data_offset(gguf_);
        const int64_t count = gguf_get_n_tensors(gguf_);
        for (int64_t index = 0; index < count; ++index) {
            const char * name = gguf_get_tensor_name(gguf_, index);
            ggml_tensor * tensor = require_tensor(name);
            const size_t tensor_size = gguf_get_tensor_size(gguf_, index);
            const uint64_t tensor_offset = data_offset + gguf_get_tensor_offset(gguf_, index);
            file.clear();
            file.seekg(static_cast<std::streamoff>(tensor_offset), std::ios::beg);
            if (!file) {
                throw load_error(path_, "seek failed for tensor '" + std::string(name) + "'");
            }
            if (tensor->buffer != nullptr &&
                ggml_backend_buffer_get_type(tensor->buffer) == repack_buft) {
                std::vector<char> repack_staging(tensor_size);
                file.read(repack_staging.data(), static_cast<std::streamsize>(tensor_size));
                if (file.gcount() != static_cast<std::streamsize>(tensor_size)) {
                    throw load_error(path_, "short read for repacked tensor '" + std::string(name) + "'");
                }
                ggml_backend_tensor_set(tensor, repack_staging.data(), 0, tensor_size);
                continue;
            }
            size_t copied = 0;
            while (copied < tensor_size) {
                const size_t current = std::min(staging.size(), tensor_size - copied);
                file.read(staging.data(), static_cast<std::streamsize>(current));
                if (file.gcount() != static_cast<std::streamsize>(current)) {
                    throw load_error(path_, "short read for tensor '" + std::string(name) + "'");
                }
                ggml_backend_tensor_set(tensor, staging.data(), copied, current);
                copied += current;
            }
        }
        ggml_backend_synchronize(backend);
    } catch (...) {
        free_buffers(&owned_buffers_);
        buffer_ = nullptr;
        throw;
    }
}

ggml_tensor * GgufWeights::require_tensor(std::string_view name) const {
    ggml_tensor * tensor = find_tensor(name);
    if (tensor == nullptr) {
        throw std::invalid_argument("required GGUF tensor is missing: " + std::string(name));
    }
    return tensor;
}

ggml_tensor * GgufWeights::find_tensor(std::string_view name) const noexcept {
    const auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : it->second;
}

bool GgufWeights::contains(std::string_view name) const noexcept {
    return find_tensor(name) != nullptr;
}

ggml_backend_buffer_type_t GgufWeights::buffer_type() const noexcept {
    return buffer_ == nullptr ? nullptr : ggml_backend_buffer_get_type(buffer_);
}

size_t GgufWeights::resident_buffer_bytes() const noexcept {
    size_t result = 0;
    for (ggml_backend_buffer_t buffer : owned_buffers_) {
        const size_t size = ggml_backend_buffer_get_size(buffer);
        if (result > std::numeric_limits<size_t>::max() - size) {
            return std::numeric_limits<size_t>::max();
        }
        result += size;
    }
    return result;
}

std::vector<std::string> GgufWeights::tensor_names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto & [name, tensor] : tensors_) {
        (void) tensor;
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace nanovllm::native
