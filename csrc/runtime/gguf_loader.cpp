#include "runtime/gguf_loader.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nanovllm::native {
namespace {

std::runtime_error load_error(const std::string & path, const std::string & detail) {
    return std::runtime_error("GGUF load failed for '" + path + "': " + detail);
}

}  // namespace

GgufWeights::GgufWeights(std::string path) : path_(std::move(path)) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &tensor_ctx_;
    gguf_ = gguf_init_from_file(path_.c_str(), params);
    if (gguf_ == nullptr || tensor_ctx_ == nullptr) {
        if (tensor_ctx_ != nullptr) {
            ggml_free(tensor_ctx_);
            tensor_ctx_ = nullptr;
        }
        throw load_error(path_, "could not parse metadata and tensor descriptors");
    }

    try {
        const int64_t count = gguf_get_n_tensors(gguf_);
        tensors_.reserve(static_cast<size_t>(count));
        for (int64_t index = 0; index < count; ++index) {
            const char * name = gguf_get_tensor_name(gguf_, index);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx_, name);
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
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
    }
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

    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(tensor_ctx_, buft);
    if (buffer_ == nullptr) {
        throw load_error(path_, "backend weight-buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (ggml_backend_buffer_get_type(buffer_) != buft) {
        throw load_error(path_, "backend returned an unexpected buffer type");
    }

    try {
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
        ggml_backend_buffer_free(buffer_);
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
    return buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(buffer_);
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
