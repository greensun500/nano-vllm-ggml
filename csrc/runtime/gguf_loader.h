#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nanovllm::native {

// Owns GGUF metadata, tensor descriptors, and one persistent backend weight
// buffer.  It intentionally has no dependency on llama_model or llama_context.
class GgufWeights {
public:
    explicit GgufWeights(std::string path);
    ~GgufWeights();

    GgufWeights(const GgufWeights &) = delete;
    GgufWeights & operator=(const GgufWeights &) = delete;
    GgufWeights(GgufWeights &&) = delete;
    GgufWeights & operator=(GgufWeights &&) = delete;

    // Allocates every GGUF tensor in the backend's persistent default buffer
    // and uploads the file in bounded chunks. Must be called exactly once.
    void load(ggml_backend_t backend, size_t chunk_bytes = 16 * 1024 * 1024);

    ggml_tensor * require_tensor(std::string_view name) const;
    ggml_tensor * find_tensor(std::string_view name) const noexcept;
    bool contains(std::string_view name) const noexcept;

    const std::string & path() const noexcept { return path_; }
    const gguf_context * metadata() const noexcept { return gguf_; }
    ggml_context * tensor_context() const noexcept { return tensor_ctx_; }
    ggml_backend_buffer_t buffer() const noexcept { return buffer_; }
    ggml_backend_buffer_type_t buffer_type() const noexcept;
    size_t file_tensor_bytes() const noexcept { return file_tensor_bytes_; }
    size_t resident_buffer_bytes() const noexcept;
    bool loaded() const noexcept { return buffer_ != nullptr; }
    std::vector<std::string> tensor_names() const;

private:
    std::string path_;
    gguf_context * gguf_ = nullptr;
    ggml_context * tensor_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    size_t file_tensor_bytes_ = 0;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};

}  // namespace nanovllm::native
