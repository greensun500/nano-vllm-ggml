#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_vocab;

namespace nanovllm::native {

// A vocabulary-only adapter around the pinned upstream llama.cpp tokenizer.
// It reads GGUF metadata, never creates llama_model/llama_context, and owns no
// model weights or backend buffers.
class GgufTokenizer {
public:
    explicit GgufTokenizer(const std::string & gguf_path);
    ~GgufTokenizer();

    GgufTokenizer(const GgufTokenizer &) = delete;
    GgufTokenizer & operator=(const GgufTokenizer &) = delete;
    GgufTokenizer(GgufTokenizer &&) noexcept;
    GgufTokenizer & operator=(GgufTokenizer &&) noexcept;

    std::vector<std::int32_t> tokenize(const std::string & text) const;
    std::string detokenize(const std::vector<std::int32_t> & token_ids) const;
    std::vector<std::int32_t> eog_token_ids() const;
    std::uint32_t vocabulary_size() const;

private:
    std::unique_ptr<llama_vocab> vocab_;
};

}  // namespace nanovllm::native
