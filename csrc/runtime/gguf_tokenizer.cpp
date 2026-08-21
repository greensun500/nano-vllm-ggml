#include "runtime/gguf_tokenizer.h"

#include "llama-arch.h"
#include "llama-model-loader.h"
#include "llama-vocab.h"

#include <llama.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace nanovllm::native {
namespace {

[[noreturn]] void fail(const std::string & detail) {
    throw std::runtime_error("native GGUF tokenizer: " + detail);
}

// llama_model_loader and llama_vocab report a full metadata dump at INFO
// level.  Tokenizer initialization is an internal native-runtime detail, so
// avoid turning every CLI startup into a multi-page llama.cpp log.  The
// upstream logger is process-global; serialize only this short initialization
// window and restore the exact callback afterwards.
std::mutex upstream_logger_mutex;

void discard_upstream_log(ggml_log_level, const char *, void *) {}

class ScopedUpstreamTokenizerLogSilencer {
public:
    ScopedUpstreamTokenizerLogSilencer() : lock_(upstream_logger_mutex) {
        llama_log_get(&previous_callback_, &previous_user_data_);
        llama_log_set(discard_upstream_log, nullptr);
    }

    ~ScopedUpstreamTokenizerLogSilencer() {
        llama_log_set(previous_callback_, previous_user_data_);
    }

    ScopedUpstreamTokenizerLogSilencer(const ScopedUpstreamTokenizerLogSilencer &) = delete;
    ScopedUpstreamTokenizerLogSilencer & operator=(const ScopedUpstreamTokenizerLogSilencer &) = delete;

private:
    std::unique_lock<std::mutex> lock_;
    ggml_log_callback previous_callback_ = nullptr;
    void * previous_user_data_ = nullptr;
};

}  // namespace

GgufTokenizer::GgufTokenizer(const std::string & gguf_path) {
    if (gguf_path.empty()) {
        fail("GGUF path is empty");
    }
    try {
        ScopedUpstreamTokenizerLogSilencer silence_logs;
        std::vector<std::string> splits;
        // The upstream loader owns only temporary GGUF metadata descriptors in
        // this use.  It does not allocate backend buffers or construct a
        // llama_model/context; llama_vocab copies the tokenizer state it needs.
        llama_model_loader loader(
            nullptr,
            nullptr,
            nullptr,
            gguf_path,
            splits,
            nullptr,
            false,
            false,
            false,
            true,
            nullptr,
            nullptr);
        auto vocab = std::make_unique<llama_vocab>();
        vocab->load(loader, LLM_KV(loader.get_arch()));
        vocab_ = std::move(vocab);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

GgufTokenizer::~GgufTokenizer() = default;
GgufTokenizer::GgufTokenizer(GgufTokenizer &&) noexcept = default;
GgufTokenizer & GgufTokenizer::operator=(GgufTokenizer &&) noexcept = default;

std::vector<std::int32_t> GgufTokenizer::tokenize(const std::string & text) const {
    if (vocab_ == nullptr) {
        fail("tokenizer is not initialized");
    }
    const std::vector<llama_token> tokens = vocab_->tokenize(
        text,
        true,   // honor GGUF add_bos/add_eos metadata
        true);  // recognize GGUF-declared special tokens in chat prompts
    std::vector<std::int32_t> result;
    result.reserve(tokens.size());
    for (const llama_token token : tokens) {
        result.push_back(static_cast<std::int32_t>(token));
    }
    return result;
}

std::string GgufTokenizer::detokenize(const std::vector<std::int32_t> & token_ids) const {
    if (vocab_ == nullptr) {
        fail("tokenizer is not initialized");
    }
    std::vector<llama_token> tokens;
    tokens.reserve(token_ids.size());
    const std::uint32_t vocabulary = vocab_->n_tokens();
    for (const std::int32_t token : token_ids) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocabulary) {
            fail("token id is outside the GGUF vocabulary");
        }
        tokens.push_back(static_cast<llama_token>(token));
    }
    // Preserve special token text, matching HF decode(skip_special_tokens=False).
    return vocab_->detokenize(tokens, true);
}

std::vector<std::int32_t> GgufTokenizer::eog_token_ids() const {
    if (vocab_ == nullptr) {
        fail("tokenizer is not initialized");
    }
    std::vector<std::int32_t> result;
    const std::uint32_t vocabulary = vocab_->n_tokens();
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
        if (vocab_->is_eog(static_cast<llama_token>(token))) {
            result.push_back(static_cast<std::int32_t>(token));
        }
    }
    if (result.empty()) {
        fail("GGUF declares no EOG token");
    }
    return result;
}

std::uint32_t GgufTokenizer::vocabulary_size() const {
    if (vocab_ == nullptr) {
        fail("tokenizer is not initialized");
    }
    return vocab_->n_tokens();
}

}  // namespace nanovllm::native
