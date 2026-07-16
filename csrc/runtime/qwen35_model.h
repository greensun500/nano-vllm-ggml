#pragma once

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nanovllm::native::qwen35 {

// The native runtime intentionally supports one exact architecture contract
// first: the dense Qwen3.5-2B trunk with its bundled one-layer MTP head.  All
// values are read from GGUF; the validator rejects a file whose metadata does
// not describe this profile instead of silently building a wrong graph.
struct Config {
    std::string architecture;

    std::uint32_t block_count = 0;          // main decoder + bundled MTP blocks
    std::uint32_t nextn_predict_layers = 0;
    std::uint32_t main_layers = 0;          // block_count - nextn_predict_layers
    std::uint32_t context_length = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t feed_forward_length = 0;
    std::uint64_t vocabulary_size = 0;

    std::uint32_t attention_head_count = 0;
    std::uint32_t attention_head_count_kv = 0;
    std::uint32_t attention_key_length = 0;
    std::uint32_t attention_value_length = 0;
    float attention_layer_norm_rms_epsilon = 0.0f;

    std::array<std::int32_t, 4> rope_dimension_sections{};
    std::uint32_t rope_dimension_count = 0;
    float rope_freq_base = 0.0f;

    std::uint32_t ssm_conv_kernel = 0;
    std::uint32_t ssm_state_size = 0;
    std::uint32_t ssm_group_count = 0;
    std::uint32_t ssm_time_step_rank = 0;
    std::uint32_t ssm_inner_size = 0;
    std::uint32_t full_attention_interval = 0;

    bool is_main_layer(std::uint32_t layer) const noexcept;
    bool is_recurrent_layer(std::uint32_t layer) const noexcept;
    bool is_full_attention_layer(std::uint32_t layer) const noexcept;
    bool is_mtp_layer(std::uint32_t layer) const noexcept;
};

class ContractError : public std::runtime_error {
public:
    explicit ContractError(const std::string & message);
};

// GGML stores dimensions in ne[] order.  For example, a PyTorch [out, in]
// linear weight is represented here as shape {in, out}.
struct TensorSpec {
    enum class Role {
        TokenEmbedding,
        LinearWeight,
        FloatParameter,
    };

    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<ggml_type> allowed_types;
    Role role = Role::LinearWeight;
};

// A validated descriptor contains everything a later mmap/backend uploader
// needs to bind the contract name to the corresponding GGUF payload.
struct TensorDescriptor {
    TensorSpec spec;
    std::int64_t tensor_index = -1;
    ggml_type type = GGML_TYPE_COUNT;
    std::size_t file_offset = 0;  // relative to the GGUF tensor-data section
    std::size_t byte_size = 0;
};

struct Manifest {
    enum class QuantProfile {
        // Every tensor type is executable by the native runtime, but it does
        // not have the exact type assignment of the requested Q4_0 artifact.
        SupportedMixed,
        // token_embd=Q6_K, every linear weight=Q4_0, and every norm/bias/
        // recurrent scalar/conv tensor=F32.
        Q4_0WithQ6KEmbedding,
    };

    Config config;
    std::vector<TensorDescriptor> tensors;
    QuantProfile quant_profile = QuantProfile::SupportedMixed;

    const TensorDescriptor * find(std::string_view name) const noexcept;
    const TensorDescriptor & require(std::string_view name) const;
    bool strict_current_profile() const noexcept {
        return quant_profile == QuantProfile::Q4_0WithQ6KEmbedding;
    }
};

// These APIs only depend on the public GGUF/GGML surface.  A weight loader can
// keep its own contexts alive and call validate() before allocating any device
// buffers.  inspect_file() is the owning convenience path for tools/tests.
Config read_config(const gguf_context * gguf);
std::vector<TensorSpec> required_tensor_specs(const Config & config);
Manifest validate(const gguf_context * gguf, ggml_context * tensor_context);
Manifest inspect_file(std::string_view path);

}  // namespace nanovllm::native::qwen35
