#pragma once

#include "runtime/qwen35_model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanovllm::native {

class GgufWeights;

namespace qwen35 {

// Non-owning pointers to the model-wide tensors. Qwen3.5-2B ties the output
// projection to token_embd.weight; output_head therefore aliases token_embd.
struct GlobalWeights {
    ggml_tensor * token_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output_head = nullptr;

    bool output_head_is_tied() const noexcept {
        return output_head != nullptr && output_head == token_embd;
    }
};

enum class LayerKind {
    Recurrent,
    FullAttention,
    Mtp,
};

// A non-owning typed view of one Qwen3.5 block. Fields which do not apply to
// the layer's kind remain null. An MTP block is a full-attention block, so its
// full-attention fields and nextn fields are both populated.
struct LayerWeights {
    std::uint32_t index = 0;
    LayerKind kind = LayerKind::Recurrent;

    // Common decoder tensors.
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * post_attention_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;

    // Full-attention tensors (also used by the MTP decoder block).
    ggml_tensor * attn_q = nullptr;
    ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;

    // Gated Delta Net / recurrent tensors.
    ggml_tensor * attn_qkv = nullptr;
    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * ssm_conv1d = nullptr;
    ggml_tensor * ssm_dt_bias = nullptr;
    ggml_tensor * ssm_a = nullptr;
    ggml_tensor * ssm_beta = nullptr;
    ggml_tensor * ssm_alpha = nullptr;
    ggml_tensor * ssm_norm = nullptr;
    ggml_tensor * ssm_out = nullptr;

    // Required bundled-MTP tensors.
    ggml_tensor * nextn_eh_proj = nullptr;
    ggml_tensor * nextn_enorm = nullptr;
    ggml_tensor * nextn_hnorm = nullptr;
    ggml_tensor * nextn_shared_head_norm = nullptr;

    // Optional dedicated MTP matrices. They are null when absent from GGUF.
    ggml_tensor * nextn_embed_tokens = nullptr;
    ggml_tensor * nextn_shared_head_head = nullptr;

    // Always-resolved graph inputs. These explicitly alias the global tied
    // matrices when the corresponding dedicated MTP tensor is absent.
    ggml_tensor * mtp_token_embd = nullptr;
    ggml_tensor * mtp_output_head = nullptr;
    ggml_tensor * mtp_output_norm = nullptr;
    bool mtp_token_embd_uses_global = false;
    bool mtp_output_head_uses_global = false;

    bool is_recurrent() const noexcept { return kind == LayerKind::Recurrent; }
    bool is_full_attention() const noexcept {
        return kind == LayerKind::FullAttention || kind == LayerKind::Mtp;
    }
    bool is_mtp() const noexcept { return kind == LayerKind::Mtp; }
};

// Typed, non-owning view over tensors owned by GgufWeights. The source loader
// and its backend buffer must outlive this object. The view owns only its
// Config copy and the vector of pointer records; it never allocates tensor
// storage and never builds a GGML graph.
class Qwen35Weights {
public:
    static constexpr std::size_t kRequiredTensorCount = 335;

    // Validates a fresh manifest directly from the source GGUF, then binds it.
    explicit Qwen35Weights(const GgufWeights & weights);

    // Rechecks the supplied manifest against the source GGUF and binds every
    // one of its 335 required tensors exactly once into the typed view.
    Qwen35Weights(const GgufWeights & weights, const Manifest & manifest);

    const Config & config() const noexcept { return config_; }
    const GlobalWeights & global() const noexcept { return global_; }
    const std::vector<LayerWeights> & layers() const noexcept { return layers_; }
    const LayerWeights & layer(std::size_t index) const;
    const GgufWeights & source() const noexcept { return *source_; }
    Manifest::QuantProfile quant_profile() const noexcept { return quant_profile_; }
    std::size_t bound_tensor_count() const noexcept { return bound_tensor_count_; }

private:
    const GgufWeights * source_ = nullptr;
    Config config_;
    GlobalWeights global_;
    std::vector<LayerWeights> layers_;
    Manifest::QuantProfile quant_profile_ = Manifest::QuantProfile::SupportedMixed;
    std::size_t bound_tensor_count_ = 0;
};

}  // namespace qwen35
}  // namespace nanovllm::native
