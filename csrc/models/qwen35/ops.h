#pragma once

#include "runtime/qwen35_model.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace nanovllm::native::qwen35::ops {

struct QueryGate {
    ggml_tensor * query = nullptr;  // [head_dim, query_heads, tokens]
    ggml_tensor * gate = nullptr;   // [embedding, tokens]
};

ggml_tensor * linear(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    const char * name = nullptr);

ggml_tensor * rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float epsilon,
    const char * name = nullptr);

ggml_tensor * parallel_swiglu_ffn(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * gate_weight,
    ggml_tensor * up_weight,
    ggml_tensor * down_weight,
    const Config & config);

QueryGate project_query_and_gate(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * query_gate_weight,
    const Config & config,
    std::int64_t tokens);

ggml_tensor * project_key_or_value(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    const Config & config,
    std::int64_t tokens,
    const char * name = nullptr);

ggml_tensor * normalize_attention_heads(
    ggml_context * ctx,
    ggml_tensor * heads,
    ggml_tensor * norm_weight,
    const Config & config,
    const char * name = nullptr);

ggml_tensor * apply_imrope(
    ggml_context * ctx,
    ggml_tensor * heads,
    ggml_tensor * positions_4d,
    const Config & config,
    const char * name = nullptr);

// GGML IMRoPE consumes a flat [4 * tokens] position tensor laid out by
// channel, not interleaved by token: [p...][p...][p...][0...].
std::vector<std::int32_t> expand_text_positions(
    const std::vector<std::int32_t> & positions);

ggml_tensor * mtp_merge_embedding_and_hidden(
    ggml_context * ctx,
    ggml_tensor * token_embedding,
    ggml_tensor * target_hidden,
    ggml_tensor * embedding_norm,
    ggml_tensor * hidden_norm,
    ggml_tensor * projection,
    const Config & config);

}  // namespace nanovllm::native::qwen35::ops
