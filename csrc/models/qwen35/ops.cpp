#include "models/qwen35/ops.h"

#include <array>
#include <stdexcept>
#include <string>

namespace nanovllm::native::qwen35::ops {
namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw ContractError("Qwen3.5 graph: " + message);
    }
}

void require_tensor(const ggml_tensor * tensor, const char * label) {
    require(tensor != nullptr, std::string(label) + " tensor is null");
}

void name_tensor(ggml_tensor * tensor, const char * name) {
    if (tensor != nullptr && name != nullptr && name[0] != '\0') {
        ggml_set_name(tensor, name);
    }
}

void require_input_embedding(const ggml_tensor * input, const Config & config, const char * label) {
    require_tensor(input, label);
    require(
        input->ne[0] == static_cast<std::int64_t>(config.embedding_length),
        std::string(label) + " first dimension must equal embedding_length");
}

}  // namespace

ggml_tensor * linear(
    ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * input,
    const char * name) {
    require(ctx != nullptr, "linear context is null");
    require_tensor(weight, "linear weight");
    require_tensor(input, "linear input");
    require(weight->ne[0] == input->ne[0], "linear weight/input reduction dimensions differ");
    ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
    name_tensor(result, name);
    return result;
}

ggml_tensor * rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float epsilon,
    const char * name) {
    require(ctx != nullptr, "RMSNorm context is null");
    require_tensor(input, "RMSNorm input");
    require_tensor(weight, "RMSNorm weight");
    require(weight->ne[0] == input->ne[0], "RMSNorm weight/input dimensions differ");
    require(epsilon > 0.0f, "RMSNorm epsilon must be positive");
    ggml_tensor * result = ggml_rms_norm(ctx, input, epsilon);
    result = ggml_mul(ctx, result, weight);
    name_tensor(result, name);
    return result;
}

ggml_tensor * parallel_swiglu_ffn(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * gate_weight,
    ggml_tensor * up_weight,
    ggml_tensor * down_weight,
    const Config & config) {
    require(ctx != nullptr, "FFN context is null");
    require_input_embedding(input, config, "FFN input");
    require_tensor(gate_weight, "FFN gate weight");
    require_tensor(up_weight, "FFN up weight");
    require_tensor(down_weight, "FFN down weight");
    const auto embedding = static_cast<std::int64_t>(config.embedding_length);
    const auto intermediate = static_cast<std::int64_t>(config.feed_forward_length);
    require(
        gate_weight->ne[0] == embedding && gate_weight->ne[1] == intermediate,
        "FFN gate weight shape is invalid");
    require(
        up_weight->ne[0] == embedding && up_weight->ne[1] == intermediate,
        "FFN up weight shape is invalid");
    require(
        down_weight->ne[0] == intermediate && down_weight->ne[1] == embedding,
        "FFN down weight shape is invalid");

    ggml_tensor * gate = linear(ctx, gate_weight, input, "qwen35.ffn_gate");
    ggml_tensor * up = linear(ctx, up_weight, input, "qwen35.ffn_up");
    ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
    name_tensor(activated, "qwen35.ffn_swiglu");
    return linear(ctx, down_weight, activated, "qwen35.ffn_down");
}

QueryGate project_query_and_gate(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * query_gate_weight,
    const Config & config,
    std::int64_t tokens) {
    require(ctx != nullptr, "query/gate context is null");
    require_input_embedding(input, config, "query/gate input");
    require_tensor(query_gate_weight, "query/gate weight");
    require(tokens > 0 && input->ne[1] == tokens, "query/gate token count is inconsistent");

    const auto head_dim = static_cast<std::int64_t>(config.attention_key_length);
    const auto heads = static_cast<std::int64_t>(config.attention_head_count);
    require(
        query_gate_weight->ne[0] == static_cast<std::int64_t>(config.embedding_length) &&
            query_gate_weight->ne[1] == head_dim * heads * 2,
        "joint query/gate weight shape is invalid");

    ggml_tensor * joint = linear(ctx, query_gate_weight, input, "qwen35.query_gate_joint"); //矩阵乘投影
    const size_t element = ggml_element_size(joint);
    QueryGate result;
    result.query = ggml_view_3d(
        ctx,
        joint,
        head_dim,
        heads,
        tokens,
        element * head_dim * 2,
        element * head_dim * 2 * heads,
        0);
    name_tensor(result.query, "qwen35.query_heads");
    ggml_tensor * gate_view = ggml_view_3d(
        ctx,
        joint,
        head_dim,
        heads,
        tokens,
        element * head_dim * 2,
        element * head_dim * 2 * heads,
        element * head_dim);
    result.gate = ggml_cont_2d(ctx, gate_view, head_dim * heads, tokens);
    name_tensor(result.gate, "qwen35.attention_gate");
    return result;
}

ggml_tensor * project_key_or_value(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    const Config & config,
    std::int64_t tokens,
    const char * name) {
    require(ctx != nullptr, "key/value context is null");
    require_input_embedding(input, config, "key/value input");
    require_tensor(weight, "key/value weight");
    require(tokens > 0 && input->ne[1] == tokens, "key/value token count is inconsistent");
    const auto head_dim = static_cast<std::int64_t>(config.attention_key_length);
    const auto kv_heads = static_cast<std::int64_t>(config.attention_head_count_kv);
    require(
        weight->ne[0] == static_cast<std::int64_t>(config.embedding_length) &&
            weight->ne[1] == head_dim * kv_heads,
        "key/value weight shape is invalid");
    ggml_tensor * result = linear(ctx, weight, input);
    result = ggml_reshape_3d(ctx, result, head_dim, kv_heads, tokens);
    name_tensor(result, name);
    return result;
}

ggml_tensor * normalize_attention_heads(
    ggml_context * ctx,
    ggml_tensor * heads,
    ggml_tensor * norm_weight,
    const Config & config,
    const char * name) {
    require_tensor(heads, "attention heads");
    require(
        heads->ne[0] == static_cast<std::int64_t>(config.attention_key_length),
        "attention head width is invalid");
    return rms_norm(
        ctx, heads, norm_weight, config.attention_layer_norm_rms_epsilon, name);
}

ggml_tensor * apply_imrope(
    ggml_context * ctx,
    ggml_tensor * heads,
    ggml_tensor * positions_4d,
    const Config & config,
    const char * name) {
    require(ctx != nullptr, "IMRoPE context is null");
    require_tensor(heads, "IMRoPE heads");
    require_tensor(positions_4d, "IMRoPE positions");
    require(positions_4d->type == GGML_TYPE_I32, "IMRoPE positions must be I32");
    require(
        positions_4d->ne[0] == heads->ne[2] * 4,
        "IMRoPE positions must contain four channels per token");
    std::array<int, GGML_MROPE_SECTIONS> sections{
        config.rope_dimension_sections[0],
        config.rope_dimension_sections[1],
        config.rope_dimension_sections[2],
        config.rope_dimension_sections[3],
    };
    ggml_tensor * result = ggml_rope_multi(
        ctx,
        heads,
        positions_4d,
        nullptr,
        static_cast<int>(config.rope_dimension_count),
        sections.data(),
        GGML_ROPE_TYPE_IMROPE,
        static_cast<int>(config.context_length),
        config.rope_freq_base,
        1.0f,
        0.0f,
        1.0f,
        32.0f,
        1.0f);
    name_tensor(result, name);
    return result;
}

std::vector<std::int32_t> expand_text_positions(
    const std::vector<std::int32_t> & positions) {
    std::vector<std::int32_t> expanded(positions.size() * 4, 0);
    for (size_t index = 0; index < positions.size(); ++index) {
        expanded[index] = positions[index];
        expanded[positions.size() + index] = positions[index];
        expanded[positions.size() * 2 + index] = positions[index];
    }
    return expanded;
}

ggml_tensor * mtp_merge_embedding_and_hidden(
    ggml_context * ctx,
    ggml_tensor * token_embedding,
    ggml_tensor * target_hidden,
    ggml_tensor * embedding_norm,
    ggml_tensor * hidden_norm,
    ggml_tensor * projection,
    const Config & config) {
    require(ctx != nullptr, "MTP merge context is null");
    require_input_embedding(token_embedding, config, "MTP token embedding");
    require_input_embedding(target_hidden, config, "MTP target hidden");
    require(
        token_embedding->ne[1] == target_hidden->ne[1],
        "MTP token embedding and target hidden token counts differ");
    ggml_tensor * embedding = rms_norm(
        ctx,
        token_embedding,
        embedding_norm,
        config.attention_layer_norm_rms_epsilon,
        "qwen35.mtp_embedding_norm");
    ggml_tensor * hidden = rms_norm(
        ctx,
        target_hidden,
        hidden_norm,
        config.attention_layer_norm_rms_epsilon,
        "qwen35.mtp_hidden_norm");
    ggml_tensor * merged = ggml_concat(ctx, embedding, hidden, 0);
    name_tensor(merged, "qwen35.mtp_eh_concat");
    require_tensor(projection, "MTP e/h projection");
    require(
        projection->ne[0] == static_cast<std::int64_t>(config.embedding_length) * 2 &&
            projection->ne[1] == static_cast<std::int64_t>(config.embedding_length),
        "MTP e/h projection shape is invalid");
    return linear(ctx, projection, merged, "qwen35.mtp_eh_projection");
}

}  // namespace nanovllm::native::qwen35::ops
