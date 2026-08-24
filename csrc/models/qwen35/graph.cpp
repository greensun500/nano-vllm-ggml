#include "models/qwen35/graph.h"

#include "models/qwen35/ops.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nanovllm::native::qwen35 {
namespace {

constexpr std::int64_t kExpectedRecurrentLayers = 18;
constexpr std::int64_t kExpectedAttentionLayers = 6;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw ContractError("Qwen3.5 token graph: " + message);
    }
}

void require_shape(
    const ggml_tensor * tensor,
    std::initializer_list<std::int64_t> shape,
    const char * label) {
    require(tensor != nullptr, std::string(label) + " tensor is null");
    std::size_t dimension = 0;
    for (const std::int64_t extent : shape) {
        require(
            tensor->ne[dimension] == extent,
            std::string(label) + " tensor has the wrong shape at dimension " +
                std::to_string(dimension));
        ++dimension;
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        require(
            tensor->ne[dimension] == 1,
            std::string(label) + " tensor has an unexpected trailing dimension " +
                std::to_string(dimension));
    }
}

void set_name(ggml_tensor * tensor, const std::string & name) {
    if (tensor != nullptr) {
        ggml_set_name(tensor, name.c_str());
    }
}

void add_critical(TokenGraph & result, ggml_tensor * tensor) {
    require(tensor != nullptr, "critical compute node is null");
    result.critical_compute_nodes.push_back(tensor);
}

void add_critical(TargetChunkGraph & result, ggml_tensor * tensor) {
    require(tensor != nullptr, "critical compute node is null");
    result.critical_compute_nodes.push_back(tensor);
}

void add_critical(MtpKvUpdateGraph & result, ggml_tensor * tensor) {
    require(tensor != nullptr, "critical compute node is null");
    result.critical_compute_nodes.push_back(tensor);
}

ggml_type causal_mask_type(AttentionImplementation implementation) {
    return implementation == AttentionImplementation::Flash
        ? GGML_TYPE_F16
        : GGML_TYPE_F32;
}

void require_causal_mask(
    const ggml_tensor * mask,
    std::initializer_list<std::int64_t> shape,
    AttentionImplementation implementation,
    const std::string & prefix) {
    if (mask == nullptr) {
        return;
    }
    require_shape(mask, shape, (prefix + ".causal_mask").c_str());
    require(
        mask->type == causal_mask_type(implementation),
        prefix + " causal mask type does not match the attention implementation");
}

struct GraphInputs {
    ggml_tensor * token = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * write_slot = nullptr;
    ggml_tensor * read_slots = nullptr;
    ggml_tensor * causal_mask = nullptr;
};

GraphInputs make_inputs(
    ggml_context * ctx,
    std::size_t n_kv,
    bool use_causal_mask,
    AttentionImplementation attention_implementation) {
    require(ctx != nullptr, "GGML context is null");
    require(n_kv > 0, "attention read length must be positive");
    GraphInputs inputs;
    inputs.token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    inputs.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    inputs.write_slot = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    inputs.read_slots = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, static_cast<std::int64_t>(n_kv));
    if (use_causal_mask) {
        inputs.causal_mask = ggml_new_tensor_2d(
            ctx,
            causal_mask_type(attention_implementation),
            static_cast<std::int64_t>(n_kv),
            1);
    }
    ggml_set_input(inputs.token);
    ggml_set_input(inputs.positions);
    ggml_set_input(inputs.write_slot);
    ggml_set_input(inputs.read_slots);
    if (inputs.causal_mask != nullptr) {
        ggml_set_input(inputs.causal_mask);
    }
    set_name(inputs.token, "qwen35.input.token");
    set_name(inputs.positions, "qwen35.input.positions");
    set_name(inputs.write_slot, "qwen35.input.write_slot");
    set_name(inputs.read_slots, "qwen35.input.read_slots");
    set_name(inputs.causal_mask, "qwen35.input.causal_mask");
    return inputs;
}

struct ChunkGraphInputs {
    ggml_tensor * tokens = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * write_slots = nullptr;
    ggml_tensor * read_slots = nullptr;
    ggml_tensor * causal_mask = nullptr;
};

ChunkGraphInputs make_chunk_inputs(
    ggml_context * ctx,
    std::size_t n_tokens,
    std::size_t n_kv,
    bool force_causal_mask,
    AttentionImplementation attention_implementation) {
    require(ctx != nullptr, "chunk GGML context is null");
    require(n_tokens > 0, "chunk token count must be positive");
    require(n_kv >= n_tokens, "chunk attention context is shorter than its token count");

    const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
    const std::int64_t kv_count = static_cast<std::int64_t>(n_kv);
    ChunkGraphInputs inputs;
    inputs.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    inputs.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count * 4);
    inputs.write_slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    inputs.read_slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv_count);
    if (n_tokens > 1 || force_causal_mask) {
        inputs.causal_mask = ggml_new_tensor_2d(
            ctx, causal_mask_type(attention_implementation), kv_count, token_count);
    }
    //把tensor标记为graph input，说明其值不是内部计算的，是外部必须输入的
    ggml_set_input(inputs.tokens);
    ggml_set_input(inputs.positions);
    ggml_set_input(inputs.write_slots);
    ggml_set_input(inputs.read_slots);
    if (inputs.causal_mask != nullptr) {
        ggml_set_input(inputs.causal_mask);
    }
    set_name(inputs.tokens, "qwen35.target_chunk.tokens");
    set_name(inputs.positions, "qwen35.target_chunk.positions");
    set_name(inputs.write_slots, "qwen35.target_chunk.write_slots");
    set_name(inputs.read_slots, "qwen35.target_chunk.read_slots");
    set_name(inputs.causal_mask, "qwen35.target_chunk.causal_mask");
    return inputs;
}

struct StoredKeyValue {
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
};

StoredKeyValue store_attention_key_value(
    ggml_context * ctx,
    ggml_tensor * input,
    const LayerWeights & layer,
    const AttentionCacheView & cache,
    ggml_tensor * positions,
    ggml_tensor * write_slots,
    const Config & config,
    std::int64_t n_tokens,
    const std::string & prefix) {
    require(ctx != nullptr, prefix + " GGML context is null");
    require(layer.is_full_attention(), prefix + " is not a full-attention layer");
    require(n_tokens > 0, prefix + " token count must be positive");
    require(input != nullptr, prefix + " input tensor is null");
    require(
        input->ne[0] == static_cast<std::int64_t>(config.embedding_length) &&
            input->ne[1] == n_tokens,
        prefix + " input shape is invalid");
    require_shape(
        positions,
        {n_tokens * 4},
        (prefix + ".positions").c_str());
    require(positions->type == GGML_TYPE_I32, prefix + " positions must be I32");
    require_shape(
        write_slots,
        {n_tokens},
        (prefix + ".write_slots").c_str());
    require(write_slots->type == GGML_TYPE_I32, prefix + " write slots must be I32");

    const std::int64_t key_width = static_cast<std::int64_t>(   //config.attention_key_length * config.attention_head_count_kv;
        config.attention_key_length * config.attention_head_count_kv);
    const std::int64_t value_width = static_cast<std::int64_t>(
        config.attention_value_length * config.attention_head_count_kv);
    require_shape(
        cache.key,
        {key_width, cache.key != nullptr ? cache.key->ne[1] : 0},
        (prefix + ".key_cache").c_str());
    require_shape(
        cache.value,
        {value_width, cache.value != nullptr ? cache.value->ne[1] : 0},
        (prefix + ".value_cache").c_str());
    require(cache.key->type == GGML_TYPE_F32, prefix + " key cache must be F32");
    require(cache.value->type == GGML_TYPE_F32, prefix + " value cache must be F32");
    require(cache.key->ne[1] == cache.value->ne[1], prefix + " cache sizes differ");

    ggml_tensor * key = ops::project_key_or_value(  //投影计算key
        ctx, input, layer.attn_k, config, n_tokens, (prefix + ".key").c_str());
    ggml_tensor * value = ops::project_key_or_value(
        ctx, input, layer.attn_v, config, n_tokens, (prefix + ".value").c_str());
    key = ops::normalize_attention_heads(
        ctx, key, layer.attn_k_norm, config, (prefix + ".key_norm").c_str());
    key = ops::apply_imrope(    //对key应用IMRoPE位置编码
        ctx, key, positions, config, (prefix + ".key_rope").c_str());

    key = ggml_cont_2d(ctx, key, key_width, n_tokens);
    value = ggml_cont_2d(ctx, value, value_width, n_tokens);    //进行复制和重排，保证key和value在内存中是连续的，便于后续写入pagedkv

    set_name(key, prefix + ".key_contiguous");
    set_name(value, prefix + ".value_contiguous");

    StoredKeyValue stored;
    stored.key = ggml_set_rows(ctx, cache.key, key, write_slots);
    stored.value = ggml_set_rows(ctx, cache.value, value, write_slots);
    set_name(stored.key, prefix + ".key_store");
    set_name(stored.value, prefix + ".value_store");
    return stored;
}

ggml_tensor * build_attention(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TokenGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const AttentionCacheView & cache,
    const GraphInputs & inputs,
    const Config & config,
    std::size_t n_kv,
    AttentionImplementation attention_implementation,
    const std::string & prefix) {
    require(layer.is_full_attention(), prefix + " is not a full-attention layer");
    ops::QueryGate query_gate =
        ops::project_query_and_gate(ctx, input, layer.attn_q, config, 1);
    query_gate.query = ops::normalize_attention_heads(
        ctx, query_gate.query, layer.attn_q_norm, config,
        (prefix + ".query_norm").c_str());
    query_gate.query = ops::apply_imrope(
        ctx, query_gate.query, inputs.positions, config,
        (prefix + ".query_rope").c_str());

    const StoredKeyValue stored = store_attention_key_value(
        ctx,
        input,
        layer,
        cache,
        inputs.positions,
        inputs.write_slot,
        config,
        1,
        prefix);

    const float scale = 1.0f /
        std::sqrt(static_cast<float>(config.attention_key_length));
    require_causal_mask(
        inputs.causal_mask,
        {static_cast<std::int64_t>(n_kv), 1},
        attention_implementation,
        prefix);
    // SET_ROWS returns a view of the persistent destination and establishes an
    // explicit dependency for the subsequent GET_ROWS. The current token is
    // therefore visible to its own causal attention without relying on graph
    // insertion order as an implicit memory barrier.
    ggml_tensor * gathered_key = ggml_get_rows(ctx, stored.key, inputs.read_slots);
    ggml_tensor * gathered_value = ggml_get_rows(ctx, stored.value, inputs.read_slots);
    ggml_tensor * attended = nullptr;
        gathered_key = ggml_reshape_3d(
            ctx,
            gathered_key,
            config.attention_key_length,
            config.attention_head_count_kv,
            static_cast<std::int64_t>(n_kv));
        gathered_value = ggml_reshape_3d(
            ctx,
            gathered_value,
            config.attention_value_length,
            config.attention_head_count_kv,
            static_cast<std::int64_t>(n_kv));

        // Match llama.cpp's non-Flash MHA layout. GQA broadcasting is handled by
        // GGML because 8 query heads are an integer multiple of 2 KV heads.
        ggml_tensor * query = ggml_permute(ctx, query_gate.query, 0, 2, 1, 3);
        ggml_tensor * keys = ggml_permute(ctx, gathered_key, 0, 2, 1, 3);
        ggml_tensor * values = ggml_permute(ctx, gathered_value, 0, 2, 1, 3);

        if (attention_implementation == AttentionImplementation::Flash) {
        attended = ggml_flash_attn_ext(
            ctx, query, keys, values, inputs.causal_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        set_name(attended, prefix + ".flash_attention");
        add_critical(result, attended);
        attended = ggml_reshape_2d(
            ctx,
            attended,
            static_cast<std::int64_t>(
                config.attention_value_length * config.attention_head_count),
            1);
        } else {
        ggml_tensor * scores = ggml_mul_mat(ctx, keys, query);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        set_name(scores, prefix + ".scores");
        add_critical(result, scores);
        scores = ggml_soft_max_ext(ctx, scores, inputs.causal_mask, scale, 0.0f);
        set_name(scores, prefix + ".probabilities");

        values = ggml_cont(ctx, ggml_transpose(ctx, values));
        attended = ggml_mul_mat(ctx, values, scores);
        set_name(attended, prefix + ".weighted_values");
        add_critical(result, attended);
        attended = ggml_permute(ctx, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(
            ctx,
            attended,
            static_cast<std::int64_t>(
                config.attention_value_length * config.attention_head_count),
            1);
        }
    attended = ggml_mul(ctx, attended, ggml_sigmoid(ctx, query_gate.gate));
    ggml_tensor * output = ops::linear(
        ctx, layer.attn_output, attended, (prefix + ".output").c_str());
    add_critical(result, output);

    // Build the store branches before downstream residual work. They also
    // remain dependencies of gathered_key/value, but explicitly expanding
    // them makes persistent side effects auditable in graph dumps.
    ggml_build_forward_expand(graph, stored.key);
    ggml_build_forward_expand(graph, stored.value);
    return output;
}

ggml_tensor * build_chunk_attention(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TargetChunkGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const AttentionCacheView & cache,   //持久保存的kvcache
    const ChunkGraphInputs & inputs,
    const Config & config,
    std::size_t n_tokens,
    std::size_t n_kv,
    AttentionImplementation attention_implementation,
    const std::string & prefix) {
    require(layer.is_full_attention(), prefix + " is not a full-attention layer");
    require(n_tokens > 0, prefix + " token count must be positive");
    require(n_kv >= n_tokens, prefix + " attention context is too short");
    const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
    const std::int64_t kv_count = static_cast<std::int64_t>(n_kv);
    require_shape(
        input,
        {static_cast<std::int64_t>(config.embedding_length), token_count},
        (prefix + ".input").c_str());
    require_causal_mask(
        inputs.causal_mask,
        {kv_count, token_count},
        attention_implementation,
        prefix);

    ops::QueryGate query_gate = ops::project_query_and_gate(
        ctx, input, layer.attn_q, config, token_count);
    query_gate.query = ops::normalize_attention_heads(
        ctx,
        query_gate.query,
        layer.attn_q_norm,
        config,
        (prefix + ".query_norm").c_str());
    query_gate.query = ops::apply_imrope(
        ctx,
        query_gate.query,
        inputs.positions,
        config,
        (prefix + ".query_rope").c_str());

    const StoredKeyValue stored = store_attention_key_value(    //把当前inpiut投影出来的k和v写入到cache中
        ctx,
        input,
        layer,
        cache,
        inputs.positions,
        inputs.write_slots,
        config,
        token_count,
        prefix);

    const float scale = 1.0f /
        std::sqrt(static_cast<float>(config.attention_key_length));
    ggml_tensor * gathered_key = ggml_get_rows(ctx, stored.key, inputs.read_slots);
    ggml_tensor * gathered_value = ggml_get_rows(ctx, stored.value, inputs.read_slots);
    ggml_tensor * attended = nullptr;
        gathered_key = ggml_reshape_3d(
            ctx,
            gathered_key,
            config.attention_key_length,
            config.attention_head_count_kv,
            kv_count);
        gathered_value = ggml_reshape_3d(
            ctx,
            gathered_value,
            config.attention_value_length,
            config.attention_head_count_kv,
            kv_count);

        ggml_tensor * query = ggml_permute(ctx, query_gate.query, 0, 2, 1, 3);
        ggml_tensor * keys = ggml_permute(ctx, gathered_key, 0, 2, 1, 3);
        ggml_tensor * values = ggml_permute(ctx, gathered_value, 0, 2, 1, 3);

        if (attention_implementation == AttentionImplementation::Flash) {
        attended = ggml_flash_attn_ext(
            ctx, query, keys, values, inputs.causal_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        set_name(attended, prefix + ".flash_attention");
        add_critical(result, attended);
        attended = ggml_reshape_2d(
            ctx,
            attended,
            static_cast<std::int64_t>(
                config.attention_value_length * config.attention_head_count),
            token_count);
        } else {
        ggml_tensor * scores = ggml_mul_mat(ctx, keys, query);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        set_name(scores, prefix + ".scores");
        add_critical(result, scores);
        require_shape(
            scores,
            {kv_count,
             token_count,
             static_cast<std::int64_t>(config.attention_head_count)},
            (prefix + ".scores").c_str());
        scores = ggml_soft_max_ext(
            ctx, scores, inputs.causal_mask, scale, 0.0f);
        set_name(scores, prefix + ".probabilities");

        values = ggml_cont(ctx, ggml_transpose(ctx, values));
        attended = ggml_mul_mat(ctx, values, scores);
        set_name(attended, prefix + ".weighted_values");
        add_critical(result, attended);
        attended = ggml_permute(ctx, attended, 0, 2, 1, 3);
        attended = ggml_cont_2d(
            ctx,
            attended,
            static_cast<std::int64_t>(
                config.attention_value_length * config.attention_head_count),
            token_count);
        }
    attended = ggml_mul(ctx, attended, ggml_sigmoid(ctx, query_gate.gate));
    ggml_tensor * output = ops::linear(
        ctx, layer.attn_output, attended, (prefix + ".output").c_str());
    add_critical(result, output);

    ggml_build_forward_expand(graph, stored.key);
    ggml_build_forward_expand(graph, stored.value);
    return output;
}

ggml_tensor * build_recurrent(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TokenGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const RecurrentStateView & persistent,
    const Config & config,
    const std::string & prefix) {
    require(layer.is_recurrent(), prefix + " is not a recurrent layer");
    const std::int64_t kernel = config.ssm_conv_kernel;
    const std::int64_t state_width = config.ssm_state_size;
    const std::int64_t key_heads = config.ssm_group_count;
    const std::int64_t value_heads = config.ssm_time_step_rank;
    const std::int64_t inner = config.ssm_inner_size;
    const std::int64_t value_width = inner / value_heads;
    const std::int64_t qkv_width =
        state_width * key_heads * 2 + value_width * value_heads;

    require(kernel == 4, "only the Qwen3.5 convolution kernel size 4 is supported");
    require(value_width == state_width, "GDN key/value state widths differ");
    require(key_heads == value_heads, "first native GDN path requires equal key/value heads");
    require_shape(
        persistent.convolution,
        {kernel - 1, qkv_width},
        (prefix + ".convolution_state").c_str());
    require_shape(
        persistent.delta,
        {state_width, state_width, value_heads, 1},
        (prefix + ".delta_state").c_str());
    require(
        persistent.convolution->type == GGML_TYPE_F32,
        prefix + " convolution state must be F32");
    require(persistent.delta->type == GGML_TYPE_F32, prefix + " delta state must be F32");
    ggml_tensor * convolution_destination = persistent.convolution_output != nullptr
        ? persistent.convolution_output
        : persistent.convolution;
    ggml_tensor * delta_destination =
        persistent.delta_output != nullptr ? persistent.delta_output : persistent.delta;
    require_shape(
        convolution_destination,
        {kernel - 1, qkv_width},
        (prefix + ".convolution_state_destination").c_str());
    require_shape(
        delta_destination,
        {state_width, state_width, value_heads, 1},
        (prefix + ".delta_state_destination").c_str());
    require(
        convolution_destination->type == GGML_TYPE_F32,
        prefix + " convolution destination must be F32");
    require(
        delta_destination->type == GGML_TYPE_F32,
        prefix + " delta destination must be F32");

    ggml_tensor * qkv = ops::linear(ctx, layer.attn_qkv, input,
                                    (prefix + ".qkv").c_str());
    qkv = ggml_reshape_3d(ctx, qkv, qkv_width, 1, 1);
    ggml_tensor * z = ops::linear(
        ctx, layer.attn_gate, input, (prefix + ".z").c_str());

    ggml_tensor * beta = ops::linear(
        ctx, layer.ssm_beta, input, (prefix + ".beta_linear").c_str());
    beta = ggml_reshape_4d(ctx, beta, 1, value_heads, 1, 1);
    beta = ggml_sigmoid(ctx, beta);

    ggml_tensor * alpha = ops::linear(
        ctx, layer.ssm_alpha, input, (prefix + ".alpha_linear").c_str());
    alpha = ggml_reshape_3d(ctx, alpha, value_heads, 1, 1);
    alpha = ggml_softplus(ctx, ggml_add(ctx, alpha, layer.ssm_dt_bias));
    ggml_tensor * gate = ggml_mul(ctx, alpha, layer.ssm_a);
    gate = ggml_reshape_4d(ctx, gate, 1, value_heads, 1, 1);

    ggml_tensor * old_conv = ggml_reshape_3d(
        ctx, persistent.convolution, kernel - 1, qkv_width, 1);
    ggml_tensor * qkv_rows = ggml_transpose(ctx, qkv);  // [1, 6144, 1]
    ggml_tensor * conv_input = ggml_concat(ctx, old_conv, qkv_rows, 0);
    set_name(conv_input, prefix + ".convolution_input");

    ggml_tensor * new_conv = ggml_view_2d(
        ctx,
        conv_input,
        kernel - 1,
        qkv_width,
        conv_input->nb[1],
        ggml_row_size(conv_input->type, 1));
    ggml_tensor * conv_update = ggml_cpy(ctx, new_conv, convolution_destination);
    set_name(conv_update, prefix + ".convolution_state_update");
    ggml_build_forward_expand(graph, conv_update);

    ggml_tensor * mixed = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, layer.ssm_conv1d));
    set_name(mixed, prefix + ".convolution_output");
    add_critical(result, mixed);

    const std::size_t scalar_bytes = ggml_element_size(mixed);
    const std::size_t row_stride = ggml_row_size(mixed->type, qkv_width);
    ggml_tensor * query = ggml_view_4d(
        ctx, mixed, state_width, key_heads, 1, 1,
        ggml_row_size(mixed->type, state_width), row_stride, row_stride, 0);
    ggml_tensor * key = ggml_view_4d(
        ctx, mixed, state_width, key_heads, 1, 1,
        ggml_row_size(mixed->type, state_width), row_stride, row_stride,
        static_cast<std::size_t>(state_width * key_heads) * scalar_bytes);
    ggml_tensor * value = ggml_view_4d(
        ctx, mixed, value_width, value_heads, 1, 1,
        ggml_row_size(mixed->type, value_width), row_stride, row_stride,
        static_cast<std::size_t>(state_width * key_heads * 2) * scalar_bytes);
    query = ggml_l2_norm(ctx, query, config.attention_layer_norm_rms_epsilon);
    key = ggml_l2_norm(ctx, key, config.attention_layer_norm_rms_epsilon);

    ggml_tensor * gdn = ggml_gated_delta_net(
        ctx, query, key, value, gate, beta, persistent.delta, 1);
    set_name(gdn, prefix + ".gated_delta_net");
    add_critical(result, gdn);
    const std::int64_t output_elements = value_width * value_heads;
    ggml_tensor * output = ggml_view_4d(
        ctx,
        gdn,
        value_width,
        value_heads,
        1,
        1,
        ggml_row_size(gdn->type, value_width),
        ggml_row_size(gdn->type, value_width * value_heads),
        ggml_row_size(gdn->type, value_width * value_heads),
        0);
    ggml_tensor * new_delta = ggml_view_4d(
        ctx,
        gdn,
        state_width,
        state_width,
        value_heads,
        1,
        ggml_row_size(gdn->type, state_width),
        ggml_row_size(gdn->type, state_width * state_width),
        ggml_row_size(gdn->type, state_width * state_width * value_heads),
        static_cast<std::size_t>(output_elements) * ggml_element_size(gdn));
    ggml_tensor * delta_update = ggml_cpy(ctx, new_delta, delta_destination);
    set_name(delta_update, prefix + ".delta_state_update");
    ggml_build_forward_expand(graph, delta_update);

    z = ggml_reshape_4d(ctx, z, value_width, value_heads, 1, 1);
    output = ops::rms_norm(
        ctx,
        output,
        layer.ssm_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".state_norm").c_str());
    output = ggml_mul(ctx, output, ggml_silu(ctx, z));
    output = ggml_reshape_2d(ctx, output, inner, 1);
    output = ops::linear(
        ctx, layer.ssm_out, output, (prefix + ".output").c_str());
    add_critical(result, output);
    return output;
}

ggml_tensor * build_chunk_recurrent(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TargetChunkGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const RecurrentChunkStateView & persistent,
    const Config & config,
    std::size_t n_tokens,
    std::size_t snapshot_count,
    const std::string & prefix) {
    require(layer.is_recurrent(), prefix + " is not a recurrent layer");
    require(n_tokens > 0, prefix + " token count must be positive");
    require(
        snapshot_count > 0 && snapshot_count <= n_tokens,
        prefix + " snapshot count must be in [1, n_tokens]");
//获取模型参数
    const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);   //本次计算多少token
    const std::int64_t snapshots = static_cast<std::int64_t>(snapshot_count);   //mtp的校验token数量
    const std::int64_t kernel = config.ssm_conv_kernel; //卷积核大小
    const std::int64_t state_width = config.ssm_state_size;
    const std::int64_t key_heads = config.ssm_group_count;
    const std::int64_t value_heads = config.ssm_time_step_rank;
    const std::int64_t inner = config.ssm_inner_size;
    const std::int64_t value_width = inner / value_heads;
    const std::int64_t qkv_width =
        state_width * key_heads * 2 + value_width * value_heads;

    require(kernel == 4, "only the Qwen3.5 convolution kernel size 4 is supported");
    require(value_width == state_width, "GDN key/value state widths differ");
    require(key_heads == value_heads, "Qwen3.5 GDN requires equal key/value heads");
    require_shape(
        input,
        {static_cast<std::int64_t>(config.embedding_length), token_count},
        (prefix + ".input").c_str());
    require_shape(
        persistent.convolution,
        {kernel - 1, qkv_width},
        (prefix + ".convolution_state").c_str());
    require_shape(
        persistent.delta,
        {state_width, state_width, value_heads, 1},
        (prefix + ".delta_state").c_str());
    require(
        persistent.convolution->type == GGML_TYPE_F32,
        prefix + " convolution state must be F32");
    require(
        persistent.delta->type == GGML_TYPE_F32,
        prefix + " delta state must be F32");
    require(
        persistent.convolution_outputs.size() == snapshot_count,
        prefix + " convolution snapshot destination count differs from the graph request");
    require(
        persistent.delta_outputs.size() == snapshot_count,
        prefix + " delta snapshot destination count differs from the graph request");
    for (std::size_t snapshot = 0; snapshot < snapshot_count; ++snapshot) {
        const std::string snapshot_prefix =
            prefix + ".snapshot." + std::to_string(snapshot);
        require_shape(
            persistent.convolution_outputs[snapshot],
            {kernel - 1, qkv_width},
            (snapshot_prefix + ".convolution_destination").c_str());
        require(
            persistent.convolution_outputs[snapshot]->type == GGML_TYPE_F32,
            snapshot_prefix + " convolution destination must be F32");
        require_shape(
            persistent.delta_outputs[snapshot],
            {state_width, state_width, value_heads, 1},
            (snapshot_prefix + ".delta_destination").c_str());
        require(
            persistent.delta_outputs[snapshot]->type == GGML_TYPE_F32,
            snapshot_prefix + " delta destination must be F32");
    }
//开始手写graph
//按照线性注意力，第一步是把输入投影到qkv空间，形成一个qkv_like矩阵
    ggml_tensor * qkv = ops::linear(
        ctx, layer.attn_qkv, input, (prefix + ".qkv").c_str());//实际就是input与权重矩阵相乘，得到qkv矩阵[6144,T]
    qkv = ggml_reshape_3d(ctx, qkv, qkv_width, token_count, 1);//reshape成[6144,T,1]，方便后续操作

    //z、beta、alpha、gate都是GDN的参数，都是通过线性变换得到的
    ggml_tensor * z = ops::linear(
        ctx, layer.attn_gate, input, (prefix + ".z").c_str());//z：[2048,T]

    ggml_tensor * beta = ops::linear(
        ctx, layer.ssm_beta, input, (prefix + ".beta_linear").c_str());
    beta = ggml_reshape_4d(ctx, beta, 1, value_heads, token_count, 1);//[1,16,T,1]
    beta = ggml_sigmoid(ctx, beta); //范围变成[0,1]

    ggml_tensor * alpha = ops::linear(
        ctx, layer.ssm_alpha, input, (prefix + ".alpha_linear").c_str());
    alpha = ggml_reshape_3d(ctx, alpha, value_heads, token_count, 1);   //16，T,1
    alpha = ggml_softplus(ctx, ggml_add(ctx, alpha, layer.ssm_dt_bias));

    ggml_tensor * gate = ggml_mul(ctx, alpha, layer.ssm_a);
    gate = ggml_reshape_4d(ctx, gate, 1, value_heads, token_count, 1);

    ggml_tensor * old_conv = ggml_reshape_3d(
        ctx, persistent.convolution, kernel - 1, qkv_width, 1);//从[3,6144]变成[3,6144,1]
    ggml_tensor * qkv_rows = ggml_transpose(ctx, qkv);//[6144,T,1] -> [T,6144,1]，方便取
    ggml_tensor * conv_input = ggml_concat(ctx, old_conv, qkv_rows, 0);//矩阵拼接，conv_input: [3 + T, 6144, 1]，过去3个token的qkv和当前chunk的token的qkv拼接在一起，形成一个新的矩阵
    set_name(conv_input, prefix + ".convolution_input");

    for (std::size_t snapshot = 0; snapshot < snapshot_count; ++snapshot) //更新conv_state
    {
        const std::int64_t window_start =
            token_count - static_cast<std::int64_t>(snapshot);
        ggml_tensor * new_conv = ggml_view_2d(
            ctx,
            conv_input,
            kernel - 1,
            qkv_width,
            conv_input->nb[1],
            ggml_row_size(conv_input->type, window_start));
        ggml_tensor * update = ggml_cpy(
            ctx, new_conv, persistent.convolution_outputs[snapshot]);
        set_name(
            update,
            prefix + ".convolution_state_update." + std::to_string(snapshot));
        ggml_build_forward_expand(graph, update);
    }

    ggml_tensor * mixed = ggml_silu(
        ctx, ggml_ssm_conv(ctx, conv_input, layer.ssm_conv1d));
    set_name(mixed, prefix + ".convolution_output");
    add_critical(result, mixed);
    require_shape(
        mixed,
        {qkv_width, token_count},
        (prefix + ".convolution_output").c_str());

    const std::size_t scalar_bytes = ggml_element_size(mixed);
    const std::size_t head_stride = ggml_row_size(mixed->type, state_width);
    const std::size_t token_stride = ggml_row_size(mixed->type, qkv_width);
    const std::size_t sequence_stride =
        token_stride * static_cast<std::size_t>(token_count);
    ggml_tensor * query = ggml_view_4d(     //从ssm之后的qkv-like矩阵中取出query部分，形成一个4维张量[64,16,T,1]
        ctx,
        mixed,
        state_width,
        key_heads,
        token_count,
        1,
        head_stride,
        token_stride,
        sequence_stride,
        0);
    ggml_tensor * key = ggml_view_4d(
        ctx,
        mixed,
        state_width,
        key_heads,
        token_count,
        1,
        head_stride,
        token_stride,
        sequence_stride,
        static_cast<std::size_t>(state_width * key_heads) * scalar_bytes);
    ggml_tensor * value = ggml_view_4d(
        ctx,
        mixed,
        value_width,
        value_heads,
        token_count,
        1,
        ggml_row_size(mixed->type, value_width),
        token_stride,
        sequence_stride,
        static_cast<std::size_t>(state_width * key_heads * 2) * scalar_bytes);

        
    query = ggml_l2_norm(ctx, query, config.attention_layer_norm_rms_epsilon);
    key = ggml_l2_norm(ctx, key, config.attention_layer_norm_rms_epsilon);

    ggml_tensor * gdn = ggml_gated_delta_net(
        ctx, query, key, value, gate, beta, persistent.delta, snapshots);
    set_name(gdn, prefix + ".gated_delta_net");
    add_critical(result, gdn);
    const std::int64_t output_elements_per_token = value_width * value_heads;
    const std::int64_t attention_output_elements =
        output_elements_per_token * token_count;
    const std::int64_t state_elements = state_width * state_width * value_heads;
    ggml_tensor * output = ggml_view_4d(
        ctx,
        gdn,
        value_width,
        value_heads,
        token_count,
        1,
        ggml_row_size(gdn->type, value_width),
        ggml_row_size(gdn->type, output_elements_per_token),
        ggml_row_size(gdn->type, attention_output_elements),
        0);
    for (std::size_t snapshot = 0; snapshot < snapshot_count; ++snapshot) {
            const std::int64_t state_offset = attention_output_elements +
                static_cast<std::int64_t>(snapshot) * state_elements;
            ggml_tensor * new_delta = ggml_view_4d(
                ctx,
                gdn,
                state_width,
                state_width,
                value_heads,
                1,
                ggml_row_size(gdn->type, state_width),
                ggml_row_size(gdn->type, state_width * state_width),
                ggml_row_size(gdn->type, state_elements),
                static_cast<std::size_t>(state_offset) * ggml_element_size(gdn));
            ggml_tensor * update = ggml_cpy(
                ctx, new_delta, persistent.delta_outputs[snapshot]);
            set_name(
                update,
                prefix + ".delta_state_update." + std::to_string(snapshot));
            ggml_build_forward_expand(graph, update);
    }

    z = ggml_reshape_4d(
        ctx, z, value_width, value_heads, token_count, 1);
    output = ops::rms_norm(
        ctx,
        output,
        layer.ssm_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".state_norm").c_str());
    output = ggml_mul(ctx, output, ggml_silu(ctx, z));
    output = ggml_reshape_2d(ctx, output, inner, token_count);
    output = ops::linear(
        ctx, layer.ssm_out, output, (prefix + ".output").c_str());
    add_critical(result, output);
    return output;
}

ggml_tensor * decoder_block(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TokenGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const RecurrentStateView * recurrent,
    const AttentionCacheView * attention,
    const GraphInputs & inputs,
    const Config & config,
    std::size_t n_kv,
    AttentionImplementation attention_implementation,
    const std::string & prefix) {
    ggml_tensor * normalized = ops::rms_norm(
        ctx,
        input,
        layer.attn_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".attention_norm").c_str());
    ggml_tensor * attention_output = nullptr;
    if (layer.is_recurrent()) {
        require(recurrent != nullptr && attention == nullptr,
                prefix + " persistent-state kind mismatch");
        attention_output = build_recurrent(
            ctx, graph, result, normalized, layer, *recurrent, config, prefix);
    } else {
        require(attention != nullptr && recurrent == nullptr,
                prefix + " persistent-cache kind mismatch");
        attention_output = build_attention(
            ctx,
            graph,
            result,
            normalized,
            layer,
            *attention,
            inputs,
            config,
            n_kv,
            attention_implementation,
            prefix);
    }
    ggml_tensor * residual = ggml_add(ctx, attention_output, input);
    ggml_tensor * ffn_input = ops::rms_norm(
        ctx,
        residual,
        layer.post_attention_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".post_attention_norm").c_str());
    ggml_tensor * ffn = ops::parallel_swiglu_ffn(
        ctx, ffn_input, layer.ffn_gate, layer.ffn_up, layer.ffn_down, config);
    add_critical(result, ffn);
    ggml_tensor * output = ggml_add(ctx, ffn, residual);
    set_name(output, prefix + ".block_output");
    return output;
}

ggml_tensor * decoder_chunk_block(
    ggml_context * ctx,
    ggml_cgraph * graph,
    TargetChunkGraph & result,
    ggml_tensor * input,
    const LayerWeights & layer,
    const RecurrentChunkStateView * recurrent,
    const AttentionCacheView * attention,       //持久保存的kvcache
    const ChunkGraphInputs & inputs,
    const Config & config,
    std::size_t n_tokens,
    std::size_t n_kv,
    std::size_t snapshot_count,
    AttentionImplementation attention_implementation,
    const std::string & prefix) {
    ggml_tensor * normalized = ops::rms_norm(       //每一个decode block的输入都要经过rms_norm
        ctx,
        input,
        layer.attn_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".attention_norm").c_str());
    ggml_tensor * attention_output = nullptr;
    if (layer.is_recurrent()) {
        require(
            recurrent != nullptr && attention == nullptr,
            prefix + " persistent-state kind mismatch");
        attention_output = build_chunk_recurrent(
            ctx,
            graph,
            result,
            normalized,
            layer,
            *recurrent,
            config,
            n_tokens,
            snapshot_count,
            prefix);
    } else {
        require(
            attention != nullptr && recurrent == nullptr,
            prefix + " persistent-cache kind mismatch");
        attention_output = build_chunk_attention(
            ctx,
            graph,
            result,
            normalized,
            layer,
            *attention,
            inputs,
            config,
            n_tokens,
            n_kv,
            attention_implementation,
            prefix);
    }
    ggml_tensor * residual = ggml_add(ctx, attention_output, input);
    ggml_tensor * ffn_input = ops::rms_norm(
        ctx,
        residual,
        layer.post_attention_norm,
        config.attention_layer_norm_rms_epsilon,
        (prefix + ".post_attention_norm").c_str());
    ggml_tensor * ffn = ops::parallel_swiglu_ffn(
        ctx, ffn_input, layer.ffn_gate, layer.ffn_up, layer.ffn_down, config);
    add_critical(result, ffn);
    ggml_tensor * output = ggml_add(ctx, ffn, residual);
    set_name(output, prefix + ".block_output");
    return output;
}

void finalize_outputs(
    ggml_context * ctx,
    TokenGraph & result,
    ggml_tensor * hidden,
    ggml_tensor * head,
    bool emit_greedy,
    bool retain_hidden,
    const char * prefix) {
    result.hidden = hidden;
    if (retain_hidden) {
        ggml_set_output(result.hidden);
    }
    set_name(result.hidden, std::string(prefix) + ".hidden");
    if (emit_greedy) {
        ggml_tensor * logits = ops::linear(
            ctx, head, hidden, (std::string(prefix) + ".logits").c_str());
        add_critical(result, logits);
        result.greedy_token = ggml_argmax(ctx, logits);
        ggml_set_output(result.greedy_token);
        set_name(result.greedy_token, std::string(prefix) + ".greedy_token");
        ggml_build_forward_expand(result.graph, result.greedy_token);
    } else {
        ggml_build_forward_expand(result.graph, result.hidden);
    }
}

void finalize_chunk_outputs(
    ggml_context * ctx,
    TargetChunkGraph & result,
    ggml_tensor * hidden,
    ggml_tensor * head,
    std::size_t n_tokens,
    TargetChunkOutputMode output_mode,
    bool retain_hidden) {
    const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
    require(head != nullptr, "target chunk output head is null");
    require_shape(
        hidden,
        {head->ne[0], token_count},
        "target_chunk.hidden");
    require(
        ggml_is_contiguous(hidden),
        "target chunk hidden tensor must be contiguous");
    result.hidden = hidden;
    set_name(result.hidden, "qwen35.target_chunk.hidden");
    // Keep the full target hidden live through the final-token vocabulary
    // projection.  On Vulkan, a Last-mode view otherwise permits the planner
    // to recycle the parent hidden buffer too early for a multi-token graph.
    // This is a graph-liveness pin only: callers still copy it back to host
    // solely when retain_hidden is requested (MTP prefill/verification).
    if (retain_hidden ||
        (output_mode == TargetChunkOutputMode::Last && n_tokens > 1)) {
        ggml_set_output(result.hidden);
    }

    if (output_mode == TargetChunkOutputMode::None) {
        if (retain_hidden) {
            ggml_build_forward_expand(result.graph, result.hidden);
        }
        return;
    }

    ggml_tensor * head_input = hidden;
    if (output_mode == TargetChunkOutputMode::Last && n_tokens > 1) {
        head_input = ggml_view_2d(
            ctx,
            hidden,
            hidden->ne[0],
            1,
            hidden->nb[1],
            static_cast<std::size_t>(token_count - 1) * hidden->nb[1]);
        set_name(head_input, "qwen35.target_chunk.last_hidden");
    }

    ggml_tensor * logits = ops::linear(
        ctx, head, head_input, "qwen35.target_chunk.logits");
    add_critical(result, logits);
    result.greedy_tokens = ggml_argmax(ctx, logits);
    ggml_set_output(result.greedy_tokens);
    set_name(result.greedy_tokens, "qwen35.target_chunk.greedy_tokens");
    ggml_build_forward_expand(result.graph, result.greedy_tokens);
}

}  // namespace

TokenGraph build_target_token_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const TargetPersistentView & persistent,
    std::size_t n_kv,
    bool emit_greedy,
    bool retain_hidden,
    bool use_causal_mask,
    AttentionImplementation attention_implementation) {
    const Config & config = weights.config();
    require(config.main_layers == 24, "target graph requires 24 decoder layers");
    require(
        persistent.recurrent.size() == kExpectedRecurrentLayers,
        "target graph requires 18 recurrent state records");
    require(
        persistent.attention.size() == kExpectedAttentionLayers,
        "target graph requires 6 attention-cache records");

    TokenGraph result;
    result.graph = ggml_new_graph_custom(ctx, 4096, false);
    const GraphInputs inputs = make_inputs(
        ctx, n_kv, use_causal_mask, attention_implementation);
    result.token = inputs.token;
    result.positions = inputs.positions;
    result.write_slot = inputs.write_slot;
    result.read_slots = inputs.read_slots;
    result.causal_mask = inputs.causal_mask;

    ggml_tensor * current = ggml_get_rows(ctx, weights.global().token_embd, inputs.token);
    set_name(current, "qwen35.target.token_embedding");
    std::size_t recurrent_index = 0;
    std::size_t attention_index = 0;
    for (std::uint32_t index = 0; index < config.main_layers; ++index) {
        const LayerWeights & layer = weights.layer(index);
        const std::string prefix = "qwen35.target.layer." + std::to_string(index);
        if (layer.is_recurrent()) {
            current = decoder_block(
                ctx,
                result.graph,
                result,
                current,
                layer,
                &persistent.recurrent.at(recurrent_index++),
                nullptr,
                inputs,
                config,
                n_kv,
                attention_implementation,
                prefix);
        } else {
            current = decoder_block(
                ctx,
                result.graph,
                result,
                current,
                layer,
                nullptr,
                &persistent.attention.at(attention_index++),
                inputs,
                config,
                n_kv,
                attention_implementation,
                prefix);
        }
    }
    require(recurrent_index == persistent.recurrent.size(), "unused recurrent state record");
    require(attention_index == persistent.attention.size(), "unused attention cache record");

    current = ops::rms_norm(
        ctx,
        current,
        weights.global().output_norm,
        config.attention_layer_norm_rms_epsilon,
        "qwen35.target.output_norm");
    finalize_outputs(
        ctx,
        result,
        current,
        weights.global().output_head,
        emit_greedy,
        retain_hidden,
        "qwen35.target");
    return result;
}

TargetChunkGraph build_target_chunk_graph(  //搭建计算图
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const TargetChunkPersistentView & persistent,
    std::size_t n_tokens,
    std::size_t n_kv,
    std::size_t snapshot_count,
    TargetChunkOutputMode output_mode,
    bool retain_hidden,
    bool force_causal_mask,
    AttentionImplementation attention_implementation) {

    //执行校验，保证参数有效，随时准备抛出异常
    require(ctx != nullptr, "target chunk GGML context is null");
    require(n_tokens > 0, "target chunk requires at least one token");
    require(n_kv >= n_tokens, "target chunk attention context is too short");
    require(
        snapshot_count > 0 && snapshot_count <= n_tokens,
        "target chunk snapshot count must be in [1, n_tokens]");
    require(
        output_mode == TargetChunkOutputMode::None ||
            output_mode == TargetChunkOutputMode::Last ||
            output_mode == TargetChunkOutputMode::All,
        "target chunk output mode is invalid");
    const Config & config = weights.config();   //模型配置
    require(config.main_layers == 24, "target chunk graph requires 24 decoder layers");
    require(
        persistent.recurrent.size() == kExpectedRecurrentLayers,
        "target chunk graph requires 18 recurrent state records");
    require(
        persistent.attention.size() == kExpectedAttentionLayers,
        "target chunk graph requires 6 attention-cache records");

    TargetChunkGraph result;    //graph描述对象
    result.graph = ggml_new_graph_custom(ctx, 4096, false);//创建新的ggml计算图，挂载到本次ctx，设置为4096个节点，不进行梯度计算
    const ChunkGraphInputs inputs = make_chunk_inputs(      //此处是建图过程，这个input只是一个shape描述，实际的值在运行时才会传入
        ctx, n_tokens, n_kv, force_causal_mask, attention_implementation);
    result.tokens = inputs.tokens;
    result.positions = inputs.positions;
    result.write_slots = inputs.write_slots;
    result.read_slots = inputs.read_slots;
    result.causal_mask = inputs.causal_mask;

    ggml_tensor * current = ggml_get_rows(
        ctx, weights.global().token_embd, inputs.tokens);
    set_name(current, "qwen35.target_chunk.token_embedding");
    std::size_t recurrent_index = 0;    //当前是第几个recurrent层
    std::size_t attention_index = 0;    //当前是第几个attention层
    for (std::uint32_t index = 0; index < config.main_layers; ++index)  //开始逐层进行计算
    {
        const LayerWeights & layer = weights.layer(index);  //找出该层的权重
        const std::string prefix =
            "qwen35.target_chunk.layer." + std::to_string(index);   
        if (layer.is_recurrent()) { //对于reccuent的处理
            current = decoder_chunk_block(
                ctx,    //上下文
                result.graph,   //计算图
                result, 
                current,    //输入embedding
                layer,  //该层的权重
                &persistent.recurrent.at(recurrent_index++),    //当前recurrent层的状态
                nullptr,
                inputs, //graph 输入，包括 positions / write_slots / read_slots / mask
                config,
                n_tokens,
                n_kv,
                snapshot_count,
                attention_implementation,
                prefix);
        } else {
            current = decoder_chunk_block(
                ctx,
                result.graph,
                result,
                current,
                layer,
                nullptr,
                &persistent.attention.at(attention_index++),
                inputs,
                config,
                n_tokens,
                n_kv,
                snapshot_count,
                attention_implementation,
                prefix);
        }
    }
    require(
        recurrent_index == persistent.recurrent.size(),
        "target chunk graph has an unused recurrent state record");
    require(
        attention_index == persistent.attention.size(),
        "target chunk graph has an unused attention cache record");

    // A non-final MTP-off prefill slice only needs its persistent KV/recurrent
    // side effects. Its final hidden, output norm and 248K-row vocabulary head
    // have no consumer.
    if (output_mode != TargetChunkOutputMode::None || retain_hidden) {
        current = ops::rms_norm(
            ctx,
            current,
            weights.global().output_norm,
            config.attention_layer_norm_rms_epsilon,
            "qwen35.target_chunk.output_norm");
        if (output_mode != TargetChunkOutputMode::None || retain_hidden) {
            finalize_chunk_outputs(
                ctx,
                result,
                current,
                weights.global().output_head,
                n_tokens,
                output_mode,
                retain_hidden);
        }
    }
    return result;
}

TokenGraph build_mtp_token_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const AttentionCacheView & cache,
    std::size_t n_kv,
    bool emit_greedy,
    bool retain_hidden,
    bool use_causal_mask,
    AttentionImplementation attention_implementation) {
    const Config & config = weights.config();
    require(config.nextn_predict_layers == 1, "MTP graph requires one bundled MTP layer");
    const LayerWeights & layer = weights.layer(config.main_layers);
    require(layer.is_mtp(), "last Qwen3.5 layer is not the bundled MTP block");

    TokenGraph result;
    result.graph = ggml_new_graph_custom(ctx, 1024, false);
    const GraphInputs inputs = make_inputs(
        ctx, n_kv, use_causal_mask, attention_implementation);
    result.token = inputs.token;
    result.positions = inputs.positions;
    result.write_slot = inputs.write_slot;
    result.read_slots = inputs.read_slots;
    result.causal_mask = inputs.causal_mask;
    result.hidden_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, config.embedding_length, 1);
    ggml_set_input(result.hidden_input);
    set_name(result.hidden_input, "qwen35.mtp.hidden_input");

    ggml_tensor * token_embedding =
        ggml_get_rows(ctx, layer.mtp_token_embd, inputs.token);
    ggml_tensor * current = ops::mtp_merge_embedding_and_hidden(
        ctx,
        token_embedding,
        result.hidden_input,
        layer.nextn_enorm,
        layer.nextn_hnorm,
        layer.nextn_eh_proj,
        config);
    current = decoder_block(
        ctx,
        result.graph,
        result,
        current,
        layer,
        nullptr,
        &cache,
        inputs,
        config,
        n_kv,
        attention_implementation,
        "qwen35.mtp.layer");
    current = ops::rms_norm(
        ctx,
        current,
        layer.mtp_output_norm,
        config.attention_layer_norm_rms_epsilon,
        "qwen35.mtp.output_norm");
    finalize_outputs(
        ctx,
        result,
        current,
        layer.mtp_output_head,
        emit_greedy,
        retain_hidden,
        "qwen35.mtp");
    return result;
}

MtpKvUpdateGraph build_mtp_kv_update_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const AttentionCacheView & cache,
    std::size_t n_tokens) {
    require(ctx != nullptr, "MTP KV-update graph context is null");
    require(n_tokens > 0, "MTP KV-update graph requires at least one token");
    const Config & config = weights.config();
    require(config.nextn_predict_layers == 1, "MTP graph requires one bundled MTP layer");
    const LayerWeights & layer = weights.layer(config.main_layers);
    require(layer.is_mtp(), "last Qwen3.5 layer is not the bundled MTP block");

    const std::int64_t token_count = static_cast<std::int64_t>(n_tokens);
    MtpKvUpdateGraph result;
    result.graph = ggml_new_graph_custom(ctx, 512, false);
    result.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    result.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count * 4);
    result.write_slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    result.hidden_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, config.embedding_length, token_count);
    ggml_set_input(result.tokens);
    ggml_set_input(result.positions);
    ggml_set_input(result.write_slots);
    ggml_set_input(result.hidden_input);
    set_name(result.tokens, "qwen35.mtp.kv_update.tokens");
    set_name(result.positions, "qwen35.mtp.kv_update.positions");
    set_name(result.write_slots, "qwen35.mtp.kv_update.write_slots");
    set_name(result.hidden_input, "qwen35.mtp.kv_update.hidden_input");

    ggml_tensor * token_embedding = ggml_get_rows(
        ctx, layer.mtp_token_embd, result.tokens);
    ggml_tensor * current = ops::mtp_merge_embedding_and_hidden(
        ctx,
        token_embedding,
        result.hidden_input,
        layer.nextn_enorm,
        layer.nextn_hnorm,
        layer.nextn_eh_proj,
        config);
    current = ops::rms_norm(
        ctx,
        current,
        layer.attn_norm,
        config.attention_layer_norm_rms_epsilon,
        "qwen35.mtp.kv_update.attention_norm");

    const StoredKeyValue stored = store_attention_key_value(
        ctx,
        current,
        layer,
        cache,
        result.positions,
        result.write_slots,
        config,
        token_count,
        "qwen35.mtp.kv_update");
    result.key_store = stored.key;
    result.value_store = stored.value;
    add_critical(result, result.key_store);
    add_critical(result, result.value_store);
    ggml_build_forward_expand(result.graph, result.key_store);
    ggml_build_forward_expand(result.graph, result.value_store);
    return result;
}

}  // namespace nanovllm::native::qwen35
