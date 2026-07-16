#include "runtime/qwen35_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

namespace nanovllm::native::qwen35 {
namespace {

constexpr std::uint32_t kExpectedBlockCount = 25;
constexpr std::uint32_t kExpectedNextnLayers = 1;
constexpr std::uint32_t kExpectedMainLayers = 24;
constexpr std::uint32_t kExpectedContextLength = 262144;
constexpr std::uint32_t kExpectedEmbeddingLength = 2048;
constexpr std::uint32_t kExpectedFeedForwardLength = 6144;
constexpr std::uint64_t kExpectedVocabularySize = 248320;
constexpr std::uint32_t kExpectedHeadCount = 8;
constexpr std::uint32_t kExpectedHeadCountKv = 2;
constexpr std::uint32_t kExpectedKeyLength = 256;
constexpr std::uint32_t kExpectedValueLength = 256;
constexpr std::uint32_t kExpectedRopeDimensionCount = 64;
constexpr std::array<std::int32_t, 4> kExpectedRopeSections = {11, 11, 10, 0};
constexpr float kExpectedRopeFreqBase = 10000000.0f;
constexpr float kExpectedRmsEpsilon = 1.0e-6f;
constexpr std::uint32_t kExpectedSsmConvKernel = 4;
constexpr std::uint32_t kExpectedSsmStateSize = 128;
constexpr std::uint32_t kExpectedSsmGroupCount = 16;
constexpr std::uint32_t kExpectedSsmTimeStepRank = 16;
constexpr std::uint32_t kExpectedSsmInnerSize = 2048;
constexpr std::uint32_t kExpectedFullAttentionInterval = 4;

std::string contract_message(std::string_view detail) {
    return "Qwen3.5-2B GGUF contract: " + std::string(detail);
}

const char * safe_gguf_type_name(gguf_type type) {
    const char * name = gguf_type_name(type);
    return name == nullptr ? "unknown" : name;
}

const char * safe_ggml_type_name(ggml_type type) {
    const char * name = ggml_type_name(type);
    return name == nullptr ? "unknown" : name;
}

std::int64_t require_key(const gguf_context * gguf, const char * key, gguf_type expected) {
    if (gguf == nullptr) {
        throw ContractError(contract_message("GGUF context is null"));
    }
    const std::int64_t index = gguf_find_key(gguf, key);
    if (index < 0) {
        throw ContractError(contract_message("missing metadata key '" + std::string(key) + "'"));
    }
    const gguf_type actual = gguf_get_kv_type(gguf, index);
    if (actual != expected) {
        std::ostringstream message;
        message << "metadata key '" << key << "' has type " << safe_gguf_type_name(actual)
                << ", expected " << safe_gguf_type_name(expected);
        throw ContractError(contract_message(message.str()));
    }
    return index;
}

std::uint32_t read_u32(const gguf_context * gguf, const char * key) {
    return gguf_get_val_u32(gguf, require_key(gguf, key, GGUF_TYPE_UINT32));
}

float read_f32(const gguf_context * gguf, const char * key) {
    return gguf_get_val_f32(gguf, require_key(gguf, key, GGUF_TYPE_FLOAT32));
}

std::string read_string(const gguf_context * gguf, const char * key) {
    return gguf_get_val_str(gguf, require_key(gguf, key, GGUF_TYPE_STRING));
}

std::uint64_t read_vocabulary_size(const gguf_context * gguf) {
    constexpr const char * key = "tokenizer.ggml.tokens";
    const std::int64_t index = require_key(gguf, key, GGUF_TYPE_ARRAY);
    const gguf_type element_type = gguf_get_arr_type(gguf, index);
    if (element_type != GGUF_TYPE_STRING) {
        std::ostringstream message;
        message << "metadata array '" << key << "' has element type "
                << safe_gguf_type_name(element_type) << ", expected string";
        throw ContractError(contract_message(message.str()));
    }
    const std::size_t count = gguf_get_arr_n(gguf, index);
    if (count > std::numeric_limits<std::uint64_t>::max()) {
        throw ContractError(contract_message("tokenizer vocabulary size overflows uint64"));
    }
    return static_cast<std::uint64_t>(count);
}

std::array<std::int32_t, 4> read_rope_sections(const gguf_context * gguf) {
    constexpr const char * key = "qwen35.rope.dimension_sections";
    const std::int64_t index = require_key(gguf, key, GGUF_TYPE_ARRAY);
    const gguf_type element_type = gguf_get_arr_type(gguf, index);
    if (element_type != GGUF_TYPE_INT32) {
        std::ostringstream message;
        message << "metadata array '" << key << "' has element type "
                << safe_gguf_type_name(element_type) << ", expected int32";
        throw ContractError(contract_message(message.str()));
    }
    const std::size_t count = gguf_get_arr_n(gguf, index);
    if (count != 4) {
        std::ostringstream message;
        message << "metadata array '" << key << "' has " << count << " entries, expected 4";
        throw ContractError(contract_message(message.str()));
    }
    const auto * data = static_cast<const std::int32_t *>(gguf_get_arr_data(gguf, index));
    return {data[0], data[1], data[2], data[3]};
}

template <typename T>
void expect_equal(const char * key, T actual, T expected) {
    if (actual == expected) {
        return;
    }
    std::ostringstream message;
    message << "metadata key '" << key << "' is " << actual << ", expected " << expected;
    throw ContractError(contract_message(message.str()));
}

void expect_float(const char * key, float actual, float expected) {
    const float tolerance = std::max(std::abs(expected) * 1.0e-6f, 1.0e-12f);
    if (std::isfinite(actual) && std::abs(actual - expected) <= tolerance) {
        return;
    }
    std::ostringstream message;
    message << "metadata key '" << key << "' is " << actual << ", expected " << expected;
    throw ContractError(contract_message(message.str()));
}

void validate_supported_profile(const Config & config) {
    if (config.architecture != "qwen35") {
        throw ContractError(contract_message(
            "general.architecture is '" + config.architecture + "', expected 'qwen35'"));
    }
    expect_equal("qwen35.block_count", config.block_count, kExpectedBlockCount);
    expect_equal(
        "qwen35.nextn_predict_layers", config.nextn_predict_layers, kExpectedNextnLayers);
    expect_equal("derived main_layers", config.main_layers, kExpectedMainLayers);
    expect_equal("qwen35.context_length", config.context_length, kExpectedContextLength);
    expect_equal(
        "qwen35.embedding_length", config.embedding_length, kExpectedEmbeddingLength);
    expect_equal(
        "qwen35.feed_forward_length",
        config.feed_forward_length,
        kExpectedFeedForwardLength);
    expect_equal("tokenizer.ggml.tokens length", config.vocabulary_size, kExpectedVocabularySize);
    expect_equal(
        "qwen35.attention.head_count", config.attention_head_count, kExpectedHeadCount);
    expect_equal(
        "qwen35.attention.head_count_kv",
        config.attention_head_count_kv,
        kExpectedHeadCountKv);
    expect_equal(
        "qwen35.attention.key_length", config.attention_key_length, kExpectedKeyLength);
    expect_equal(
        "qwen35.attention.value_length", config.attention_value_length, kExpectedValueLength);
    expect_float(
        "qwen35.attention.layer_norm_rms_epsilon",
        config.attention_layer_norm_rms_epsilon,
        kExpectedRmsEpsilon);
    expect_equal(
        "qwen35.rope.dimension_count",
        config.rope_dimension_count,
        kExpectedRopeDimensionCount);
    expect_float("qwen35.rope.freq_base", config.rope_freq_base, kExpectedRopeFreqBase);
    if (config.rope_dimension_sections != kExpectedRopeSections) {
        std::ostringstream message;
        message << "metadata key 'qwen35.rope.dimension_sections' is ["
                << config.rope_dimension_sections[0] << ", "
                << config.rope_dimension_sections[1] << ", "
                << config.rope_dimension_sections[2] << ", "
                << config.rope_dimension_sections[3] << "], expected [11, 11, 10, 0]";
        throw ContractError(contract_message(message.str()));
    }
    expect_equal("qwen35.ssm.conv_kernel", config.ssm_conv_kernel, kExpectedSsmConvKernel);
    expect_equal("qwen35.ssm.state_size", config.ssm_state_size, kExpectedSsmStateSize);
    expect_equal("qwen35.ssm.group_count", config.ssm_group_count, kExpectedSsmGroupCount);
    expect_equal(
        "qwen35.ssm.time_step_rank",
        config.ssm_time_step_rank,
        kExpectedSsmTimeStepRank);
    expect_equal("qwen35.ssm.inner_size", config.ssm_inner_size, kExpectedSsmInnerSize);
    expect_equal(
        "qwen35.full_attention_interval",
        config.full_attention_interval,
        kExpectedFullAttentionInterval);

    if (config.attention_head_count_kv == 0 ||
        config.attention_head_count % config.attention_head_count_kv != 0) {
        throw ContractError(contract_message(
            "attention head_count must be divisible by head_count_kv"));
    }
    if (config.attention_head_count * config.attention_value_length !=
        config.embedding_length) {
        throw ContractError(contract_message(
            "attention head_count * value_length must equal embedding_length"));
    }
    if (config.ssm_state_size * config.ssm_time_step_rank != config.ssm_inner_size) {
        throw ContractError(contract_message(
            "ssm state_size * time_step_rank must equal inner_size"));
    }
    std::int64_t rope_sum = 0;
    for (const std::int32_t section : config.rope_dimension_sections) {
        if (section < 0) {
            throw ContractError(contract_message("rope dimension sections cannot be negative"));
        }
        rope_sum += section;
    }
    if (rope_sum * 2 != config.rope_dimension_count) {
        throw ContractError(contract_message(
            "twice the rope dimension-section sum must equal rope.dimension_count"));
    }
}

std::vector<ggml_type> matrix_types() {
    // These are the portable weight formats exercised by GGML CPU/Vulkan
    // mul_mat. Q4_0 and Q6_K cover the requested model; float and other common
    // quants keep this manifest reusable for lossless/debug conversions.
    return {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    };
}

std::vector<ggml_type> float_types() {
    return {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16};
}

void add_spec(
    std::vector<TensorSpec> & specs,
    std::string name,
    std::vector<std::int64_t> shape,
    std::vector<ggml_type> allowed_types,
    TensorSpec::Role role) {
    specs.push_back({std::move(name), std::move(shape), std::move(allowed_types), role});
}

std::string block_name(std::uint32_t layer, std::string_view suffix) {
    return "blk." + std::to_string(layer) + "." + std::string(suffix);
}

void add_common_block_specs(
    std::vector<TensorSpec> & specs,
    const Config & config,
    std::uint32_t layer) {
    const std::int64_t embedding = config.embedding_length;
    const std::int64_t feed_forward = config.feed_forward_length;
    add_spec(
        specs,
        block_name(layer, "attn_norm.weight"),
        {embedding},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "post_attention_norm.weight"),
        {embedding},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "ffn_gate.weight"),
        {embedding, feed_forward},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "ffn_up.weight"),
        {embedding, feed_forward},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "ffn_down.weight"),
        {feed_forward, embedding},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
}

void add_full_attention_specs(
    std::vector<TensorSpec> & specs,
    const Config & config,
    std::uint32_t layer) {
    const std::int64_t embedding = config.embedding_length;
    const std::int64_t key_length = config.attention_key_length;
    const std::int64_t value_length = config.attention_value_length;
    const std::int64_t query_output =
        key_length * config.attention_head_count * 2;  // Q + per-head output gate
    const std::int64_t key_output = key_length * config.attention_head_count_kv;
    const std::int64_t value_output = value_length * config.attention_head_count_kv;
    const std::int64_t attention_output = value_length * config.attention_head_count;

    add_spec(
        specs,
        block_name(layer, "attn_q.weight"),
        {embedding, query_output},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "attn_k.weight"),
        {embedding, key_output},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "attn_v.weight"),
        {embedding, value_output},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "attn_output.weight"),
        {attention_output, embedding},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "attn_q_norm.weight"),
        {key_length},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "attn_k_norm.weight"),
        {key_length},
        float_types(),
        TensorSpec::Role::FloatParameter);
}

void add_recurrent_specs(
    std::vector<TensorSpec> & specs,
    const Config & config,
    std::uint32_t layer) {
    const std::int64_t embedding = config.embedding_length;
    const std::int64_t key_dimension =
        static_cast<std::int64_t>(config.ssm_state_size) * config.ssm_group_count;
    const std::int64_t value_dimension =
        static_cast<std::int64_t>(config.ssm_state_size) * config.ssm_time_step_rank;
    const std::int64_t convolution_dimension = key_dimension * 2 + value_dimension;

    add_spec(
        specs,
        block_name(layer, "attn_qkv.weight"),
        {embedding, convolution_dimension},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "attn_gate.weight"),
        {embedding, value_dimension},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "ssm_conv1d.weight"),
        {config.ssm_conv_kernel, convolution_dimension},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "ssm_dt.bias"),
        {config.ssm_time_step_rank},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "ssm_a"),
        {config.ssm_time_step_rank},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "ssm_beta.weight"),
        {embedding, config.ssm_time_step_rank},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "ssm_alpha.weight"),
        {embedding, config.ssm_time_step_rank},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
    add_spec(
        specs,
        block_name(layer, "ssm_norm.weight"),
        {config.ssm_state_size},
        float_types(),
        TensorSpec::Role::FloatParameter);
    add_spec(
        specs,
        block_name(layer, "ssm_out.weight"),
        {value_dimension, embedding},
        matrix_types(),
        TensorSpec::Role::LinearWeight);
}

std::string shape_string(const std::vector<std::int64_t> & shape) {
    std::ostringstream result;
    result << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            result << ", ";
        }
        result << shape[index];
    }
    result << ']';
    return result.str();
}

std::string tensor_shape_string(const ggml_tensor * tensor) {
    std::vector<std::int64_t> shape;
    const int dimensions = ggml_n_dims(tensor);
    shape.reserve(static_cast<std::size_t>(dimensions));
    for (int index = 0; index < dimensions; ++index) {
        shape.push_back(tensor->ne[index]);
    }
    return shape_string(shape);
}

bool type_allowed(ggml_type type, const std::vector<ggml_type> & allowed) {
    return std::find(allowed.begin(), allowed.end(), type) != allowed.end();
}

std::string allowed_types_string(const std::vector<ggml_type> & types) {
    std::ostringstream result;
    for (std::size_t index = 0; index < types.size(); ++index) {
        if (index != 0) {
            result << ", ";
        }
        result << safe_ggml_type_name(types[index]);
    }
    return result.str();
}

struct GgufDeleter {
    void operator()(gguf_context * context) const noexcept {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

struct GgmlDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

}  // namespace

bool Config::is_main_layer(std::uint32_t layer) const noexcept {
    return layer < main_layers;
}

bool Config::is_recurrent_layer(std::uint32_t layer) const noexcept {
    return is_main_layer(layer) && full_attention_interval != 0 &&
           (layer + 1) % full_attention_interval != 0;
}

bool Config::is_full_attention_layer(std::uint32_t layer) const noexcept {
    return is_main_layer(layer) && !is_recurrent_layer(layer);
}

bool Config::is_mtp_layer(std::uint32_t layer) const noexcept {
    return layer >= main_layers && layer < block_count;
}

ContractError::ContractError(const std::string & message) : std::runtime_error(message) {}

const TensorDescriptor * Manifest::find(std::string_view name) const noexcept {
    const auto it = std::find_if(
        tensors.begin(), tensors.end(), [&](const TensorDescriptor & descriptor) {
            return descriptor.spec.name == name;
        });
    return it == tensors.end() ? nullptr : &*it;
}

const TensorDescriptor & Manifest::require(std::string_view name) const {
    const TensorDescriptor * descriptor = find(name);
    if (descriptor == nullptr) {
        throw ContractError(contract_message(
            "validated manifest does not contain tensor '" + std::string(name) + "'"));
    }
    return *descriptor;
}

Config read_config(const gguf_context * gguf) {
    Config config;
    config.architecture = read_string(gguf, "general.architecture");
    if (config.architecture != "qwen35") {
        throw ContractError(contract_message(
            "general.architecture is '" + config.architecture + "', expected 'qwen35'"));
    }
    config.block_count = read_u32(gguf, "qwen35.block_count");
    config.nextn_predict_layers = read_u32(gguf, "qwen35.nextn_predict_layers");
    if (config.nextn_predict_layers == 0) {
        throw ContractError(contract_message(
            "qwen35.nextn_predict_layers must be positive for bundled MTP"));
    }
    if (config.nextn_predict_layers >= config.block_count) {
        throw ContractError(contract_message(
            "qwen35.nextn_predict_layers must be less than qwen35.block_count"));
    }
    config.main_layers = config.block_count - config.nextn_predict_layers;

    config.context_length = read_u32(gguf, "qwen35.context_length");
    config.embedding_length = read_u32(gguf, "qwen35.embedding_length");
    config.feed_forward_length = read_u32(gguf, "qwen35.feed_forward_length");
    config.vocabulary_size = read_vocabulary_size(gguf);
    config.attention_head_count = read_u32(gguf, "qwen35.attention.head_count");
    config.attention_head_count_kv = read_u32(gguf, "qwen35.attention.head_count_kv");
    config.attention_key_length = read_u32(gguf, "qwen35.attention.key_length");
    config.attention_value_length = read_u32(gguf, "qwen35.attention.value_length");
    config.attention_layer_norm_rms_epsilon =
        read_f32(gguf, "qwen35.attention.layer_norm_rms_epsilon");
    config.rope_dimension_sections = read_rope_sections(gguf);
    config.rope_dimension_count = read_u32(gguf, "qwen35.rope.dimension_count");
    config.rope_freq_base = read_f32(gguf, "qwen35.rope.freq_base");
    config.ssm_conv_kernel = read_u32(gguf, "qwen35.ssm.conv_kernel");
    config.ssm_state_size = read_u32(gguf, "qwen35.ssm.state_size");
    config.ssm_group_count = read_u32(gguf, "qwen35.ssm.group_count");
    config.ssm_time_step_rank = read_u32(gguf, "qwen35.ssm.time_step_rank");
    config.ssm_inner_size = read_u32(gguf, "qwen35.ssm.inner_size");
    config.full_attention_interval = read_u32(gguf, "qwen35.full_attention_interval");

    validate_supported_profile(config);
    return config;
}

std::vector<TensorSpec> required_tensor_specs(const Config & config) {
    validate_supported_profile(config);

    std::vector<TensorSpec> specs;
    specs.reserve(335);
    add_spec(
        specs,
        "token_embd.weight",
        {static_cast<std::int64_t>(config.embedding_length),
         static_cast<std::int64_t>(config.vocabulary_size)},
        matrix_types(),
        TensorSpec::Role::TokenEmbedding);
    add_spec(
        specs,
        "output_norm.weight",
        {static_cast<std::int64_t>(config.embedding_length)},
        float_types(),
        TensorSpec::Role::FloatParameter);

    for (std::uint32_t layer = 0; layer < config.main_layers; ++layer) {
        add_common_block_specs(specs, config, layer);
        if (config.is_recurrent_layer(layer)) {
            add_recurrent_specs(specs, config, layer);
        } else {
            add_full_attention_specs(specs, config, layer);
        }
    }

    for (std::uint32_t layer = config.main_layers; layer < config.block_count; ++layer) {
        add_common_block_specs(specs, config, layer);
        add_full_attention_specs(specs, config, layer);
        add_spec(
            specs,
            block_name(layer, "nextn.eh_proj.weight"),
            {static_cast<std::int64_t>(config.embedding_length) * 2,
             static_cast<std::int64_t>(config.embedding_length)},
            matrix_types(),
            TensorSpec::Role::LinearWeight);
        add_spec(
            specs,
            block_name(layer, "nextn.enorm.weight"),
            {static_cast<std::int64_t>(config.embedding_length)},
            float_types(),
            TensorSpec::Role::FloatParameter);
        add_spec(
            specs,
            block_name(layer, "nextn.hnorm.weight"),
            {static_cast<std::int64_t>(config.embedding_length)},
            float_types(),
            TensorSpec::Role::FloatParameter);
        // The bundled 2B head uses the trunk token embedding/output matrix but
        // has its own final norm. Dedicated embed/head matrices are optional
        // GGUF extensions and therefore are not part of this required set.
        add_spec(
            specs,
            block_name(layer, "nextn.shared_head_norm.weight"),
            {static_cast<std::int64_t>(config.embedding_length)},
            float_types(),
            TensorSpec::Role::FloatParameter);
    }

    if (specs.size() != 335) {
        std::ostringstream message;
        message << "internal tensor manifest has " << specs.size() << " entries, expected 335";
        throw ContractError(contract_message(message.str()));
    }
    return specs;
}

Manifest validate(const gguf_context * gguf, ggml_context * tensor_context) {
    if (gguf == nullptr) {
        throw ContractError(contract_message("GGUF context is null"));
    }
    if (tensor_context == nullptr) {
        throw ContractError(contract_message("GGML tensor metadata context is null"));
    }

    Manifest manifest;
    manifest.config = read_config(gguf);
    std::vector<TensorSpec> specs = required_tensor_specs(manifest.config);
    manifest.tensors.reserve(specs.size());

    for (TensorSpec & spec : specs) {
        const std::int64_t tensor_index = gguf_find_tensor(gguf, spec.name.c_str());
        if (tensor_index < 0) {
            throw ContractError(contract_message("missing required tensor '" + spec.name + "'"));
        }
        ggml_tensor * tensor = ggml_get_tensor(tensor_context, spec.name.c_str());
        if (tensor == nullptr) {
            throw ContractError(contract_message(
                "GGML metadata context is missing descriptor for required tensor '" +
                spec.name + "'"));
        }

        const int actual_dimensions = ggml_n_dims(tensor);
        bool shape_matches = actual_dimensions == static_cast<int>(spec.shape.size());
        if (shape_matches) {
            for (int dimension = 0; dimension < actual_dimensions; ++dimension) {
                if (tensor->ne[dimension] != spec.shape[static_cast<std::size_t>(dimension)]) {
                    shape_matches = false;
                    break;
                }
            }
        }
        if (!shape_matches) {
            throw ContractError(contract_message(
                "tensor '" + spec.name + "' has shape " + tensor_shape_string(tensor) +
                ", expected " + shape_string(spec.shape)));
        }

        const ggml_type gguf_type = gguf_get_tensor_type(gguf, tensor_index);
        if (tensor->type != gguf_type) {
            throw ContractError(contract_message(
                "tensor '" + spec.name + "' type differs between GGUF and GGML metadata"));
        }
        if (!type_allowed(gguf_type, spec.allowed_types)) {
            throw ContractError(contract_message(
                "tensor '" + spec.name + "' has type " + safe_ggml_type_name(gguf_type) +
                ", allowed types: " + allowed_types_string(spec.allowed_types)));
        }

        const std::size_t byte_size = gguf_get_tensor_size(gguf, tensor_index);
        if (byte_size != ggml_nbytes(tensor)) {
            std::ostringstream message;
            message << "tensor '" << spec.name << "' has GGUF byte size " << byte_size
                    << " but GGML descriptor requires " << ggml_nbytes(tensor);
            throw ContractError(contract_message(message.str()));
        }
        manifest.tensors.push_back({
            std::move(spec),
            tensor_index,
            gguf_type,
            gguf_get_tensor_offset(gguf, tensor_index),
            byte_size,
        });
    }
    const bool strict_profile = std::all_of(
        manifest.tensors.begin(),
        manifest.tensors.end(),
        [](const TensorDescriptor & descriptor) {
            switch (descriptor.spec.role) {
                case TensorSpec::Role::TokenEmbedding:
                    return descriptor.type == GGML_TYPE_Q6_K;
                case TensorSpec::Role::LinearWeight:
                    return descriptor.type == GGML_TYPE_Q4_0;
                case TensorSpec::Role::FloatParameter:
                    return descriptor.type == GGML_TYPE_F32;
            }
            return false;
        });
    manifest.quant_profile = strict_profile
        ? Manifest::QuantProfile::Q4_0WithQ6KEmbedding
        : Manifest::QuantProfile::SupportedMixed;
    return manifest;
}

Manifest inspect_file(std::string_view path) {
    if (path.empty()) {
        throw ContractError(contract_message("model path is empty"));
    }
    const std::string owned_path(path);
    ggml_context * raw_tensor_context = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &raw_tensor_context;
    std::unique_ptr<gguf_context, GgufDeleter> gguf(
        gguf_init_from_file(owned_path.c_str(), params));
    std::unique_ptr<ggml_context, GgmlDeleter> tensor_context(raw_tensor_context);
    if (gguf == nullptr || tensor_context == nullptr) {
        throw ContractError(contract_message(
            "failed to parse metadata and tensor descriptors from '" + owned_path + "'"));
    }
    return validate(gguf.get(), tensor_context.get());
}

}  // namespace nanovllm::native::qwen35
