#include "models/qwen35/weights.h"

#include "runtime/gguf_loader.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nanovllm::native::qwen35 {
namespace {

std::string bind_message(std::string_view detail) {
    return "Qwen3.5-2B weight binding: " + std::string(detail);
}

[[noreturn]] void fail(std::string_view detail) {
    throw ContractError(bind_message(detail));
}

std::string block_name(std::uint32_t layer, std::string_view suffix) {
    return "blk." + std::to_string(layer) + "." + std::string(suffix);
}

bool configs_equal(const Config & left, const Config & right) noexcept {
    return left.architecture == right.architecture &&
           left.block_count == right.block_count &&
           left.nextn_predict_layers == right.nextn_predict_layers &&
           left.main_layers == right.main_layers &&
           left.context_length == right.context_length &&
           left.embedding_length == right.embedding_length &&
           left.feed_forward_length == right.feed_forward_length &&
           left.vocabulary_size == right.vocabulary_size &&
           left.attention_head_count == right.attention_head_count &&
           left.attention_head_count_kv == right.attention_head_count_kv &&
           left.attention_key_length == right.attention_key_length &&
           left.attention_value_length == right.attention_value_length &&
           left.attention_layer_norm_rms_epsilon ==
               right.attention_layer_norm_rms_epsilon &&
           left.rope_dimension_sections == right.rope_dimension_sections &&
           left.rope_dimension_count == right.rope_dimension_count &&
           left.rope_freq_base == right.rope_freq_base &&
           left.ssm_conv_kernel == right.ssm_conv_kernel &&
           left.ssm_state_size == right.ssm_state_size &&
           left.ssm_group_count == right.ssm_group_count &&
           left.ssm_time_step_rank == right.ssm_time_step_rank &&
           left.ssm_inner_size == right.ssm_inner_size &&
           left.full_attention_interval == right.full_attention_interval;
}

bool specs_equal(const TensorSpec & left, const TensorSpec & right) noexcept {
    return left.name == right.name && left.shape == right.shape &&
           left.allowed_types == right.allowed_types && left.role == right.role;
}

bool tensor_has_shape(const ggml_tensor * tensor, const std::vector<std::int64_t> & shape) {
    if (tensor == nullptr || ggml_n_dims(tensor) != static_cast<int>(shape.size())) {
        return false;
    }
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (tensor->ne[dimension] != shape[dimension]) {
            return false;
        }
    }
    return true;
}

std::string shape_string(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "<null>";
    }
    std::ostringstream result;
    result << '[';
    const int dimensions = ggml_n_dims(tensor);
    for (int dimension = 0; dimension < dimensions; ++dimension) {
        if (dimension != 0) {
            result << ", ";
        }
        result << tensor->ne[dimension];
    }
    result << ']';
    return result.str();
}

std::string shape_string(const std::vector<std::int64_t> & shape) {
    std::ostringstream result;
    result << '[';
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (dimension != 0) {
            result << ", ";
        }
        result << shape[dimension];
    }
    result << ']';
    return result.str();
}

bool is_supported_matrix_type(ggml_type type) noexcept {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

class Binder {
public:
    Binder(const GgufWeights & weights, const Manifest & manifest)
        : weights_(weights), manifest_(manifest), parsed_config_(read_config(weights.metadata())) {
        if (!configs_equal(parsed_config_, manifest_.config)) {
            fail("manifest Config does not describe the supplied GgufWeights source");
        }
        expected_specs_ = required_tensor_specs(parsed_config_);
        if (expected_specs_.size() != Qwen35Weights::kRequiredTensorCount) {
            fail("required tensor specification count is " +
                 std::to_string(expected_specs_.size()) + ", expected " +
                 std::to_string(Qwen35Weights::kRequiredTensorCount));
        }
        if (manifest_.tensors.size() != Qwen35Weights::kRequiredTensorCount) {
            fail("manifest tensor count is " + std::to_string(manifest_.tensors.size()) +
                 ", expected " + std::to_string(Qwen35Weights::kRequiredTensorCount));
        }

        expected_by_name_.reserve(expected_specs_.size());
        for (const TensorSpec & spec : expected_specs_) {
            const auto [unused, inserted] = expected_by_name_.emplace(spec.name, &spec);
            (void) unused;
            if (!inserted) {
                fail("duplicate internal required tensor name '" + spec.name + "'");
            }
        }

        std::unordered_set<std::string> manifest_names;
        manifest_names.reserve(manifest_.tensors.size());
        for (const TensorDescriptor & descriptor : manifest_.tensors) {
            if (!manifest_names.emplace(descriptor.spec.name).second) {
                fail("duplicate manifest tensor name '" + descriptor.spec.name + "'");
            }
            const auto expected = expected_by_name_.find(descriptor.spec.name);
            if (expected == expected_by_name_.end()) {
                fail("manifest contains unexpected tensor '" + descriptor.spec.name + "'");
            }
            if (!specs_equal(descriptor.spec, *expected->second)) {
                fail("manifest specification differs from the Qwen3.5 contract for tensor '" +
                     descriptor.spec.name + "'");
            }
        }
    }

    const Config & config() const noexcept { return parsed_config_; }

    ggml_tensor * bind(std::string_view name) {
        const std::string owned_name(name);
        const auto expected_it = expected_by_name_.find(owned_name);
        if (expected_it == expected_by_name_.end()) {
            fail("typed view requested non-required tensor '" + owned_name + "'");
        }
        if (!bound_names_.emplace(owned_name).second) {
            fail("typed view bound tensor '" + owned_name + "' more than once");
        }

        const TensorSpec & expected = *expected_it->second;
        const TensorDescriptor & descriptor = manifest_.require(owned_name);
        ggml_tensor * tensor = weights_.require_tensor(owned_name);

        const std::int64_t tensor_index =
            gguf_find_tensor(weights_.metadata(), owned_name.c_str());
        if (tensor_index < 0) {
            fail("source GGUF is missing manifest tensor '" + owned_name + "'");
        }
        if (descriptor.tensor_index != tensor_index) {
            fail("GGUF index differs from manifest for tensor '" + owned_name + "'");
        }
        if (descriptor.type != gguf_get_tensor_type(weights_.metadata(), tensor_index) ||
            descriptor.type != tensor->type) {
            fail("GGUF/GGML type differs from manifest for tensor '" + owned_name + "'");
        }
        if (std::find(
                expected.allowed_types.begin(), expected.allowed_types.end(), tensor->type) ==
            expected.allowed_types.end()) {
            fail("tensor '" + owned_name + "' has a type outside its allowed contract");
        }
        if (!tensor_has_shape(tensor, expected.shape)) {
            fail("tensor '" + owned_name + "' has shape " + shape_string(tensor) +
                 ", expected " + shape_string(expected.shape));
        }
        if (descriptor.file_offset !=
            gguf_get_tensor_offset(weights_.metadata(), tensor_index)) {
            fail("GGUF file offset differs from manifest for tensor '" + owned_name + "'");
        }
        const std::size_t byte_size =
            gguf_get_tensor_size(weights_.metadata(), tensor_index);
        if (descriptor.byte_size != byte_size || byte_size != ggml_nbytes(tensor)) {
            fail("GGUF/GGML byte size differs from manifest for tensor '" + owned_name + "'");
        }
        if (ggml_get_tensor(weights_.tensor_context(), owned_name.c_str()) != tensor) {
            fail("loader lookup and GGML context disagree for tensor '" + owned_name + "'");
        }
        const char * tensor_name = ggml_get_name(tensor);
        if (tensor_name == nullptr || owned_name != tensor_name) {
            fail("GGML descriptor name differs from manifest tensor '" + owned_name + "'");
        }
        if (!bound_pointers_.emplace(tensor).second) {
            fail("two required tensor names resolve to the same GGML descriptor at '" +
                 owned_name + "'");
        }
        return tensor;
    }

    std::size_t finish() const {
        if (bound_names_.size() != expected_specs_.size()) {
            std::string missing;
            for (const TensorSpec & spec : expected_specs_) {
                if (bound_names_.find(spec.name) == bound_names_.end()) {
                    if (!missing.empty()) {
                        missing += ", ";
                    }
                    missing += spec.name;
                    if (missing.size() > 240) {
                        missing += ", ...";
                        break;
                    }
                }
            }
            fail("typed view bound " + std::to_string(bound_names_.size()) + " of " +
                 std::to_string(expected_specs_.size()) + " required tensors; missing: " +
                 missing);
        }
        return bound_names_.size();
    }

private:
    const GgufWeights & weights_;
    const Manifest & manifest_;
    Config parsed_config_;
    std::vector<TensorSpec> expected_specs_;
    std::unordered_map<std::string, const TensorSpec *> expected_by_name_;
    std::unordered_set<std::string> bound_names_;
    std::unordered_set<ggml_tensor *> bound_pointers_;
};

void bind_common(Binder & binder, LayerWeights & layer) {
    layer.attn_norm = binder.bind(block_name(layer.index, "attn_norm.weight"));
    layer.post_attention_norm =
        binder.bind(block_name(layer.index, "post_attention_norm.weight"));
    layer.ffn_gate = binder.bind(block_name(layer.index, "ffn_gate.weight"));
    layer.ffn_up = binder.bind(block_name(layer.index, "ffn_up.weight"));
    layer.ffn_down = binder.bind(block_name(layer.index, "ffn_down.weight"));
}

void bind_full_attention(Binder & binder, LayerWeights & layer) {
    layer.attn_q = binder.bind(block_name(layer.index, "attn_q.weight"));
    layer.attn_k = binder.bind(block_name(layer.index, "attn_k.weight"));
    layer.attn_v = binder.bind(block_name(layer.index, "attn_v.weight"));
    layer.attn_output = binder.bind(block_name(layer.index, "attn_output.weight"));
    layer.attn_q_norm = binder.bind(block_name(layer.index, "attn_q_norm.weight"));
    layer.attn_k_norm = binder.bind(block_name(layer.index, "attn_k_norm.weight"));
}

void bind_recurrent(Binder & binder, LayerWeights & layer) {
    layer.attn_qkv = binder.bind(block_name(layer.index, "attn_qkv.weight"));
    layer.attn_gate = binder.bind(block_name(layer.index, "attn_gate.weight"));
    layer.ssm_conv1d = binder.bind(block_name(layer.index, "ssm_conv1d.weight"));
    layer.ssm_dt_bias = binder.bind(block_name(layer.index, "ssm_dt.bias"));
    layer.ssm_a = binder.bind(block_name(layer.index, "ssm_a"));
    layer.ssm_beta = binder.bind(block_name(layer.index, "ssm_beta.weight"));
    layer.ssm_alpha = binder.bind(block_name(layer.index, "ssm_alpha.weight"));
    layer.ssm_norm = binder.bind(block_name(layer.index, "ssm_norm.weight"));
    layer.ssm_out = binder.bind(block_name(layer.index, "ssm_out.weight"));
}

ggml_tensor * find_optional_matrix(
    const GgufWeights & weights,
    const std::string & name,
    const ggml_tensor * shape_reference) {
    ggml_tensor * tensor = weights.find_tensor(name);
    if (tensor == nullptr) {
        return nullptr;
    }
    const std::vector<std::int64_t> expected_shape = {
        shape_reference->ne[0],
        shape_reference->ne[1],
    };
    if (!tensor_has_shape(tensor, expected_shape)) {
        fail("optional tensor '" + name + "' has shape " + shape_string(tensor) +
             ", expected " + shape_string(expected_shape));
    }
    if (!is_supported_matrix_type(tensor->type)) {
        fail("optional tensor '" + name + "' has an unsupported matrix type");
    }
    const std::int64_t tensor_index = gguf_find_tensor(weights.metadata(), name.c_str());
    if (tensor_index < 0 || gguf_get_tensor_type(weights.metadata(), tensor_index) != tensor->type ||
        gguf_get_tensor_size(weights.metadata(), tensor_index) != ggml_nbytes(tensor) ||
        ggml_get_tensor(weights.tensor_context(), name.c_str()) != tensor) {
        fail("optional tensor '" + name + "' is inconsistent between GGUF and GGML");
    }
    return tensor;
}

void bind_mtp(
    Binder & binder,
    const GgufWeights & weights,
    const GlobalWeights & global,
    LayerWeights & layer) {
    layer.nextn_eh_proj =
        binder.bind(block_name(layer.index, "nextn.eh_proj.weight"));
    layer.nextn_enorm = binder.bind(block_name(layer.index, "nextn.enorm.weight"));
    layer.nextn_hnorm = binder.bind(block_name(layer.index, "nextn.hnorm.weight"));
    layer.nextn_shared_head_norm =
        binder.bind(block_name(layer.index, "nextn.shared_head_norm.weight"));

    layer.nextn_embed_tokens = find_optional_matrix(
        weights,
        block_name(layer.index, "nextn.embed_tokens.weight"),
        global.token_embd);
    layer.nextn_shared_head_head = find_optional_matrix(
        weights,
        block_name(layer.index, "nextn.shared_head_head.weight"),
        global.output_head);

    layer.mtp_token_embd =
        layer.nextn_embed_tokens != nullptr ? layer.nextn_embed_tokens : global.token_embd;
    layer.mtp_output_head = layer.nextn_shared_head_head != nullptr
        ? layer.nextn_shared_head_head
        : global.output_head;
    layer.mtp_output_norm = layer.nextn_shared_head_norm;
    layer.mtp_token_embd_uses_global = layer.nextn_embed_tokens == nullptr;
    layer.mtp_output_head_uses_global = layer.nextn_shared_head_head == nullptr;
}

}  // namespace

Qwen35Weights::Qwen35Weights(const GgufWeights & weights)
    : Qwen35Weights(weights, validate(weights.metadata(), weights.tensor_context())) {}

Qwen35Weights::Qwen35Weights(const GgufWeights & weights, const Manifest & manifest)
    : source_(&weights), config_(manifest.config), quant_profile_(manifest.quant_profile) {
    Binder binder(weights, manifest);
    config_ = binder.config();

    global_.token_embd = binder.bind("token_embd.weight");
    global_.output_norm = binder.bind("output_norm.weight");
    // This exact Qwen3.5-2B contract has no required output.weight. Its LM head
    // is intentionally tied to the token embedding matrix.
    global_.output_head = global_.token_embd;

    layers_.reserve(config_.block_count);
    for (std::uint32_t index = 0; index < config_.block_count; ++index) {
        LayerWeights layer;
        layer.index = index;
        if (config_.is_mtp_layer(index)) {
            layer.kind = LayerKind::Mtp;
        } else if (config_.is_recurrent_layer(index)) {
            layer.kind = LayerKind::Recurrent;
        } else if (config_.is_full_attention_layer(index)) {
            layer.kind = LayerKind::FullAttention;
        } else {
            fail("layer " + std::to_string(index) + " has no supported Qwen3.5 kind");
        }

        bind_common(binder, layer);
        if (layer.is_recurrent()) {
            bind_recurrent(binder, layer);
        } else {
            bind_full_attention(binder, layer);
        }
        if (layer.is_mtp()) {
            bind_mtp(binder, weights, global_, layer);
        }
        layers_.push_back(layer);
    }

    bound_tensor_count_ = binder.finish();
}

const LayerWeights & Qwen35Weights::layer(std::size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Qwen3.5 layer index " + std::to_string(index) + " is outside [0, " +
            std::to_string(layers_.size()) + ")");
    }
    return layers_[index];
}

}  // namespace nanovllm::native::qwen35
