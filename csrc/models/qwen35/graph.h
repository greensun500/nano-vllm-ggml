#pragma once

#include "models/qwen35/weights.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nanovllm::native::qwen35 {

// Non-owning persistent storage used while building one target-token graph.
// The runtime owns these tensors in backend buffers which outlive every
// transient graph context.
struct RecurrentStateView {
    ggml_tensor * convolution = nullptr;  // input [3, 6144], F32
    ggml_tensor * delta = nullptr;        // input [128, 128, 16, 1], F32

    // Optional distinct snapshot destinations. Null means update the input
    // view in place. MTP target verification uses separate planes so K-a can
    // be selected without copying multi-megabyte states through the host.
    ggml_tensor * convolution_output = nullptr;
    ggml_tensor * delta_output = nullptr;
};

struct AttentionCacheView {
    ggml_tensor * key = nullptr;    // [512, total_slots], F32
    ggml_tensor * value = nullptr;  // [512, total_slots], F32
};

struct TargetPersistentView {
    // Ordered by target-layer occurrence, not by absolute layer number:
    // 18 recurrent entries and 6 full-attention entries for Qwen3.5-2B.
    std::vector<RecurrentStateView> recurrent;
    std::vector<AttentionCacheView> attention;
};

enum class TargetChunkOutputMode {
    None,
    Last,
    All,
};

// Persistent state used by one single-sequence, multi-token target graph.
// Snapshot destinations are ordered newest first and must contain exactly the
// number of planes requested from build_target_chunk_graph().
struct RecurrentChunkStateView {
    ggml_tensor * convolution = nullptr;  // input [3, 6144], F32
    ggml_tensor * delta = nullptr;        // input [128, 128, 16, 1], F32

    std::vector<ggml_tensor *> convolution_outputs;
    std::vector<ggml_tensor *> delta_outputs;
};

struct TargetChunkPersistentView {
    // Ordered by target-layer occurrence, matching TargetPersistentView.
    std::vector<RecurrentChunkStateView> recurrent;
    std::vector<AttentionCacheView> attention;
};

// Inputs and outputs of a graph which consumes exactly one token from exactly
// one nano-vLLM sequence. Processing requests token-by-token is intentional in
// the first correctness runtime: it preserves nano-vLLM's batching/Paged-KV
// contract while avoiding a second, llama-style ubatch scheduler.
struct TokenGraph {
    ggml_cgraph * graph = nullptr;

    ggml_tensor * token = nullptr;       // I32 [1]
    ggml_tensor * positions = nullptr;   // I32 [4], IMRoPE channels
    ggml_tensor * write_slot = nullptr;  // I32 [1]
    ggml_tensor * read_slots = nullptr;  // I32 [n_kv]
    ggml_tensor * hidden_input = nullptr;  // MTP only: F32 [2048, 1]

    ggml_tensor * hidden = nullptr;       // F32 [2048, 1], pre-LM-head
    ggml_tensor * greedy_token = nullptr; // I32 [1], optional

    // A narrow diagnostic subset retained for compatibility. Strict Vulkan
    // execution pins and audits every real compute node in the graph, including
    // cache/state CPY and SET_ROWS operations; it does not rely on this list.
    std::vector<ggml_tensor *> critical_compute_nodes;
};

// One target graph for T contiguous tokens from exactly one sequence. The
// read-slot vector is the complete logical prefix through the final token, so
// the caller-provided mask can hide the future T-token suffix per query row.
struct TargetChunkGraph {
    ggml_cgraph * graph = nullptr;

    ggml_tensor * tokens = nullptr;       // I32 [T]
    ggml_tensor * positions = nullptr;    // I32 [4 * T], IMRoPE channels
    ggml_tensor * write_slots = nullptr;  // I32 [T]
    ggml_tensor * read_slots = nullptr;   // I32 [n_kv]
    ggml_tensor * causal_mask = nullptr;  // F32 [n_kv, T], null for T == 1

    // Hidden remains readable only when retain_hidden=true was requested.
    ggml_tensor * hidden = nullptr;        // F32 [2048, T]
    // Null for None, I32 [1] for Last, or I32 [T] for All.
    ggml_tensor * greedy_tokens = nullptr;

    std::vector<ggml_tensor *> critical_compute_nodes;
};

// Batched state-maintenance graph for the bundled MTP attention layer.  MTP
// prefill and post-verification catch-up only need target-conditioned K/V rows;
// constructing the query, attending over the cache, running the FFN and
// materializing a hidden output would be dead work in those paths.
struct MtpKvUpdateGraph {
    ggml_cgraph * graph = nullptr;

    ggml_tensor * tokens = nullptr;        // I32 [n_tokens]
    ggml_tensor * positions = nullptr;     // I32 [4 * n_tokens], IMRoPE channels
    ggml_tensor * write_slots = nullptr;   // I32 [n_tokens]
    ggml_tensor * hidden_input = nullptr;  // F32 [2048, n_tokens]

    // Persistent-cache update nodes.  They are graph roots so executing this
    // graph commits every supplied row without producing a host-visible output.
    ggml_tensor * key_store = nullptr;
    ggml_tensor * value_store = nullptr;

    std::vector<ggml_tensor *> critical_compute_nodes;
};

TokenGraph build_target_token_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const TargetPersistentView & persistent,
    std::size_t n_kv,
    bool emit_greedy);

TargetChunkGraph build_target_chunk_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const TargetChunkPersistentView & persistent,
    std::size_t n_tokens,
    std::size_t n_kv,
    std::size_t snapshot_count,
    TargetChunkOutputMode output_mode,
    bool retain_hidden);

TokenGraph build_mtp_token_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const AttentionCacheView & cache,
    std::size_t n_kv,
    bool emit_greedy);

MtpKvUpdateGraph build_mtp_kv_update_graph(
    ggml_context * ctx,
    const Qwen35Weights & weights,
    const AttentionCacheView & cache,
    std::size_t n_tokens);

}  // namespace nanovllm::native::qwen35
