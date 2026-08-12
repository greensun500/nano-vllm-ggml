#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nanovllm::native {

struct Qwen35RuntimeOptions {
    std::string model_path;
    std::string backend = "cpu";
    std::size_t max_model_len = 0;
    std::size_t max_num_batched_tokens = 0;
    std::size_t max_num_seqs = 0;
    std::size_t block_size = 0;
    std::size_t num_blocks = 0;
    int cpu_threads = 1;
    std::size_t device_index = 0;
    bool enable_mtp = false;
    std::size_t mtp_max_draft_tokens = 0;
    bool enable_graph_reuse = true;
};

struct Qwen35GraphReuseStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t active_entries = 0;
};

// Owning C++ representation of the nano-vLLM Python execution-plan ABI.
// Tokens belonging to a sequence are contiguous and scheduled_token_counts
// partitions every token-level vector in sequence-row order.
struct Qwen35ExecutionPlan {
    bool is_prefill = false;
    std::int32_t n_tokens = 0;
    std::int32_t n_seqs = 0;
    std::int32_t block_size = 0;
    std::int32_t block_table_cols = 0;

    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> positions;
    std::vector<std::int32_t> seq_ids;
    std::vector<std::int32_t> scheduled_token_counts;
    std::vector<std::int32_t> slot_mapping;
    std::vector<std::int32_t> block_tables;
    std::vector<std::int32_t> context_lens;
    std::vector<std::int32_t> num_cached_tokens;
};

struct Qwen35MtpResult {
    // Row-major [n_seqs, token_capacity]. Unused entries are -1.
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> output_counts;
    std::vector<std::int32_t> draft_counts;
};

// nano-vLLM's model runtime. Its private implementation owns embedded GGML
// backends, GGUF weights, model graphs, Paged KV and recurrent state; it never
// creates a llama_model/llama_context and never calls llama_decode.
class Qwen35Runtime {
public:
    explicit Qwen35Runtime(Qwen35RuntimeOptions options);
    ~Qwen35Runtime();

    Qwen35Runtime(const Qwen35Runtime &) = delete;
    Qwen35Runtime & operator=(const Qwen35Runtime &) = delete;
    Qwen35Runtime(Qwen35Runtime &&) = delete;
    Qwen35Runtime & operator=(Qwen35Runtime &&) = delete;

    std::vector<std::int32_t> run(const Qwen35ExecutionPlan & plan);
    Qwen35MtpResult run_mtp(
        const Qwen35ExecutionPlan & plan,
        std::size_t token_capacity);

    void release_blocks(
        const std::vector<std::int32_t> & block_ids,
        const std::vector<std::int32_t> & sequence_ids,
        std::size_t block_size);
    Qwen35GraphReuseStats graph_reuse_stats() const;
    void shutdown();
    bool is_shutdown() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nanovllm::native
