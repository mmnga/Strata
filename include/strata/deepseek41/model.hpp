// Native Strata DeepSeek V4.1 artifacts. No llama model, context or graph dependency.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::deepseek41 {

struct EngramConfig {
    int heads = 0, head_dim = 0, max_ngram = 0;
    uint64_t pad_id = 0;
    std::vector<uint64_t> layers, multipliers, primes, offsets, token_map;
    size_t buckets() const { return size_t(heads) * size_t(max_ngram - 1); }
};

struct Config {
    int layers = 0, hidden = 0, vocab = 0, heads = 0, kv_heads = 0, head_dim = 0;
    int rope_dim = 0, q_rank = 0, o_rank = 0, o_groups = 0, sliding_window = 0;
    int experts = 0, top_k = 0, expert_ff = 0, shared_experts = 0;
    int hc = 0, sinkhorn_iterations = 0;
    int index_heads = 0, index_dim = 0, index_top_k = 0;
    int context_train = 0, rope_context_original = 0;
    double rms_eps = 0, hc_eps = 0, expert_scale = 0;
    double rope_base = 0, compress_rope_base = 0, rope_scale = 0, beta_fast = 0, beta_slow = 0;
    std::vector<uint64_t> compress_ratios;
    std::vector<double> clamp_expert, clamp_shared;
    EngramConfig engram;

    static Config read(const GgufFile& f);
};

struct LayerStreams {
    int kv_source = -1, key_source = -1, top_k_source = -1, engram_index = -1;
    bool publishes_kv = false, publishes_keys = false, publishes_top_k = false;
};

struct TensorRef {
    const GgufFile* file = nullptr;
    const TensorInfo* tensor = nullptr;
    const uint8_t* data() const { return file->tensor_data(*tensor); }
    uint64_t bytes() const { return tensor_payload_bytes(*tensor); }
};

class Model {
public:
    explicit Model(const std::string& any_shard);
    const Config& config() const { return config_; }
    const std::vector<LayerStreams>& streams() const { return streams_; }
    const GgufModel& gguf() const { return gguf_; }
    TensorRef tensor(const std::string& name) const;
    size_t tensor_count() const { return checked_.size(); }
    uint64_t routed_bytes() const { return routed_bytes_; }
    uint64_t engram_bytes() const { return engram_bytes_; }
    uint64_t dense_bytes() const { return dense_bytes_; }

private:
    GgufModel gguf_;
    Config config_;
    std::vector<LayerStreams> streams_;
    std::set<std::string> checked_;
    uint64_t routed_bytes_ = 0, engram_bytes_ = 0, dense_bytes_ = 0;
    void validate();
    void require(const std::string& name, const std::vector<uint64_t>& shape, int group = 0,
                 const std::vector<uint64_t>& alternative = {});
};

/// Original64-bit Engram XOR/modulo hash. History is oldest first; a negative
/// predecessor ends look-back, and every older position is padding. No history
/// mutation, so callers can retain/discard speculative branches transactionally.
std::vector<int32_t> engram_rows(const EngramConfig& c, size_t engram, int32_t token,
                                const std::vector<int32_t>& history);

}  // namespace strata::deepseek41
