#include "strata/deepseek41/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("DeepSeek V4.1: " + s); }
constexpr const char* prefix = "deepseek41.";
std::string key(const char* s) { return std::string(prefix) + s; }
bool integral(MetaType t) {
    return t == MetaType::U8 || t == MetaType::I8 || t == MetaType::U16 || t == MetaType::I16 ||
           t == MetaType::U32 || t == MetaType::I32 || t == MetaType::U64 || t == MetaType::I64;
}
uint64_t integer(const GgufFile& f, const char* suffix, bool zero = false) {
    const auto* v = f.get(key(suffix));
    if (!v || !integral(v->type) || v->u > uint64_t(std::numeric_limits<int>::max()) || (!zero && !v->u))
        fail(key(suffix) + " must be a " + (zero ? "nonnegative" : "positive") + " integer");
    return v->u;
}
double real(const GgufFile& f, const char* suffix, bool zero = false) {
    const auto* v = f.get(key(suffix));
    if (!v || !v->is_num() || v->type == MetaType::BOOL || !std::isfinite(v->num()) ||
        (zero ? v->num() < 0 : v->num() <= 0)) fail(key(suffix) + " must be finite and positive");
    return v->num();
}
std::vector<double> per_layer(const GgufFile& f, const char* suffix, size_t n) {
    const auto* v = f.get(key(suffix));
    if (!v) fail("missing " + key(suffix));
    auto out = v->type == MetaType::ARRAY ? f.real_array(key(suffix)) : std::vector<double>(n, v->num());
    if (out.size() < n || (v->type != MetaType::ARRAY && (!v->is_num() || v->type == MetaType::BOOL)))
        fail(key(suffix) + " must provide one clamp per layer");
    out.resize(n);
    for (double x : out) if (!std::isfinite(x) || x <= 0) fail(key(suffix) + " has an invalid clamp");
    return out;
}
std::vector<uint64_t> integers(const GgufFile& f, const char* suffix) { return f.integer_array(key(suffix)); }
void count(size_t actual, size_t expected, const char* what) {
    if (actual != expected) fail(std::string(what) + " length " + std::to_string(actual) +
                                 ", expected " + std::to_string(expected));
}
std::string block(int layer) { return "blk." + std::to_string(layer) + "."; }
}

Config Config::read(const GgufFile& f) {
    const auto* arch = f.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING || arch->s != "deepseek41") fail("expected architecture deepseek41");
    Config c;
    c.layers = (int) integer(f, "block_count");
    c.hidden = (int) integer(f, "embedding_length");
    c.heads = (int) integer(f, "attention.head_count");
    c.kv_heads = (int) integer(f, "attention.head_count_kv");
    c.head_dim = (int) integer(f, "attention.key_length");
    if (integer(f, "attention.value_length") != uint64_t(c.head_dim)) fail("key/value head dimensions differ");
    c.rope_dim = (int) integer(f, "rope.dimension_count");
    c.q_rank = (int) integer(f, "attention.q_lora_rank");
    c.o_rank = (int) integer(f, "attention.output_lora_rank");
    c.o_groups = (int) integer(f, "attention.output_group_count");
    c.sliding_window = (int) integer(f, "attention.sliding_window");
    c.experts = (int) integer(f, "expert_count");
    c.top_k = (int) integer(f, "expert_used_count");
    c.expert_ff = (int) integer(f, "expert_feed_forward_length");
    c.shared_experts = (int) integer(f, "expert_shared_count");
    c.expert_scale = real(f, "expert_weights_scale");
    const auto* normalized = f.get(key("expert_weights_norm"));
    if (!normalized || normalized->type != MetaType::BOOL || !normalized->u || integer(f, "expert_gating_func") != 4)
        fail("requires normalized sqrtsoftplus expert routing");
    if (const auto* hash = f.get(key("hash_layer_count")); hash && hash->u != 0) fail("hash routing is not V4.1 routing");
    c.hc = (int) integer(f, "hyper_connection.count");
    c.sinkhorn_iterations = (int) integer(f, "hyper_connection.sinkhorn_iterations");
    c.hc_eps = real(f, "hyper_connection.epsilon");
    c.rms_eps = real(f, "attention.layer_norm_rms_epsilon");
    c.index_heads = (int) integer(f, "attention.indexer.head_count");
    c.index_dim = (int) integer(f, "attention.indexer.key_length");
    c.index_top_k = (int) integer(f, "attention.indexer.top_k");
    c.context_train = (int) integer(f, "context_length");
    c.rope_context_original = (int) integer(f, "rope.scaling.original_context_length");
    c.rope_base = real(f, "rope.freq_base");
    c.compress_rope_base = real(f, "attention.compress_rope_freq_base");
    c.rope_scale = real(f, "rope.scaling.factor");
    c.beta_fast = real(f, "rope.scaling.yarn_beta_fast");
    c.beta_slow = real(f, "rope.scaling.yarn_beta_slow");
    const auto* scaling = f.get(key("rope.scaling.type"));
    if (!scaling || scaling->s != "yarn") fail("requires YaRN compressed-stream RoPE");
    const auto* tokens = f.get("tokenizer.ggml.tokens");
    if (!tokens || tokens->type != MetaType::ARRAY || tokens->elem != MetaType::STRING ||
        !tokens->count || tokens->count > uint64_t(std::numeric_limits<int>::max())) fail("invalid vocabulary");
    c.vocab = (int) tokens->count;
    if (c.layers > 1024 || c.hc > 16 || c.kv_heads != 1 || c.heads % c.o_groups ||
        c.top_k > c.experts || c.top_k > 64 || c.rope_dim % 2 || c.rope_dim > c.head_dim || c.rope_dim > c.index_dim)
        fail("incompatible attention, routing or hyper-connection dimensions");
    // CUDA launch dimensions and strides are signed32-bit, including flattened HC/output projections.
    for (const uint64_t n : {uint64_t(c.hidden)*(c.hc+1), uint64_t(c.hc)*(c.hc+2),
                           uint64_t(c.heads)*c.head_dim, uint64_t(c.o_rank)*c.o_groups,
                           uint64_t(c.index_heads)*c.index_dim, uint64_t(c.expert_ff)*c.shared_experts})
        if (n > uint64_t(std::numeric_limits<int>::max())) fail("flattened dimension overflows int32");
    c.compress_ratios = integers(f, "attention.compress_ratios");
    if (c.compress_ratios.size() < size_t(c.layers)) fail("compress_ratios is shorter than block_count");
    c.compress_ratios.resize(c.layers);
    for (auto n : c.compress_ratios) if (n > 2) fail("V4.1 compress ratio must be0,1,2");
    c.clamp_expert = per_layer(f, "swiglu_clamp_exp", c.layers);
    c.clamp_shared = f.get(key("swiglu_clamp_shexp")) ? per_layer(f, "swiglu_clamp_shexp", c.layers) : c.clamp_expert;
    auto& e = c.engram;
    e.heads = (int) integer(f, "engram.head_count"); e.head_dim = (int) integer(f, "engram.key_length");
    e.max_ngram = (int) integer(f, "engram.max_ngram_size"); e.pad_id = integer(f, "engram.pad_id", true);
    if (e.heads > 1024 || e.max_ngram < 2 || e.max_ngram > 128) fail("invalid Engram bucket geometry");
    e.layers = integers(f, "engram.layer_ids"); e.multipliers = integers(f, "engram.multipliers");
    e.primes = integers(f, "engram.primes"); e.offsets = integers(f, "engram.offsets");
    e.token_map = integers(f, "engram.token_map");
    if (e.layers.empty() || e.layers.size() > size_t(c.layers)) fail("invalid Engram layer count");
    count(e.multipliers.size(), e.layers.size()*size_t(e.max_ngram), "Engram multipliers");
    count(e.primes.size(), e.layers.size()*e.buckets(), "Engram primes");
    count(e.offsets.size(), e.primes.size(), "Engram offsets"); count(e.token_map.size(), size_t(c.vocab), "Engram token_map");
    std::set<uint64_t> seen;
    for (uint64_t layer : e.layers) if (layer >= uint64_t(c.layers) || !seen.insert(layer).second) fail("invalid/duplicate Engram layer");
    for (size_t i = 0; i < e.primes.size(); ++i)
        if (!e.primes[i] || e.offsets[i] >= uint64_t(INT32_MAX) || e.primes[i] > uint64_t(INT32_MAX) - e.offsets[i])
            fail("Engram bucket exceeds int32 row range");
    if (uint64_t(e.head_dim)*e.buckets() > uint64_t(INT32_MAX)) fail("Engram projection dimension overflows");
    return c;
}

Model::Model(const std::string& any_shard) : gguf_(gguf_split_paths(any_shard)), config_(Config::read(gguf_.meta())) { validate(); }

TensorRef Model::tensor(const std::string& name) const {
    size_t shard = 0;
    const auto* t = gguf_.find(name, &shard);
    if (!t) fail("missing tensor " + name);
    if (!gguf_.in_bounds(*t, shard)) fail("truncated/unsupported tensor " + name);
    return {&gguf_.shard(shard), t};
}

void Model::require(const std::string& name, const std::vector<uint64_t>& shape, int group,
                    const std::vector<uint64_t>& alternative) {
    const auto ref = tensor(name);
    if (ref.tensor->shape != shape && (alternative.empty() || ref.tensor->shape != alternative)) fail("wrong shape: " + name);
    const auto type = ref.tensor->type;
    if (type != 0 && type != 1 && type != 8 && type != 12 && type != 14 && type != 30)
        fail("unsupported type " + std::string(ref.tensor->type_name()) + ": " + name);
    if (!checked_.insert(name).second) fail("duplicate layout entry " + name);
    auto& bytes = group == 1 ? routed_bytes_ : (group == 2 ? engram_bytes_ : dense_bytes_);
    if (ref.bytes() > UINT64_MAX - bytes) fail("tensor byte count overflow");
    bytes += ref.bytes();
}

void Model::validate() {
    const auto& c = config_;
    const uint64_t H = c.hidden, V = c.vocab, A = c.heads, D = c.head_dim, Q = c.q_rank;
    const uint64_t G = c.o_groups, R = c.o_rank, HC = c.hc, E = c.experts, F = c.expert_ff;
    const uint64_t I = c.index_dim, N = c.index_heads, C = HC*(HC+2);
    require("token_embd.weight", {H,V}); require("output_norm.weight", {H}); require("output.weight", {H,V});
    streams_.resize(c.layers);
    int last_kv = -1, last_key = -1, last_topk = -1;
    for (int l = 0; l < c.layers; ++l) {
        const auto b = block(l);
        auto need = [&](const char* suffix, const std::vector<uint64_t>& shape, int group = 0) { require(b+suffix,shape,group); };
        auto has = [&](const char* suffix) { return gguf_.find(b+suffix) != nullptr; };
        need("attn_norm.weight",{H}); need("attn_sinks.weight",{A});
        need("attn_q_a.weight",{H,Q}); need("attn_q_a_norm.weight",{Q}); need("attn_q_b.weight",{Q,A*D});
        need("attn_kv.weight",{H,D}); need("attn_kv_a_norm.weight",{D});
        require(b+"attn_output_a.weight",{A*D/G,R*G},0,{A*D/G,R,G}); need("attn_output_b.weight",{R*G,H});
        for (const auto* sublayer : {"attn","ffn"}) {
            const auto h = "hc_" + std::string(sublayer);
            require(b+h+"_fn.weight",{H*HC,C}); require(b+h+"_base.weight",{C}); require(b+h+"_scale.weight",{3});
        }
        need("ffn_norm.weight",{H}); need("ffn_gate_inp.weight",{H,E}); need("exp_probs_b.bias",{E});
        if (has("exp_probs_b_vl.bias")) need("exp_probs_b_vl.bias",{E});
        need("ffn_gate_exps.weight",{H,F,E},1); need("ffn_up_exps.weight",{H,F,E},1); need("ffn_down_exps.weight",{F,H,E},1);
        need("ffn_gate_shexp.weight",{H,F*uint64_t(c.shared_experts)});
        need("ffn_up_shexp.weight",{H,F*uint64_t(c.shared_experts)});
        need("ffn_down_shexp.weight",{F*uint64_t(c.shared_experts),H});
        auto& s = streams_[l];
        s.publishes_kv = has("attn_compressor_kv.weight");
        s.publishes_keys = has("indexer.attn_k.weight");
        s.publishes_top_k = has("indexer.attn_q_b.weight");
        const auto ratio = c.compress_ratios[l];
        if (s.publishes_kv) {
            if (!ratio) fail(b+"compressor on pure sliding-window layer");
            need("attn_compressor_kv.weight",{H,D}); need("attn_compressor_norm.weight",{D});
            if (ratio > 1 && !has("attn_compressor_gate.weight")) fail(b+"pooling compressor has no gate");
            if (has("attn_compressor_gate.weight")) need("attn_compressor_gate.weight",{H,D});
            last_kv = l;
        } else if (has("attn_compressor_norm.weight") || has("attn_compressor_gate.weight")) fail(b+"orphan compressor weights");
        if (s.publishes_keys) {
            if (!s.publishes_kv) fail(b+"index keys have no KV source");
            need("indexer.attn_k.weight",{D,I}); need("indexer.k_norm.weight",{I}); last_key = l;
        } else if (has("indexer.k_norm.weight")) fail(b+"orphan index key norm");
        if (s.publishes_top_k) {
            need("indexer.attn_q_b.weight",{Q,N*I}); need("indexer.proj.weight",{H,N}); last_topk = l;
        } else if (has("indexer.proj.weight")) fail(b+"orphan index query projection");
        if (ratio) {
            if (last_kv < 0 || last_key < 0 || last_topk < 0) fail(b+"reads compressed state before a source publishes it");
            if (c.compress_ratios[last_kv] != ratio) fail(b+"compression ratio disagrees with KV source");
            s.kv_source = last_kv; s.key_source = last_key; s.top_k_source = last_topk;
        } else if (s.publishes_keys || s.publishes_top_k) fail(b+"indexer on pure sliding-window layer");
        const auto& ec = c.engram;
        const auto it = std::find(ec.layers.begin(),ec.layers.end(),uint64_t(l));
        if (it != ec.layers.end()) {
            s.engram_index = (int) (it-ec.layers.begin());
            const auto table = tensor(b+"engram_embd.weight");
            if (table.tensor->shape.size()!=2 || table.tensor->shape[0]!=uint64_t(ec.head_dim)) fail(b+"invalid Engram table shape");
            uint64_t max_row = 0;
            for (size_t k = 0; k < ec.buckets(); ++k) {
                const size_t at = size_t(s.engram_index)*ec.buckets()+k;
                max_row = std::max(max_row,ec.offsets[at]+ec.primes[at]);
            }
            if (max_row > table.tensor->shape[1]) fail(b+"Engram buckets exceed table rows");
            require(b+"engram_embd.weight",table.tensor->shape,2);
            need("engram_wkv.weight",{ec.buckets()*uint64_t(ec.head_dim),H*(HC+1)});
            need("engram_q.weight",{H,HC}); need("engram_k.weight",{H,HC});
        }
    }
    for (size_t shard = 0; shard < gguf_.size(); ++shard)
        for (const auto& t : gguf_.shard(shard).tensors())
            if (!checked_.count(t.name)) fail("tensor is not part of the native V4.1 layout: "+t.name);
}

std::vector<int32_t> engram_rows(const EngramConfig& c, size_t engram, int32_t token,
                                const std::vector<int32_t>& history) {
    if (c.heads <= 0 || c.max_ngram < 2 || engram >= c.layers.size() ||
        c.multipliers.size() != c.layers.size()*size_t(c.max_ngram) ||
        c.primes.size() != c.layers.size()*c.buckets() || c.offsets.size() != c.primes.size()) fail("invalid Engram hash configuration");
    auto mapped = [&](int32_t id) { return id < 0 || uint64_t(id) >= c.token_map.size() ? c.pad_id : c.token_map[size_t(id)]; };
    const auto* mult = c.multipliers.data()+engram*size_t(c.max_ngram);
    uint64_t rolling = mapped(token)*mult[0];
    bool blocked = false;
    std::vector<int32_t> result(c.buckets());
    for (int s = 1; s < c.max_ngram; ++s) {
        const int32_t prior = size_t(s) <= history.size() ? history[history.size()-size_t(s)] : -1;
        blocked = blocked || prior < 0;
        rolling ^= (blocked ? c.pad_id : mapped(prior))*mult[s];
        for (int h = 0; h < c.heads; ++h) {
            const size_t bucket = size_t(s-1)*size_t(c.heads)+size_t(h), at = engram*c.buckets()+bucket;
            if (!c.primes[at] || c.offsets[at] >= uint64_t(INT32_MAX) || c.primes[at] > uint64_t(INT32_MAX)-c.offsets[at])
                fail("invalid Engram hash bucket");
            result[bucket] = (int32_t) (rolling%c.primes[at]+c.offsets[at]);
        }
    }
    return result;
}
}  // namespace strata::deepseek41
