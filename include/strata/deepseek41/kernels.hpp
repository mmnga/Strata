#pragma once

#include <cstdint>
#include <cstddef>

namespace strata::deepseek41::cuda {

// Native Strata operators. Device pointers, column/token-major contiguous F32,
// explicit non-null stream, no allocation or synchronization in these calls.
// Hyper-connections use four streams, with dst varying fastest in comb[4][4].
void rms(const float* x, const float* weight, float* y, int width, int rows, float eps, void* stream);
void hc_init(const float* embedding, float* residual, float* pre, int hidden, int tokens, void* stream);
void hc_coeff(const float* mixes, const float* scale, const float* base, float* pre, float* post,
              float* comb, int tokens, float eps, int iterations, void* stream);
void hc_pre(const float* residual, const float* pre, float* y, int hidden, int tokens, void* stream);
void hc_post(const float* x, const float* residual, const float* post, const float* comb,
             float* y, int hidden, int tokens, void* stream);
// q/k weights are dequantized once at load. kv contains four keys, then one value.
// Signed square root uses training's copysign, including a positive zero score.
void engram_add(float* residual, const float* kv, const float* qw, const float* kw,
                float* gates, int hidden, int tokens, float eps, void* stream);
void route(const float* logits, const float* bias, int32_t* ids, float* weights,
           int experts, int top_k, int tokens, float scale, void* stream);
void swiglu(const float* gate, const float* up, float* y, int count, float limit, void* stream);
void combine(const float* parts, const float* weights, const float* shared, float* y,
             int hidden, int top_k, int tokens, void* stream);
// Identical route-order arithmetic, selecting cached device rows or completed
// mapped host rows. The mask is device data so graph replay can change routes.
void combine_sources(const float* device_parts,const float* host_parts,const uint64_t* gpu_mask,
                     const float* weights,const float* shared,float* y,
                     int hidden,int top_k,int tokens,void* stream);
// Same FMA order, coherent mapped CPU reads after a device-side Host wait.
void combine_sources_system(const float* device_parts,const float* host_parts,const uint64_t* gpu_mask,
                            const float* weights,const float* shared,float* y,
                            int hidden,int top_k,int tokens,void* stream);
// Original route-order combine followed by in-place four-stream HC post;
// output retains the rounded F32 FFN values needed by the other TP rank.
void combine_sources_post_system(const float* parts,const float* cpu,const uint64_t* mask,
    const float* weights,const float* shared,float* output,float* residual,
    const float* post,const float* comb,int h,int k,int tokens,void* stream);

// Exact sequence equality also handles uint32 rollover. Abort releases a wait.
void wait_host(const uint32_t* ready,const uint32_t* expected,const uint32_t* abort,void* stream);
void dense(const float* w, const float* x, float* y, int n_in, int n_out, int tokens, void* stream);
void add(const float* x, const float* y, float* out, int count, void* stream);
// Original CPU rank-order sum, after Qwen's mapped partial publication.
// Own/peer sequence counters advance independently of token positions/rollback.
// Setting abort releases device waits during an exception or shutdown.
void tp_sum_wait(const float* local,const float* peer,const uint32_t* own_ready,
                 const uint32_t* peer_ready,const uint32_t* abort,float* out,
                 int count,int rank,void* stream);
void strided_copy(const float* x,float* y,int width,int rows,int src_stride,int dst_stride,void* stream);
// Device-side Markov chain input and original DSpark acceptance head.
void gather_row(const float* rows,const int32_t* id,float* out,int width,int row_count,void* stream);
void draft_confidence(const float* hidden,const float* markov,const float* weight,const float* bias,
                      float* out,int hidden_width,int markov_width,void* stream);
void hc_mean(const float* residual,float* y,int hidden,int tokens,int dst_stride,void* stream);
// Original DSpark MXFP4 blocks (17 bytes / 32 values). No weight conversion.
void mxfp4_mmv(const void* w,const float* x,float* y,int n_in,int n_out,int tokens,void* stream,bool alu=false);
void mxfp4_gu(const void* gate,const void* up,const float* x,const int32_t* ids,float* hidden,
              int n_in,int n_ff,int top_k,int tokens,float limit,void* stream,bool alu=false);
void mxfp4_down(const void* down,const float* hidden,const int32_t* ids,float* out,
                int n_ff,int n_out,int top_k,int tokens,void* stream,bool alu=false);

struct Rope {
    int dim = 64, original_context = 0;
    float base = 10000, scale = 1, beta_fast = 0, beta_slow = 0;
    bool yarn = false;
};
// Adjacent pairs in the last rope.dim coordinates of every head; positions are
// device data so captured graphs can be replayed at a different position.
void rope(float* x, int head_dim, int heads, int tokens, const int32_t* positions,
          Rope config, bool inverse, void* stream);
void store_f16(const float* x, uint16_t* cache, int width, int row, void* stream);
// Two source projections, channelwise softmax pooling. ratio=1 is a plain copy.
void pool(const float* values, const float* scores, float* y, int width, int ratio, void* stream);
// Capturable cache operations: positions and completion masks are device data.
// ratio=0 stores the raw ring; ratio=1/2 stores completed compressed rows.
void store_f16_positions(const float* x,uint16_t* cache,int width,int tokens,
                         const int32_t* positions,int capacity,int ratio,void* stream);
void projection_ring(const float* values,const float* scores,float* ring_values,float* ring_scores,
                     int width,int tokens,const int32_t* positions,int capacity,void* stream);
void pool_positions(const float* ring_values,const float* ring_scores,float* y,int width,int tokens,
                    const int32_t* positions,int capacity,int ratio,void* stream);
// One decode query, F16 raw ring and F16 compressed rows; the sink contributes
// to the denominator but has no value. comp_ids may be null for all comp rows.
void attention(const float* q, const uint16_t* raw, const uint16_t* compressed,
               const int32_t* comp_ids, const float* sinks, float* out,
               int dim, int heads, int window, int position, int selected, void* stream,int raw_capacity = 0);
// DSpark's noncausal noise block sees every position through block_end.
void draft_attention(const float* q,const uint16_t* raw,const float* sinks,float* out,
                     int dim,int heads,int window,int position,int block_end,int capacity,void* stream);
void draft_attention_positions(const float* q,const uint16_t* raw,const float* sinks,float* out,
                               int dim,int heads,int tokens,int window,const int32_t* positions,
                               int capacity,void* stream);
void index_scores(const float* q, const float* weights, const uint16_t* keys, float* scores,
                  int dim, int heads, int rows, void* stream);
// A fixed bucket bounds graph capture. Nonvisible rows score -infinity; all
// visible scores and their stable order match the ordinary indexer exactly.
void index_scores_positions(const float* q,const float* weights,const uint16_t* keys,float* scores,
                            int dim,int heads,int bucket,int tokens,const int32_t* positions,
                            int ratio,int stride,void* stream);
void attention_positions(const float* q,const uint16_t* raw,const uint16_t* compressed,
                         const int32_t* ids,const float* sinks,float* out,int dim,int heads,int tokens,
                         int window,const int32_t* positions,int ratio,int max_selected,int ids_stride,
                         int raw_capacity,void* stream);
// Reuse each original rounded softmax weight across all value channels.
void attention_positions_normalized(const float* q,const uint16_t* raw,const uint16_t* compressed,
                         const int32_t* ids,const float* sinks,float* out,int dim,int heads,int tokens,
                         int window,const int32_t* positions,int ratio,int max_selected,int ids_stride,
                         int raw_capacity,void* stream);
// Qwen's 64-cell split-K design adapted to original D512 shared MLA KV.
// Scratch: tokens * heads * ceil((window+max_selected)/64) * 544 floats.
void attention_positions_chunked(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,float* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream);
// Original rounded weights and sequential value FMAs, parallel output dimensions.
// Scratch: tokens * heads * (window+max_selected) * 12 bytes (score plane aligned256).
void attention_positions_values(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream);
// Eight independent FP16 reads before unchanged sequential value FMAs.
void attention_positions_values_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream);
// Original D512 dot order with query registers and independent FP16 key reads.
void attention_positions_query_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,bool value_prefetch);


// Independent QK rows, original whole-head softmax tree and value FMA order.
void attention_positions_score_grid(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,bool pair_heads=false);

// Destructive scratch scores; stable ties prefer lower row id.
void top_k(float* scores, int32_t* ids, int rows, int k, void* stream);
// Stable radix-sort path for long contexts; ids has capacity for all rows.
size_t top_k_scratch_bytes(int rows);
void top_k_sorted(const float* scores,int32_t* ids,int rows,int k,void* scratch,size_t bytes,void* stream);

}  // namespace strata::deepseek41::cuda
