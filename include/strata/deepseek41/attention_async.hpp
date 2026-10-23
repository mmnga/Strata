#pragma once
#include <cstdint>
namespace strata::deepseek41::cuda {
// Query once on the current device, before capture; no profiling or kernel launch.
int attention_score_reuse_capacity(int* multiprocessors=nullptr,int* blocks_per_sm=nullptr);
void attention_positions_score_grid_async(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* positions,int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,int qk_capacity=0);
// Exact adjacent-head key reuse; same original softmax/value kernels and workspace.
int attention_score_pair_keys_capacity(int* multiprocessors=nullptr,int* blocks_per_sm=nullptr);
void attention_positions_score_grid_async_pair_keys(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* positions,int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,int qk_capacity);
}
