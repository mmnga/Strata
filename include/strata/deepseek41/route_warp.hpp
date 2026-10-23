#pragma once
#include <cstdint>
namespace strata::deepseek41::cuda {
// Stable warp selection, original DeepSeek sqrt(softplus) + bias for
// ranking and unbiased, rank-ordered F32 weights. Original E384/top6 main
// and E128/top3 DSpark only; finite logits/bias, explicit non-null stream.
void route_warp(const float* logits,const float* bias,int32_t* ids,float* weights,
                int experts,int top_k,int tokens,float scale,void* stream);
}
