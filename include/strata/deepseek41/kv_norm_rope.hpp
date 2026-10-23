#pragma once
#include "strata/deepseek41/kernels.hpp"
namespace strata::deepseek41::cuda {
// Original D512 KV RMS, last64 RoPE and F16 raw-ring publication in one
// explicit-stream operation. Keeps the rotated F32 KV as well as cache bits.
// Positions are device data and the caller owns the original raw ring.
void kv_norm_rope_store(float* kv,const float* gamma,uint16_t* raw,
    const int32_t* positions,int width,int tokens,Rope rope,int capacity,
    float eps,void* stream);
}
