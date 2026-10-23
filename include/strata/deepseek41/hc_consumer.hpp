#pragma once
namespace strata::kernels {
// CUDA-only original HC Q4_K[20480,24]. One consumer-owned normalization,
// Q8_1 and exact row dot per row/token, with private shared quant bytes.
// Output tensors must be disjoint from the input and from one another.
// Row0 alone publishes the original normalized F32/Q81/inverse buffers.
void native_deepseek41_hc_consumer(const void* weights,const float* residual,
    float* normalized,void* quant,float* inverse,float* mix,int width,int tokens,
    float eps,void* stream);
}
