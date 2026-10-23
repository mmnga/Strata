#pragma once
#include "strata/kernels/native_mmvq.hpp"
namespace strata::deepseek41 {
// Original Q8_0 draft output-A, eight token-major independent groups. No
// weight repack or arithmetic change; caller owns graph-stable quant storage.
inline void draft_output_a(const void* weights,const float* input,void* quant,
                           float* output,int tokens,void* stream) {
    namespace K=strata::kernels;
    K::native_quantize_q8_1(input,quant,32768,tokens,stream);
    K::native_mmvq_grouped(8,weights,quant,nullptr,output,4096,1024,8,tokens,stream);
}
}
