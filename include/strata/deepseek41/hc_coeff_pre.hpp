#pragma once
namespace strata::deepseek41::cuda {
// Independent original HC coefficient recurrence and lagged HC pre/RMS.
// pre_in and pre_out must be disjoint. normalize=false writes inverse[t],
// true writes normalized[t][5120], with optional original F32 gamma.
// One coefficient warp plus the original eight RMS warps per token.
void hc_coeff_pre(const float* mix,const float* scale,const float* base,
    float* pre_out,float* post,float* comb,const float* residual,
    const float* pre_in,const float* gamma,float* x,float* normalized_or_inverse,
    int width,int tokens,float hc_eps,int iterations,float rms_eps,
    bool normalize,void* stream);
}
