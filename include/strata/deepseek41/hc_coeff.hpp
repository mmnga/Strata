#pragma once
namespace strata::deepseek41::cuda {
// Original four-stream HC coefficients and twenty Sinkhorn iterations. Each
// matrix entry keeps the original ordered four-term row/column sums.
void hc_coeff_warp(const float* mix,const float* scale,const float* base,
    float* pre,float* post,float* comb,int tokens,float eps,int iterations,void* stream);
}
