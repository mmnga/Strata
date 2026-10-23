#pragma once
namespace strata::deepseek41::cuda {
void hc_pre_stats(const float* residual,const float* pre,float* x,float* inv,int width,int tokens,float eps,void* stream);
void norm_quant_q8(const float* x,const float* inv,const float* weight,float* normalized,void* q8,void* il,int width,int tokens,int cp,void* stream);
}
