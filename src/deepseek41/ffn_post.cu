#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__global__ void combine_post(const float* device_parts,const volatile float* host_parts,
    const uint64_t* gpu_mask,const float* weights,const float* shared,float* output,
    float* residual,const float* post,const float* comb,int h,int k) {
    const int t=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=h)return;
    const uint64_t mask=*gpu_mask; weights+=t*k;
    float value=0;
    for(int j=0;j<k;++j) {
        const int row=t*k+j;const size_t at=size_t(row)*h+i;
        const float part=(mask&(uint64_t(1)<<row)) ? device_parts[at] : host_parts[at];
        value=fmaf(part,weights[j],value);
    }
    // Explicit rounding preserves the original stored F32 combine result.
    const float ffn=__fadd_rn(value,shared ? shared[size_t(t)*h+i] : 0.0f);
    output[size_t(t)*h+i]=ffn;
    residual+=size_t(t)*h*4+i;post+=t*4;comb+=t*16;
    float src[4];for(int j=0;j<4;++j)src[j]=residual[j*h];
    for(int dst=0;dst<4;++dst) {
        float result=__fmul_rn(ffn,post[dst]);
        for(int j=0;j<4;++j)result=fmaf(src[j],comb[dst+4*j],result);
        residual[dst*h]=result;
    }
}
}
void combine_sources_post_system(const float* parts,const float* cpu,const uint64_t* mask,
    const float* weights,const float* shared,float* output,float* residual,
    const float* post,const float* comb,int h,int k,int tokens,void* stream) {
    if(h<1||k<1||tokens<1||int64_t(k)*tokens>64)throw std::invalid_argument("fused FFN post geometry");
    combine_post<<<dim3((h+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(
        parts,cpu,mask,weights,shared,output,residual,post,comb,h,k);
}
}
