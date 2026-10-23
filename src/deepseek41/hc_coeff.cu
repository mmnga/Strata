#include "strata/deepseek41/hc_coeff.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__device__ float sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
__global__ void coefficient(const float* mix,const float* scale,const float* base,
    float* pre,float* post,float* comb,float eps,int iterations){
    const int lane=threadIdx.x;
    if(lane>=16)return;
    const int t=blockIdx.x;mix+=t*24;pre+=t*4;post+=t*4;comb+=t*16;
    if(lane<4){
        pre[lane]=sigmoid(fmaf(mix[lane],scale[0],base[lane]))+eps;
        post[lane]=2*sigmoid(fmaf(mix[4+lane],scale[1],base[4+lane]));
    }
    constexpr unsigned mask=0x0000ffffu;
    const int src=lane/4,dst=lane%4;
    float value=fmaf(mix[8+lane],scale[2],base[8+lane]);
    float maximum=-INFINITY,total=0;
    #pragma unroll
    for(int j=0;j<4;++j)maximum=fmaxf(maximum,__shfl_sync(mask,value,4*src+j));
    value=expf(value-maximum);
    #pragma unroll
    for(int j=0;j<4;++j)total+=__shfl_sync(mask,value,4*src+j);
    float inv=1.0f/total;
    value=value*inv+eps;
    for(int it=0;it<iterations;++it){
        if(it){
            total=eps;
            #pragma unroll
            for(int j=0;j<4;++j)total+=__shfl_sync(mask,value,4*src+j);
            inv=1.0f/total;value*=inv;
        }
        total=eps;
        #pragma unroll
        for(int j=0;j<4;++j)total+=__shfl_sync(mask,value,dst+4*j);
        inv=1.0f/total;value*=inv;
    }
    comb[lane]=value;
}
}
void hc_coeff_warp(const float* mix,const float* scale,const float* base,
    float* pre,float* post,float* comb,int tokens,float eps,int iterations,void* stream){
    if(tokens<1||tokens>8||iterations!=20||!std::isfinite(eps)||eps<=0||!stream)
        throw std::invalid_argument("original four-stream HC coefficient geometry");
    coefficient<<<tokens,32,0,static_cast<cudaStream_t>(stream)>>>(mix,scale,base,pre,post,comb,eps,iterations);
}
}
