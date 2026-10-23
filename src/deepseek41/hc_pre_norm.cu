#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__device__ float hc_sum(float v) {
    __shared__ float partial[8];
    for (int n = 16; n; n >>= 1) v += __shfl_down_sync(0xffffffffu, v, n);
    if (!(threadIdx.x & 31)) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
    if (threadIdx.x < 32) for (int n = 16; n; n >>= 1) v += __shfl_down_sync(0xffffffffu, v, n);
    if (!threadIdx.x) partial[0] = v;
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}

__global__ void pre_rms(const float* residual,const float* pre,const float* weight,
    float* x,float* normalized,int width,float eps) {
    constexpr int H=5120,Threads=256,Rows=H/Threads;
    const int token=blockIdx.x,lane=threadIdx.x;
    residual+=size_t(token)*H*4;pre+=token*4;
    x+=size_t(token)*H;normalized+=size_t(token)*H;
    float values[Rows],s=0;
    #pragma unroll
    for(int row=0;row<Rows;++row) {
        const int i=lane+row*Threads;float v=0;
        for(int h=0;h<4;++h)v=fmaf(residual[size_t(h)*H+i],pre[h],v);
        values[row]=v;x[i]=v;
        s=fmaf(v,v,s);
    }
    // Keep the original 256-thread reduction and runtime divisor.
    const float inv=rsqrtf(hc_sum(s)/width+eps);
    #pragma unroll
    for(int row=0;row<Rows;++row) {
        const int i=lane+row*Threads;
        normalized[i]=__fmul_rn(__fmul_rn(values[row],inv),weight ? weight[i] : 1.0f);
    }
}
}
void hc_pre_rms(const float* residual,const float* pre,const float* weight,
    float* x,float* normalized,int width,int tokens,float eps,void* stream) {
    if(width!=5120||tokens<1||tokens>8)throw std::invalid_argument("fused HC pre/RMS requires H5120 NT1..8");
    pre_rms<<<tokens,256,0,static_cast<cudaStream_t>(stream)>>>(residual,pre,weight,x,normalized,width,eps);
}
}
