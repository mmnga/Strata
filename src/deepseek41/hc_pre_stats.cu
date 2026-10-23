#include "strata/deepseek41/norm_quant.hpp"
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

__global__ void pre_stats(const float* residual,const float* pre,float* x,float* inverse,int width,float eps){
 constexpr int H=5120,Threads=256,Rows=H/Threads;
 const int token=blockIdx.x,lane=threadIdx.x;
 residual+=size_t(token)*H*4;pre+=token*4;x+=size_t(token)*H;float s=0;
 #pragma unroll
 for(int row=0;row<Rows;++row){const int i=lane+row*Threads;float v=0;
  for(int h=0;h<4;++h)v=fmaf(residual[size_t(h)*H+i],pre[h],v);
  x[i]=v;s=fmaf(v,v,s);
 }
 const float inv=rsqrtf(hc_sum(s)/width+eps);
 if(!lane)inverse[token]=inv;
}
}
void hc_pre_stats(const float*r,const float*p,float*x,float*inv,int width,int tokens,float eps,void*stream){
 if(width!=5120||tokens<1||tokens>8||!stream)throw std::invalid_argument("HC/RMS stats geometry");
 pre_stats<<<tokens,256,0,static_cast<cudaStream_t>(stream)>>>(r,p,x,inv,width,eps);
}
}
