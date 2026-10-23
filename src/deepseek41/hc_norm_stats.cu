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

__global__ void rms_stats(const float* input,float* inverse,int width,float eps){
    input+=size_t(blockIdx.x)*width;
    float value=0;
    for(int i=threadIdx.x;i<width;i+=blockDim.x)value+=input[i]*input[i];
    const float inv=rsqrtf(hc_sum(value)/width+eps);
    if(!threadIdx.x)inverse[blockIdx.x]=inv;
}
}
void hc_rms_stats(const float* x,float* inv,int width,int tokens,float eps,void* stream){
    if(width!=20480||tokens<1||tokens>8||!stream)throw std::invalid_argument("original flat HC RMS geometry");
    rms_stats<<<tokens,256,0,static_cast<cudaStream_t>(stream)>>>(x,inv,width,eps);
}
}
