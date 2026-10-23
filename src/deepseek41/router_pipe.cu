#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__device__ float pipe_sum(float v) {
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

__global__ void router_pipe_kernel(const float* weight,const float* input,float* output,int width,int columns) {
    const int row=blockIdx.x,token=blockIdx.y;
    weight+=size_t(row)*width;input+=size_t(token)*width;
    float value=0;
    for(int i=threadIdx.x;i<width;i+=4*256) {
        // Issue independent immutable operand loads before the dependent FMAs.
        const float w0=weight[i],w1=weight[i+256],w2=weight[i+512],w3=weight[i+768];
        const float x0=input[i],x1=input[i+256],x2=input[i+512],x3=input[i+768];
        value=fmaf(w0,x0,value);value=fmaf(w1,x1,value);
        value=fmaf(w2,x2,value);value=fmaf(w3,x3,value);
    }
    value=pipe_sum(value);
    if(!threadIdx.x)output[size_t(token)*columns+row]=value;
}
}
void router_dense_pipe4(const float* weight,const float* input,float* output,int width,int columns,int tokens,void* stream) {
    if(!(width==5120&&columns==384)||tokens<1||tokens>8)
        throw std::invalid_argument("pipelined dense requires original router geometry");
    router_pipe_kernel<<<dim3(columns,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(weight,input,output,width,columns);
}
}
