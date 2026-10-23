#include "strata/deepseek41/kv_norm_rope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__device__ float sum(float v) {
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

// Qwen's fused norm/RoPE dataflow, retaining the original DeepSeek
// 256-lane RMS tree, last64 coordinates and precise YaRN trigonometry.
__global__ void kv_prepare(float* kv,const float* gamma,half* raw,const int32_t* positions,
    int width,Rope c,int capacity,float eps,float low,float high){
    __shared__ float values[512];
    const int tid=threadIdx.x,t=blockIdx.x;
    kv+=size_t(t)*width;
    float s=0;
    for(int i=tid;i<width;i+=blockDim.x)s+=kv[i]*kv[i];
    const float inv=rsqrtf(sum(s)/width+eps);
    for(int i=tid;i<width;i+=blockDim.x)values[i]=__fmul_rn(__fmul_rn(kv[i],inv),gamma?gamma[i]:1.0f);
    __syncthreads();
    if(tid<c.dim/2){const int i=tid;
        const float theta_scale=powf(c.base,-2.0f/c.dim);
        const float extrap=float(positions[t])*powf(theta_scale,float(i));
        float theta=c.scale*extrap;
        if(c.yarn){const float ramp=1.0f-fminf(1.0f,fmaxf(0.0f,(i-low)/fmaxf(0.001f,high-low)));theta=theta*(1.0f-ramp)+extrap*ramp;}
        const float cs=cosf(theta),sn=sinf(theta)*(false?-1.0f:1.0f);
        const int at=width-c.dim+2*i;
        const float a=values[at],b=values[at+1];values[at]=a*cs-b*sn;values[at+1]=a*sn+b*cs;
    }
    __syncthreads();
    const int row=positions[t]%capacity;
    for(int i=tid;i<width;i+=blockDim.x){const float v=values[i];kv[i]=v;raw[size_t(row)*width+i]=__float2half_rn(v);}
}
}
void kv_norm_rope_store(float* kv,const float* gamma,uint16_t* raw,
    const int32_t* positions,int width,int tokens,Rope c,int capacity,float eps,void* stream){
    if(width!=512||tokens<1||tokens>8||c.dim!=64||capacity<128+tokens||!stream||!std::isfinite(eps)||eps<=0)
        throw std::invalid_argument("original D512 KV preparation geometry");
    if (c.dim < 2 || c.dim%2 || c.dim > width || c.base <= 1 || c.scale <= 0) throw std::invalid_argument("DeepSeek RoPE geometry");
    float low = 0, high = 0;
    if (c.yarn) {
        if (c.original_context <= 0 || c.beta_fast <= 0 || c.beta_slow <= 0) throw std::invalid_argument("DeepSeek YaRN configuration");
        auto corr = [&](float beta) { return c.dim*std::log(c.original_context/(beta*2.0f*3.14159265358979323846f))/(2*std::log(c.base)); };
        low = std::fmax(0.0f,std::floor(corr(c.beta_fast))); high = std::fmin(float(c.dim-1),std::ceil(corr(c.beta_slow)));
    }
    kv_prepare<<<tokens,256,0,static_cast<cudaStream_t>(stream)>>>(kv,gamma,reinterpret_cast<half*>(raw),positions,width,c,capacity,eps,low,high);
}
}
