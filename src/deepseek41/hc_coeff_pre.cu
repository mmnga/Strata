#include "strata/deepseek41/hc_coeff_pre.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__device__ float sigmoid(float x){return 1.0f/(1.0f+expf(-x));}
__device__ void coefficient(const float* mix,const float* scale,const float* base,
    float* pre,float* post,float* comb,float eps,int iterations,int lane,int t){
    if(lane>=16)return;
    mix+=t*24;pre+=t*4;post+=t*4;comb+=t*16;
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
template<bool Normalize> __global__ void prepare(const float* mix,const float* scale,const float* base,
    float* pre_out,float* post,float* comb,const float* residual,const float* pre_in,
    const float* gamma,float* x,float* normalized_or_inverse,int width,float hc_eps,int iterations,float rms_eps){
    constexpr int H=5120,Threads=256,Rows=H/Threads;
    __shared__ float partial[8];
    const int tid=threadIdx.x,token=blockIdx.x,lane=tid-32;
    // The new coefficients do not feed this stage's lagged pre operation.
    if(tid<16)coefficient(mix,scale,base,pre_out,post,comb,hc_eps,iterations,tid,token);
    float values[Rows],s=0;
    if(tid>=32){
        residual+=size_t(token)*H*4;pre_in+=token*4;x+=size_t(token)*H;
        #pragma unroll
        for(int row=0;row<Rows;++row){const int i=lane+row*Threads;float v=0;
            for(int h=0;h<4;++h)v=fmaf(residual[size_t(h)*H+i],pre_in[h],v);
            if constexpr(Normalize)values[row]=v;
            x[i]=v;s=fmaf(v,v,s);
        }
        for(int n=16;n;n>>=1)s+=__shfl_down_sync(0xffffffffu,s,n);
        if(!(lane&31))partial[lane>>5]=s;
    }
    __syncthreads();
    if(tid>=32&&tid<64){s=lane<8?partial[lane]:0;
        for(int n=16;n;n>>=1)s+=__shfl_down_sync(0xffffffffu,s,n);
        if(!lane)partial[0]=s;
    }
    __syncthreads();
    if(tid>=32){const float inv=rsqrtf(partial[0]/width+rms_eps);
        if constexpr(Normalize){normalized_or_inverse+=size_t(token)*H;
            #pragma unroll
            for(int row=0;row<Rows;++row){const int i=lane+row*Threads;normalized_or_inverse[i]=__fmul_rn(__fmul_rn(values[row],inv),gamma?gamma[i]:1.0f);}
        }else if(!lane)normalized_or_inverse[token]=inv;
    }
}
}
void hc_coeff_pre(const float* mix,const float* scale,const float* base,
    float* pre_out,float* post,float* comb,const float* residual,const float* pre_in,
    const float* gamma,float* x,float* normalized_or_inverse,int width,int tokens,
    float hc_eps,int iterations,float rms_eps,bool normalize,void* stream){
    if(width!=5120||tokens<1||tokens>8||iterations!=20||!stream||pre_in==pre_out||!std::isfinite(hc_eps)||hc_eps<=0||!std::isfinite(rms_eps)||rms_eps<=0)
        throw std::invalid_argument("original independent HC coefficient/lagged pre geometry");
    const auto s=static_cast<cudaStream_t>(stream);
    if(normalize)prepare<true><<<tokens,288,0,s>>>(mix,scale,base,pre_out,post,comb,residual,pre_in,gamma,x,normalized_or_inverse,width,hc_eps,iterations,rms_eps);
    else prepare<false><<<tokens,288,0,s>>>(mix,scale,base,pre_out,post,comb,residual,pre_in,gamma,x,normalized_or_inverse,width,hc_eps,iterations,rms_eps);
}
}
