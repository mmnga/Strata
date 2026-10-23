#include "strata/deepseek41/route_warp.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
// Qwen's native_router uses a stable warp argmax and register-resident
// candidates. DeepSeek retains its own probability expression and original
// rank-order sum; neither Qwen's softmax nor its lane-order sum applies.
template<int Experts,int Top> __global__ void route_warp_kernel(
    const float* logits,const float* bias,int32_t* ids,float* weights,float scale){
    constexpr int Cells=Experts/32;
    const int lane=threadIdx.x,t=blockIdx.x;
    logits+=size_t(t)*Experts;ids+=t*Top;weights+=t*Top;
    float probability[Cells],score[Cells];unsigned used=0;
    #pragma unroll
    for(int j=0;j<Cells;++j){const int id=lane+32*j;const float v=logits[id];
        probability[j]=sqrtf(fmaxf(v,0.0f)+log1pf(expf(-fabsf(v))));
        score[j]=probability[j]+bias[id];
    }
    float total=0;
    for(int rank=0;rank<Top;++rank){
        float best=-INFINITY,picked=0;int id=INT_MAX;
        #pragma unroll
        for(int j=0;j<Cells;++j){const int candidate=lane+32*j;
            if(!(used&(1u<<j))&&(score[j]>best||(score[j]==best&&candidate<id))){best=score[j];id=candidate;picked=probability[j];}
        }
        #pragma unroll
        for(int offset=16;offset;offset>>=1){
            const float other=__shfl_xor_sync(0xffffffffu,best,offset);
            const int other_id=__shfl_xor_sync(0xffffffffu,id,offset);
            const float other_p=__shfl_xor_sync(0xffffffffu,picked,offset);
            if(other>best||(other==best&&other_id<id)){best=other;id=other_id;picked=other_p;}
        }
        if((id&31)==lane)used|=1u<<(id/32);
        if(!lane){ids[rank]=id;weights[rank]=picked;total+=picked;}
    }
    if(!lane){const float inv=1.0f/fmaxf(total,6.103515625e-5f);
        for(int rank=0;rank<Top;++rank)weights[rank]=(weights[rank]*inv)*scale;
    }
}
}
void route_warp(const float* logits,const float* bias,int32_t* ids,float* weights,
                int experts,int top_k,int tokens,float scale,void* stream){
    if(tokens<1||tokens>8||!stream)throw std::invalid_argument("warp DeepSeek route geometry");
    const auto s=static_cast<cudaStream_t>(stream);
    if(experts==384&&top_k==6)route_warp_kernel<384,6><<<tokens,32,0,s>>>(logits,bias,ids,weights,scale);
    else if(experts==128&&top_k==3)route_warp_kernel<128,3><<<tokens,32,0,s>>>(logits,bias,ids,weights,scale);
    else throw std::invalid_argument("warp route requires original main384/6 or DSpark128/3");
}
}
