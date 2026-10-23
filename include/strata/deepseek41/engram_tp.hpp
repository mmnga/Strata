#pragma once
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
namespace strata::deepseek41::engram_tp {
constexpr int Input=6144,Full=25600,Half=12800,Batch=8;
constexpr size_t HalfWeightBytes=44236800;
struct View{int device;cudaStream_t stream;const uint8_t*weights;float*input,*kv,*residual;const float*query,*key;void*q,*il;bool wide=true;float*gates=nullptr;};
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline void project(const View&v,int tokens,int rows,size_t offset){
    const void*w=v.weights+offset;
    if(v.wide&&tokens>=5){
        strata::kernels::native_quantize_q8_1_il_wide(v.input,v.q,v.il,Input,tokens,v.stream);
        strata::kernels::native_mmvq_il_wide(12,w,v.il,v.kv,Input,rows,tokens,v.stream);
    }else if(tokens>1&&tokens<=4&&strata::kernels::native_mmvq_il_supported(12,tokens,rows)){
        strata::kernels::native_quantize_q8_1_il(v.input,v.q,v.il,Input,tokens,v.stream);
        strata::kernels::native_mmvq_il(12,w,v.q,v.il,v.kv,Input,rows,tokens,v.stream);
    }else{
        strata::kernels::native_quantize_q8_1(v.input,v.q,Input,tokens,v.stream);
        strata::kernels::native_mmvq(12,w,v.q,v.kv,Input,rows,tokens,v.stream);
    }
}
// Caller owns two immutable full weight matrices and 2*Half*Batch pinned
// host floats. Each Rank's original full KV buffer is the compact temporary.
// Keep host_rows alive until both streams finish their final queued copies.
inline void enqueue(const View* v,const float* input,float* host_rows,int tokens,float eps){
    if(!v||!input||!host_rows||tokens<1||tokens>Batch||v[0].device!=0||v[1].device!=1)
        throw std::invalid_argument("Engram TP2 geometry");
    for(int r=0;r<2;++r){
        check(cudaSetDevice(v[r].device));
        check(cudaMemcpyAsync(v[r].input,input,size_t(Input)*tokens*4,cudaMemcpyHostToDevice,v[r].stream));
        project(v[r],tokens,Half,size_t(r)*HalfWeightBytes);
        check(cudaMemcpyAsync(host_rows+size_t(r)*Half*Batch,v[r].kv,size_t(Half)*tokens*4,cudaMemcpyDeviceToHost,v[r].stream));
    }
    for(int r=0;r<2;++r){check(cudaSetDevice(v[r].device));check(cudaStreamSynchronize(v[r].stream));}
    for(int r=0;r<2;++r){
        check(cudaSetDevice(v[r].device));
        for(int half=0;half<2;++half)check(cudaMemcpy2DAsync(v[r].kv+half*Half,Full*4,
            host_rows+size_t(half)*Half*Batch,Half*4,Half*4,tokens,cudaMemcpyHostToDevice,v[r].stream));
        strata::deepseek41::cuda::engram_add(v[r].residual,v[r].kv,v[r].query,v[r].key,v[r].gates,5120,tokens,eps,v[r].stream);
    }
}
}
