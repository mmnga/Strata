#include "strata/deepseek41/dspark_head.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>
#include "strata/deepseek41/markov_bf16.hpp"
namespace strata::deepseek41::cuda {
namespace {
__device__ float markov_sum(float v){
    __shared__ float partial[8];
    for(int n=16;n;n>>=1)v+=__shfl_down_sync(0xffffffffu,v,n);
    if(!(threadIdx.x&31))partial[threadIdx.x>>5]=v;
    __syncthreads();v=threadIdx.x<8?partial[threadIdx.x]:0;
    if(threadIdx.x<32)for(int n=16;n;n>>=1)v+=__shfl_down_sync(0xffffffffu,v,n);
    if(!threadIdx.x)partial[0]=v;
    __syncthreads();const float result=partial[0];__syncthreads();return result;
}
__global__ void bf16_markov_kernel(const uint16_t*w,const float*x,float*y,int rows){
    const int row=blockIdx.x,t=blockIdx.y;const int i=threadIdx.x;
    const float weight=__uint_as_float(uint32_t(w[size_t(row)*256+i])<<16);
    float value=fmaf(weight,x[size_t(t)*256+i],0.f);value=markov_sum(value);
    if(!threadIdx.x)y[size_t(t)*rows+row]=value;
}
}
void markov_bf16(const uint16_t*w,const float*x,float*y,int rows,int tokens,void*stream){
    if(!w||!x||!y||!stream||rows!=129280||tokens<1||tokens>8)throw std::invalid_argument("original BF16 Markov geometry");
    bf16_markov_kernel<<<dim3(rows,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(w,x,y,rows);
}
}
namespace strata::deepseek41::cuda {

#if CUDART_VERSION>=13000
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void bf16_set_condition(cudaGraphConditionalHandle handle,const float* value,float threshold){cudaGraphSetConditional(handle,isfinite(*value)&&*value>=threshold);}
void bf16_capture_if(const float* value,float threshold,cudaStream_t parent,cudaStream_t body_stream,
    const std::function<void(cudaStream_t)>& body,void(*observe)(void*,void*),void* context) {
    cudaStreamCaptureStatus status;cudaGraph_t graph;
    ck(cudaStreamGetCaptureInfo(parent,&status,nullptr,&graph));
    if(status!=cudaStreamCaptureStatusActive)throw std::runtime_error("conditional draft head requires graph capture");
    cudaGraphConditionalHandle handle;
    ck(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    bf16_set_condition<<<1,1,0,parent>>>(handle,value,threshold);ck(cudaGetLastError());
    const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
    ck(cudaStreamGetCaptureInfo(parent,&status,nullptr,&graph,&dependencies,nullptr,&count));
    std::vector<cudaGraphNode_t> deps;if(count)deps.assign(dependencies,dependencies+count);
    cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
    parameters.conditional.handle=handle;parameters.conditional.type=cudaGraphCondTypeIf;parameters.conditional.size=1;
    cudaGraphNode_t node;ck(cudaGraphAddNode(&node,graph,deps.data(),nullptr,deps.size(),&parameters));
    ck(cudaStreamBeginCaptureToGraph(body_stream,parameters.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
    try{body(body_stream);}catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(body_stream,&abandoned);throw;}
    cudaGraph_t captured=nullptr;ck(cudaStreamEndCapture(body_stream,&captured));
    if(observe)observe(context,captured);
    ck(cudaStreamUpdateCaptureDependencies(parent,&node,nullptr,1,cudaStreamSetCaptureDependencies));
}
}
#endif
void conditional_draft_head_bf16(const DraftHeadView& v,const uint16_t* raw_w2,int count,float threshold,void* stream,void* const* body_streams,void(*observe)(void*,void*),void* context) {
#if CUDART_VERSION>=13000
    if(!raw_w2||count<1||count>5||v.hidden_width!=5120||v.vocab!=129280||!stream||!body_streams||!std::isfinite(threshold)||threshold<=0||threshold>1)
        throw std::invalid_argument("conditional draft head geometry/policy");
    const auto root=static_cast<cudaStream_t>(stream);
    // A later uncertain position is never emitted; its initialized valid ID
    // lets the unchanged host collector inspect confidence before stopping.
    ck(cudaMemsetAsync(v.picks,0,size_t(count)*4,root));
    ck(cudaMemsetAsync(v.confidence,0,size_t(count)*4,root));
    strata::kernels::SamplerParams params;params.greedy=true;params.temperature=0;
    auto select=[&](int t,cudaStream_t s){
        markov_bf16(raw_w2,v.markov,v.bias,v.vocab,1,s);
        auto* logits=v.logits+size_t(t)*v.vocab;
        add(logits,v.bias,logits,v.vocab,s);
        strata::kernels::sample_tokens(logits,1,v.vocab,nullptr,0,params,v.picks+t,s);
    };
    std::function<void(int,cudaStream_t)> step=[&](int t,cudaStream_t s){
        gather_row(v.markov_rows,t ? v.picks+t-1 : v.anchor,v.markov,256,v.vocab,s);
        draft_confidence(v.hidden+size_t(t)*v.hidden_width,v.markov,v.confidence_weight,v.confidence_bias,v.confidence+t,v.hidden_width,256,s);
        if(t==0){
            select(t,s);
            if(count>1)bf16_capture_if(v.confidence,threshold,s,static_cast<cudaStream_t>(body_streams[0]),[&](cudaStream_t child){step(1,child);},observe,context);
        }else{
            bf16_capture_if(v.confidence+t,threshold,s,static_cast<cudaStream_t>(body_streams[t]),[&](cudaStream_t child){
                select(t,child);if(t+1<count)step(t+1,child);
            },observe,context);
        }
    };
    step(0,root);
#else
    (void)v;(void)raw_w2;(void)observe;(void)context;(void)count;(void)threshold;(void)stream;(void)body_streams;
    throw std::runtime_error("conditional draft head requires CUDA 13.0 or newer");
#endif
}
}
