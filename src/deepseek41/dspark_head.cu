#include "strata/deepseek41/dspark_head.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>
namespace strata::deepseek41::cuda {
bool conditional_draft_available(){return CUDART_VERSION>=13000;}
#if CUDART_VERSION>=13000
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void set_condition(cudaGraphConditionalHandle handle,const float* value,float threshold){cudaGraphSetConditional(handle,isfinite(*value)&&*value>=threshold);}
void capture_if(const float* value,float threshold,cudaStream_t parent,cudaStream_t body_stream,
    const std::function<void(cudaStream_t)>& body) {
    cudaStreamCaptureStatus status;cudaGraph_t graph;
    ck(cudaStreamGetCaptureInfo(parent,&status,nullptr,&graph));
    if(status!=cudaStreamCaptureStatusActive)throw std::runtime_error("conditional draft head requires graph capture");
    cudaGraphConditionalHandle handle;
    ck(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    set_condition<<<1,1,0,parent>>>(handle,value,threshold);ck(cudaGetLastError());
    const cudaGraphNode_t* dependencies=nullptr;size_t count=0;
    ck(cudaStreamGetCaptureInfo(parent,&status,nullptr,&graph,&dependencies,nullptr,&count));
    std::vector<cudaGraphNode_t> deps;if(count)deps.assign(dependencies,dependencies+count);
    cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
    parameters.conditional.handle=handle;parameters.conditional.type=cudaGraphCondTypeIf;parameters.conditional.size=1;
    cudaGraphNode_t node;ck(cudaGraphAddNode(&node,graph,deps.data(),nullptr,deps.size(),&parameters));
    ck(cudaStreamBeginCaptureToGraph(body_stream,parameters.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
    try{body(body_stream);}catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(body_stream,&abandoned);throw;}
    cudaGraph_t captured=nullptr;ck(cudaStreamEndCapture(body_stream,&captured));
    ck(cudaStreamUpdateCaptureDependencies(parent,&node,nullptr,1,cudaStreamSetCaptureDependencies));
}
}
#endif
void conditional_draft_head(const DraftHeadView& v,int count,float threshold,void* stream,void* const* body_streams) {
#if CUDART_VERSION>=13000
    if(count<1||count>5||v.hidden_width<1||v.vocab<1||!stream||!body_streams||!std::isfinite(threshold)||threshold<=0||threshold>1)
        throw std::invalid_argument("conditional draft head geometry/policy");
    const auto root=static_cast<cudaStream_t>(stream);
    // A later uncertain position is never emitted; its initialized valid ID
    // lets the unchanged host collector inspect confidence before stopping.
    ck(cudaMemsetAsync(v.picks,0,size_t(count)*4,root));
    ck(cudaMemsetAsync(v.confidence,0,size_t(count)*4,root));
    strata::kernels::SamplerParams params;params.greedy=true;params.temperature=0;
    auto select=[&](int t,cudaStream_t s){
        dense(v.markov_weight,v.markov,v.bias,256,v.vocab,1,s);
        auto* logits=v.logits+size_t(t)*v.vocab;
        add(logits,v.bias,logits,v.vocab,s);
        strata::kernels::sample_tokens(logits,1,v.vocab,nullptr,0,params,v.picks+t,s);
    };
    std::function<void(int,cudaStream_t)> step=[&](int t,cudaStream_t s){
        gather_row(v.markov_rows,t ? v.picks+t-1 : v.anchor,v.markov,256,v.vocab,s);
        draft_confidence(v.hidden+size_t(t)*v.hidden_width,v.markov,v.confidence_weight,v.confidence_bias,v.confidence+t,v.hidden_width,256,s);
        if(t==0){
            select(t,s);
            if(count>1)capture_if(v.confidence,threshold,s,static_cast<cudaStream_t>(body_streams[0]),[&](cudaStream_t child){step(1,child);});
        }else{
            capture_if(v.confidence+t,threshold,s,static_cast<cudaStream_t>(body_streams[t]),[&](cudaStream_t child){
                select(t,child);if(t+1<count)step(t+1,child);
            });
        }
    };
    step(0,root);
#else
    (void)v;(void)count;(void)threshold;(void)stream;(void)body_streams;
    throw std::runtime_error("conditional draft head requires CUDA 13.0 or newer");
#endif
}
}
