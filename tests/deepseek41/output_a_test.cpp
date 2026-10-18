#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace K=strata::kernels;
namespace C=strata::deepseek41::cuda;
void ck(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    void* p=nullptr;
    explicit Device(size_t n) {ck(cudaMalloc(&p,n));}
    ~Device() {cudaFree(p);}
    float* f() const {return static_cast<float*>(p);}
};
void finite_weights(std::vector<uint8_t>& w,int type,std::mt19937& rng) {
    const size_t block=type==12 ? 144 : 210;
    for(auto& b:w) b=uint8_t(rng());
    for(size_t i=0;i<w.size();i+=block) {
        const uint16_t scale=uint16_t(0x1000+rng()%0x2000),minimum=uint16_t(0x1000+rng()%0x2000);
        std::memcpy(w.data()+i+(type==12 ? 0 : 208),&scale,2);
        if(type==12) std::memcpy(w.data()+i+2,&minimum,2);
    }
}
void check(int type,int groups,int tokens,const Device& weights,cudaStream_t s) {
    constexpr int In=4096,Out=1024;
    const int full=groups*In;
    Device input(size_t(full)*tokens*4),quant(K::native_q8_1_bytes(full,tokens)),il(K::native_q8_1_il_bytes(full,4));
    Device actual(size_t(groups)*Out*tokens*4),expected(size_t(groups)*Out*tokens*4);
    Device group_input(size_t(In)*tokens*4),group_output(size_t(Out)*tokens*4);
    Device group_quant(K::native_q8_1_bytes(In,tokens)),group_il(K::native_q8_1_il_bytes(In,4));
    const bool interleaved=tokens>1 && tokens<=4 && K::native_mmvq_il_supported(type,tokens,Out);
    auto candidate=[&] {
        if(interleaved) K::native_quantize_q8_1_il(input.f(),quant.p,il.p,full,tokens,s);
        else K::native_quantize_q8_1(input.f(),quant.p,full,tokens,s);
        K::native_mmvq_grouped(type,weights.p,quant.p,interleaved ? il.p : nullptr,actual.f(),In,Out,groups,tokens,s);
    };
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));candidate();
    ck(cudaStreamEndCapture(s,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    const size_t group_bytes=K::native_mmvq_weight_bytes(type,In,Out);
    for(int replay=0;replay<3;++replay) {
        std::vector<float> x(size_t(full)*tokens);
        for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*0.017f+replay*0.23f)*float(1+(i/32)%7)*0.2f;
        ck(cudaMemcpyAsync(input.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,s));
        ck(cudaGraphLaunch(exec,s));
        // The former group-by-group path, including its per-group gather,
        // quantization, original column selector and token-major output scatter.
        for(int g=0;g<groups;++g) {
            const float* src=input.f()+g*In;
            float* dst=expected.f()+g*Out;
            if(tokens>1) C::strided_copy(src,group_input.f(),In,tokens,full,In,s);
            const float* values=tokens>1 ? group_input.f() : src;
            float* result=tokens>1 ? group_output.f() : dst;
            const auto* w=static_cast<const uint8_t*>(weights.p)+size_t(g)*group_bytes;
            if(interleaved) {
                K::native_quantize_q8_1_il(values,group_quant.p,group_il.p,In,tokens,s);
                K::native_mmvq_il(type,w,group_quant.p,group_il.p,result,In,Out,tokens,s);
            } else {
                K::native_quantize_q8_1(values,group_quant.p,In,tokens,s);
                K::native_mmvq(type,w,group_quant.p,result,In,Out,tokens,s);
            }
            if(tokens>1) C::strided_copy(group_output.f(),dst,Out,tokens,Out,groups*Out,s);
        }
        ck(cudaStreamSynchronize(s));std::vector<float> a(size_t(groups)*Out*tokens),b(a.size());
        ck(cudaMemcpy(a.data(),actual.p,a.size()*4,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b.data(),expected.p,b.size()*4,cudaMemcpyDeviceToHost));
        if(!std::all_of(a.begin(),a.end(),[](float v){return std::isfinite(v);}) || std::memcmp(a.data(),b.data(),a.size()*4)) {
            size_t different=0;double max_abs=0;int shown=0;
            for(size_t i=0;i<a.size();++i) if(std::memcmp(&a[i],&b[i],4)) {
                ++different;max_abs=std::max(max_abs,std::abs(double(a[i])-b[i]));
                if(shown++<5) std::cerr<<"index="<<i<<" actual="<<a[i]<<" expected="<<b[i]<<'\n';
            }
            std::cerr<<"different="<<different<<"/"<<a.size()<<" max_abs="<<max_abs<<'\n';
            if(groups==1) {
                std::vector<uint8_t> qa(K::native_q8_1_bytes(In,tokens)),qb(qa.size());
                ck(cudaMemcpy(qa.data(),quant.p,qa.size(),cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(qb.data(),group_quant.p,qb.size(),cudaMemcpyDeviceToHost));
                std::cerr<<"standard quant bytes identical="<<(qa==qb)<<'\n';
                K::native_mmvq(type,weights.p,quant.p,expected.f(),In,Out,tokens,s);ck(cudaStreamSynchronize(s));
                ck(cudaMemcpy(b.data(),expected.p,b.size()*4,cudaMemcpyDeviceToHost));
                std::cerr<<"direct old call on candidate quant matches candidate="<<(std::memcmp(a.data(),b.data(),a.size()*4)==0)<<'\n';
                candidate();ck(cudaStreamSynchronize(s));ck(cudaMemcpy(b.data(),actual.p,b.size()*4,cudaMemcpyDeviceToHost));
                std::cerr<<"uncaptured candidate matches captured="<<(std::memcmp(a.data(),b.data(),a.size()*4)==0)<<'\n';
            }
            throw std::runtime_error("grouped output-A differs from original per-group projection: type="+std::to_string(type)+" groups="+std::to_string(groups)+" tokens="+std::to_string(tokens));
        }
    }
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
}
int main() {
    try {
        int devices=0;ck(cudaGetDeviceCount(&devices));if(devices<2) throw std::runtime_error("TP2 test needs both GPUs");
        K::native_mmvq_set_multi_exact(true);
        for(int type:{12,14}) {
            std::vector<uint8_t> raw(K::native_mmvq_weight_bytes(type,4096,1024)*8);std::mt19937 rng(1024+type);finite_weights(raw,type,rng);
            for(int device=0;device<2;++device) {
                ck(cudaSetDevice(device));Device weights(raw.size());ck(cudaMemcpy(weights.p,raw.data(),raw.size(),cudaMemcpyHostToDevice));
                cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
                for(int groups:{1,4,8}) for(int tokens=1;tokens<=8;++tokens) check(type,groups,tokens,weights,s);
                ck(cudaStreamDestroy(s));std::cout<<"device="<<device<<" type="<<type<<" original 4096x1024 groups1/4/8 NT1..8, 3 changing graph replays bitwise PASS\n";
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
