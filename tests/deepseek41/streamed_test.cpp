#include "strata/deepseek41/streamed_experts.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace D=strata::deepseek41;
namespace K=strata::kernels;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool ok,const char* s) { if(!ok) throw std::runtime_error(s); }
struct Device {
    void* p=nullptr;
    explicit Device(size_t n) {ck(cudaMalloc(&p,n));}
    ~Device() {cudaFree(p);}
};
template<class T> struct Host {
    T* p=nullptr; size_t size;
    explicit Host(size_t n):size(n) {ck(cudaMallocHost(&p,n*sizeof(T)));}
    ~Host() {cudaFreeHost(p);}
};
void original_rows(uint8_t* out,size_t bytes,int type,std::mt19937& rng) {
    const size_t block=type==12 ? 144 : 210;
    for(size_t i=0;i<bytes;++i) out[i]=uint8_t(rng());
    for(size_t i=0;i<bytes;i+=block) {
        const uint16_t scale=0x1800, minimum=0x1400;
        std::memcpy(out+i+(type==12 ? 0 : 208),&scale,2);
        if(type==12) std::memcpy(out+i+2,&minimum,2);
    }
}
void check(int down_type) {
    constexpr int H=5120,F=2304,Experts=6;
    const auto layout=K::native_expert_layout(12,down_type,H,F);
    const size_t gu=layout.up_off, down=layout.bytes-layout.down_off;
    Host<uint8_t> gate(gu*Experts), up(gu*Experts), down_weight(down*Experts);
    std::mt19937 rng(773+down_type);
    original_rows(gate.p,gate.size,12,rng); original_rows(up.p,up.size,12,rng);
    original_rows(down_weight.p,down_weight.size,down_type,rng);
    D::StreamLayerWeights layer; layer.experts=Experts;
    layer.base={gate.p,up.p,down_weight.p,12,down_type,H,F,gu/F,down/H,1.75f};
    D::StreamedExperts streams({layer},2,3);
    ck(cudaSetDevice(0)); cudaStream_t s=nullptr; ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    Device blobs(layout.bytes*Experts), input(size_t(H)*8*4), quant(K::native_q8_1_bytes(H,8));
    for(int e=0;e<Experts;++e) {
        auto* p=static_cast<uint8_t*>(blobs.p)+size_t(e)*layout.bytes;
        ck(cudaMemcpy(p,gate.p+size_t(e)*gu,gu,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(p+gu,up.p+size_t(e)*gu,gu,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(p+2*gu,down_weight.p+size_t(e)*down,down,cudaMemcpyHostToDevice));
    }
    Device ptr(Experts*8), start((Experts+1)*4), count(4), dst(Experts*8*4), tok(Experts*8*4);
    Device scratch(K::native_expert_scratch_bytes(Experts*8,F)), expected(size_t(Experts)*8*H*4);
    Host<float> x(size_t(H)*8), actual(size_t(Experts)*8*H);
    for(size_t i=0;i<x.size;++i) x.p[i]=std::sin(float(i)*0.11f);
    for(int tokens:{1,2,5,8}) for(int owner:{0,1}) {
        ck(cudaSetDevice(0)); const int entries=Experts*tokens;
        std::vector<D::ExpertBatch> jobs(Experts);
        std::vector<unsigned long long> pointers(Experts); std::vector<int32_t> starts(Experts+1), destinations(entries), token(entries);
        for(int e=0;e<Experts;++e) {
            auto& job=jobs[size_t(e)]; job.weights=layer.base;
            job.weights.gate+=size_t(e)*gu; job.weights.up+=size_t(e)*gu; job.weights.down+=size_t(e)*down; job.count=tokens;
            pointers[size_t(e)]=reinterpret_cast<unsigned long long>(static_cast<uint8_t*>(blobs.p)+size_t(e)*layout.bytes);
            starts[size_t(e)]=e*tokens;
            for(int t=0;t<tokens;++t) {
                const int i=e*tokens+t; job.tokens[t]=(t+e)%tokens; job.destinations[t]=entries-1-i;
                destinations[size_t(i)]=job.destinations[t]; token[size_t(i)]=job.tokens[t];
            }
        }
        starts[Experts]=entries;
        ck(cudaMemcpy(input.p,x.p,size_t(H)*tokens*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(ptr.p,pointers.data(),Experts*8,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(start.p,starts.data(),(Experts+1)*4,cudaMemcpyHostToDevice));
        const int n=Experts; ck(cudaMemcpy(count.p,&n,4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dst.p,destinations.data(),size_t(entries)*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(tok.p,token.data(),size_t(entries)*4,cudaMemcpyHostToDevice));
        K::native_quantize_q8_1(static_cast<float*>(input.p),quant.p,H,tokens,s);
        K::native_expert_grouped_clamp(layout,static_cast<unsigned long long*>(ptr.p),static_cast<int32_t*>(start.p),
            static_cast<int32_t*>(count.p),static_cast<int32_t*>(dst.p),static_cast<int32_t*>(tok.p),Experts,entries,
            quant.p,scratch.p,static_cast<float*>(expected.p),s,1.75f);
        ck(cudaStreamSynchronize(s)); std::vector<float> reference(size_t(entries)*H);
        ck(cudaMemcpy(reference.data(),expected.p,reference.size()*4,cudaMemcpyDeviceToHost));
        std::fill(actual.p,actual.p+actual.size,123.5f);
        const auto cpu=streams.submit(0,owner,jobs,x.p,tokens);
        require(!cpu.empty() && cpu.size()<jobs.size(),"stream planner failed to exercise split work");
        streams.finish(actual.p);
        std::vector<bool> cpu_destination(size_t(entries),false);
        for(const auto& job:cpu) for(int t=0;t<job.count;++t) cpu_destination[size_t(job.destinations[t])]=true;
        for(int row=0;row<entries;++row) for(int j=0;j<H;++j) {
            const size_t i=size_t(row)*H+j;
            if(cpu_destination[size_t(row)]) require(actual.p[i]==123.5f,"stream finish overwrote CPU output");
            else require(std::isfinite(actual.p[i]) && std::memcmp(actual.p+i,reference.data()+i,4)==0,"streamed original FFN differs from cached native FFN");
        }
    }
    streams.report(); ck(cudaSetDevice(0)); ck(cudaStreamDestroy(s));
}
int main() {
    try { for(int type:{12,14}) check(type); std::cout<<"Native streamed experts vs cached FFN, both owners, NT1/2/5/8, scattered routes and CPU preservation bitwise PASS\n"; }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
