#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace K=strata::kernels;
void ck(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
template<class T> struct Device {
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count) {ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()) {set(v);}
    ~Device() {cudaFree(p);}
    void set(const std::vector<T>& v) {
        require(v.size()==n,"upload size");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
        ck(cudaStreamSynchronize(nullptr)); // Complete pageable uploads before nonblocking compute.
    }
    std::vector<T> get(cudaStream_t s) {
        ck(cudaStreamSynchronize(s));std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;
    }
};
template<class T> void exact(const std::vector<T>& a,const std::vector<T>& b,const char* s) {
    require(a.size()==b.size(),"comparison size");require(!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)),s);
}
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    template<class F> Graph(cudaStream_t s,F body) {
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));body();
        ck(cudaStreamEndCapture(s,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph() {cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    void run(cudaStream_t s) {ck(cudaGraphLaunch(exec,s));ck(cudaStreamSynchronize(s));}
};
void check(cudaStream_t stream,int type,int ni,int no,int nt) {
    const size_t block=type==12 ? 144 : 210;
    std::vector<uint8_t> bytes(size_t(ni/256)*no*block);uint32_t rng=1234567+type;
    for(auto& b:bytes) {rng=1664525*rng+1013904223;b=uint8_t(rng>>24);}
    for(size_t i=0;i<bytes.size();i+=block) {
        const uint16_t d=0x1800,minimum=0x1400;
        std::memcpy(bytes.data()+i+(type==12 ? 0 : 208),&d,2);
        if(type==12) std::memcpy(bytes.data()+i+2,&minimum,2);
    }
    Device<uint8_t> weights(bytes),plain(K::native_q8_1_bytes(ni,nt)),wide_plain(plain.n),il(K::native_q8_1_il_bytes(ni,nt));
    std::vector<float> input(size_t(ni)*nt);
    for(size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*0.1739f);
    Device<float> x(input),actual(size_t(nt)*no),expected(actual.n);
    auto body=[&] {
        K::native_quantize_q8_1(x.p,plain.p,ni,nt,stream);
        K::native_mmvq(type,weights.p,plain.p,expected.p,ni,no,nt,stream);
        K::native_quantize_q8_1_il_wide(x.p,wide_plain.p,il.p,ni,nt,stream);
        K::native_mmvq_il_wide(type,weights.p,il.p,actual.p,ni,no,nt,stream);
    };
    body();ck(cudaStreamSynchronize(stream));Graph graph(stream,body);
    for(int replay=0;replay<2;++replay) {
        for(size_t i=0;i<input.size();++i) {
            input[i]=-0.91f*input[i]+float(replay)*0.01327f;
            if(replay==0 && i<size_t(ni)) input[i]=(i&1) ? -0.0f : 0.0f;
            else if(i%257==0) {const uint32_t b=(i&1) ? 0x80000001u : 1u;std::memcpy(&input[i],&b,4);}
        }
        x.set(input);graph.run(stream);
        exact(plain.get(stream),wide_plain.get(stream),"wide IL changed original Q8_1 quantization bytes");
        const auto output=actual.get(stream);for(float v:output) require(std::isfinite(v),"nonfinite wide IL output");
        exact(output,expected.get(stream),"wide IL changed original multi-column MMVQ arithmetic");
    }
}
int main() {
    try {
        int devices=0;ck(cudaGetDeviceCount(&devices));require(devices>=2,"wide IL test requires TP2 GPUs");
        for(int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            for(int type:{12,14}) for(int nt:{5,6,7,8})
                for(const auto& shape:std::vector<std::pair<int,int>>{{5120,512},{2304,513},{5120,2304}})
                    check(s,type,shape.first,shape.second,nt);
            ck(cudaStreamDestroy(s));
            std::cout<<"device="<<device<<" wide IL Q4_K/Q6_K, NT5..8, 24 shape cells/48 changed graph replays, "
                     <<"zero rows, signed zero, subnormal inputs, partial row group: original Q8_1 bytes and MMVQ outputs bitwise PASS\n";
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
