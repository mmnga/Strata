#include "strata/deepseek41/kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace K = strata::deepseek41::cuda;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b,const char* what) { if (!b) throw std::runtime_error(what); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t size) : n(size) { ck(cudaMalloc(&p,n*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { set(v); }
    ~Device() { cudaFree(p); }
    void set(const std::vector<T>& v) { require(v.size()==n,"upload size"); ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(cudaStream_t s) { ck(cudaStreamSynchronize(s)); std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
std::vector<float> values(size_t n) { std::vector<float> v(n); for (size_t i=0;i<n;++i) v[i]=std::sin(float(i)*0.1789f+0.71f); return v; }
template<class T> void exact(const std::vector<T>& a,const std::vector<T>& b,const char* what) {
    require(a.size()==b.size(),"comparison size");
    if (std::memcmp(a.data(),b.data(),a.size()*sizeof(T))) throw std::runtime_error(what);
}
struct Graph {
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    template<class F> Graph(cudaStream_t s,F body) {
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal)); body();
        ck(cudaStreamEndCapture(s,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run(cudaStream_t s) { ck(cudaGraphLaunch(exec,s)); ck(cudaStreamSynchronize(s)); }
};
uint16_t half_bits(float v) { const auto h=__float2half_rn(v); uint16_t b; std::memcpy(&b,&h,2); return b; }

void normalized_weights(cudaStream_t s,int tokens,int ratio,int first) {
    constexpr int D=512,Heads=32,Window=128,Capacity=136,Top=512;
    auto qv=values(size_t(tokens)*D*Heads);Device<float> q(qv),sinks(values(Heads));
    Device<float> actual(q.n),expected(q.n),prefetched(q.n);Device<uint8_t> scratch(size_t(tokens)*Heads*640*12);Device<int32_t> positions(tokens),ids(size_t(tokens)*Top);
    auto rv=values(size_t(Capacity)*D),cv=values(size_t(Top)*D);
    std::vector<uint16_t> rh(rv.size()),ch(cv.size());
    for(size_t i=0;i<rv.size();++i) rh[i]=half_bits(rv[i]);
    for(size_t i=0;i<cv.size();++i) ch[i]=half_bits(cv[i]);
    Device<uint16_t> raw(rh),comp(ch);ck(cudaDeviceSynchronize());
    Graph graph(s,[&] {
        K::attention_positions(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,expected.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s);
        K::attention_positions_query_prefetch(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,actual.p,scratch.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s,false);
        K::attention_positions_query_prefetch(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,prefetched.p,scratch.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s,true);
    });
    for(int replay=0;replay<2;++replay) {
        const int start=std::min(first+replay,262144-tokens);
        std::vector<int32_t> pos(tokens),index(size_t(tokens)*Top);
        std::iota(pos.begin(),pos.end(),start);
        for(int t=0;t<tokens;++t) {
            const int n=ratio ? std::min((pos[t]+1)/ratio,Top) : 0;
            for(int j=0;j<n;++j) index[size_t(t)*Top+j]=(j+replay*(t+7))%n;
        }
        for(size_t i=0;i<qv.size();++i) qv[i]=qv[i]*-0.91f+float(replay)*0.013f;
        ck(cudaMemcpyAsync(q.p,qv.data(),qv.size()*4,cudaMemcpyHostToDevice,s));
        ck(cudaMemcpyAsync(positions.p,pos.data(),pos.size()*4,cudaMemcpyHostToDevice,s));
        ck(cudaMemcpyAsync(ids.p,index.data(),index.size()*4,cudaMemcpyHostToDevice,s));
        graph.run(s);
        exact(actual.get(s),expected.get(s),"query/key prefetch changed original dot/FMA order");
        exact(prefetched.get(s),expected.get(s),"query/key plus value prefetch changed original dot/FMA order");
    }
}
int main() {
    try {
        int devices=0;ck(cudaGetDeviceCount(&devices));require(devices>=2,"native attention test requires TP2 GPUs");
        for(int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            for(int tokens=1;tokens<=8;++tokens) for(int ratio:{0,1,2}) for(int first:{0,127,511,1023,262136})
                normalized_weights(s,tokens,ratio,first);
            ck(cudaStreamDestroy(s));std::cout<<"device="<<device<<" query/key prefetch with/without value prefetch bitwise PASS, D512/head32, NT1..8, ratios0/1/2, sink, ring wrap, full context, two changed graph replays\n";
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
