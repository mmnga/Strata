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

struct Error {double max_abs=0,relative_rms=0;};
Error worst_original,worst_fp64,original_fp64;
int fp64_cases=0;
template<class T> Error error(const std::vector<float>& a,const std::vector<T>& b) {
    require(a.size()==b.size(),"error vector size");Error e;double ss=0,ref=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(double(b[i])),"non-finite attention result");
        const double d=double(a[i])-double(b[i]);e.max_abs=std::max(e.max_abs,std::abs(d));ss+=d*d;ref+=double(b[i])*double(b[i]);
    }
    e.relative_rms=std::sqrt(ss/std::max(ref,1e-30));return e;
}
void merge(Error& worst,Error e) {worst.max_abs=std::max(worst.max_abs,e.max_abs);worst.relative_rms=std::max(worst.relative_rms,e.relative_rms);}
void bounded(Error e,const char* what) {
    if(e.max_abs>1e-5 || e.relative_rms>1e-5) {
        std::cerr<<what<<" max_abs="<<e.max_abs<<" relative_rms="<<e.relative_rms<<'\n';throw std::runtime_error(what);
    }
}
float from_half(uint16_t bits) {__half h;std::memcpy(&h,&bits,2);return __half2float(h);}
std::vector<double> double_reference(const std::vector<float>& q,const std::vector<uint16_t>& raw,
    const std::vector<uint16_t>& comp,const std::vector<float>& sinks,int pos,const std::vector<int32_t>& ids) {
    constexpr int D=512,Heads=32,Window=128,Capacity=136;
    const int first=std::max(0,pos-Window+1),nr=pos-first+1,n=nr+std::min((pos+1)/2,512);
    auto value=[&](int row,int dim) {
        return double(row<nr ? from_half(raw[size_t((first+row)%Capacity)*D+dim]) : from_half(comp[size_t(ids[row-nr])*D+dim]));
    };
    std::vector<double> result(size_t(Heads)*D),score(n);
    for(int h=0;h<Heads;++h) {
        double m=sinks[h];
        for(int row=0;row<n;++row) {double v=0;for(int i=0;i<D;++i) v+=double(q[size_t(h)*D+i])*value(row,i);score[row]=v/std::sqrt(double(D));m=std::max(m,score[row]);}
        double denom=std::exp(double(sinks[h])-m);
        for(auto& v:score) {v=std::exp(v-m);denom+=v;}
        for(int i=0;i<D;++i) {double v=0;for(int row=0;row<n;++row) v+=value(row,i)*(score[row]/denom);result[size_t(h)*D+i]=v;}
    }
    return result;
}
void normalized_weights(cudaStream_t s,int tokens,int ratio,int first) {
    constexpr int D=512,Heads=32,Window=128,Capacity=136,Top=512;
    auto qv=values(size_t(tokens)*D*Heads);Device<float> q(qv),sinks(values(Heads));
    Device<float> actual(q.n),expected(q.n),scratch(size_t(tokens)*Heads*10*544);Device<int32_t> positions(tokens),ids(size_t(tokens)*Top);
    auto rv=values(size_t(Capacity)*D),cv=values(size_t(Top)*D);
    std::vector<uint16_t> rh(rv.size()),ch(cv.size());
    for(size_t i=0;i<rv.size();++i) rh[i]=half_bits(rv[i]);
    for(size_t i=0;i<cv.size();++i) ch[i]=half_bits(cv[i]);
    Device<uint16_t> raw(rh),comp(ch);ck(cudaDeviceSynchronize());
    Graph graph(s,[&] {
        K::attention_positions(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,expected.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s);
        K::attention_positions_chunked(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,actual.p,scratch.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s);
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
        const auto a=actual.get(s),b=expected.get(s);const auto e=error(a,b);
        bounded(e,"split attention differs from original beyond FP32 bounds");merge(worst_original,e);
        if(tokens==1 && ratio==2 && first==127) {
            const auto ref=double_reference(qv,rh,ch,values(Heads),pos[0],index);
            const auto ea=error(a,ref),eb=error(b,ref);bounded(ea,"split attention differs from FP64");bounded(eb,"original attention differs from FP64");
            merge(worst_fp64,ea);merge(original_fp64,eb);++fp64_cases;
        }
    }
}
int main() {
    try {
        int devices=0;ck(cudaGetDeviceCount(&devices));require(devices>=2,"native attention test requires TP2 GPUs");
        for(int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            worst_original={};worst_fp64={};original_fp64={};fp64_cases=0;
            for(int tokens=1;tokens<=8;++tokens) for(int ratio:{0,1,2}) for(int first:{0,127,511,1023,262136})
                normalized_weights(s,tokens,ratio,first);
            ck(cudaStreamDestroy(s));std::cout<<"device="<<device<<" Qwen split64/head-group4 D512, NT1..8/ratios0..2/five starts/two replays PASS"
                <<" original_max_abs="<<worst_original.max_abs<<" original_relative_rms="<<worst_original.relative_rms
                <<" fp64_cases="<<fp64_cases<<" candidate_fp64_max_abs="<<worst_fp64.max_abs<<" candidate_fp64_relative_rms="<<worst_fp64.relative_rms
                <<" baseline_fp64_max_abs="<<original_fp64.max_abs<<" baseline_fp64_relative_rms="<<original_fp64.relative_rms<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
