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

void cache_positions(cudaStream_t s,int tokens,int ratio) {
    constexpr int W=17, Capacity=136, Ring=16, Context=262144;
    const int rows=(Context+ratio-1)/ratio;
    const auto input=values(W*tokens), initial=values(W*Ring);
    auto score=values(W*tokens), initial_score=values(W*Ring);
    for (auto& v : score) v=v*-2.3f+0.19f;
    for (auto& v : initial_score) v=v*1.7f-0.3f;
    Device<float> x(input), scores(score), rv(initial), rs(initial_score), pool(W*tokens), reference(W*tokens);
    Device<int32_t> pos(tokens); Device<uint16_t> raw(W*Capacity), comp(size_t(W)*rows);
    Graph graph(s,[&] {
        K::store_f16_positions(x.p,raw.p,W,tokens,pos.p,Capacity,0,s);
        K::projection_ring(x.p,scores.p,rv.p,rs.p,W,tokens,pos.p,Ring,s);
        K::pool_positions(rv.p,rs.p,pool.p,W,tokens,pos.p,Ring,ratio,s);
        K::store_f16_positions(pool.p,comp.p,W,tokens,pos.p,rows,ratio,s);
    });
    for (int first : {0,1,15,16,127,128,511,512,262136}) {
        if (first+tokens>Context) continue;
        std::vector<int32_t> p(tokens); std::iota(p.begin(),p.end(),first); pos.set(p);
        rv.set(initial); rs.set(initial_score); ck(cudaMemsetAsync(raw.p,0x7f,raw.n*2,s)); ck(cudaMemsetAsync(comp.p,0x7f,comp.n*2,s));
        graph.run(s);
        auto expected_ring=initial, expected_score=initial_score;
        std::vector<uint16_t> expected_raw(raw.n,0x7f7f), expected_comp(comp.n,0x7f7f);
        for (int t=0;t<tokens;++t) for (int i=0;i<W;++i) {
            expected_ring[size_t(p[t]%Ring)*W+i]=input[size_t(t)*W+i];
            expected_score[size_t(p[t]%Ring)*W+i]=score[size_t(t)*W+i];
            expected_raw[size_t(p[t]%Capacity)*W+i]=half_bits(input[size_t(t)*W+i]);
        }
        exact(rv.get(s),expected_ring,"projection ring replay differs");
        exact(rs.get(s),expected_score,"projection score ring replay differs");
        exact(raw.get(s),expected_raw,"raw position store differs");
        ck(cudaMemsetAsync(reference.p,0,reference.n*4,s));
        for (int t=0;t<tokens;++t) if ((p[t]+1)%ratio==0) {
            const int offset=(p[t]+1-ratio)%Ring;
            K::pool(rv.p+offset*W,rs.p+offset*W,reference.p+t*W,W,ratio,s);
        }
        const auto pooled=reference.get(s); exact(pool.get(s),pooled,"position pool differs from legacy pool");
        for (int t=0;t<tokens;++t) if ((p[t]+1)%ratio==0) for (int i=0;i<W;++i)
            expected_comp[size_t(p[t]/ratio)*W+i]=half_bits(pooled[size_t(t)*W+i]);
        exact(comp.get(s),expected_comp,"compressed store changed incomplete or wrong row");
    }
}

void indexed_attention(cudaStream_t s,int tokens,int ratio,int first,int max_rows) {
    constexpr int D=512, Heads=32, I=128, IndexHeads=32, Window=128, Capacity=136, Top=512;
    const int stride=max_rows;
    int bucket=1; if (ratio) while (bucket<(first+tokens)/ratio) bucket*=2;
    require(bucket<=max_rows,"index test capacity");
    Device<float> q(values(size_t(tokens)*D*Heads)), iq(values(size_t(tokens)*I*IndexHeads));
    Device<float> weights(values(size_t(tokens)*IndexHeads)), sinks(values(Heads));
    Device<float> actual(size_t(tokens)*Heads*D), expected(actual.n), scores(size_t(stride)*tokens), old_scores(stride);
    Device<int32_t> ids(size_t(stride)*tokens), old_ids(stride), positions(tokens);
    Device<uint16_t> raw(size_t(Capacity)*D), comp(size_t(max_rows)*D), keys(size_t(max_rows)*I);
    ck(cudaMemset(raw.p,0x30,raw.n*2)); ck(cudaMemset(comp.p,0x28,comp.n*2)); ck(cudaMemset(keys.p,0,keys.n*2));
    const int seed_rows=std::min(max_rows,1025);
    auto rv=values(size_t(Capacity)*D), cv=values(size_t(seed_rows)*D), kv=values(size_t(seed_rows)*I);
    std::vector<uint16_t> raw_host(rv.size()), comp_host(cv.size()), key_host(kv.size());
    for(size_t i=0;i<rv.size();++i) raw_host[i]=half_bits(rv[i]);
    for(size_t i=0;i<cv.size();++i) comp_host[i]=half_bits(cv[i]);
    for(size_t i=0;i<kv.size();++i) key_host[i]=half_bits(kv[i]);
    raw.set(raw_host); ck(cudaMemcpy(comp.p,comp_host.data(),comp_host.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(keys.p,key_host.data(),key_host.size()*2,cudaMemcpyHostToDevice));
    Device<uint8_t> scratch(K::top_k_scratch_bytes(max_rows));
    Graph graph(s,[&] {
        if (ratio) {
            K::index_scores_positions(iq.p,weights.p,keys.p,scores.p,I,IndexHeads,bucket,tokens,positions.p,ratio,stride,s);
            for(int t=0;t<tokens;++t) K::top_k_sorted(scores.p+size_t(t)*stride,ids.p+size_t(t)*stride,bucket,std::min(Top,bucket),scratch.p,scratch.n,s);
        }
        K::attention_positions(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,actual.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,stride,Capacity,s);
    });
    for(int replay=0;replay<2;++replay) {
        const int start=first+replay;
        if (start+tokens>262144 || (ratio && (start+tokens)/ratio>bucket)) continue;
        std::vector<int32_t> p(tokens); std::iota(p.begin(),p.end(),start); positions.set(p); graph.run(s);
        for(int t=0;t<tokens;++t) {
            const int rows=ratio ? (p[t]+1)/ratio : 0, selected=std::min(rows,Top);
            if (ratio && bucket>rows) {
                std::vector<float> padding(bucket-rows);
                ck(cudaMemcpy(padding.data(),scores.p+size_t(t)*stride+rows,padding.size()*4,cudaMemcpyDeviceToHost));
                require(std::all_of(padding.begin(),padding.end(),[](float v) { return std::isinf(v) && v<0; }),"future index rows were not masked");
            }
            if (rows) {
                K::index_scores(iq.p+size_t(t)*I*IndexHeads,weights.p+size_t(t)*IndexHeads,keys.p,old_scores.p,I,IndexHeads,rows,s);
                K::top_k_sorted(old_scores.p,old_ids.p,rows,selected,scratch.p,scratch.n,s); ck(cudaStreamSynchronize(s));
                std::vector<int32_t> a(selected), b(selected);
                ck(cudaMemcpy(a.data(),ids.p+size_t(t)*stride,selected*4,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(b.data(),old_ids.p,selected*4,cudaMemcpyDeviceToHost)); exact(a,b,"masked bucket changed stable selected IDs");
                std::vector<float> x(rows),y(rows);
                ck(cudaMemcpy(x.data(),scores.p+size_t(t)*stride,rows*4,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(y.data(),old_scores.p,rows*4,cudaMemcpyDeviceToHost)); exact(x,y,"dynamic visible index scores differ");
            }
            K::attention(q.p+size_t(t)*Heads*D,raw.p,ratio ? comp.p : nullptr,rows ? old_ids.p : nullptr,sinks.p,
                expected.p+size_t(t)*Heads*D,D,Heads,Window,p[t],selected,s,Capacity);
        }
        exact(actual.get(s),expected.get(s),"position attention differs from legacy scalar calls");
    }
}
void draft_block(cudaStream_t s,int tokens) {
    constexpr int H=5120, M=256, D=512, Heads=64, Window=128, Capacity=136;
    Device<float> q(values(size_t(tokens)*D*Heads)), sinks(values(Heads));
    Device<float> actual(size_t(tokens)*D*Heads), expected(actual.n);
    auto rv=values(size_t(Capacity)*D); std::vector<uint16_t> raw_host(rv.size());
    for(size_t i=0;i<rv.size();++i) raw_host[i]=half_bits(rv[i]);
    Device<uint16_t> raw(raw_host); Device<int32_t> pos(tokens), id(1);
    auto hv=values(H), mv=values(M*7), wv=values(H+M);
    for(auto& v:wv) v*=0.001f;
    Device<float> hidden(hv), markov(mv), gathered(M), weight(wv), bias(std::vector<float>{0.17f});
    Device<float> conf(1), feat(H+M), dot(1);
    Graph graph(s,[&] {
        K::draft_attention_positions(q.p,raw.p,sinks.p,actual.p,D,Heads,tokens,Window,pos.p,Capacity,s);
        K::gather_row(markov.p,id.p,gathered.p,M,7,s);
        K::draft_confidence(hidden.p,gathered.p,weight.p,bias.p,conf.p,H,M,s);
    });
    int replay=0;
    for(int first : {0,1,127,128,511,262144-tokens}) {
        const int row=(replay++*3)%7;
        std::vector<int32_t> positions(tokens); std::iota(positions.begin(),positions.end(),first);
        pos.set(positions); id.set({row}); graph.run(s);
        for(int t=0;t<tokens;++t)
            K::draft_attention(q.p+size_t(t)*D*Heads,raw.p,sinks.p,expected.p+size_t(t)*D*Heads,
                D,Heads,Window,first+t,first+tokens-1,Capacity,s);
        exact(actual.get(s),expected.get(s),"DSpark noncausal graph attention differs");
        std::vector<float> gathered_expected(mv.begin()+row*M,mv.begin()+(row+1)*M);
        exact(gathered.get(s),gathered_expected,"device Markov predecessor gather differs");
        auto f=hv; f.insert(f.end(),gathered_expected.begin(),gathered_expected.end()); feat.set(f);
        K::dense(weight.p,feat.p,dot.p,H+M,1,1,s);
        const float reference=1.0f/(1.0f+std::exp(-(dot.get(s)[0]+0.17f)));
        require(std::abs(conf.get(s)[0]-reference)<2e-7f,"DSpark confidence differs from original concatenated dense head");
    }
}
int main() {
    try {
        int devices=0; ck(cudaGetDeviceCount(&devices));
        for(int d=0;d<devices;++d) {
            ck(cudaSetDevice(d)); cudaStream_t s=nullptr; ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            for(int t : {1,2,5,8}) for(int ratio : {1,2}) cache_positions(s,t,ratio);
            for(int t : {1,2,8}) for(int ratio : {0,1,2}) for(int first : {0,1,127,128,511,512,1023})
                indexed_attention(s,t,ratio,first,2048);
            for(int ratio : {1,2}) indexed_attention(s,8,ratio,262136,262144);
            for(int tokens : {1,5}) draft_block(s,tokens);
            ck(cudaStreamDestroy(s)); std::cout<<"device="<<d<<" cache masks, wrap, graph replay, padded stable sort and attention bitwise PASS\n";
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
