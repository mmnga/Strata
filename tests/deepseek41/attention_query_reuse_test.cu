#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/attention_async.hpp"
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace K = strata::deepseek41::cuda;
int reuse_capacity=0;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b,const char* what) { if (!b) throw std::runtime_error(what); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t size) : n(size) { ck(cudaMalloc(&p,n*sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { set(v); }
    ~Device() { cudaFree(p); }
    void set(const std::vector<T>& v) { require(v.size()==n,"upload size"); ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); ck(cudaDeviceSynchronize()); }
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

void normalized_weights(cudaStream_t s,int tokens,int ratio,int first,bool pair_heads) {
    constexpr int D=512,Heads=32,Window=128,Capacity=136,Top=512;
    auto qv=values(size_t(tokens)*D*Heads);Device<float> q(qv),sinks(values(Heads));
    Device<float> actual(q.n),expected(q.n),gridded(q.n);Device<uint8_t> scratch(size_t(tokens)*Heads*640*12);Device<int32_t> positions(tokens),ids(size_t(tokens)*Top);
    auto rv=values(size_t(Capacity)*D),cv=values(size_t(Top)*D);
    std::vector<uint16_t> rh(rv.size()),ch(cv.size());
    for(size_t i=0;i<rv.size();++i) rh[i]=half_bits(rv[i]);
    for(size_t i=0;i<cv.size();++i) ch[i]=half_bits(cv[i]);
    const uint16_t edges[8]={0x0001,0x8001,0x0000,0x8000,0x03ff,0x83ff,0x0400,0x8400};
    for(int i=0;i<8;++i) {rh[i]=edges[i];ch[i]=edges[7-i];}
    Device<uint16_t> raw(rh),comp(ch);ck(cudaDeviceSynchronize());
    Graph graph(s,[&] {
        K::attention_positions(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,expected.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s);
        K::attention_positions_score_grid_async(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,actual.p,scratch.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s,0);
        K::attention_positions_score_grid_async(q.p,raw.p,ratio ? comp.p : nullptr,ratio ? ids.p : nullptr,sinks.p,gridded.p,scratch.p,
            D,Heads,tokens,Window,positions.p,ratio,ratio ? Top : 0,Top,Capacity,s,reuse_capacity);
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
        exact(actual.get(s),expected.get(s),"original paired head baseline changed original attention bits");
        exact(gridded.get(s),expected.get(s),"query_reuse head changed original dot/whole-head softmax/FMA order");
    }
}

bool timing(cudaStream_t s){
 constexpr int D=512,H=32,N=3,W=128,Cap=136,Top=512;auto qv=values(size_t(N)*D*H);Device<float>q(qv),sinks(values(H));
 Device<float>out[2]={Device<float>(q.n),Device<float>(q.n)};Device<uint8_t>scratch(size_t(N)*H*640*12);Device<int32_t>pos(std::vector<int32_t>{1021,1022,1023});
 std::vector<int32_t>index(N*Top);for(int t=0;t<N;++t)for(int j=0;j<Top;++j)index[t*Top+j]=(37*j+7*t)%Top;Device<int32_t>ids(index);
 auto rv=values(size_t(Cap)*D),cv=values(size_t(Top)*D);std::vector<uint16_t>rh(rv.size()),ch(cv.size());for(size_t i=0;i<rh.size();++i)rh[i]=half_bits(rv[i]);for(size_t i=0;i<ch.size();++i)ch[i]=half_bits(cv[i]);Device<uint16_t>raw(rh),comp(ch);
 Graph baseline(s,[&]{K::attention_positions_score_grid_async(q.p,raw.p,comp.p,ids.p,sinks.p,out[0].p,scratch.p,D,H,N,W,pos.p,2,Top,Top,Cap,s,0);});
 Graph candidate(s,[&]{K::attention_positions_score_grid_async(q.p,raw.p,comp.p,ids.p,sinks.p,out[1].p,scratch.p,D,H,N,W,pos.p,2,Top,Top,Cap,s,reuse_capacity);});
 auto launch=[&](int v){if(v)candidate.run(s);else baseline.run(s);};for(int i=0;i<16;++i)launch(i%2);exact(out[0].get(s),out[1].get(s),"paired/query_reuse timing oracle changed bits");double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto start=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/16;total[v]+=ms[v];}faster+=ms[1]<ms[0];exact(out[0].get(s),out[1].get(s),"timed query_reuse attention changed bits");std::cout<<"pair="<<pair<<" adopted_async_ms="<<ms[0]<<" query_reuse_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE full QK+original softmax+value attention NT3/head32/D512/ratio2/640rows sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; excludes other projections/index ranking; not whole-model gain\n";
 return total[0]/total[1]>1.05&&faster==16;
}
int main(int argc,char**argv){try{bool gate_pass=true;for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));int sms=0,blocks=0;reuse_capacity=K::attention_score_reuse_capacity(&sms,&blocks);std::cout<<"CAP gpu="<<gpu<<" SMs="<<sms<<" active_blocks_per_SM="<<blocks<<" total="<<reuse_capacity<<std::endl;cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));for(int n=1;n<=8;++n)for(int ratio:{0,1,2})for(int first:{0,127,511,1023,262136})normalized_weights(s,n,ratio,first,true);std::cout<<"NUM gpu="<<gpu<<" original/pair/query_reuse attention BITWISE NT1..8 ratio0/1/2 ring/fullcontext/signedzero/subnormal/changedcaptures PASS"<<std::endl;if(gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only"))gate_pass=timing(s);ck(cudaStreamDestroy(s));}std::cout<<"QUERY_REUSE attention numeric PASS 35389440 query_reuse values bitwise original, plus paired baseline same values; 786432 additional paired gate values if timing enabled\n";if(!gate_pass)throw std::runtime_error("query_reuse full attention speed gate failed >1.05 and16/16 required");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
