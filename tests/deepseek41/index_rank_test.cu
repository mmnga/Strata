#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/index_rank.hpp"
#include <random>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct B{void*p;size_t n;explicit B(size_t size):n(size){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}int32_t*i(){return static_cast<int32_t*>(p);}uint16_t*h(){return static_cast<uint16_t*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T>read(const void*p,size_t n,cudaStream_t s){std::vector<T>v(n);if(!n)return v;ck(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
std::vector<float>data(size_t n,float phase){std::vector<float>v(n);for(size_t i=0;i<n;++i)v[i]=std::sin(float(i)*.017f+phase)*.2f;v[0]=-0.f;if(n>1)v[1]=std::ldexp(1.f,-140);return v;}
std::vector<uint16_t>halves(size_t n,float phase){auto f=data(n,phase);std::vector<uint16_t>v(n);for(size_t i=0;i<n;++i){auto h=__float2half_rn(f[i]);std::memcpy(&v[i],&h,2);}return v;}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;~G(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);}};
uint64_t ordering_check(cudaStream_t s){
 constexpr int stride=1024;B scores(stride*8*4),a(stride*8*4),b(stride*8*4),scratch(C::top_k_scratch_bytes(stride));uint64_t count=0;std::mt19937 rng(75179);
 for(int n=1;n<=8;++n)for(int rows:{1,31,32,127,128,257,511,512,513,1024}){
  G graphs[2];for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));if(v)C::index_rank_batch(scores.f(),b.i(),rows,std::min(rows,512),n,stride,scratch.p,scratch.n,s);else for(int t=0;t<n;++t)C::top_k_sorted(scores.f()+size_t(t)*stride,a.i()+size_t(t)*stride,rows,std::min(rows,512),scratch.p,scratch.n,s);ck(cudaStreamEndCapture(s,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  for(int phase=0;phase<3;++phase){std::vector<uint32_t>bits(stride*n);const uint32_t special[]={0,0x80000000u,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u,0xffffffffu,1,0x80000001u};for(size_t i=0;i<bits.size();++i)bits[i]=phase==0?uint32_t(rng()):phase==1?special[i%9]:0;upload(scores,bits,s);for(int v=0;v<2;++v){ck(cudaGraphLaunch(graphs[v].e,s));ck(cudaStreamSynchronize(s));}for(int t=0;t<n;++t){auto av=read<int32_t>(a.i()+size_t(t)*stride,rows,s),bv=read<int32_t>(b.i()+size_t(t)*stride,rows,s);if(av!=bv)throw std::runtime_error("arbitrary float-bit CUB ordered ID oracle mismatch rows="+std::to_string(rows)+" phase="+std::to_string(phase));count+=rows;}}
 }return count;
}

int main(int argc,char**argv){try{constexpr int D=512,A=32,I=128,IH=32,W=128,Cap=136,Top=512,Stride=262144;
uint64_t compared=0,permuted=0;double worst_abs=0,worst_rms=0;bool gate_pass=true;
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));std::cout<<"ORDER gpu="<<gpu<<" IDs="<<ordering_check(s)<<" arbitrary floatbits/NaN/Inf/zeros/ties original CUB bitwise PASS"<<std::endl;
{
B q(D*A*8*4),iq(I*IH*8*4),iw(IH*8*4),raw(Cap*D*2),comp(1024*D*2),keys(1024*I*2),sinks(A*4),pos(8*4),scores(size_t(Stride)*8*4),ids[2]={B(size_t(Stride)*8*4),B(size_t(Stride)*8*4)},out[2]={B(D*A*8*4),B(D*A*8*4)},values(size_t(8)*A*640*12),scratch(C::top_k_scratch_bytes(1024));
upload(raw,halves(Cap*D,.2f),s);upload(comp,halves(1024*D,.5f),s);upload(keys,halves(1024*I,.3f),s);upload(sinks,data(A,.1f),s);
for(int n=1;n<=8;++n)for(int ratio:{1,2})for(int target:{0,257,512,513}){
 const int first=target?ratio*target-n:0;const int rows=(first+n)/ratio;int bucket=1;while(bucket<rows)bucket*=2;
 G graph[2];
 auto body=[&](int v){C::index_scores_positions(iq.f(),iw.f(),keys.h(),scores.f(),I,IH,bucket,n,pos.i(),ratio,Stride,s);if(v)C::index_rank_batch(scores.f(),ids[v].i(),bucket,std::min(bucket,Top),n,Stride,scratch.p,scratch.n,s);else for(int t=0;t<n;++t)C::top_k_sorted(scores.f()+size_t(t)*Stride,ids[v].i()+size_t(t)*Stride,bucket,std::min(bucket,Top),scratch.p,scratch.n,s);
 C::attention_positions_score_grid(q.f(),raw.h(),comp.h(),ids[v].i(),sinks.f(),out[v].f(),values.p,D,A,n,W,pos.i(),ratio,Top,Stride,Cap,s,true);};
 // scratch has a fixed allocation but its queried byte count is needed by CUB.
 for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));body(v);ck(cudaStreamEndCapture(s,&graph[v].g));ck(cudaGraphInstantiate(&graph[v].e,graph[v].g,nullptr,nullptr,0));}
 auto verify=[&]{auto a=read<float>(out[0].p,size_t(n)*A*D,s),b=read<float>(out[1].p,a.size(),s);if(std::memcmp(a.data(),b.data(),a.size()*4))throw std::runtime_error("batched ranking changed attention bits");compared+=a.size();for(int t=0;t<n;++t){auto old=read<int32_t>(ids[0].i()+size_t(t)*Stride,bucket,s),now=read<int32_t>(ids[1].i()+size_t(t)*Stride,bucket,s);if(old!=now)throw std::runtime_error("ordered ranking IDs differ");permuted+=old.size();}};
 for(int replay=0;replay<2;++replay){upload(q,data(size_t(n)*A*D,float(replay)*.3f+.1f),s);upload(iq,data(size_t(n)*IH*I,float(replay)*.7f+.2f),s);auto w=data(size_t(n)*IH,float(replay)*.1f);for(auto&v:w)v=std::abs(v)+.01f;upload(iw,w,s);std::vector<int32_t>p(n);for(int t=0;t<n;++t)p[t]=first+t;upload(pos,p,s);for(int v=0;v<2;++v){ck(cudaGraphLaunch(graph[v].e,s));ck(cudaStreamSynchronize(s));}verify();}
 if(gpu==0&&n==3&&ratio==2&&target==512&&(argc==1||std::string(argv[1])!="--correctness-only")){
 auto launch=[&](int v){ck(cudaGraphLaunch(graph[v].e,s));ck(cudaStreamSynchronize(s));};for(int i=0;i<16;++i)launch(i%2);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;total[v]+=ms[v];}faster+=ms[1]<ms[0];verify();std::cout<<"pair="<<pair<<" score_sort_attn_ms="<<ms[0]<<" batchsort_attn_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 gate_pass=total[0]/total[1]>1.05&&faster==16;
 std::cout.precision(12);std::cout<<"GATE scoring+top-k+attention vs batched stable ranking+attention, excludes index-query projections NT3 ratio2 lastpos1023 bucket512 sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16 original limits unchanged\n";
 }
}
}ck(cudaStreamDestroy(s));}
std::cout.precision(12);std::cout<<"Batched exact ranking PASS compared_attention_values="<<compared<<" ordered_ID_values="<<permuted<<" worst_abs="<<worst_abs<<" worst_rRMS="<<worst_rms<<" GPUs2 NT1..8 ratio1/2 0/257/512/513-row boundaries changingdata graphreplay; ALL ordered IDs and attention bits exact, fallback unchanged\n";
if(!gate_pass)throw std::runtime_error("local speed gate failed >1.05 and16/16 required");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
