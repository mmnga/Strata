#include "strata/deepseek41/dspark_head.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;namespace K=strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct B{void*p;explicit B(size_t n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}int32_t*i(){return static_cast<int32_t*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T> read(B&b,size_t n,cudaStream_t s){std::vector<T>v(n);ck(cudaMemcpyAsync(v.data(),b.p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
void equal(const void*a,const void*b,size_t bytes){if(std::memcmp(a,b,bytes))throw std::runtime_error("conditional head changed emitted prefix/confidence or mutated skipped logits");}
std::vector<float> data(size_t n,float phase){std::vector<float>v(n);for(size_t i=0;i<n;++i)v[i]=std::sin(float(i)*.017f+phase)*.1f;return v;}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;~G(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);}};
int main(int argc,char**argv){try{if(!C::conditional_draft_available())return 77;uint64_t compared=0;
constexpr int V=129280,H=5120,M=256,Rows=5;
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t stream,aux[5];void* bodies[5];ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));for(int i=0;i<5;++i){ck(cudaStreamCreateWithFlags(&aux[i],cudaStreamNonBlocking));bodies[i]=aux[i];}
{
B w1(size_t(V)*M*4),w2(size_t(V)*M*4),cw((H+M)*4),cb(4),input(H*Rows*4),base(size_t(V)*Rows*4),anchor(4),markov(M*4),bias(V*4);
B logits[2]={B(size_t(V)*Rows*4),B(size_t(V)*Rows*4)},pick[2]={B(Rows*4),B(Rows*4)},conf[2]={B(Rows*4),B(Rows*4)};
upload(w1,data(size_t(V)*M,.1f),stream);upload(w2,data(size_t(V)*M,.6f),stream);
std::vector<float>weights(H+M,0);weights[0]=1;for(int i=0;i<M;++i)weights[H+i]=float(i%5)*.005f;upload(cw,weights,stream);upload(cb,std::vector<float>{0.f},stream);
K::SamplerParams params;params.greedy=true;params.temperature=0;ck(cudaMemsetAsync(logits[0].p,0,V*4,stream));K::sample_tokens(logits[0].f(),1,V,nullptr,0,params,pick[0].i(),stream);ck(cudaStreamSynchronize(stream));
auto view=[&](int v){return C::DraftHeadView{input.f(),w1.f(),w2.f(),cw.f(),cb.f(),anchor.i(),logits[v].f(),markov.f(),bias.f(),conf[v].f(),pick[v].i(),H,V};};
auto original=[&](int n){for(int t=0;t<n;++t){C::gather_row(w1.f(),t ? pick[0].i()+t-1:anchor.i(),markov.f(),M,V,stream);C::dense(w2.f(),markov.f(),bias.f(),M,V,1,stream);C::add(logits[0].f()+size_t(t)*V,bias.f(),logits[0].f()+size_t(t)*V,V,stream);C::draft_confidence(input.f()+size_t(t)*H,markov.f(),cw.f(),cb.f(),conf[0].f()+t,H,M,stream);K::sample_tokens(logits[0].f()+size_t(t)*V,1,V,nullptr,0,params,pick[0].i()+t,stream);}};
auto populate=[&](int n,int stop,int replay){auto x=data(H*Rows,float(replay)*.2f),b=data(size_t(V)*Rows,.3f+float(replay)*.11f);for(int t=0;t<n;++t)x[t*H]=(t==stop ? -2.f:2.f);upload(input,x,stream);upload(base,b,stream);upload(anchor,std::vector<int32_t>{42+replay*137+stop+1},stream);};
auto verify=[&](int n){auto a=read<int32_t>(pick[0],n,stream),b=read<int32_t>(pick[1],n,stream);auto p=read<float>(conf[0],n,stream),q=read<float>(conf[1],n,stream);auto old=read<float>(logits[0],size_t(n)*V,stream),now=read<float>(logits[1],size_t(n)*V,stream),raw=read<float>(base,size_t(n)*V,stream);
int take=0,computed=n;for(int t=0;t<n;++t){if(!std::isfinite(p[t]))throw std::runtime_error("nonfinite fixture confidence");if(p[t]<.75f){computed=t+1;take=std::max(1,t);break;}take=t+1;}
equal(a.data(),b.data(),take*4);equal(p.data(),q.data(),computed*4);equal(old.data(),now.data(),size_t(take)*V*4);if(take<n)equal(raw.data()+size_t(take)*V,now.data()+size_t(take)*V,size_t(n-take)*V*4);for(int t=take;t<n;++t)if(b[t]!=0)throw std::runtime_error("unreturned ID not reset on replay");for(int t=computed;t<n;++t)if(q[t]!=0)throw std::runtime_error("unexecuted confidence not reset");compared+=size_t(take)*V+take+computed+size_t(n-take)*V;};
for(int n=1;n<=Rows;++n){G graph[2];for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaMemcpyAsync(logits[v].p,base.p,size_t(n)*V*4,cudaMemcpyDeviceToDevice,stream));if(v)C::conditional_draft_head(view(v),n,.75f,stream,bodies);else original(n);ck(cudaStreamEndCapture(stream,&graph[v].g));ck(cudaGraphInstantiate(&graph[v].e,graph[v].g,nullptr,nullptr,0));}
for(int stop=-1;stop<n;++stop)for(int replay=0;replay<2;++replay){populate(n,stop,replay);for(int v=0;v<2;++v){ck(cudaGraphLaunch(graph[v].e,stream));ck(cudaStreamSynchronize(stream));}verify(n);}
if(gpu==0&&n==5&&(argc==1||std::string(argv[1])!="--correctness-only")){populate(n,2,1);auto launch=[&](int v){ck(cudaGraphLaunch(graph[v].e,stream));ck(cudaStreamSynchronize(stream));};for(int i=0;i<16;++i)launch(i%2);double totals[2]={};int faster=0;
for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;totals[v]+=ms[v];}faster+=ms[1]<ms[0];verify(n);std::cout<<"pair="<<pair<<" original_ms="<<ms[0]<<" conditional_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
std::cout.precision(12);std::cout<<"GATE original5-position Markov chain vs same2-emitted-prefix conditional sum_old_ms="<<totals[0]<<" sum_new_ms="<<totals[1]<<" ratio="<<totals[0]/totals[1]<<" faster="<<faster<<"/16; includes common base-logit D2D reset, submission/sync; not model throughput\n";
}
}
}for(auto s:aux)ck(cudaStreamDestroy(s));ck(cudaStreamDestroy(stream));}
std::cout<<"Conditional original DSpark head PASS compared_values="<<compared<<" GPUs2 count1..5 every-stopping-position/allhigh twochanginginputs/anchors graphreplay emittedIDs/conf/logits bitwise and skippedlogits immutable; no stale reuse\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
