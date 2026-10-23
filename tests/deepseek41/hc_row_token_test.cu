#include "strata/deepseek41/hc_coeff.hpp"
#include "strata/deepseek41/norm_quant.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;namespace K=strata::kernels;
constexpr int H=5120,Q=1280,MaxT=8;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
struct B{void*p=nullptr;size_t bytes;explicit B(size_t n):bytes(n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){need(v.size()*sizeof(T)<=b.bytes,"upload bounds");ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T>read(const void*p,size_t n,cudaStream_t s){std::vector<T>v(n);ck(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
template<class T>void exact(const std::vector<T>&a,const std::vector<T>&b,const char*message){need(a.size()==b.size(),"comparison bounds");if(std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(message);}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;template<class F>G(cudaStream_t s,F f){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));f();ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));}~G(){cudaGraphExecDestroy(e);cudaGraphDestroy(g);}void run(cudaStream_t s){ck(cudaGraphLaunch(e,s));ck(cudaStreamSynchronize(s));}
 void require_weight(const void*w){size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));bool found=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));if(p.kernelParams&&p.kernelParams[0]){void*value=nullptr;std::memcpy(&value,p.kernelParams[0],sizeof(value));found|=value==w;}}need(found,"cold graph does not reference its actual weight bank");}
};


void require_row_grid(G&g,const void*weights,int tokens){size_t count=0;ck(cudaGraphGetNodes(g.g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g.g,nodes.data(),&count));bool found=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));if(!p.kernelParams||!p.kernelParams[0])continue;void*w=nullptr;std::memcpy(&w,p.kernelParams[0],sizeof(w));if(w==weights&&p.gridDim.x==24&&p.gridDim.y==unsigned(tokens)&&p.blockDim.x==32&&p.blockDim.y==4)found=true;}need(found,"HC graph did not capture actual matrix with24xtokens row/token CTAs");}
constexpr int Flat=20480,Mix=24;
struct Workspace {
 B residual{size_t(Flat)*MaxT*4},scale{3*4},base{24*4},inverse{MaxT*4};
 B norm[2]={B(size_t(Flat)*MaxT*4),B(size_t(Flat)*MaxT*4)},mix[2]={B(Mix*MaxT*4),B(Mix*MaxT*4)};
 B q[2]={B(K::native_q8_1_bytes(Flat,MaxT)),B(K::native_q8_1_bytes(Flat,MaxT))},il[2]={B(K::native_q8_1_il_bytes(Flat,MaxT)),B(K::native_q8_1_il_bytes(Flat,MaxT))};
 B pre[2]={B(4*MaxT*4),B(4*MaxT*4)},post[2]={B(4*MaxT*4),B(4*MaxT*4)},comb[2]={B(16*MaxT*4),B(16*MaxT*4)};
 void input(int n,int phase,cudaStream_t s){std::vector<float>x(size_t(Flat)*n),sc={.3f,-.27f,.43f},b(24);for(size_t i=0;i<x.size();++i)x[i]=phase?std::sin(i*.013f+.31f*phase)*.2f:0.f;for(int t=0;t<n;++t){x[size_t(t)*Flat]=1.f;x[size_t(t)*Flat+33]=-0.f;x[size_t(t)*Flat+34]=std::ldexp(1.f,-140);}for(int i=0;i<24;++i)b[i]=std::sin(i*.73f+.19f*phase)*.5f;upload(residual,x,s);upload(scale,sc,s);upload(base,b,s);for(auto&v:il)ck(cudaMemsetAsync(v.p,0xa5,v.bytes,s));ck(cudaStreamSynchronize(s));}
 void body(int v,const void*weights,int n,cudaStream_t s){
  C::hc_rms_stats(residual.f(),inverse.f(),Flat,n,1e-20f,s);
  const int cp=!v&&n>=5?K::native_q8_1_il_cp(n):0;
  C::norm_quant_q8(residual.f(),inverse.f(),nullptr,norm[v].f(),q[v].p,cp?il[v].p:nullptr,Flat,n,cp,s);
  if(v)K::native_hc_q4_k_row_tokens(weights,q[1].p,mix[1].f(),n,s);
  else if(n>=5)K::native_mmvq_il_wide(12,weights,il[0].p,mix[0].f(),Flat,Mix,n,s);
  else K::native_mmvq(12,weights,q[0].p,mix[0].f(),Flat,Mix,n,s);
  C::hc_coeff_warp(mix[v].f(),scale.f(),base.f(),pre[v].f(),post[v].f(),comb[v].f(),n,1e-6f,20,s);
 }
 void compare(B(&v)[2],size_t n,const char*msg,cudaStream_t s){exact(read<float>(v[0].p,n,s),read<float>(v[1].p,n,s),msg);}
 void verify(int n,cudaStream_t s,bool observable=true){auto a=read<float>(norm[0].p,size_t(n)*Flat,s);exact(a,read<float>(norm[1].p,size_t(n)*Flat,s),"HC normalized F32 differs");if(observable){auto inv=read<float>(inverse.p,n,s);for(int t=0;t<n;++t)need(!std::memcmp(&inv[t],&a[size_t(t)*Flat],4),"RMS inverse differs from original observable x0=1");}
  exact(read<uint8_t>(q[0].p,K::native_q8_1_bytes(Flat,n),s),read<uint8_t>(q[1].p,K::native_q8_1_bytes(Flat,n),s),"HC plain Q81 bytes differ");// Candidate intentionally emits no IL; compare all original plain bytes instead.
  compare(mix,n*Mix,"HC Q4K projection differs",s);compare(pre,n*4,"HC pre differs",s);compare(post,n*4,"HC post differs",s);compare(comb,n*16,"HC ordered Sinkhorn differs",s);
 }
};
std::vector<uint8_t>weights(size_t n){std::vector<uint8_t>w(n);uint32_t state=800179;for(auto&b:w){state^=state<<13;state^=state>>17;state^=state<<5;b=uint8_t(state);}__half a=__float2half_rn(.001f),b=__float2half_rn(.0007f);for(size_t i=0;i<n;i+=144){std::memcpy(w.data()+i,&a,2);std::memcpy(w.data()+i+2,&b,2);}return w;}


int main(int argc,char**argv){try{bool gate=true;uint64_t floats=0,quant=0;
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{
 Workspace w;size_t matrix=K::native_mmvq_weight_bytes(12,Flat,Mix);need(matrix==276480,"actual HC matrix geometry");int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));bool timing=gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only");int banks=timing?int(2*size_t(l2)/matrix)+1:1;B all_weights(matrix*banks);upload(all_weights,weights(all_weights.bytes),s);
 for(int n=1;n<=8;++n){if(n>1&&n<=4)need(!K::native_mmvq_il_supported(12,n,Mix),"original HC NT2..4 must use plain layout");G g(s,[&]{w.body(0,all_weights.p,n,s);w.body(1,all_weights.p,n,s);});for(int phase=0;phase<3;++phase){w.input(n,phase,s);require_row_grid(g,all_weights.p,n);g.run(s);w.verify(n,s);floats+=size_t(n)*(Flat+Mix+24+1);quant+=K::native_q8_1_bytes(Flat,n);}}
 w.input(1,0,s);ck(cudaMemsetAsync(w.residual.p,0,Flat*4,s));ck(cudaStreamSynchronize(s));{G zero(s,[&]{w.body(0,all_weights.p,1,s);w.body(1,all_weights.p,1,s);});zero.run(s);w.verify(1,s,false);floats+=Flat+Mix+24;quant+=K::native_q8_1_bytes(Flat,1);}
 std::cout<<"NUM gpu="<<gpu<<" N88 vs one-row/token HC Q4K BITWISE NT1..8 three changed captures plus allzero, signedzero/subnorm/inverseobservable/NULLgamma/plain bytes, baseline wideCP8 NT5..8, candidate plain-only PASS"<<std::endl;
 if(timing){constexpr int n=3;w.input(n,2,s);std::vector<std::unique_ptr<G>>g[2];for(int b=0;b<banks;++b){const void*weight=static_cast<uint8_t*>(all_weights.p)+size_t(b)*matrix;for(int v=0;v<2;++v){g[v].push_back(std::make_unique<G>(s,[&,v,weight]{w.body(v,weight,n,s);}));g[v].back()->require_weight(weight);if(v)require_row_grid(*g[v].back(),weight,n);g[v].back()->run(s);}w.verify(n,s);}
 auto cycle=[&](int v){for(auto&graph:g[v])graph->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];w.verify(n,s);std::cout<<"pair="<<pair<<" original_ms="<<ms[0]<<" row_token_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE complete N88 HC RMS/normQ/Q4K20480x24/coeff vs row-token CTAs NT3 banks="<<banks<<" bank_bytes="<<all_weights.bytes<<" reportedL2="<<l2<<" captured matrix pointers and allbank outputs verified sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; both4kernels no model gain claim\n";gate=total[0]/total[1]>1.05&&faster==16;}
}ck(cudaStreamDestroy(s));}
std::cout<<"HC_ROW_TOKEN numeric PASS original_F32_values="<<floats<<" original_quant_bytes="<<quant<<" GPUs2 NT1..8 plus cold/paired oracle checks\n";if(!gate)throw std::runtime_error("complete HC row/token speed gate failed >1.05 and16/16 required");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
