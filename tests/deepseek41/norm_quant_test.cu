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
struct Workspace{
 B residual{size_t(H)*4*MaxT*4},pre{4*MaxT*4},weight{H*4},inv{MaxT*4};
 B x[2]={B(H*MaxT*4),B(H*MaxT*4)},norm[2]={B(H*MaxT*4),B(H*MaxT*4)},q[2]={B(K::native_q8_1_bytes(H,MaxT)),B(K::native_q8_1_bytes(H,MaxT))},il[2]={B(K::native_q8_1_il_bytes(H,MaxT)),B(K::native_q8_1_il_bytes(H,MaxT))},out[2]={B(Q*MaxT*4),B(Q*MaxT*4)};
 bool null_weight=false;
 void input(int n,int phase,cudaStream_t s){std::vector<float>r(size_t(n)*4*H),p(n*4),w(H);for(size_t i=0;i<r.size();++i)r[i]=std::sin(float(i)*.011f+.31f*phase)*.2f;for(int t=0;t<n;++t){for(int h=0;h<4;++h)p[t*4+h]=h?(.13f*float(h)-.09f*phase):1.f;if(!phase)std::fill(r.begin()+size_t(t)*4*H,r.begin()+size_t(t+1)*4*H,0.f);for(int h=0;h<4;++h)r[(size_t(t)*4+h)*H]=h?0.f:1.f;r[size_t(t)*4*H+33]=-0.f;r[size_t(t)*4*H+34]=std::ldexp(1.f,-140);}for(int i=0;i<H;++i)w[i]=1.f+.3f*std::sin(i*.007f);w[0]=1.f;upload(residual,r,s);upload(pre,p,s);upload(weight,w,s);null_weight=phase==1;for(auto&b:il)ck(cudaMemsetAsync(b.p,0xa5,b.bytes,s));ck(cudaStreamSynchronize(s));}
 static int cp(int n){if(n>=5)return K::native_q8_1_il_cp(n);return n>1&&K::native_mmvq_il_supported(12,n,Q)?K::native_q8_1_il_cp(n):0;}
 void body(int v,const void*weights,int n,cudaStream_t s){int interleave=cp(n);const float*w=null_weight?nullptr:weight.f();if(v){C::hc_pre_stats(residual.f(),pre.f(),x[1].f(),inv.f(),H,n,1e-20f,s);C::norm_quant_q8(x[1].f(),inv.f(),w,norm[1].f(),q[1].p,interleave?il[1].p:nullptr,H,n,interleave,s);}else{C::hc_pre_rms(residual.f(),pre.f(),w,x[0].f(),norm[0].f(),H,n,1e-20f,s);if(n>=5)K::native_quantize_q8_1_il_wide(norm[0].f(),q[0].p,il[0].p,H,n,s);else if(interleave)K::native_quantize_q8_1_il(norm[0].f(),q[0].p,il[0].p,H,n,s);else K::native_quantize_q8_1(norm[0].f(),q[0].p,H,n,s);}
 if(n>=5)K::native_mmvq_il_wide(12,weights,il[v].p,out[v].f(),H,Q,n,s);else if(interleave)K::native_mmvq_il(12,weights,q[v].p,il[v].p,out[v].f(),H,Q,n,s);else K::native_mmvq(12,weights,q[v].p,out[v].f(),H,Q,n,s);}
 void verify(int n,cudaStream_t s){exact(read<float>(x[0].p,H*n,s),read<float>(x[1].p,H*n,s),"HC collapse bits differ");auto a=read<float>(norm[0].p,H*n,s);exact(a,read<float>(norm[1].p,H*n,s),"normalized F32 bits differ");auto inverse=read<float>(inv.p,n,s);for(int t=0;t<n;++t)need(!std::memcmp(&inverse[t],&a[size_t(t)*H],4),"inverse differs from original observable x0=w0=1");exact(read<uint8_t>(q[0].p,K::native_q8_1_bytes(H,n),s),read<uint8_t>(q[1].p,K::native_q8_1_bytes(H,n),s),"native plain Q81 bytes differ");if(cp(n))exact(read<uint8_t>(il[0].p,K::native_q8_1_il_bytes(H,n),s),read<uint8_t>(il[1].p,K::native_q8_1_il_bytes(H,n),s),"native interleaved Q81 bytes differ");exact(read<float>(out[0].p,Q*n,s),read<float>(out[1].p,Q*n,s),"Q4K QA projection bits differ");}
};
std::vector<uint8_t>weights(size_t n){std::vector<uint8_t>w(n);uint32_t state=800179;for(auto&b:w){state^=state<<13;state^=state>>17;state^=state<<5;b=uint8_t(state);}__half a=__float2half_rn(.001f),b=__float2half_rn(.0007f);for(size_t i=0;i<n;i+=144){std::memcpy(w.data()+i,&a,2);std::memcpy(w.data()+i+2,&b,2);}return w;}
int main(int argc,char**argv){try{uint64_t floats=0,bytes=0;bool gate=true;
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{
 const size_t matrix=K::native_mmvq_weight_bytes(12,H,Q);need(matrix==3686400,"actual QA matrix geometry");int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));need(l2>0,"L2 capacity unavailable");int banks=gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only")?int((2*size_t(l2))/matrix)+1:1;B all_weights(matrix*banks);upload(all_weights,weights(all_weights.bytes),s);Workspace w;
 for(int n=1;n<=8;++n){G graph(s,[&]{w.body(0,all_weights.p,n,s);w.body(1,all_weights.p,n,s);});for(int phase=0;phase<3;++phase){w.input(n,phase,s);graph.run(s);w.verify(n,s);floats+=size_t(n)*(2*H+Q+1);bytes+=K::native_q8_1_bytes(H,n)+(Workspace::cp(n)?K::native_q8_1_il_bytes(H,n):0);}}
 // Null is a captured pointer, so exercise it in its own graph, not by
 // changing the host flag after a graph already captured the non-null pointer.
 w.input(1,1,s);{G null_graph(s,[&]{w.body(0,all_weights.p,1,s);w.body(1,all_weights.p,1,s);});null_graph.run(s);w.verify(1,s);floats+=2*H+Q+1;bytes+=K::native_q8_1_bytes(H,1);}
 std::cout<<"NUM gpu="<<gpu<<" HC/x/norm/inverse/plainQ81/IL/QA BITWISE NT1..8 three changed graphs zero blocks/signedzero/F32subnorm/nullnormweight PASS"<<std::endl;
 if(banks>1){constexpr int n=3;w.input(n,2,s);std::vector<std::unique_ptr<G>>graphs[2];for(int b=0;b<banks;++b){const void*weight=static_cast<uint8_t*>(all_weights.p)+size_t(b)*matrix;for(int v=0;v<2;++v){graphs[v].push_back(std::make_unique<G>(s,[&,v,weight]{w.body(v,weight,n,s);}));graphs[v].back()->require_weight(weight);graphs[v].back()->run(s);}w.verify(n,s);}
 auto cycle=[&](int v){for(auto&g:graphs[v])g->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];w.verify(n,s);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" fused_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE full HCpre+RMS+Q81+QA Q4K5120x1280 NT3 banks="<<banks<<" bank_bytes="<<all_weights.bytes<<" reportedL2="<<l2<<" captured weight pointers verified, all bank outputs exact sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; 3 kernels BOTH paths no model speed claim\n";gate=total[0]/total[1]>1.05&&faster==16;}
}ck(cudaStreamDestroy(s));}
std::cout<<"NORM_Q81 numeric PASS original_F32_values="<<floats<<" original_quant_bytes="<<bytes<<" exact GPUs2 NT1..8; cold bank/paired checks additional\n";if(!gate)throw std::runtime_error("complete HC/norm/quant/QA speed gate failed >1.05 and16/16 required");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
