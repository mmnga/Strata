#include "strata/deepseek41/hc_coeff.hpp"
#include "strata/deepseek41/hc_coeff_pre.hpp"
#include "strata/deepseek41/route_warp.hpp"
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


constexpr int Flat=20480,Mix=24,Router=384,Top=6;
constexpr size_t HCBytes=276480,QABytes=3686400,RouterBytes=7864320,BankBytes=2*HCBytes+QABytes+RouterBytes;
uint64_t floats=0,quant_bytes=0,ids_compared=0;
void inspect(G&graph,int n,int expected,bool candidate,const std::vector<const void*>&matrices){size_t count=0;ck(cudaGraphGetNodes(graph.g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(graph.g,nodes.data(),&count));int kernels=0,fused=0;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;++kernels;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));if(p.blockDim.x==288){const char*name=nullptr;ck(cudaFuncGetName(&name,p.func));need(name&&std::string(name).find("prepare")!=std::string::npos&&p.gridDim.x==unsigned(n),"wrong fused HC launch");++fused;}}need(kernels==expected,"complete entry-pair kernel count differs");need(fused==(candidate?2:0),"fused stats/norm dispatch absent");for(auto w:matrices)graph.require_weight(w);}
struct Stage{
 bool norm_mode;
 B residual{size_t(Flat)*MaxT*4},scale{3*4},base{24*4},lagged{4*MaxT*4},gamma{H*4},bias{Router*4};
 B flat[2]={B(Flat*MaxT*4),B(Flat*MaxT*4)},hc_inv[2]={B(MaxT*4),B(MaxT*4)},q8[2]={B(K::native_q8_1_bytes(Flat,MaxT)),B(K::native_q8_1_bytes(Flat,MaxT))},il[2]={B(K::native_q8_1_il_bytes(H,MaxT)),B(K::native_q8_1_il_bytes(H,MaxT))};
 B mix[2]={B(Mix*MaxT*4),B(Mix*MaxT*4)},pre[2]={B(4*MaxT*4),B(4*MaxT*4)},post[2]={B(4*MaxT*4),B(4*MaxT*4)},comb[2]={B(16*MaxT*4),B(16*MaxT*4)},x[2]={B(H*MaxT*4),B(H*MaxT*4)},normal[2]={B(H*MaxT*4),B(H*MaxT*4)},inv[2]={B(MaxT*4),B(MaxT*4)},out[2]={B(Q*MaxT*4),B(Q*MaxT*4)},rw[2]={B(Top*MaxT*4),B(Top*MaxT*4)},ids[2]={B(Top*MaxT*4),B(Top*MaxT*4)};
 explicit Stage(bool mode):norm_mode(mode){}
 int cp(int n){return n>=5?8:n>1&&K::native_mmvq_il_supported(12,n,Q)?K::native_q8_1_il_cp(n):0;}
 void input(int n,int phase,cudaStream_t s){std::vector<float>r(size_t(Flat)*n),sc={.3f,-.27f,.43f},b(24),p(4*n),g(H),rb(Router);for(size_t i=0;i<r.size();++i)r[i]=phase?std::sin(float(i)*.013f+.31f*phase+(norm_mode?.17f:0))*.2f:0;for(int t=0;t<n;++t){for(int h=0;h<4;++h)p[t*4+h]=h?(.17f*h-.09f*phase):1.0f;if(phase){r[size_t(t)*Flat+33]=-0.f;r[size_t(t)*Flat+34]=std::ldexp(1.f,-140);r[size_t(t)*Flat+35]=-std::ldexp(1.f,-140);}}for(int j=0;j<24;++j)b[j]=phase==2?(j%2?80.f:-80.f):std::sin(j*.73f+.19f*phase)*.5f;for(int i=0;i<H;++i)g[i]=1.f+.2f*std::sin(i*.007f);for(int i=0;i<Router;++i)rb[i]=.01f*std::sin(i*.03f);upload(residual,r,s);upload(scale,sc,s);upload(base,b,s);upload(lagged,p,s);upload(gamma,g,s);upload(bias,rb,s);}
 void body(int v,const void*hc,const void*matrix,int n,bool nullgamma,cudaStream_t s){
  C::hc_rms_stats(residual.f(),hc_inv[v].f(),Flat,n,1e-20f,s);C::norm_quant_q8(residual.f(),hc_inv[v].f(),nullptr,flat[v].f(),q8[v].p,nullptr,Flat,n,0,s);K::native_hc_q4_k_row_tokens(hc,q8[v].p,mix[v].f(),n,s);
  const float* gm=nullgamma?nullptr:gamma.f();
  if(v)C::hc_coeff_pre(mix[v].f(),scale.f(),base.f(),pre[v].f(),post[v].f(),comb[v].f(),residual.f(),lagged.f(),gm,x[v].f(),norm_mode?normal[v].f():inv[v].f(),H,n,1e-6f,20,1e-20f,norm_mode,s);
  else{C::hc_coeff_warp(mix[v].f(),scale.f(),base.f(),pre[v].f(),post[v].f(),comb[v].f(),n,1e-6f,20,s);if(norm_mode)C::hc_pre_rms(residual.f(),lagged.f(),gm,x[v].f(),normal[v].f(),H,n,1e-20f,s);else C::hc_pre_stats(residual.f(),lagged.f(),x[v].f(),inv[v].f(),H,n,1e-20f,s);}
  if(norm_mode){C::router_dense_pipe4(static_cast<const float*>(matrix),normal[v].f(),out[v].f(),H,Router,n,s);C::route_warp(out[v].f(),bias.f(),static_cast<int32_t*>(ids[v].p),rw[v].f(),Router,Top,n,1.5f,s);}
  else{const int c=cp(n);C::norm_quant_q8(x[v].f(),inv[v].f(),gm,normal[v].f(),q8[v].p,c?il[v].p:nullptr,H,n,c,s);if(n>=5)K::native_mmvq_il_wide(12,matrix,il[v].p,out[v].f(),H,Q,n,s);else if(c)K::native_mmvq_il(12,matrix,q8[v].p,il[v].p,out[v].f(),H,Q,n,s);else K::native_mmvq(12,matrix,q8[v].p,out[v].f(),H,Q,n,s);}
 }
 void compare(B(&a)[2],size_t n,const char*msg,cudaStream_t s){exact(read<float>(a[0].p,n,s),read<float>(a[1].p,n,s),msg);floats+=n;}
 void verify(int n,cudaStream_t s){compare(flat,Flat*n,"flat HC norm differs",s);compare(hc_inv,n,"HC inverse differs",s);compare(mix,Mix*n,"HC mix differs",s);compare(pre,4*n,"new pre differs",s);compare(post,4*n,"post differs",s);compare(comb,16*n,"Sinkhorn differs",s);compare(x,H*n,"lagged collapse differs",s);compare(normal,H*n,"lagged normalized differs",s);compare(out,(norm_mode?Router:Q)*n,"consumer matrix differs",s);
  if(norm_mode){compare(rw,Top*n,"route weights differ",s);exact(read<int32_t>(ids[0].p,Top*n,s),read<int32_t>(ids[1].p,Top*n,s),"route IDs differ");ids_compared+=Top*n;}else compare(inv,n,"lagged inverse differs",s);
  const size_t bytes=K::native_q8_1_bytes(norm_mode?Flat:H,n);exact(read<uint8_t>(q8[0].p,bytes,s),read<uint8_t>(q8[1].p,bytes,s),"Q81 bytes differ");quant_bytes+=bytes;if(!norm_mode&&cp(n)){size_t bytes=K::native_q8_1_il_bytes(H,n);exact(read<uint8_t>(il[0].p,bytes,s),read<uint8_t>(il[1].p,bytes,s),"QA IL bytes differ");quant_bytes+=bytes;}
 }
};
std::vector<uint8_t>weights(int banks){std::vector<uint8_t>w(size_t(banks)*BankBytes);uint32_t state=9917101;auto random=[&]{state^=state<<13;state^=state>>17;state^=state<<5;return state;};const __half d=__float2half_rn(.001f),m=__float2half_rn(.0007f);for(int b=0;b<banks;++b){auto*p=w.data()+size_t(b)*BankBytes;const size_t quant=2*HCBytes+QABytes;for(size_t i=0;i<quant;++i)p[i]=uint8_t(random());for(size_t i=0;i<quant;i+=144){std::memcpy(p+i,&d,2);std::memcpy(p+i+2,&m,2);}for(size_t i=quant;i<BankBytes;i+=4){uint32_t bits=0x3a000000u|(random()&0x807f0000u);std::memcpy(p+i,&bits,4);}}return w;}
int main(int argc,char**argv){try{bool gate=true;for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));const bool timing=gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only");int banks=timing?int((2*size_t(l2))/BankBytes)+1:1;B all(size_t(banks)*BankBytes);upload(all,weights(banks),s);Stage attn(false),ffn(true);auto pointers=[&](int b){auto*p=static_cast<uint8_t*>(all.p)+size_t(b)*BankBytes;return std::vector<const void*>{p,p+HCBytes,p+2*HCBytes,p+2*HCBytes+QABytes};};auto body=[&](int v,int b,int n,bool nullgamma){auto p=pointers(b);attn.body(v,p[0],p[2],n,nullgamma,s);ffn.body(v,p[1],p[3],n,nullgamma,s);};
 for(int n=1;n<=8;++n)for(bool nullgamma:{false,true}){G graph(s,[&]{body(0,0,n,nullgamma);body(1,0,n,nullgamma);});for(int phase=0;phase<3;++phase){attn.input(n,phase,s);ffn.input(n,phase,s);graph.run(s);attn.verify(n,s);ffn.verify(n,s);}}
 bool rejected=false;try{C::hc_coeff_pre(nullptr,nullptr,nullptr,attn.lagged.f(),nullptr,nullptr,nullptr,attn.lagged.f(),nullptr,nullptr,nullptr,H,1,1e-6f,20,1e-20f,false,s);}catch(const std::invalid_argument&){rejected=true;}need(rejected,"aliasing lagged/new pre must be rejected");
 std::cout<<"NUM gpu="<<gpu<<" BITWISE both full HC+ATTN_QA and HC+FFNrouter+route NT1..8/gamma+NULL/threechangedcaptures/wholezero/signedzero/subnorm/coeffextremes/aliasguard PASS"<<std::endl;
 if(timing){constexpr int n=3;attn.input(n,1,s);ffn.input(n,1,s);std::vector<std::unique_ptr<G>>g[2];for(int b=0;b<banks;++b){for(int v=0;v<2;++v){g[v].push_back(std::make_unique<G>(s,[&,v,b]{body(v,b,n,false);}));inspect(*g[v].back(),n,v?12:14,v,pointers(b));g[v].back()->run(s);}attn.verify(n,s);ffn.verify(n,s);}
 auto cycle=[&](int v){for(auto&graph:g[v])graph->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto start=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];attn.verify(n,s);ffn.verify(n,s);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" combined_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<std::endl;}
 std::cout.precision(12);std::cout<<"GATE complete paired HC+ATTN_QA and HC+FFNrouter/route fixedNT3 banks="<<banks<<" bytes="<<all.bytes<<" >2L2="<<2*size_t(l2)<<"; all4actualmatrixpointers/allbankoutputs/two288threadCTAs proved; old14/new12kernels originalMath sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; no model gain claim"<<std::endl;gate=total[0]/total[1]>1.05&&faster==16;}
 }ck(cudaStreamDestroy(s));}std::cout<<"HC_COEFF_PRE numeric PASS F32_values="<<floats<<" quant_bytes="<<quant_bytes<<" route_IDs="<<ids_compared<<" GPUs2"<<std::endl;if(!gate)throw std::runtime_error("full paired HC consumer gate failed >1.05 and16/16 required");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
