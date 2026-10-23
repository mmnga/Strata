#include "strata/deepseek41/kv_norm_rope.hpp"
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;namespace K=strata::kernels;
constexpr int H=5120,D=512,MaxT=8,Capacity=136;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
struct B{void*p=nullptr;size_t bytes;explicit B(size_t n):bytes(n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){need(v.size()*sizeof(T)<=b.bytes,"upload bounds");ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T>read(const void*p,size_t n,cudaStream_t s){std::vector<T>v(n);ck(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
template<class T>void exact(const std::vector<T>&a,const std::vector<T>&b,const char*message){need(a.size()==b.size(),"comparison bounds");if(std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(message);}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;template<class F>G(cudaStream_t s,F f){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));f();ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));}~G(){cudaGraphExecDestroy(e);cudaGraphDestroy(g);}void run(cudaStream_t s){ck(cudaGraphLaunch(e,s));ck(cudaStreamSynchronize(s));}
 void require_weight(const void*w){size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));bool found=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));if(p.kernelParams&&p.kernelParams[0]){void*value=nullptr;std::memcpy(&value,p.kernelParams[0],sizeof(value));found|=value==w;}}need(found,"cold graph does not reference its actual weight bank");}
};
uint64_t floats=0,rawbytes=0;
C::Rope config(bool yarn){C::Rope c;c.dim=64;if(yarn){c.base=160000;c.scale=1.f/16;c.original_context=65536;c.beta_fast=32;c.beta_slow=1;c.yarn=true;}return c;}
void inspect(G&g,int n,bool candidate,const void*weight){size_t count=0;ck(cudaGraphGetNodes(g.g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g.g,nodes.data(),&count));int kernels=0;bool fused=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;++kernels;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));const char*name=nullptr;ck(cudaFuncGetName(&name,p.func));if(name&&std::string(name).find("kv_prepare")!=std::string::npos){need(p.blockDim.x==256&&p.gridDim.x==unsigned(n),"fused KV mapping mismatch");fused=true;}}need(kernels==(candidate?3:5)&&fused==candidate,"full KV projection+preparation path not captured");g.require_weight(weight);}
struct W{
 B x{H*MaxT*4},gamma{D*4},positions{MaxT*4};
 B kv[2]={B(D*MaxT*4),B(D*MaxT*4)},cache[2]={B(Capacity*D*2),B(Capacity*D*2)},q[2]={B(K::native_q8_1_bytes(H,MaxT)),B(K::native_q8_1_bytes(H,MaxT))},il[2]={B(K::native_q8_1_il_bytes(H,MaxT)),B(K::native_q8_1_il_bytes(H,MaxT))};
 static int cp(int n){return n>=5?K::native_q8_1_il_cp(n):n>1&&K::native_mmvq_il_supported(12,n,D)?K::native_q8_1_il_cp(n):0;}
 void input(int n,int start,int phase,cudaStream_t s){std::vector<float>v(size_t(H)*n),g(D);std::vector<int32_t>p(n);std::iota(p.begin(),p.end(),std::min(start+phase,262144-n));for(size_t i=0;i<v.size();++i)v[i]=phase?std::sin(i*.011f+.31f*phase)*.2f:0;for(int i=0;i<D;++i)g[i]=1.f+.3f*std::sin(i*.007f+phase*.1f);if(phase){v[33]=-0.f;v[34]=std::ldexp(1.f,-140);v[35]=-v[34];}upload(x,v,s);upload(gamma,g,s);upload(positions,p,s);for(auto&b:cache)ck(cudaMemsetAsync(b.p,0xa5,b.bytes,s));ck(cudaStreamSynchronize(s));}
 void prepare(int v,int n,C::Rope c,bool nullgamma,cudaStream_t s){auto*g=nullgamma?nullptr:gamma.f();auto*p=static_cast<int32_t*>(positions.p);auto*raw=static_cast<uint16_t*>(cache[v].p);if(v)C::kv_norm_rope_store(kv[v].f(),g,raw,p,D,n,c,Capacity,1e-20f,s);else{C::rms(kv[v].f(),g,kv[v].f(),D,n,1e-20f,s);C::rope(kv[v].f(),D,1,n,p,c,false,s);C::store_f16_positions(kv[v].f(),raw,D,n,p,Capacity,0,s);}}
 void body(int v,const void*weight,int n,C::Rope c,bool nullgamma,cudaStream_t s){int interleave=cp(n);if(n>=5)K::native_quantize_q8_1_il_wide(x.f(),q[v].p,il[v].p,H,n,s);else if(interleave)K::native_quantize_q8_1_il(x.f(),q[v].p,il[v].p,H,n,s);else K::native_quantize_q8_1(x.f(),q[v].p,H,n,s);if(n>=5)K::native_mmvq_il_wide(12,weight,il[v].p,kv[v].f(),H,D,n,s);else if(interleave)K::native_mmvq_il(12,weight,q[v].p,il[v].p,kv[v].f(),H,D,n,s);else K::native_mmvq(12,weight,q[v].p,kv[v].f(),H,D,n,s);prepare(v,n,c,nullgamma,s);}
 void verify(int n,cudaStream_t s){const auto a=read<float>(kv[0].p,n*D,s),b=read<float>(kv[1].p,n*D,s);for(float f:b)need(std::isfinite(f),"nonfinite KV fixture");exact(a,b,"rotated F32 KV differs");exact(read<uint8_t>(cache[0].p,cache[0].bytes,s),read<uint8_t>(cache[1].p,cache[1].bytes,s),"raw F16 ring/padding differs");floats+=n*D;rawbytes+=cache[0].bytes;}
};
std::vector<uint8_t>weights(size_t n){std::vector<uint8_t>w(n);uint32_t state=870173;for(auto&v:w){state^=state<<13;state^=state>>17;state^=state<<5;v=uint8_t(state);}__half d=__float2half_rn(.001f),m=__float2half_rn(.0007f);for(size_t i=0;i<n;i+=144){std::memcpy(w.data()+i,&d,2);std::memcpy(w.data()+i+2,&m,2);}return w;}
int main(int argc,char**argv){try{bool gate=true;for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));const size_t matrix=K::native_mmvq_weight_bytes(12,H,D);need(matrix==1474560,"actual KV projection geometry");bool timing=gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only");const int banks=timing?int(2*size_t(l2)/matrix)+1:1;B all(matrix*banks);upload(all,weights(all.bytes),s);W w;
 for(int n=1;n<=8;++n)for(bool yarn:{false,true})for(bool nullgamma:{false,true}){C::Rope c=config(yarn);G graph(s,[&]{w.body(0,all.p,n,c,nullgamma,s);w.body(1,all.p,n,c,nullgamma,s);});for(int start:{0,1,125,127,128,1023,8191,65535,262136})for(int phase=0;phase<3;++phase){w.input(n,start,phase,s);graph.run(s);w.verify(n,s);}}
 // Direct raw KV controls retain F32 subnormal and signed-zero values that
 // the unchanged projection's activation quantizer can otherwise flush.
 for(bool yarn:{false,true})for(bool nullgamma:{false,true}){const auto c=config(yarn);G g(s,[&]{w.prepare(0,8,c,nullgamma,s);w.prepare(1,8,c,nullgamma,s);});for(int phase=0;phase<3;++phase){w.input(8,262136,phase,s);std::vector<float>raw(8*D);for(int t=0;t<8;++t){raw[t*D]=phase?1.f:0;raw[t*D+1]=-0.f;for(int j=2;j<D;++j)raw[t*D+j]=std::ldexp(j%2?1.f:-1.f,phase==2?-125:-140);}upload(w.kv[0],raw,s);upload(w.kv[1],raw,s);g.run(s);w.verify(8,s);}}
 std::cout<<"NUM gpu="<<gpu<<" complete KV Q4projection/norm/RoPE/rawring BITWISE NT1..8 plain+actualYaRN gamma+NULL nine starts threechangedcaptures/wrap/fullctx/zero; direct F32 signedzero/subnorm/near-normal controls PASS"<<std::endl;
 if(timing){constexpr int n=3;auto c=config(true);w.input(n,1022,1,s);std::vector<std::unique_ptr<G>>g[2];for(int b=0;b<banks;++b){const void*weight=static_cast<uint8_t*>(all.p)+size_t(b)*matrix;for(int v=0;v<2;++v){g[v].push_back(std::make_unique<G>(s,[&,v,weight]{w.body(v,weight,n,c,false,s);}));inspect(*g[v].back(),n,v,weight);g[v].back()->run(s);}w.verify(n,s);}
 auto cycle=[&](int v){for(auto&graph:g[v])graph->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto start=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];w.verify(n,s);std::cout<<"pair="<<pair<<" original_ms="<<ms[0]<<" fused_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<std::endl;}
 std::cout.precision(12);std::cout<<"GATE full KV inputquant/Q4projection5120x512/RMS512/actualYaRN/F16rawstore NT3/start1023 banks="<<banks<<" bytes="<<all.bytes<<" >2L2="<<2*size_t(l2)<<"; allactualweightpointers/allbankF32+F16outputs/fused256CTAproved old5/new3kernels sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; no model gain claim"<<std::endl;gate=total[0]/total[1]>1.05&&faster==16;}
 }ck(cudaStreamDestroy(s));}
 std::cout<<"KV_NORM_ROPE numeric PASS rotated_F32_values="<<floats<<" raw_F16_bytes="<<rawbytes<<" GPUs2"<<std::endl;if(!gate)throw std::runtime_error("full KV projection/preparation gate failed >1.05 and16/16 required");}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
