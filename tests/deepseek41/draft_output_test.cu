#include "strata/deepseek41/draft_output.hpp"
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

namespace D=strata::deepseek41;
constexpr int In=4096,Out=1024,Groups=8,Full=In*Groups,OA=Out*Groups;
struct Workspace{
 B input{size_t(Full)*MaxT*4},positions{MaxT*4};
 B attn[2]={B(size_t(Full)*MaxT*4),B(size_t(Full)*MaxT*4)},oa[2]={B(size_t(OA)*MaxT*4),B(size_t(OA)*MaxT*4)},ob[2]={B(size_t(H)*MaxT*4),B(size_t(H)*MaxT*4)};
 B groupIn{size_t(In)*MaxT*4},groupOut{size_t(Out)*MaxT*4};
 B q[2]={B(K::native_q8_1_bytes(Full,MaxT)),B(K::native_q8_1_bytes(Full,MaxT))};
 size_t wa=K::native_mmvq_weight_bytes(8,In,Out)*Groups,wb=K::native_mmvq_weight_bytes(8,OA,H);
 void input_data(int n,int phase,cudaStream_t s){std::vector<float>x(size_t(Full)*n);std::vector<int32_t>p(n);for(size_t i=0;i<x.size();++i)x[i]=phase?std::sin(float(i)*.017f+.23f*phase)*float(1+(i/32)%7)*.2f:0.f;for(int t=0;t<n;++t){p[t]=t+phase*37;auto base=size_t(t)*Full;x[base+33]=-0.f;x[base+34]=std::ldexp(1.f,-140);}upload(input,x,s);upload(positions,p,s);}
 void body(int v,const void*weights,int n,cudaStream_t s){
  ck(cudaMemcpyAsync(attn[v].p,input.p,size_t(Full)*n*4,cudaMemcpyDeviceToDevice,s));
  C::rope(attn[v].f(),512,64,n,static_cast<int32_t*>(positions.p),C::Rope{},true,s);
  if(v)D::draft_output_a(weights,attn[v].f(),q[v].p,oa[v].f(),n,s);
  else for(int g=0;g<Groups;++g){
   C::strided_copy(attn[0].f()+g*In,groupIn.f(),In,n,Full,In,s);
   K::native_quantize_q8_1(groupIn.f(),q[0].p,In,n,s);
   K::native_mmvq(8,static_cast<const uint8_t*>(weights)+size_t(g)*wa/Groups,q[0].p,groupOut.f(),In,Out,n,s);
   C::strided_copy(groupOut.f(),oa[0].f()+g*Out,Out,n,Out,OA,s);
  }
  K::native_quantize_q8_1(oa[v].f(),q[v].p,OA,n,s);
  K::native_mmvq(8,static_cast<const uint8_t*>(weights)+wa,q[v].p,ob[v].f(),OA,H,n,s);
 }
 uint64_t verify(int n,cudaStream_t s){exact(read<float>(attn[0].p,size_t(Full)*n,s),read<float>(attn[1].p,size_t(Full)*n,s),"inverse rope changed");exact(read<float>(oa[0].p,size_t(OA)*n,s),read<float>(oa[1].p,size_t(OA)*n,s),"grouped Q8 OA differs from independent gather/quant/mat/scatter");exact(read<float>(ob[0].p,size_t(H)*n,s),read<float>(ob[1].p,size_t(H)*n,s),"full draft output OB differs");return size_t(Full+OA+H)*n;}
};
std::vector<uint8_t>weights(size_t bytes){std::vector<uint8_t>w(bytes);uint32_t state=800183;for(auto&x:w){state^=state<<13;state^=state>>17;state^=state<<5;x=uint8_t(state);}for(size_t i=0;i<bytes;i+=34){__half d=__float2half_rn(.0001f*float(1+(i/34)%7));std::memcpy(w.data()+i,&d,2);}return w;}
int main(int argc,char**argv){try{bool gate=true;uint64_t floats=0;K::native_mmvq_set_multi_exact(true);
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{
 Workspace w;const size_t bank=w.wa+w.wb;need(w.wa==35651584&&w.wb==44564480&&bank==80216064,"actual original draft OA/OB geometry");int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));need(l2>0,"L2 unavailable");bool timing=gpu==1&&(argc==1||std::string(argv[1])!="--correctness-only");int banks=timing?int(2*size_t(l2)/bank)+1:1;B matrices(bank*banks);upload(matrices,weights(matrices.bytes),s);
 for(int n=1;n<=8;++n){G g(s,[&]{w.body(0,matrices.p,n,s);w.body(1,matrices.p,n,s);});for(int phase=0;phase<3;++phase){w.input_data(n,phase,s);g.run(s);floats+=w.verify(n,s);}}
 std::cout<<"NUM gpu="<<gpu<<" original Q8_0 draft inverseRope/OA/OB BITWISE NT1..8 three changed captured inputs zero/signedzero/subnorm PASS"<<std::endl;
 if(timing){constexpr int n=5;w.input_data(n,2,s);std::vector<std::unique_ptr<G>>graphs[2];for(int b=0;b<banks;++b){const void*weight=static_cast<uint8_t*>(matrices.p)+size_t(b)*bank;for(int v=0;v<2;++v){graphs[v].push_back(std::make_unique<G>(s,[&,v,weight]{w.body(v,weight,n,s);}));if(v)graphs[v].back()->require_weight(weight);else for(int group=0;group<Groups;++group)graphs[v].back()->require_weight(static_cast<const uint8_t*>(weight)+size_t(group)*w.wa/Groups);graphs[v].back()->require_weight(static_cast<const uint8_t*>(weight)+w.wa);graphs[v].back()->run(s);}w.verify(n,s);}
 auto cycle=[&](int v){for(auto&g:graphs[v])g->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];w.verify(n,s);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" grouped_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE full inverseRope/OA/OB original Q8_0 NT5 eight4096x1024groups OB8192x5120 banks="<<banks<<" bank_bytes="<<matrices.bytes<<" reportedL2="<<l2<<" captured all OA/OB weight pointers and all bank outputs verified sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; 35->5 kernels plus common input D2D reset both included no model speed claim\n";gate=total[0]/total[1]>1.05&&faster==16;
 }
}ck(cudaStreamDestroy(s));}
std::cout<<"DRAFT_GROUPED_OUTPUT numeric PASS original_F32_values="<<floats<<" exact GPUs2 NT1..8 original Q8_0; cold paired checks additional\n";if(!gate)throw std::runtime_error("full draft output speed gate failed >1.05 and16/16 required");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
