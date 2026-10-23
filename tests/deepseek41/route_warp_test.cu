#include "strata/deepseek41/norm_quant.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/route_warp.hpp"
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
namespace C=strata::deepseek41::cuda;
constexpr int H=5120,Q=384,Top=6,MaxT=8;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
struct B{void*p=nullptr;size_t bytes;explicit B(size_t n):bytes(n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){need(v.size()*sizeof(T)<=b.bytes,"upload bounds");ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T>read(const void*p,size_t n,cudaStream_t s){std::vector<T>v(n);ck(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
template<class T>void exact(const std::vector<T>&a,const std::vector<T>&b,const char*message){need(a.size()==b.size(),"comparison bounds");if(std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(message);}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;template<class F>G(cudaStream_t s,F f){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));f();ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));}~G(){cudaGraphExecDestroy(e);cudaGraphDestroy(g);}void run(cudaStream_t s){ck(cudaGraphLaunch(e,s));ck(cudaStreamSynchronize(s));}
 void require_weight(const void*w){size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));bool found=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));if(p.kernelParams&&p.kernelParams[0]){void*value=nullptr;std::memcpy(&value,p.kernelParams[0],sizeof(value));found|=value==w;}}need(found,"cold graph does not reference its actual weight bank");}
 void require_selector(int nt){size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));bool found=false;for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));const char*name=nullptr;ck(cudaFuncGetName(&name,p.func));if(name&&std::string(name).find("route_warp_kernel")!=std::string::npos){need(p.blockDim.x==32&&p.gridDim.x==unsigned(nt),"warp selector launch mismatch");found=true;}}need(found,"warp selector not captured");}
};

struct Workspace{
 B residual{size_t(H)*4*MaxT*4},pre{4*MaxT*4},weight{H*4},inv{MaxT*4},bias{Q*4};
 B x[2]={B(H*MaxT*4),B(H*MaxT*4)},norm[2]={B(H*MaxT*4),B(H*MaxT*4)},out[2]={B(Q*MaxT*4),B(Q*MaxT*4)},route_weights[2]={B(Top*MaxT*4),B(Top*MaxT*4)},ids[2]={B(Top*MaxT*4),B(Top*MaxT*4)};
 void input(int n,int phase,cudaStream_t s){std::vector<float>r(size_t(n)*4*H),p(n*4),w(H),b(Q);for(size_t i=0;i<r.size();++i)r[i]=std::sin(float(i)*.011f+.31f*phase)*.2f;for(int t=0;t<n;++t){for(int h=0;h<4;++h)p[t*4+h]=h?(.13f*float(h)-.09f*phase):1.f;if(!phase)std::fill(r.begin()+size_t(t)*4*H,r.begin()+size_t(t+1)*4*H,0.f);for(int h=0;h<4;++h)r[(size_t(t)*4+h)*H]=h?0.f:1.f;r[size_t(t)*4*H+33]=-0.f;r[size_t(t)*4*H+34]=std::ldexp(1.f,-140);}for(int i=0;i<H;++i)w[i]=1.f+.3f*std::sin(i*.007f);w[0]=1.f;for(int i=0;i<Q;++i)b[i]=.01f*std::sin(i*.03f);upload(residual,r,s);upload(pre,p,s);upload(weight,w,s);upload(bias,b,s);}
 void body(int v,const void*weights,int n,cudaStream_t s){C::hc_pre_rms(residual.f(),pre.f(),weight.f(),x[v].f(),norm[v].f(),H,n,1e-20f,s);C::router_dense_pipe4(static_cast<const float*>(weights),norm[v].f(),out[v].f(),H,Q,n,s);(v?C::route_warp:C::route)(out[v].f(),bias.f(),static_cast<int32_t*>(ids[v].p),route_weights[v].f(),Q,Top,n,1.5f,s);}

 void verify(int n,cudaStream_t s){exact(read<float>(x[0].p,H*n,s),read<float>(x[1].p,H*n,s),"HC collapse bits differ");auto a=read<float>(norm[0].p,H*n,s);exact(a,read<float>(norm[1].p,H*n,s),"normalized F32 bits differ");exact(read<float>(out[0].p,Q*n,s),read<float>(out[1].p,Q*n,s),"router logit bits differ");exact(read<float>(route_weights[0].p,Top*n,s),read<float>(route_weights[1].p,Top*n,s),"route weight bits differ");exact(read<int32_t>(ids[0].p,Top*n,s),read<int32_t>(ids[1].p,Top*n,s),"route IDs differ");}
};
std::vector<float>weights(size_t n){std::vector<float>w(n/4);uint32_t state=810179;for(auto&v:w){state^=state<<13;state^=state>>17;state^=state<<5;const uint32_t bits=0x3a000000u|(state&0x807f0000u);std::memcpy(&v,&bits,4);}return w;}

void edge_routes(cudaStream_t s,uint64_t&checked_weights,uint64_t&checked_ids){
 for(int e:{128,384}){const int k=e==128?3:6;B logits(e*MaxT*4),bias(e*4),ids[2]={B(k*MaxT*4),B(k*MaxT*4)},weights[2]={B(k*MaxT*4),B(k*MaxT*4)};
 for(int n=1;n<=MaxT;++n){G graph(s,[&]{C::route(logits.f(),bias.f(),static_cast<int32_t*>(ids[0].p),weights[0].f(),e,k,n,1.5f,s);C::route_warp(logits.f(),bias.f(),static_cast<int32_t*>(ids[1].p),weights[1].f(),e,k,n,1.5f,s);});graph.require_selector(n);
 for(int phase=0;phase<10;++phase){std::vector<float>x(e*n),b(e);std::vector<int>expected;
 if(phase==1){for(int j=0;j<k;++j){expected.push_back(1+32*j);b[1+32*j]=10.f;}}
 if(phase==2){std::vector<int>edges=e==128?std::vector<int>{31,32,127}:std::vector<int>{31,32,127,128,255,256};expected=edges;for(int id:edges)b[id]=10.f;}
 if(phase==3||phase==6)for(int id=0;id<e;++id)b[id]=float(id)*.125f;
 for(int token=0;token<n;++token)for(int id=0;id<e;++id){float v=0;
  if(phase==3)v=-1000.f;
  if(phase==4){v=std::sin(id*.13f+token*.23f)*20.f;b[id]=std::cos(id*.17f)*.2f;}
  if(phase==5){v=(id%3==0?-1000.f:id%3==1?1000.f:std::ldexp(1.f,-140));b[id]=(id%7-3)*.1f;}
  if(phase==6)v=-80.f;
  if(phase==7)v=(id%2?-0.f:0.f);
  if(phase==8){v=(id%4==0?3.402823466e38f:id%4==1?-3.402823466e38f:id%4==2?std::ldexp(1.f,-140):-std::ldexp(1.f,-140));b[id]=float((id%9)-4)*1e30f;}
  if(phase==9){v=-1000.f;b[id]=0.f;}
  x[token*e+id]=v;
 }
 upload(logits,x,s);upload(bias,b,s);graph.run(s);
 const auto a=read<int32_t>(ids[0].p,n*k,s),c=read<int32_t>(ids[1].p,n*k,s);exact(a,c,"edge stable IDs differ");const auto w=read<float>(weights[0].p,n*k,s),v=read<float>(weights[1].p,n*k,s);exact(w,v,"edge rank-order weights differ");for(float z:v)need(std::isfinite(z),"edge nonfinite weights");
 if(phase==0||phase==7||phase==9){expected.clear();for(int j=0;j<k;++j)expected.push_back(j);}
 if(phase==3||phase==6){expected.clear();for(int j=0;j<k;++j)expected.push_back(e-1-j);}
 if(!expected.empty())for(int token=0;token<n;++token)for(int j=0;j<k;++j)need(a[token*k+j]==expected[j],"independent stable tie/bias winner oracle failed");
 if(phase==3||phase==9)for(float z:v)need(z==0.f,"all-underflow must retain zero probability weights");checked_weights+=n*k;checked_ids+=n*k;
 }
 }
 }
}

int main(int argc,char**argv){try{uint64_t floats=0,id_values=0,edge_weights=0,edge_ids=0;bool gate=true;
for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));{
 edge_routes(s,edge_weights,edge_ids);
 const size_t matrix=size_t(H)*Q*4;need(matrix==7864320,"actual router matrix geometry");int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,gpu));need(l2>0,"L2 capacity unavailable");int banks=gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only")?int((2*size_t(l2))/matrix)+1:1;B all_weights(matrix*banks);upload(all_weights,weights(all_weights.bytes),s);Workspace w;
 for(int n=1;n<=8;++n){G graph(s,[&]{w.body(0,all_weights.p,n,s);w.body(1,all_weights.p,n,s);});for(int phase=0;phase<3;++phase){w.input(n,phase,s);graph.run(s);w.verify(n,s);floats+=size_t(n)*(2*H+Q+Top);id_values+=Top*n;}}
 std::cout<<"NUM gpu="<<gpu<<" HC/x/norm/router logits/route weights/IDs BITWISE NT1..8 three changed graphs zero blocks/signedzero/F32subnorm PASS"<<std::endl;
 if(banks>1){constexpr int n=3;w.input(n,2,s);std::vector<std::unique_ptr<G>>graphs[2];for(int b=0;b<banks;++b){const void*weight=static_cast<uint8_t*>(all_weights.p)+size_t(b)*matrix;for(int v=0;v<2;++v){graphs[v].push_back(std::make_unique<G>(s,[&,v,weight]{w.body(v,weight,n,s);}));graphs[v].back()->require_weight(weight);if(v)graphs[v].back()->require_selector(n);graphs[v].back()->run(s);}w.verify(n,s);}
 auto cycle=[&](int v){for(auto&g:graphs[v])g->run(s);};cycle(0);cycle(1);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();cycle(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/banks;total[v]+=ms[v];}faster+=ms[1]<ms[0];w.verify(n,s);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" warp_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE full adopted HCpre+RMS+N65F32router+top6route vs warp selection H5120/E384 NT3 banks="<<banks<<" bank_bytes="<<all_weights.bytes<<" reportedL2="<<l2<<" captured weight pointers verified, all bank outputs exact sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; 3 kernels BOTH paths no model speed claim\n";gate=total[0]/total[1]>1.05&&faster==16;}
}ck(cudaStreamDestroy(s));}
std::cout<<"WARP_ROUTE numeric PASS original_pipeline_F32_values="<<floats<<" original_route_ID_values="<<id_values<<" exact GPUs2 NT1..8; cold bank/paired checks additional\n";std::cout<<"Standalone edge F32_weights="<<edge_weights<<" IDs="<<edge_ids<<" exact both384/6+128/3 NT1..8 tenchangedcaptures bias/ties/extremes/underflow/signedzero/subnorm independentboundary+same-lane winner controls PASS\n";if(!gate)throw std::runtime_error("complete HC/norm/router/route speed gate failed >1.05 and16/16 required");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
