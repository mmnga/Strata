#include "strata/deepseek41/attention_async.hpp"
#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;
constexpr int D=512,Heads=32,W=128,Cap=136,Top=512;
uint64_t outputs=0,workspace_bytes=0;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
struct B{void*p=nullptr;size_t bytes;explicit B(size_t n):bytes(n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}uint16_t*h(){return static_cast<uint16_t*>(p);}int32_t*i(){return static_cast<int32_t*>(p);}};
template<class T>void upload(B&b,const std::vector<T>&v,cudaStream_t s){need(v.size()*sizeof(T)<=b.bytes,"upload bounds");ck(cudaMemcpyAsync(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T>read(B&b,size_t n,cudaStream_t s){std::vector<T>v(n);ck(cudaMemcpyAsync(v.data(),b.p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return v;}
template<class T>void exact(const std::vector<T>&a,const std::vector<T>&b,const char*msg){need(a.size()==b.size(),"compare size");if(std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(msg);}
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;template<class F>G(cudaStream_t s,F f){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));f();ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));}~G(){cudaGraphExecDestroy(e);cudaGraphDestroy(g);}void run(cudaStream_t s){ck(cudaGraphLaunch(e,s));ck(cudaStreamSynchronize(s));}
 std::vector<void*>inspect(int n,int stride,int capacity,bool paired){size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));int kernels=0;bool score=false;std::vector<void*>common(2,nullptr);int z=std::max(1,capacity/((paired?Heads/2:Heads)*n));z=std::min(z,(stride+7)/8);for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;++kernels;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));const char*raw=nullptr;ck(cudaFuncGetName(&raw,p.func));std::string name=raw?raw:"";if(name.find("attention_score_softmax_kernel")!=std::string::npos)common[0]=p.func;else if(name.find("attention_value_async_kernel")!=std::string::npos)common[1]=p.func;else if(name.find(paired?"attention_score_pair_keys_kernel":"attention_score_grid_kernel")!=std::string::npos){need(p.blockDim.x==256&&p.gridDim.x==unsigned(paired?Heads/2:Heads)&&p.gridDim.y==unsigned(n)&&p.gridDim.z==unsigned(z),"captured score mapping wrong");score=true;}}
 need(kernels==3&&score&&common[0]&&common[1],"attention graph not complete original3kernel pipeline");return common;
 }
};
uint16_t half_bits(float x){auto h=__float2half_rn(x);uint16_t b;std::memcpy(&b,&h,2);return b;}
struct Fixture{
 int n,ratio,old_capacity,new_capacity;B q,raw,comp,sinks,pos,ids,out[3],scratch[2];
 Fixture(int t,int r,int old,int candidate):n(t),ratio(r),old_capacity(old),new_capacity(candidate),q(size_t(t)*Heads*D*4),raw(size_t(Cap)*D*2),comp(size_t(Top)*D*2),sinks(Heads*4),pos(t*4),ids(size_t(t)*Top*4),out{B(q.bytes),B(q.bytes),B(q.bytes)},scratch{B(size_t(t)*Heads*640*12),B(size_t(t)*Heads*640*12)}{}
 void input(int start,int phase,cudaStream_t s){std::vector<float>query(size_t(n)*Heads*D),sink(Heads);std::vector<uint16_t>r(size_t(Cap)*D),c(size_t(Top)*D);std::vector<int32_t>positions(n),index(size_t(n)*Top);std::iota(positions.begin(),positions.end(),std::min(start+phase,262144-n));
 for(int t=0;t<n;++t)for(int h=0;h<Heads;++h)for(int d=0;d<D;++d)query[(size_t(t)*Heads+h)*D+d]=phase?std::sin(d*.017f+h*.31f+t*.19f+phase*.71f)*.2f:0;
 for(int h=0;h<Heads;++h)sink[h]=phase==2?(h%3==0?-1000.f:h%3==1?1000.f:0):phase?std::sin(h*.13f):0;
 for(size_t i=0;i<r.size();++i)r[i]=half_bits(phase?std::sin(i*.027f+phase*.11f)*.2f:0);for(size_t i=0;i<c.size();++i)c[i]=half_bits(phase?std::cos(i*.019f+phase*.17f)*.2f:0);
 if(phase){const uint16_t edge[12]={0x0001,0x8001,0x0000,0x8000,0x03ff,0x83ff,0x0400,0x8400,0x7bff,0xfbff,0x3c00,0xbc00};for(int i=0;i<12;++i){r[i]=edge[i];c[i]=edge[11-i];}query[0]=-0.f;query[1]=std::ldexp(1.f,-140);query[2]=-query[1];}
 for(int t=0;t<n;++t){int count=ratio?std::min((positions[t]+1)/ratio,Top):0;for(int j=0;j<count;++j)index[t*Top+j]=(phase==2?count-1-j+phase*(t+7):j+phase*(t+7))%count;}
 upload(q,query,s);upload(raw,r,s);upload(comp,c,s);upload(sinks,sink,s);upload(pos,positions,s);upload(ids,index,s);for(auto&b:scratch)ck(cudaMemsetAsync(b.p,0xa5,b.bytes,s));ck(cudaStreamSynchronize(s));
 }
 void body(int variant,cudaStream_t s){const auto*r=ratio?comp.h():nullptr;const auto*i=ratio?ids.i():nullptr;const int selected=ratio?Top:0;
 if(variant==0)C::attention_positions(q.f(),raw.h(),r,i,sinks.f(),out[0].f(),D,Heads,n,W,pos.i(),ratio,selected,Top,Cap,s);
 else if(variant==1)C::attention_positions_score_grid_async(q.f(),raw.h(),r,i,sinks.f(),out[1].f(),scratch[0].p,D,Heads,n,W,pos.i(),ratio,selected,Top,Cap,s,old_capacity);
 else C::attention_positions_score_grid_async_pair_keys(q.f(),raw.h(),r,i,sinks.f(),out[2].f(),scratch[1].p,D,Heads,n,W,pos.i(),ratio,selected,Top,Cap,s,new_capacity);
 }
 void verify(cudaStream_t s,bool original){auto candidate=read<float>(out[2],q.bytes/4,s);for(float v:candidate)need(std::isfinite(v),"nonfinite candidate attention");exact(candidate,read<float>(out[1],q.bytes/4,s),"pair keys changed adopted attention bits");if(original)exact(candidate,read<float>(out[0],q.bytes/4,s),"pair keys changed original attention bits");outputs+=candidate.size();exact(read<uint8_t>(scratch[0],scratch[0].bytes,s),read<uint8_t>(scratch[1],scratch[1].bytes,s),"rounded softmax weights/key pointers/padding differ");workspace_bytes+=scratch[0].bytes;}
};
int main(int argc,char**argv){try{bool gate=true;for(int gpu=0;gpu<2;++gpu){ck(cudaSetDevice(gpu));int old_sm=0,old_blocks=0,new_sm=0,new_blocks=0;const int old=C::attention_score_reuse_capacity(&old_sm,&old_blocks),candidate=C::attention_score_pair_keys_capacity(&new_sm,&new_blocks);std::cout<<"CAP gpu="<<gpu<<" oldSM="<<old_sm<<" oldblocks="<<old_blocks<<" oldcapacity="<<old<<" pairSM="<<new_sm<<" pairblocks="<<new_blocks<<" paircapacity="<<candidate<<std::endl;cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 for(int n=1;n<=8;++n)for(int ratio:{0,1,2}){Fixture f(n,ratio,old,candidate);G graph(s,[&]{for(int v=0;v<3;++v)f.body(v,s);});for(int first:{0,1,125,127,128,511,1023,8191,262136})for(int phase=0;phase<3;++phase){f.input(first,phase,s);graph.run(s);f.verify(s,true);}}
 std::cout<<"NUM gpu="<<gpu<<" pair-key/adoptedN79/original attention BITWISE NT1..8 ratios0/1/2 nine starts threechangedcaptures; distinct queries/F16edges/sinks/wholezero/signedzero/subnorm + rounded weights/pointers/padding exact PASS"<<std::endl;
 if(gpu==0&&(argc==1||std::string(argv[1])!="--correctness-only")){Fixture f(3,2,old,candidate);f.input(1022,1,s);G baseline(s,[&]{f.body(1,s);}),paired(s,[&]{f.body(2,s);});need(baseline.inspect(3,640,old,false)==paired.inspect(3,640,candidate,true),"softmax/value kernels not the original function identities");auto run=[&](int v){if(v)paired.run(s);else baseline.run(s);};for(int i=0;i<16;++i)run(i%2);f.verify(s,false);double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto start=std::chrono::steady_clock::now();for(int i=0;i<16;++i)run(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/16;total[v]+=ms[v];}faster+=ms[1]<ms[0];f.verify(s,false);std::cout<<"pair="<<pair<<" N79_ms="<<ms[0]<<" pairkeys_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<std::endl;}
 std::cout.precision(12);std::cout<<"GATE full N79 QK+originalsoftmax+async64values NT3/head32/D512/ratio2/640rows cachewarm KV+query+workspace, both3kernels same originalsoftmax/value funcs; pair256threads/16heads/actualcapacity-derivedZ captureproved; sum_old_ms="<<total[0]<<" sum_new_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16; excludesprojections/indexranking, no model gain claim"<<std::endl;gate=total[0]/total[1]>1.05&&faster==16;}
 ck(cudaStreamDestroy(s));}
 std::cout<<"QK_PAIR_KEYS numeric PASS new_F32_values="<<outputs<<" compared_workspace_bytes="<<workspace_bytes<<" GPUs2"<<std::endl;if(!gate)throw std::runtime_error("full attention pair-key gate failed >1.05 and16/16 required");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
