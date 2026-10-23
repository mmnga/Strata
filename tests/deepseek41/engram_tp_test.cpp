#include "strata/deepseek41/engram_tp.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
namespace E=strata::deepseek41::engram_tp;namespace C=strata::deepseek41::cuda;namespace K=strata::kernels;
constexpr int H=5120,In=6144,Out=25600,Max=8,Banks=5;
int bank_count=Banks;float test_eps=1e-6f;std::string actual_file;constexpr size_t Bytes=88473600,Guard=16;
void ck(cudaError_t e){E::check(e);}void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
struct B{void*p{};int d;size_t bytes;explicit B(size_t n):bytes(n){ck(cudaGetDevice(&d));ck(cudaMalloc(&p,n));}~B(){cudaSetDevice(d);cudaFree(p);}float*f(){return static_cast<float*>(p);}};
struct Stage{
 int d;cudaStream_t s{};std::vector<std::unique_ptr<B>>weights;
 B input[2]={B(In*Max*4),B(In*Max*4)},kv[2]={B((size_t(Out)*Max+Guard)*4),B((size_t(Out)*Max+Guard)*4)},res[2]={B((size_t(H)*4*Max+Guard)*4),B((size_t(H)*4*Max+Guard)*4)};
 B gates[2]={B((4*Max+Guard)*4),B((4*Max+Guard)*4)};
 B q[2]={B(K::native_q8_1_bytes(In,Max)),B(K::native_q8_1_bytes(In,Max))},il[2]={B(K::native_q8_1_il_bytes(In,Max)),B(K::native_q8_1_il_bytes(In,Max))},initial{H*4*Max*4},query{H*4*4},key{H*4*4};
 Stage(int device):d(device){int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest));ck(cudaStreamCreateWithPriority(&s,cudaStreamNonBlocking,greatest));for(int i=0;i<bank_count;++i)weights.push_back(std::make_unique<B>(Bytes));}
 ~Stage(){cudaSetDevice(d);cudaStreamSynchronize(s);cudaStreamDestroy(s);}
 E::View view(int v,int bank){return {d,s,static_cast<uint8_t*>(weights[bank]->p),input[v].f(),kv[v].f(),res[v].f(),query.f(),key.f(),q[v].p,il[v].p,true};}
 template<class A>void upload(B&b,const std::vector<A>&x){ck(cudaSetDevice(d));need(x.size()*sizeof(A)<=b.bytes,"upload bounds");ck(cudaMemcpyAsync(b.p,x.data(),x.size()*sizeof(A),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
 void clear(){ck(cudaSetDevice(d));for(int v=0;v<2;++v){ck(cudaMemsetAsync(gates[v].p,0x5a,gates[v].bytes,s));ck(cudaMemsetAsync(kv[v].p,0x5a,kv[v].bytes,s));ck(cudaMemsetAsync(res[v].p,0x5a,res[v].bytes,s));}ck(cudaStreamSynchronize(s));}
};
std::vector<float>values(size_t n,int seed,int phase){std::vector<float>x(n);uint32_t state=193+seed;for(size_t i=0;i<n;++i){state=1664525u*state+1013904223u;float z=phase?float(int(state>>16)-32768)/32768.f:0.f;if(i%47==0)z=-0.f;if(phase&&i%71==0)z=std::bit_cast<float>(uint32_t(1));if(phase==2&&i%997==17)z=1e5f;x[i]=z;}return x;}
void original_project(const E::View&v,int n){
 if(n>=5){K::native_quantize_q8_1_il_wide(v.input,v.q,v.il,In,n,v.stream);K::native_mmvq_il_wide(12,v.weights,v.il,v.kv,In,Out,n,v.stream);}
 else if(n>1&&K::native_mmvq_il_supported(12,n,Out)){K::native_quantize_q8_1_il(v.input,v.q,v.il,In,n,v.stream);K::native_mmvq_il(12,v.weights,v.q,v.il,v.kv,In,Out,n,v.stream);}
 else{K::native_quantize_q8_1(v.input,v.q,In,n,v.stream);K::native_mmvq(12,v.weights,v.q,v.kv,In,Out,n,v.stream);}
}
struct Suite{
 std::unique_ptr<Stage>s[2];std::vector<float>inputs[Banks];float*host{};uint64_t checked=0;std::vector<float>actual_q[2],actual_k[2];
 Suite(){K::native_mmvq_set_multi_exact(true);for(int d=0;d<2;++d){ck(cudaSetDevice(d));s[d]=std::make_unique<Stage>(d);}ck(cudaMallocHost(&host,size_t(2)*E::Half*Max*4));
 if(!actual_file.empty()){
  std::ifstream f(actual_file,std::ios::binary);need(bool(f),"actual fixture missing");
  for(int bank=0;bank<2;++bank){std::vector<uint8_t>w(Bytes);f.read(reinterpret_cast<char*>(w.data()),w.size());need(bool(f),"actual WKV short");for(auto&z:s)z->upload(*z->weights[bank],w);
   for(int role=0;role<2;++role){std::vector<uint16_t>raw(H*4);f.read(reinterpret_cast<char*>(raw.data()),raw.size()*2);need(bool(f),"actual Q/K short");auto&values=role?actual_k[bank]:actual_q[bank];values.resize(raw.size());for(size_t i=0;i<raw.size();++i)values[i]=std::bit_cast<float>(uint32_t(raw[i])<<16);}
  }need(f.peek()==std::char_traits<char>::eof(),"actual fixture order/size");
 }else for(int bank=0;bank<bank_count;++bank){std::vector<uint8_t>w(Bytes);uint32_t st=193+bank;for(auto&b:w){st=1664525u*st+1013904223u;b=uint8_t(st>>24);}for(size_t at=0;at<w.size();at+=144){uint16_t a=uint16_t(0x2400+(at/144)%1024),b=uint16_t(0x2000+(at/144+137)%1024);if(at%(144*1024)==0)a=b=0;std::memcpy(w.data()+at,&a,2);std::memcpy(w.data()+at+2,&b,2);}for(auto&z:s)z->upload(*z->weights[bank],w);}
 }
 ~Suite(){for(auto&z:s){cudaSetDevice(z->d);cudaStreamSynchronize(z->s);}cudaFreeHost(host);}
 void prepare(int phase){for(int b=0;b<bank_count;++b)inputs[b]=values(In*Max,300+b*17,phase);auto r=values(H*4*Max,41,phase),q=values(H*4,71,1),k=values(H*4,101,1);for(auto&z:s){z->upload(z->initial,r);z->upload(z->query,q);z->upload(z->key,k);z->clear();}}
 void run(int v,int bank,int n,bool diagnostic=false){
  if(!actual_file.empty())for(auto&z:s){z->upload(z->query,actual_q[bank]);z->upload(z->key,actual_k[bank]);}
  for(auto&z:s){ck(cudaSetDevice(z->d));ck(cudaMemcpyAsync(z->res[v].p,z->initial.p,size_t(H)*4*n*4,cudaMemcpyDeviceToDevice,z->s));}
  if(v){E::View views[2]={s[0]->view(1,bank),s[1]->view(1,bank)};for(int r=0;r<2;++r)views[r].gates=diagnostic?s[r]->gates[1].f():nullptr;E::enqueue(views,inputs[bank].data(),host,n,test_eps);}
  else for(auto&z:s){ck(cudaSetDevice(z->d));auto w=z->view(0,bank);ck(cudaMemcpyAsync(w.input,inputs[bank].data(),size_t(In)*n*4,cudaMemcpyHostToDevice,w.stream));original_project(w,n);C::engram_add(w.residual,w.kv,w.query,w.key,diagnostic?z->gates[0].f():nullptr,H,n,test_eps,w.stream);}
  for(auto&z:s){ck(cudaSetDevice(z->d));ck(cudaStreamSynchronize(z->s));}
 }
 std::vector<uint8_t>read(B&b,Stage&z,size_t bytes){std::vector<uint8_t>x(bytes);ck(cudaSetDevice(z.d));ck(cudaMemcpyAsync(x.data(),b.p,bytes,cudaMemcpyDeviceToHost,z.s));ck(cudaStreamSynchronize(z.s));return x;}
 void verify(int n){std::vector<uint8_t>refkv,refres;for(auto&z:s){auto a=read(z->kv[0],*z,z->kv[0].bytes),b=read(z->kv[1],*z,z->kv[1].bytes);need(a==b,"complete KV/tail differs");for(size_t i=size_t(Out)*Max*4;i<a.size();++i)need(a[i]==0x5a,"KV guard clobbered");if(refkv.empty())refkv=a;else need(a==refkv,"original full KV crossGPU differs");a=read(z->res[0],*z,z->res[0].bytes);b=read(z->res[1],*z,z->res[1].bytes);need(a==b,"Engram residual/tail differs");for(size_t i=size_t(H)*4*Max*4;i<a.size();++i)need(a[i]==0x5a,"residual guard clobbered");if(refres.empty())refres=a;else need(a==refres,"original crossGPU residual differs");need(read(z->q[0],*z,K::native_q8_1_bytes(In,n))==read(z->q[1],*z,K::native_q8_1_bytes(In,n)),"Q81 differs");if(n>1)need(read(z->il[0],*z,K::native_q8_1_il_bytes(In,n))==read(z->il[1],*z,K::native_q8_1_il_bytes(In,n)),"IL differs");need(read(z->gates[0],*z,4*n*4)==read(z->gates[1],*z,4*n*4),"gates differ");checked+=uint64_t(n)*(Out+4*H+4);}}
 void proof(int bank,int n){for(int d=0;d<2;++d){auto&z=*s[d];ck(cudaSetDevice(d));auto v=z.view(1,bank);cudaGraph_t g{};ck(cudaStreamBeginCapture(z.s,cudaStreamCaptureModeThreadLocal));E::project(v,n,E::Half,size_t(d)*E::HalfWeightBytes);ck(cudaStreamEndCapture(z.s,&g));size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t>nodes(count);ck(cudaGraphGetNodes(g,nodes.data(),&count));int found=0;bool quant=false;for(auto node:nodes){cudaGraphNodeType t;ck(cudaGraphNodeGetType(node,&t));if(t!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));const char*name=nullptr;ck(cudaFuncGetName(&name,p.func));std::string k=name?name:"";void*first=nullptr;std::memcpy(&first,p.kernelParams[0],sizeof first);if(k.find("mmvq")!=std::string::npos&&k.find("quant")==std::string::npos){need(first==v.weights+size_t(d)*E::HalfWeightBytes,"actual row-half weight pointer absent");++found;}if(k.find("quant")!=std::string::npos&&first==v.input){void*qp=nullptr;std::memcpy(&qp,p.kernelParams[1],sizeof qp);need(qp==v.q,"actual plain Q81 output pointer");if(n>1){void*ip=nullptr;std::memcpy(&ip,p.kernelParams[2],sizeof ip);need(ip==v.il,"actual IL base output pointer");}quant=true;}}need(found==1&&quant,"actual native half/quant capture absent");cudaGraphExec_t exec{};ck(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));ck(cudaGraphLaunch(exec,z.s));ck(cudaStreamSynchronize(z.s));auto raw=read(z.kv[1],z,size_t(E::Half)*n*4),ref=read(z.kv[0],z,size_t(Out)*n*4);for(int t=0;t<n;++t)need(!std::memcmp(raw.data()+size_t(t)*E::Half*4,ref.data()+(size_t(t)*Out+d*E::Half)*4,E::Half*4),"captured half vs full original rows differs");checked+=uint64_t(E::Half)*n;ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(g));}}
};
int main(int argc,char**argv){try{if(argc==4&&std::string(argv[1])=="--actual"){actual_file=argv[2];test_eps=std::stof(argv[3]);bank_count=2;}Suite z;int l2=0;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0));need(size_t(Banks)*E::HalfWeightBytes>size_t(2*l2),"half working set not cold");std::cout<<"FIXTURE banks5 full_bytes="<<Bytes*Banks<<" half_bytes="<<E::HalfWeightBytes*Banks<<" L2="<<l2<<" no new GPU kernels/weights format"<<std::endl;
 for(int phase=0;phase<3;++phase){z.prepare(phase);for(int n=1;n<=8;++n)for(int bank=0;bank<bank_count;++bank){z.run(0,bank,n,true);z.run(1,bank,n,true);z.verify(n);z.proof(bank,n);}}
 std::cout<<"NUMERIC complete banks="<<bank_count<<"  NT1..8 bothGPU allKV/residual/Q81/IL/guards/crossGPU exact F32="<<z.checked<<std::endl;
 if(argc==2&&std::string(argv[1])=="--correctness-only")return 0;
 if(!actual_file.empty()){std::cout<<"ACTUAL original two WKV/Q/K bothGPU exact PASS"<<std::endl;return 0;}
 z.prepare(1);for(int warm=0;warm<4;++warm)for(int v=0;v<2;++v)for(int b=0;b<bank_count;++b)z.run(v,b,3);
 double sum[2]={};int faster=0;for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto start=std::chrono::steady_clock::now();for(int rep=0;rep<4;++rep)for(int b=0;b<bank_count;++b)z.run(v,b,3);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/20;sum[v]+=ms[v];}faster+=ms[1]<ms[0];for(int b=0;b<bank_count;++b){z.run(0,b,3,true);z.run(1,b,3,true);z.verify(3);}std::cout<<"PAIR "<<pair<<" full_ms="<<ms[0]<<" tp_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<std::endl;}
 bool pass=sum[0]/sum[1]>1.05&&faster==16;std::cout.precision(14);std::cout<<"RESULT {\"old_sum_ms\":"<<sum[0]<<",\"new_sum_ms\":"<<sum[1]<<",\"ratio\":"<<sum[0]/sum[1]<<",\"faster_pairs\":"<<faster<<",\"pairs\":16,\"bitwise_f32\":"<<z.checked<<",\"gate_pass\":"<<(pass?"true":"false")<<"}"<<std::endl;return pass?0:1;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 2;}}
