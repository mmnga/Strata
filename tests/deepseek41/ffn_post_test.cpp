#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::deepseek41::cuda;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct B{void*p=nullptr;explicit B(size_t n){ck(cudaMalloc(&p,n));}~B(){cudaFree(p);}float*f(){return static_cast<float*>(p);}};
struct H{float*p=nullptr;explicit H(size_t n){ck(cudaHostAlloc(&p,n*4,cudaHostAllocMapped|cudaHostAllocPortable));}~H(){cudaFreeHost(p);}};
struct G{cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;~G(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);}};
constexpr int Width=5120,Top=6,Max=Width*8;
void run(int device,bool gate){
 ck(cudaSetDevice(device));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 {
 B parts(size_t(Max)*Top*4),weights(Top*8*4),shared(Max*4),initial(size_t(Max)*4*4),post(4*8*4),comb(16*8*4),mask(8);
 B residual[2]={B(size_t(Max)*4*4),B(size_t(Max)*4*4)},out[2]={B(Max*4),B(Max*4)};
 H cpu(size_t(Max)*Top),ffn[2]={H(Max),H(Max)},hc[2]={H(size_t(Max)*4),H(size_t(Max)*4)};
 float*mapped=nullptr;ck(cudaHostGetDevicePointer(&mapped,cpu.p,0));
 auto upload=[&](B& dst,size_t n,float phase){std::vector<float> v(n);for(size_t i=0;i<n;++i){v[i]=std::sin(float(i)*0.137f+phase)*0.25f;if(i%47==0)v[i]=-0.0f;if(i%71==0)v[i]=std::ldexp(float(int(i%7)-3),-140);}ck(cudaMemcpyAsync(dst.p,v.data(),n*4,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));};
 upload(parts,size_t(Max)*Top,.1f);upload(weights,Top*8,.2f);upload(shared,Max,.3f);upload(initial,size_t(Max)*4,.4f);upload(post,4*8,.5f);upload(comb,16*8,.6f);
 auto body=[&](int tokens,int variant,bool timing){int n=Width*tokens;
  ck(cudaMemcpyAsync(residual[variant].p,initial.p,size_t(n)*4*4,cudaMemcpyDeviceToDevice,stream));
  if(variant)C::combine_sources_post_system(parts.f(),mapped,static_cast<uint64_t*>(mask.p),weights.f(),shared.f(),out[variant].f(),residual[variant].f(),post.f(),comb.f(),Width,Top,tokens,stream);
  else C::combine_sources_system(parts.f(),mapped,static_cast<uint64_t*>(mask.p),weights.f(),shared.f(),out[variant].f(),Width,Top,tokens,stream);
  ck(cudaMemcpyAsync(ffn[variant].p,out[variant].p,n*4,cudaMemcpyDeviceToHost,stream));
  if(!variant)C::hc_post(out[variant].f(),residual[variant].f(),post.f(),comb.f(),residual[variant].f(),Width,tokens,stream);
  if(!timing)ck(cudaMemcpyAsync(hc[variant].p,residual[variant].p,size_t(n)*4*4,cudaMemcpyDeviceToHost,stream));
 };
 auto verify=[&](int n){if(std::memcmp(ffn[0].p,ffn[1].p,n*4)||std::memcmp(hc[0].p,hc[1].p,size_t(n)*4*4))throw std::runtime_error("fused FFN/HC changed bits");for(int i=0;i<n;++i)if(!std::isfinite(ffn[1].p[i]))throw std::runtime_error("nonfinite FFN");for(int i=0;i<4*n;++i)if(!std::isfinite(hc[1].p[i]))throw std::runtime_error("nonfinite HC");};
 uint64_t compared=0;
 for(int tokens=1;tokens<=8;++tokens)for(bool capture:{false,true}){
  G graphs[2];if(capture)for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(tokens,v,false);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  for(int replay=0;replay<12;++replay){
   uint64_t bits=0;for(int row=0;row<Top*tokens;++row)if(replay%3==1||(replay%3==2&&row%2==0))bits|=uint64_t(1)<<row;
   ck(cudaMemcpyAsync(mask.p,&bits,8,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));
   for(size_t i=0;i<size_t(tokens)*Top*Width;++i){float v=std::sin(float(i)*.117f+float(replay)*.2f);if(i%17==0)v=-0.0f;if(i%23==0)v=std::ldexp(float(int(i%7)-3),-140);cpu.p[i]=v;}
   std::atomic_thread_fence(std::memory_order_release);
   for(int v=0;v<2;++v){if(capture)ck(cudaGraphLaunch(graphs[v].e,stream));else body(tokens,v,false);ck(cudaStreamSynchronize(stream));}
   verify(Width*tokens);compared+=uint64_t(Width)*tokens*5;
  }
 }
 std::cout<<"device="<<device<<" original combine+HC vs fused bitwise PASS values="<<compared<<" NT1..8 captureon/off masksnone/all/mixed changingCPU signedzero/subnormal inplaceHC\n";
 if(gate){
  constexpr int tokens=3,n=Width*tokens;G graphs[2];uint64_t bits=0;for(int row=0;row<Top*tokens;++row)if(row%2==0)bits|=uint64_t(1)<<row;ck(cudaMemcpyAsync(mask.p,&bits,8,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));
  for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(tokens,v,true);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  auto launch=[&](int v){ck(cudaGraphLaunch(graphs[v].e,stream));ck(cudaStreamSynchronize(stream));};
  for(int i=0;i<16;++i)launch(i%2);
  double total[2]={0,0};
  for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;total[v]+=ms[v];ck(cudaMemcpy(hc[v].p,residual[v].p,size_t(n)*4*4,cudaMemcpyDeviceToHost));}verify(n);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" fused_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
  std::cout.precision(12);std::cout<<"sum_old_ms="<<total[0]<<" sum_fused_ms="<<total[1]<<" speed_ratio="<<total[0]/total[1]<<" bitwise=PASS gate=original_2nodes_vs_fused_1node_plus_common_D2D_reset_and_FFN_D2H\n";
 }
 }
 ck(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{const bool correctness_only=argc==2&&std::string(argv[1])=="--correctness-only";run(0,!correctness_only);run(1,false);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
