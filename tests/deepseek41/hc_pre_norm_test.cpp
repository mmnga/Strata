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
constexpr int Width=5120,Max=Width*8;
void run(int device,bool gate){
 ck(cudaSetDevice(device));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 {
 B residual(size_t(Max)*4*4),pre(4*8*4),weight(Width*4),x[2]={B(Max*4),B(Max*4)},y[2]={B(Max*4),B(Max*4)};
 H hx[2]={H(Max),H(Max)},hy[2]={H(Max),H(Max)};
 auto upload=[&](B& dst,size_t n,float phase){std::vector<float> v(n);for(size_t i=0;i<n;++i){v[i]=std::sin(float(i)*.113f+phase)*.3f;if(i%47==0)v[i]=-0.0f;if(i%71==0)v[i]=std::ldexp(float(int(i%7)-3),-140);}ck(cudaMemcpyAsync(dst.p,v.data(),n*4,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));};
 auto body=[&](int t,int variant,bool timing){
  if(variant)C::hc_pre_rms(residual.f(),pre.f(),weight.f(),x[variant].f(),y[variant].f(),Width,t,1e-6f,stream);
  else{C::hc_pre(residual.f(),pre.f(),x[variant].f(),Width,t,stream);C::rms(x[variant].f(),weight.f(),y[variant].f(),Width,t,1e-6f,stream);}
  if(!timing){ck(cudaMemcpyAsync(hx[variant].p,x[variant].p,Width*t*4,cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(hy[variant].p,y[variant].p,Width*t*4,cudaMemcpyDeviceToHost,stream));}
 };
 auto read=[&](int n,int v){ck(cudaMemcpy(hx[v].p,x[v].p,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hy[v].p,y[v].p,n*4,cudaMemcpyDeviceToHost));};
 auto verify=[&](int n){if(std::memcmp(hx[0].p,hx[1].p,n*4)||std::memcmp(hy[0].p,hy[1].p,n*4))throw std::runtime_error("fused HC pre/RMS changed output bits");for(int i=0;i<n;++i)if(!std::isfinite(hx[1].p[i])||!std::isfinite(hy[1].p[i]))throw std::runtime_error("nonfinite HC pre/RMS");};
 uint64_t compared=0;
 for(int t=1;t<=8;++t)for(bool capture:{false,true}){
  G graphs[2];if(capture)for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(t,v,false);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  for(int replay=0;replay<4;++replay){upload(residual,size_t(Max)*4,float(replay)*.2f);upload(pre,4*8,float(replay)*.3f+.1f);upload(weight,Width,float(replay)*.4f+.2f);
   for(int v=0;v<2;++v){if(capture)ck(cudaGraphLaunch(graphs[v].e,stream));else body(t,v,false);ck(cudaStreamSynchronize(stream));}
   verify(Width*t);compared+=uint64_t(Width)*t*2;
  }
 }
 std::cout<<"device="<<device<<" original HC pre+RMS vs fused bitwise PASS values="<<compared<<" NT1..8 captureon/off four changing replays signedzero/subnormal both outputs\n";
 if(gate){
  constexpr int t=3,n=Width*t;G graphs[2];for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(t,v,true);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  auto launch=[&](int v){ck(cudaGraphLaunch(graphs[v].e,stream));ck(cudaStreamSynchronize(stream));};for(int i=0;i<16;++i)launch(i%2);
  double total[2]={0,0};for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;total[v]+=ms[v];read(n,v);}verify(n);std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" fused_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
  std::cout.precision(12);std::cout<<"sum_old_ms="<<total[0]<<" sum_fused_ms="<<total[1]<<" speed_ratio="<<total[0]/total[1]<<" bitwise=PASS gate=original_HC_pre_then_RMS_vs_fused_graph_kernel_only_NT3\n";
 }
 }
 ck(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{const bool correctness_only=argc==2&&std::string(argv[1])=="--correctness-only";run(0,!correctness_only);run(1,false);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
