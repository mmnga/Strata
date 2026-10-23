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
void run(int device,bool check,bool gate){
 constexpr int width=5120,columns=384;
 ck(cudaSetDevice(device));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 {
 const bool hc=width==20480;B input(size_t(width)*8*4),weight(size_t(width)*columns*4),norm[2]={B(size_t(width)*8*4),B(size_t(width)*8*4)},out[2]={B(columns*8*4),B(columns*8*4)};H host[2]={H(columns*8),H(columns*8)};
 auto upload=[&](B& dst,size_t n,float phase){std::vector<float> v(n);for(size_t i=0;i<n;++i){v[i]=std::sin(float(i)*.113f+phase)*.3f;if(i%47==0)v[i]=-0.0f;if(i%71==0)v[i]=std::ldexp(float(int(i%7)-3),-140);}ck(cudaMemcpyAsync(dst.p,v.data(),n*4,cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));};
 auto body=[&](int t,int variant,bool timing){const float* x=input.f();if(hc){C::rms(input.f(),nullptr,norm[variant].f(),width,t,1e-6f,stream);x=norm[variant].f();}
  if(variant)C::router_dense_pipe4(weight.f(),x,out[variant].f(),width,columns,t,stream);else C::dense(weight.f(),x,out[variant].f(),width,columns,t,stream);
  if(!timing)ck(cudaMemcpyAsync(host[variant].p,out[variant].p,columns*t*4,cudaMemcpyDeviceToHost,stream));
 };
 auto read=[&](int t,int v){ck(cudaMemcpy(host[v].p,out[v].p,columns*t*4,cudaMemcpyDeviceToHost));};
 auto verify=[&](int t){int n=columns*t;if(std::memcmp(host[0].p,host[1].p,n*4))throw std::runtime_error("dense pipeline changed output bits");for(int i=0;i<n;++i)if(!std::isfinite(host[1].p[i]))throw std::runtime_error("nonfinite projection");};
 upload(input,size_t(width)*8,.4f);upload(weight,size_t(width)*columns,.6f);
 uint64_t compared=0;
 if(check){
 for(int t=1;t<=8;++t)for(bool capture:{false,true}){
  G graphs[2];if(capture)for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(t,v,false);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  for(int replay=0;replay<4;++replay){upload(input,size_t(width)*8,float(replay)*.2f);upload(weight,size_t(width)*columns,float(replay)*.4f+.2f);
   for(int v=0;v<2;++v){if(capture)ck(cudaGraphLaunch(graphs[v].e,stream));else body(t,v,false);ck(cudaStreamSynchronize(stream));}verify(t);compared+=uint64_t(columns)*t;
  }
 }
 std::cout<<"UNIT device="<<device<<" width="<<width<<" columns="<<columns<<" values="<<compared<<" bitwise=PASS NT1..8 captureon/off four changing inputs/weights signedzero/subnormal\n";
 }
 if(gate){constexpr int t=3;G graphs[2];for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body(t,v,true);ck(cudaStreamEndCapture(stream,&graphs[v].g));ck(cudaGraphInstantiate(&graphs[v].e,graphs[v].g,nullptr,nullptr,0));}
  auto launch=[&](int v){ck(cudaGraphLaunch(graphs[v].e,stream));ck(cudaStreamSynchronize(stream));};for(int i=0;i<16;++i)launch(i%2);
  double total[2]={0,0};for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;total[v]+=ms[v];read(t,v);}verify(t);std::cout<<"pair="<<pair<<" width="<<width<<" old_ms="<<ms[0]<<" pipe_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
  std::cout.precision(12);std::cout<<"GATE width="<<width<<" columns="<<columns<<" sum_old_ms="<<total[0]<<" sum_pipe_ms="<<total[1]<<" speed_ratio="<<total[0]/total[1]<<" bitwise=PASS NT3 HC_includes_original_RMS_router_is_projection_only\n";
 }
 }
 ck(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{
 const std::string mode=argc>1?argv[1]:"";
 if(mode=="--remaining-validation"){run(0,false,true);run(1,true,false);}
 else if(mode=="--correctness-only"){run(0,true,false);run(1,true,false);}
 else{run(0,true,true);run(1,true,false);}
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
