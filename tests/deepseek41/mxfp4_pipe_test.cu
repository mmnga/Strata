#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <string>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace C = strata::deepseek41::cuda;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Device() { cudaFree(p); }
    float* f() const { return static_cast<float*>(p); }
    int32_t* i() const { return static_cast<int32_t*>(p); }
};
uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,4); return result; }
void equal(const std::vector<float>& a,const std::vector<float>& b,const char* label) {
    if (a.size()!=b.size()) throw std::runtime_error("test shape");
    for (size_t i=0;i<a.size();++i) if (bits(a[i])!=bits(b[i]))
        throw std::runtime_error(std::string(label)+" bit mismatch at "+std::to_string(i));
}
void upload(Device& d,const void* data,size_t bytes,cudaStream_t s) {
    ck(cudaMemcpyAsync(d.p,data,bytes,cudaMemcpyHostToDevice,s));
}
std::vector<float> download(const Device& d,size_t count,cudaStream_t s) {
    std::vector<float> values(count);
    ck(cudaMemcpyAsync(values.data(),d.p,count*4,cudaMemcpyDeviceToHost,s));
    ck(cudaStreamSynchronize(s)); return values;
}
std::vector<uint8_t> weights(int input,int rows,std::mt19937& random) {
    std::vector<uint8_t> result(size_t(input/32)*17*rows);
    for (auto& v:result) v=uint8_t(random());
    for (size_t i=0;i<result.size();i+=17) result[i]=uint8_t(113+random()%10);
    return result;
}
void experts(int device,cudaStream_t s) {
    constexpr int H=5120,F=2304,E=5,K=3,MaxT=8;
    std::mt19937 random(71245);
    const auto gate=weights(H,F*E,random),up=weights(H,F*E,random),down=weights(F,H*E,random);
    Device g(gate.size()),u(up.size()),d(down.size()),input(MaxT*H*4),ids(MaxT*K*4);
    Device hidden0(MaxT*K*F*4),hidden1(MaxT*K*F*4),out0(MaxT*K*H*4),out1(MaxT*K*H*4);
    upload(g,gate.data(),gate.size(),s);upload(u,up.data(),up.size(),s);upload(d,down.data(),down.size(),s);
    ck(cudaStreamSynchronize(s));
    for (int t=1;t<=MaxT;++t) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        C::mxfp4_gu(g.p,u.p,input.f(),ids.i(),hidden0.f(),H,F,K,t,1.5f,s,false);
        C::mxfp4_down(d.p,hidden0.f(),ids.i(),out0.f(),F,H,K,t,s,false);
        C::mxfp4_pipe_gu(g.p,u.p,input.f(),ids.i(),hidden1.f(),H,F,K,t,1.5f,s,true);
        C::mxfp4_pipe_down(d.p,hidden1.f(),ids.i(),out1.f(),F,H,K,t,s,true);
        ck(cudaStreamEndCapture(s,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for (int replay=0;replay<3;++replay) {
            std::vector<float> x(t*H);std::vector<int32_t> routes(t*K);
            for (auto& v:x) v=std::uniform_real_distribution<float>(-2,2)(random);
            x[0]=0.0f;x[1]=-0.0f;x[2]=std::ldexp(1.0f,-140);x[3]=-x[2];
            for (int token=0;token<t;++token) for (int k=0;k<K;++k)
                routes[token*K+k]=(token+replay+k)%E;
            upload(input,x.data(),x.size()*4,s);upload(ids,routes.data(),routes.size()*4,s);
            ck(cudaGraphLaunch(exec,s));
            const auto a=download(hidden0,t*K*F,s),b=download(hidden1,t*K*F,s);
            const auto c=download(out0,t*K*H,s),e=download(out1,t*K*H,s);
            for (float v:c) if (!std::isfinite(v)) throw std::runtime_error("nonfinite expert fixture");
            equal(a,b,"GU graph");equal(c,e,"Down graph");
        }
        ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    }
    std::cout<<"device="<<device<<" primitive_unchanged_from_N22 H=5120 F=2304 top_k=3 NT=1..8 graph_replays=24 bitwise=PASS\n";
}
void gate(){
 constexpr int H=5120,F=2304,E=16,K=3,T=5;
 ck(cudaSetDevice(0));cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 {
 std::mt19937 random(68149);const auto gv=weights(H,F*E,random),uv=weights(H,F*E,random),dv=weights(F,H*E,random);
 std::vector<float>xv(H*T);for(size_t i=0;i<xv.size();++i)xv[i]=std::sin(float(i)*.007f)*.2f;
 std::vector<int32_t> route(K*T);for(int i=0;i<K*T;++i)route[i]=i;
 Device g(gv.size()),u(uv.size()),d(dv.size()),x(xv.size()*4),ids(route.size()*4);
 Device hidden[2]={Device(F*K*T*4),Device(F*K*T*4)},out[2]={Device(H*K*T*4),Device(H*K*T*4)};
 upload(g,gv.data(),gv.size(),s);upload(u,uv.data(),uv.size(),s);upload(d,dv.data(),dv.size(),s);upload(x,xv.data(),xv.size()*4,s);upload(ids,route.data(),route.size()*4,s);ck(cudaStreamSynchronize(s));
 cudaGraph_t graph[2]={};cudaGraphExec_t exec[2]={};
 for(int v=0;v<2;++v){ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
 if(v){C::mxfp4_pipe_gu(g.p,u.p,x.f(),ids.i(),hidden[v].f(),H,F,K,T,10.f,s,true);C::mxfp4_pipe_down(d.p,hidden[v].f(),ids.i(),out[v].f(),F,H,K,T,s,true);}
 else {C::mxfp4_gu(g.p,u.p,x.f(),ids.i(),hidden[v].f(),H,F,K,T,10.f,s,true);C::mxfp4_down(d.p,hidden[v].f(),ids.i(),out[v].f(),F,H,K,T,s,true);}
 ck(cudaStreamEndCapture(s,&graph[v]));ck(cudaGraphInstantiate(&exec[v],graph[v],nullptr,nullptr,0));}
 auto launch=[&](int v){ck(cudaGraphLaunch(exec[v],s));ck(cudaStreamSynchronize(s));};
 for(int i=0;i<16;++i)launch(i%2);
 auto verify=[&]{equal(download(hidden[0],F*K*T,s),download(hidden[1],F*K*T,s),"gate hidden");equal(download(out[0],H*K*T,s),download(out[1],H*K*T,s),"gate output");};verify();
 double total[2]={};int faster=0;
 for(int pair=0;pair<16;++pair){double ms[2];for(int order=0;order<2;++order){int v=(pair+order)%2;auto begin=std::chrono::steady_clock::now();for(int i=0;i<16;++i)launch(v);ms[v]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/16;total[v]+=ms[v];}faster+=ms[1]<ms[0];verify();std::cout<<"pair="<<pair<<" old_ms="<<ms[0]<<" pipe_ms="<<ms[1]<<" ratio="<<ms[0]/ms[1]<<'\n';}
 std::cout.precision(12);std::cout<<"GATE GU+Down H5120 F2304 E16 top3 NT5 15distinctroutes fulloriginalMXFP4 weights sameFMA: old_sum_ms="<<total[0]<<" pipe_sum_ms="<<total[1]<<" ratio="<<total[0]/total[1]<<" faster="<<faster<<"/16 bitwise=PASS\n";
 for(int v=0;v<2;++v){ck(cudaGraphExecDestroy(exec[v]));ck(cudaGraphDestroy(graph[v]));}
 }
 ck(cudaStreamDestroy(s));
}

int main(int argc,char**argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--gate-only"){gate();return 0;}
        int count=0;ck(cudaGetDeviceCount(&count));
        if (count<2) throw std::runtime_error("MXFP4 TP2 test needs two CUDA devices");
        for (int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            experts(device,s);ck(cudaStreamDestroy(s));
        }
        if(argc==1 || std::string(argv[1])!="--correctness-only")gate();
        std::cout<<"Original/load-pipelined MXFP4 GU/Down FMA-order PASS\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
