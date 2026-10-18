#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
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
void exhaustive(cudaStream_t s) {
    constexpr int N=256*16;
    constexpr int integer[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    std::vector<uint8_t> rows(N*17,0);
    std::vector<float> x(32,1),reference(N);
    for (int exponent=0;exponent<256;++exponent) for (int q=0;q<16;++q) {
        const int row=exponent*16+q;
        rows[row*17]=uint8_t(exponent); rows[row*17+1]=uint8_t(q);
        // Independent double exponent construction, including subnormal and
        // overflowing FP32 products. All other nibbles contribute finite zero.
        reference[row]=float(std::ldexp(double(integer[q]),exponent-128));
    }
    Device w(rows.size()),input(x.size()*4),original(N*4),candidate(N*4);
    upload(w,rows.data(),rows.size(),s);upload(input,x.data(),x.size()*4,s);
    C::mxfp4_mmv(w.p,input.f(),original.f(),32,N,1,s,false);
    C::mxfp4_mmv(w.p,input.f(),candidate.f(),32,N,1,s,true);
    const auto a=download(original,N,s),b=download(candidate,N,s);
    equal(a,reference,"original vs independent 4096-value reference");
    equal(b,reference,"ALU vs independent 4096-value reference");
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
        C::mxfp4_gu(g.p,u.p,input.f(),ids.i(),hidden1.f(),H,F,K,t,1.5f,s,true);
        C::mxfp4_down(d.p,hidden1.f(),ids.i(),out1.f(),F,H,K,t,s,true);
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
    std::cout<<"device="<<device<<" exhaustive_scale_nibbles=4096 H=5120 F=2304 top_k=3 NT=1..8 graph_replays=24 bitwise=PASS\n";
}
int main() {
    try {
        int count=0;ck(cudaGetDeviceCount(&count));
        if (count<2) throw std::runtime_error("MXFP4 TP2 test needs two CUDA devices");
        for (int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            exhaustive(s);experts(device,s);ck(cudaStreamDestroy(s));
        }
        std::cout<<"Original/ALU MXFP4 and independent exponent reference PASS\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
