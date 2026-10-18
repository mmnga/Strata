#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace K = strata::kernels;
namespace C = strata::deepseek41::cuda;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n)); ck(cudaMemset(p,0,n)); }
    ~Device() { cudaFree(p); }
    float* f() const { return static_cast<float*>(p); }
    int32_t* i() const { return static_cast<int32_t*>(p); }
};
void upload(Device& d,const void* p,size_t bytes) { ck(cudaMemcpy(d.p,p,bytes,cudaMemcpyHostToDevice)); }
std::vector<uint8_t> rows(int type,int in,int out,std::mt19937& rng) {
    const size_t block = type == 12 ? 144 : 210;
    std::vector<uint8_t> result(size_t(in/256)*out*block);
    for (auto& v : result) v = uint8_t(rng());
    for (size_t i = 0; i < result.size(); i += block) {
        const uint16_t scale = 0x1800;
        std::memcpy(result.data()+i+(type == 12 ? 0 : 208),&scale,2);
        if (type == 12) { const uint16_t minimum = 0x1400; std::memcpy(result.data()+i+2,&minimum,2); }
    }
    return result;
}
void check(int device,int tokens,int down_type,cudaStream_t stream) {
    constexpr int H = 5120, F = 2304, groups = 3, Ktop = 6;
    constexpr float limit = 1.75f;
    const int cap = tokens*Ktop;
    const auto layout = K::native_expert_layout(12,down_type,H,F);
    std::mt19937 rng(817+tokens); std::vector<float> x(size_t(H)*tokens);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-2,2)(rng);
    Device input(x.size()*4), xq(K::native_q8_1_bytes(H,tokens)), blobs(layout.bytes*groups);
    upload(input,x.data(),x.size()*4);
    std::vector<unsigned long long> pointers(groups);
    for (int g = 0; g < groups; ++g) {
        auto blob = rows(12,H,F*2,rng), down = rows(down_type,F,H,rng); blob.insert(blob.end(),down.begin(),down.end());
        ck(cudaMemcpy(static_cast<uint8_t*>(blobs.p)+size_t(g)*layout.bytes,blob.data(),blob.size(),cudaMemcpyHostToDevice));
        pointers[g] = reinterpret_cast<unsigned long long>(static_cast<uint8_t*>(blobs.p)+size_t(g)*layout.bytes);
    }
    std::vector<int32_t> starts(groups+1), destination(cap), token(cap);
    for (int g = 0; g <= groups; ++g) starts[g] = g*tokens;
    for (int e = 0; e < cap; ++e) { destination[e] = cap-1-e; token[e] = (e+1)%tokens; }
    Device ptr(groups*8), start((groups+1)*4), ng(4), dst(cap*4), tok(cap*4), scratch(K::native_expert_scratch_bytes(cap,F));
    Device actual(size_t(cap)*H*4), reference(size_t(cap)*H*4), gate(F*4), up(F*4), hidden(F*4), hq(K::native_q8_1_bytes(F));
    upload(ptr,pointers.data(),groups*8); upload(start,starts.data(),(groups+1)*4); upload(dst,destination.data(),cap*4); upload(tok,token.data(),cap*4);
    int count = groups; upload(ng,&count,4);
    K::native_quantize_q8_1(input.f(),xq.p,H,tokens,stream);
    for (int g = 0; g < groups; ++g) for (int e = starts[g]; e < starts[g+1]; ++e) {
        const auto* blob = reinterpret_cast<const uint8_t*>(pointers[g]);
        const auto* q = static_cast<const uint8_t*>(xq.p)+size_t(token[e])*K::native_q8_1_bytes(H);
        K::native_mmvq(12,blob,q,gate.f(),H,F,1,stream);
        K::native_mmvq(12,blob+layout.up_off,q,up.f(),H,F,1,stream);
        C::swiglu(gate.f(),up.f(),hidden.f(),F,limit,stream);
        K::native_quantize_q8_1(hidden.f(),hq.p,F,1,stream);
        K::native_mmvq(down_type,blob+layout.down_off,hq.p,reference.f()+size_t(destination[e])*H,F,H,1,stream);
    }
    // Replayed graphs must read changed group counts and plans, including an
    // empty call, without touching destinations outside the active entries.
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    K::native_expert_grouped_clamp(layout,static_cast<const unsigned long long*>(ptr.p),start.i(),ng.i(),dst.i(),tok.i(),cap,cap,xq.p,scratch.p,actual.f(),stream,limit,2);
    ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    count = 0; upload(ng,&count,4); ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
    std::vector<float> empty(size_t(cap)*H); ck(cudaMemcpy(empty.data(),actual.p,empty.size()*4,cudaMemcpyDeviceToHost));
    if (std::any_of(empty.begin(),empty.end(),[](float v) { return v != 0; })) throw std::runtime_error("empty grouped graph wrote output");
    count = groups; upload(ng,&count,4); ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
    std::vector<float> a(size_t(cap)*H), b(a.size());
    ck(cudaMemcpy(a.data(),actual.p,a.size()*4,cudaMemcpyDeviceToHost)); ck(cudaMemcpy(b.data(),reference.p,b.size()*4,cudaMemcpyDeviceToHost));
    double error = 0, power = 0, max_error = 0, max_value = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite expert result");
        const double delta = double(a[i])-b[i]; error += delta*delta; power += double(b[i])*b[i];
        max_error = std::max(max_error,std::abs(delta)); max_value = std::max(max_value,std::abs(double(b[i])));
    }
    const double relative = std::sqrt(error/std::max(power,1e-30));
    std::cout << "device=" << device << " tokens=" << tokens << " down_type=" << down_type << " grouped clamp vs independent matvec relative_rms=" << relative << " max_abs=" << max_error << '\n';
    if (relative > 3e-3 || max_error > 0.02*max_value) throw std::runtime_error("grouped expert differs from independent path");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
}
int main() {
    try {
        int devices = 0; ck(cudaGetDeviceCount(&devices));
        for (int d = 0; d < devices; ++d) {
            ck(cudaSetDevice(d)); cudaStream_t stream = nullptr; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for (int type : {12,14}) for (int n : {1,4,8}) check(d,n,type,stream);
            ck(cudaStreamDestroy(stream));
        }
        std::cout << "DeepSeek grouped clamp, scattered entries and graph plan updates PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
