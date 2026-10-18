#include "strata/deepseek41/ffn_plan.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace D=strata::deepseek41;
namespace C=D::cuda;
namespace K=strata::kernels;
void ck(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
struct Device {
    void* p=nullptr;
    explicit Device(size_t bytes) {ck(cudaMalloc(&p,bytes));}
    ~Device() {cudaFree(p);}
    float* f() const {return static_cast<float*>(p);}
};
template<class T> struct Host {
    T* p=nullptr; size_t n;
    explicit Host(size_t size,unsigned flags=0):n(size) {ck(cudaHostAlloc(&p,n*sizeof(T),flags));}
    ~Host() {cudaFreeHost(p);}
};
void finite_rows(uint8_t* p,size_t bytes,int type,std::mt19937& rng) {
    const size_t block=type==12 ? 144 : 210;
    for(size_t i=0;i<bytes;++i) p[i]=uint8_t(rng());
    for(size_t i=0;i<bytes;i+=block) {
        const uint16_t scale=0x1800,minimum=0x1400;
        std::memcpy(p+i+(type==12 ? 0 : 208),&scale,2);
        if(type==12) std::memcpy(p+i+2,&minimum,2);
    }
}
void check(int tokens,int down_type,cudaStream_t stream,const std::vector<uint8_t>& raw,Host<float>& cpu_parts) {
    constexpr int H=5120,F=2304,Top=6,Banks=3;
    const int entries=Top*tokens;
    const auto layout=K::native_expert_layout(12,down_type,H,F);
    Device blobs(raw.size()),input(size_t(tokens)*H*4),quant(K::native_q8_1_bytes(H,tokens));
    Device scratch(K::native_expert_scratch_bytes(entries,F)),parts(size_t(entries)*H*4),old_parts(size_t(entries)*H*4);
    Device route_weights(size_t(entries)*4),shared(size_t(tokens)*H*4),actual(size_t(tokens)*H*4),expected(size_t(tokens)*H*4);
    Device packet(sizeof(D::FfnPlan)),pointers(Top*8),starts((Top+1)*4),destinations(size_t(entries)*4),source_tokens(size_t(entries)*4),groups(4);
    Host<float> output(size_t(tokens)*H);
    Host<D::FfnPlan> plan(1); *plan.p=D::FfnPlan{};
    float* mapped=nullptr;ck(cudaHostGetDevicePointer(&mapped,cpu_parts.p,0));
    ck(cudaMemcpy(blobs.p,raw.data(),raw.size(),cudaMemcpyHostToDevice));
    std::vector<float> x(size_t(tokens)*H),w(entries),sh(size_t(tokens)*H);
    for(size_t i=0;i<x.size();++i) {x[i]=std::sin(float(i)*0.11f);sh[i]=std::cos(float(i)*0.07f)*0.3f;}
    for(int i=0;i<entries;++i) w[size_t(i)]=float(i%6+1)*0.071f;
    ck(cudaMemcpy(input.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(route_weights.p,w.data(),w.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(shared.p,sh.data(),sh.size()*4,cudaMemcpyHostToDevice));
    K::native_quantize_q8_1(input.f(),quant.p,H,tokens,stream);ck(cudaStreamSynchronize(stream));
    auto* gpu=static_cast<D::FfnPlan*>(packet.p);
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    ck(cudaMemcpyAsync(packet.p,plan.p,sizeof(D::FfnPlan),cudaMemcpyHostToDevice,stream));
    K::native_expert_grouped_clamp(layout,gpu->pointers,gpu->starts,&gpu->groups,gpu->destinations,gpu->tokens,
        entries,entries,quant.p,scratch.p,parts.f(),stream,1.75f);
    C::combine_sources(parts.f(),mapped,&gpu->gpu_mask,route_weights.f(),shared.f(),actual.f(),H,Top,tokens,stream);
    ck(cudaMemcpyAsync(output.p,actual.p,output.n*4,cudaMemcpyDeviceToHost,stream));
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int replay=0;replay<12;++replay) {
        auto& p=*plan.p;p.groups=0;p.starts[0]=0;p.gpu_mask=0;int ne=0;
        for(int expert=0;expert<Top;++expert) {
            const int begin=ne;
            for(int t=0;t<tokens;++t) {
                const int dst=entries-1-(expert*tokens+t);
                const int pattern=replay%6;
                const bool cached=pattern==1 || (pattern==2 && dst%2==0) || (pattern==3 && dst%3==1) ||
                    (pattern==4 && dst==0) || (pattern==5 && dst==entries-1);
                if(cached) {p.gpu_mask|=uint64_t(1)<<dst;p.destinations[ne]=dst;p.tokens[ne]=(t+expert)%tokens;++ne;}
            }
            if(ne>begin) {p.pointers[p.groups]=reinterpret_cast<unsigned long long>(static_cast<uint8_t*>(blobs.p)+size_t(expert%Banks)*layout.bytes);p.starts[++p.groups]=ne;}
        }
        // Fill mapped CPU results after all preceding CUDA calls. The actual
        // graph is launched before the independent reference uploads anything.
        for(size_t i=0;i<cpu_parts.n;++i) {
            float v=std::sin(float(i)*0.137f+float(replay)*0.2f);
            if(i%17==0) v=-0.0f;
            if(i%23==0) v=std::ldexp(float(int(i%7)-3),-140);
            cpu_parts.p[i]=v;
        }
        std::atomic_thread_fence(std::memory_order_release);
        ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(pointers.p,p.pointers,size_t(p.groups)*8,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(starts.p,p.starts,size_t(p.groups+1)*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(destinations.p,p.destinations,size_t(ne)*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(source_tokens.p,p.tokens,size_t(ne)*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(groups.p,&p.groups,4,cudaMemcpyHostToDevice));
        // Ordinary full-device reference retains the old descriptor arrays,
        // CPU-row uploads, grouped kernel and original route-order combine.
        for(int row=0;row<entries;++row) if(!(p.gpu_mask & (uint64_t(1)<<row)))
            ck(cudaMemcpyAsync(old_parts.f()+size_t(row)*H,cpu_parts.p+size_t(row)*H,size_t(H)*4,cudaMemcpyHostToDevice,stream));
        K::native_expert_grouped_clamp(layout,static_cast<unsigned long long*>(pointers.p),static_cast<int32_t*>(starts.p),
            static_cast<int32_t*>(groups.p),static_cast<int32_t*>(destinations.p),static_cast<int32_t*>(source_tokens.p),
            entries,entries,quant.p,scratch.p,old_parts.f(),stream,1.75f);
        C::combine(old_parts.f(),route_weights.f(),shared.f(),expected.f(),H,Top,tokens,stream);
        ck(cudaStreamSynchronize(stream));std::vector<float> reference(output.n);
        ck(cudaMemcpy(reference.data(),expected.p,reference.size()*4,cudaMemcpyDeviceToHost));
        require(std::all_of(reference.begin(),reference.end(),[](float v){return std::isfinite(v);}),"nonfinite reference");
        require(std::memcmp(reference.data(),output.p,reference.size()*4)==0,"packed descriptors/mapped CPU outputs changed FFN bits");
    }
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
}
int main() {
    try {
        int devices=0;ck(cudaGetDeviceCount(&devices));require(devices>=2,"TP2 test requires both GPUs");
        ck(cudaSetDevice(0));Host<float> cpu_parts(size_t(6)*8*5120,cudaHostAllocMapped|cudaHostAllocPortable);
        for(int type:{12,14}) {
            constexpr int H=5120,F=2304;const auto layout=K::native_expert_layout(12,type,H,F);
            std::vector<uint8_t> raw(size_t(layout.bytes)*3);std::mt19937 rng(900+type);
            for(int b=0;b<3;++b) {
                auto* p=raw.data()+size_t(b)*layout.bytes;
                finite_rows(p,layout.up_off,12,rng);finite_rows(p+layout.up_off,layout.down_off-layout.up_off,12,rng);
                finite_rows(p+layout.down_off,layout.bytes-layout.down_off,type,rng);
            }
            for(int d=0;d<2;++d) {ck(cudaSetDevice(d));cudaStream_t s=nullptr;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
                for(int t:{1,2,5,8}) check(t,type,s,raw,cpu_parts);
                ck(cudaStreamDestroy(s));std::cout<<"device="<<d<<" down_type="<<type<<" original 5120/2304 FFN, NT1/2/5/8, changing/all/no/mixed GPU routes and CPU publication bitwise PASS\n";
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
