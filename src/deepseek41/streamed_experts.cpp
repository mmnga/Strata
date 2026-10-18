#include "strata/deepseek41/streamed_experts.hpp"
#include "strata/deepseek41/resident.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
namespace K=strata::kernels;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p=nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Device() { cudaFree(p); }
};
struct Plan {
    unsigned long long pointer=0;
    int32_t start[2]={0,0}, count=1, destination[8]={}, token[8]={};
};
}
struct StreamedExperts::Impl {
    struct Gpu {
        int device, used=0, peak=0, hidden, ff;
        cudaStream_t stream=nullptr;
        std::unique_ptr<Device> weights,input,quant,plan,scratch,output;
        Plan* host_plan=nullptr;
        float* result=nullptr;
        std::array<ExpertBatch,3> jobs;
        Gpu(int d,int h,int f,size_t bytes,int cap) : device(d),hidden(h),ff(f) {
            ck(cudaSetDevice(d)); ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            weights=std::make_unique<Device>(bytes);
            input=std::make_unique<Device>(size_t(h)*8*4);
            quant=std::make_unique<Device>(K::native_q8_1_bytes(h,8));
            plan=std::make_unique<Device>(sizeof(Plan));
            scratch=std::make_unique<Device>(K::native_expert_scratch_bytes(8,f));
            output=std::make_unique<Device>(size_t(h)*8*4);
            ck(cudaMallocHost(&host_plan,sizeof(Plan)*cap));
            ck(cudaMallocHost(&result,size_t(cap)*8*h*4));
        }
        ~Gpu() {
            cudaSetDevice(device); cudaStreamSynchronize(stream);
            weights.reset(); input.reset(); quant.reset(); plan.reset(); scratch.reset(); output.reset();
            cudaFreeHost(host_plan); cudaFreeHost(result); cudaStreamDestroy(stream);
        }
    };
    std::vector<StreamLayerWeights> layers;
    std::vector<std::unique_ptr<Gpu>> gpus;
    int capacity;
    bool active=false;
    uint64_t calls=0, selected=0, routes=0, bytes=0;
    Impl(std::vector<StreamLayerWeights> weights,int devices,int cap) : layers(std::move(weights)),capacity(cap) {
        if(layers.empty() || devices!=2 || cap<1 || cap>3) throw std::invalid_argument("native streamed experts need TP2 and 1..3 jobs/GPU");
        const int h=layers.front().base.hidden, f=layers.front().base.ff;
        size_t max_bytes=0;
        for(const auto& layer:layers) {
            const auto& w=layer.base;
            if(w.hidden!=h || w.ff!=f || layer.experts<1 || !w.gate || !w.up || !w.down ||
               !K::native_expert_supported(w.gate_type,w.down_type,h,f)) throw std::invalid_argument("unsupported streamed expert layout");
            const auto layout=K::native_expert_layout(w.gate_type,w.down_type,h,f);
            if(layout.up_off!=w.gate_row*size_t(f) || layout.down_off!=2*layout.up_off ||
               layout.bytes!=layout.down_off+w.down_row*size_t(h)) throw std::invalid_argument("streamed expert raw offsets differ");
            max_bytes=std::max(max_bytes,size_t(layout.bytes));
        }
        for(int d=0;d<devices;++d) gpus.push_back(std::make_unique<Gpu>(d,h,f,max_bytes,cap));
        std::fprintf(stderr,"strata-deepseek41: native cold expert streams TP2 jobs_per_gpu=%d raw_scratch_MiB=%.3f, original cost planner\n",cap,double(max_bytes)/(1<<20));
    }
    void launch(int layer,int device,const ExpertBatch& job,const float* x,int tokens) {
        const auto& source=layers.at(size_t(layer)); const auto& base=source.base;
        auto& g=*gpus.at(size_t(device)); ck(cudaSetDevice(device));
        const int slot=g.used++; g.peak=std::max(g.peak,g.used); g.jobs[size_t(slot)]=job;
        const size_t gu=base.gate_row*size_t(base.ff), down=base.down_row*size_t(base.hidden);
        const uintptr_t offset=reinterpret_cast<uintptr_t>(job.weights.gate)-reinterpret_cast<uintptr_t>(base.gate);
        if(offset%gu || offset/gu>=size_t(source.experts) || job.weights.up!=base.up+offset ||
           job.weights.down!=base.down+(offset/gu)*down || job.weights.gate_type!=base.gate_type ||
           job.weights.down_type!=base.down_type || job.weights.limit!=base.limit || job.count<1 || job.count>8)
            throw std::runtime_error("streamed expert does not match its registered original layer");
        if(slot==0) {
            ck(cudaMemcpyAsync(g.input->p,x,size_t(tokens)*base.hidden*4,cudaMemcpyHostToDevice,g.stream));
            K::native_quantize_q8_1(static_cast<const float*>(g.input->p),g.quant->p,base.hidden,tokens,g.stream);
        }
        auto* dst=static_cast<uint8_t*>(g.weights->p);
        copy_registered_slice(dst,base.gate,offset,gu,g.stream);
        copy_registered_slice(dst+gu,base.up,offset,gu,g.stream);
        copy_registered_slice(dst+2*gu,base.down,(offset/gu)*down,down,g.stream);
        auto& p=g.host_plan[slot]; p=Plan{}; p.pointer=reinterpret_cast<unsigned long long>(dst); p.start[1]=job.count;
        for(int t=0;t<job.count;++t) {
            if(job.tokens[t]<0 || job.tokens[t]>=tokens || job.destinations[t]<0) throw std::invalid_argument("invalid streamed route");
            p.destination[t]=t; p.token[t]=job.tokens[t];
        }
        ck(cudaMemcpyAsync(g.plan->p,&p,sizeof(p),cudaMemcpyHostToDevice,g.stream));
        auto* plan=static_cast<Plan*>(g.plan->p);
        const auto layout=K::native_expert_layout(base.gate_type,base.down_type,base.hidden,base.ff);
        K::native_expert_grouped_clamp(layout,&plan->pointer,plan->start,&plan->count,plan->destination,plan->token,
            1,8,g.quant->p,g.scratch->p,static_cast<float*>(g.output->p),g.stream,base.limit,1);
        ck(cudaMemcpyAsync(g.result+size_t(slot)*8*base.hidden,g.output->p,size_t(job.count)*base.hidden*4,cudaMemcpyDeviceToHost,g.stream));
        ++selected; routes+=job.count; bytes+=2*gu+down;
    }
};
StreamedExperts::StreamedExperts(std::vector<StreamLayerWeights> layers,int devices,int cap) : p_(std::make_unique<Impl>(std::move(layers),devices,cap)) {}
StreamedExperts::~StreamedExperts() = default;
std::vector<ExpertBatch> StreamedExperts::submit(int layer,int owner,const std::vector<ExpertBatch>& jobs,const float* x,int tokens) {
    auto& p=*p_; if(p.active || owner<0 || owner>=2 || tokens<1 || tokens>8) throw std::logic_error("invalid native streamed epoch");
    for(auto& gpu:p.gpus) gpu->used=0;
    std::vector<int> order(jobs.size()); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return jobs[size_t(a)].count>jobs[size_t(b)].count;});
    const auto& base=p.layers.at(size_t(layer)).base;
    const double bytes=double(2*base.gate_row*size_t(base.ff)+base.down_row*size_t(base.hidden));
    const auto cost=[&](const ExpertBatch& job){return bytes/185000.0*(5+3*(job.count-1))/5;};
    double cpu=0; for(const auto& job:jobs) cpu+=cost(job);
    std::array<double,2> gpu{}; double remaining=cpu, submit=0, best=cpu*0.9; int chosen=0;
    // Copy the already-adopted streamed-miss planner, including its ten-percent
    // bound, alternating devices, three-job cap and at least one CPU job.
    for(int k=0;k<2*p.capacity && k+1<int(order.size());++k) {
        const auto& job=jobs[size_t(order[size_t(k)])]; const int device=(owner+1+k)%2;
        remaining-=cost(job); gpu[size_t(device)]+=bytes/50000.0+20+5*job.count; submit+=35+2*job.count;
        const double elapsed=submit+std::max(remaining,std::max(gpu[0],gpu[1]));
        if(elapsed<best) {best=elapsed; chosen=k+1;}
    }
    std::vector<bool> selected(jobs.size()); p.active=true; ++p.calls;
    for(int k=0;k<chosen;++k) {
        const int j=order[size_t(k)]; selected[size_t(j)]=true;
        p.launch(layer,(owner+1+k)%2,jobs[size_t(j)],x,tokens);
    }
    std::vector<ExpertBatch> cpu_jobs; cpu_jobs.reserve(jobs.size()-size_t(chosen));
    for(size_t j=0;j<jobs.size();++j) if(!selected[j]) cpu_jobs.push_back(jobs[j]);
    return cpu_jobs;
}
void StreamedExperts::finish(float* parts) {
    auto& p=*p_; if(!p.active) throw std::logic_error("no native streamed epoch");
    for(auto& gpu:p.gpus) {
        auto& g=*gpu; if(!g.used) continue; ck(cudaSetDevice(g.device)); ck(cudaStreamSynchronize(g.stream));
        for(int slot=0;slot<g.used;++slot) {
            const auto& job=g.jobs[size_t(slot)];
            for(int t=0;t<job.count;++t) std::memcpy(parts+size_t(job.destinations[t])*g.hidden,
                g.result+(size_t(slot)*8+t)*g.hidden,size_t(g.hidden)*4);
        }
    }
    p.active=false;
}
void StreamedExperts::report() const {
    const auto& p=*p_;
    std::fprintf(stderr,"strata-deepseek41: native cold expert streams calls=%llu experts=%llu routes=%llu raw_GiB=%.3f peak=%d/%d\n",
        (unsigned long long)p.calls,(unsigned long long)p.selected,(unsigned long long)p.routes,double(p.bytes)/(1ull<<30),p.gpus[0]->peak,p.gpus[1]->peak);
}
} // namespace strata::deepseek41
