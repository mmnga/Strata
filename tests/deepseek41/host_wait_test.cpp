#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace C = strata::deepseek41::cuda;
using Clock = std::chrono::steady_clock;
void ck(cudaError_t error) { if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t count) { ck(cudaMalloc(&p,count*sizeof(T))); }
    ~Device() { cudaFree(p); }
};
template<class T> struct Host {
    T* p=nullptr;T* device=nullptr;
    explicit Host(size_t count) {
        ck(cudaHostAlloc(&p,count*sizeof(T),cudaHostAllocMapped|cudaHostAllocPortable));
        ck(cudaHostGetDevicePointer(&device,p,0));
        std::memset(p,0,count*sizeof(T));
    }
    ~Host() { cudaFreeHost(p); }
};
struct Join {
    std::thread& thread;
    ~Join() { if(thread.joinable()) thread.join(); }
};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    template<class F> Graph(cudaStream_t stream,F body) {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));body();
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph() { cudaGraphExecDestroy(exec);cudaGraphDestroy(graph); }
};

void check(int device,int tokens,cudaStream_t stream) {
    constexpr int H=5120,K=6;
    const int entries=tokens*K;const size_t parts=size_t(entries)*H,outputs=size_t(tokens)*H;
    Host<float> cpu(parts),actual(outputs);Host<uint32_t> words(3);
    Device<float> source(parts),cached(parts),weights(entries),shared(outputs),out(outputs),reference(outputs);
    Device<uint64_t> mask(1);
    std::vector<float> values(parts),w(entries),s(outputs),expected(outputs);
    for(size_t i=0;i<parts;++i) values[i]=float(int(i%53)-26)*0.03125f;
    for(int i=0;i<entries;++i) w[i]=float(i%K+1)*0.03125f;
    for(size_t i=0;i<outputs;++i) s[i]=float(int(i%17)-8)*0.015625f;
    ck(cudaMemcpyAsync(source.p,values.data(),parts*4,cudaMemcpyHostToDevice,stream));
    ck(cudaMemcpyAsync(weights.p,w.data(),entries*4,cudaMemcpyHostToDevice,stream));
    ck(cudaMemcpyAsync(shared.p,s.data(),outputs*4,cudaMemcpyHostToDevice,stream));
    ck(cudaStreamSynchronize(stream));
    Graph graph(stream,[&] {
        // An actual GPU kernel precedes the Host wait, as cached expert work
        // does in the engine. Everything after it was queued before CPU writes.
        C::strided_copy(source.p,cached.p,H,entries,H,H,stream);
        C::wait_host(words.device,words.device+1,words.device+2,stream);
        C::combine_sources_system(cached.p,cpu.device,mask.p,weights.p,shared.p,out.p,H,K,tokens,stream);
        ck(cudaMemcpyAsync(actual.p,out.p,outputs*4,cudaMemcpyDeviceToHost,stream));
    });
    const uint64_t all=(uint64_t(1)<<entries)-1;
    const uint32_t sequences[]={1,2,17,UINT32_MAX,0,1};
    int iteration=0;
    for(uint64_t bits:{uint64_t(0),all,all&uint64_t(0xaaaaaaaaaaaaaaaaULL)}) {
        ck(cudaMemcpyAsync(mask.p,&bits,8,cudaMemcpyHostToDevice,stream));
        ck(cudaStreamSynchronize(stream));
        words.p[0]=0;
        for(uint32_t want:sequences) {
            words.p[1]=want;words.p[2]=0;
            std::atomic<bool> launched{false},timeout{false};
            std::thread producer([&] {
                const auto deadline=Clock::now()+std::chrono::seconds(2);
                while(!launched.load(std::memory_order_acquire)) {
                    if(Clock::now()>=deadline) {
                        timeout.store(true);words.p[2]=1;
                        std::atomic_thread_fence(std::memory_order_seq_cst);return;
                    }
                    std::this_thread::yield();
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                for(size_t i=0;i<parts;++i) cpu.p[i]=float(int(i%47)-23)*float(iteration+1)*0.015625f;
                cpu.p[0]=-0.0f;cpu.p[1]=0.0f;
                cpu.p[2]=std::numeric_limits<float>::denorm_min();cpu.p[3]=-cpu.p[2];
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *static_cast<volatile uint32_t*>(words.p)=want;
            });
            {
                Join join{producer};
                ck(cudaGraphLaunch(graph.exec,stream));
                launched.store(true,std::memory_order_release);
                ck(cudaStreamSynchronize(stream));
            }
            require(!timeout.load(),"graph launch blocked before Host could publish completion");
            C::combine_sources(cached.p,cpu.device,mask.p,weights.p,shared.p,reference.p,H,K,tokens,stream);
            ck(cudaMemcpyAsync(expected.data(),reference.p,outputs*4,cudaMemcpyDeviceToHost,stream));
            ck(cudaStreamSynchronize(stream));
            require(!std::memcmp(actual.p,expected.data(),outputs*4),"queued mapped combine differs from original completed-Host combine");
            ++iteration;
        }
    }
    // Missing CPU completion must be releasable by an exception/shutdown abort.
    words.p[0]=1;words.p[1]=42;words.p[2]=0;
    std::atomic<bool> launched{false},timeout{false};
    std::thread aborter([&] {
        const auto deadline=Clock::now()+std::chrono::seconds(2);
        while(!launched.load(std::memory_order_acquire)) {
            if(Clock::now()>=deadline) {timeout.store(true);break;}
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        *static_cast<volatile uint32_t*>(words.p+2)=1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    });
    {
        Join join{aborter};ck(cudaGraphLaunch(graph.exec,stream));
        launched.store(true,std::memory_order_release);ck(cudaStreamSynchronize(stream));
    }
    require(!timeout.load(),"abort graph launch blocked its Host publisher");
    std::cout<<"device="<<device<<" NT="<<tokens<<" masks=none/all/mixed sequences=1/2/17/MAX/0/1 exact=PASS delayed Host+abort=PASS\n";
}
int main() {
    try {
        int count=0;ck(cudaGetDeviceCount(&count));require(count>=2,"Host wait test requires TP2 GPUs");
        for(int device=0;device<2;++device) {
            ck(cudaSetDevice(device));cudaStream_t stream=nullptr;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            for(int tokens:{1,2,5,8}) check(device,tokens,stream);
            ck(cudaStreamDestroy(stream));
        }
        std::cout<<"Mapped Host wait + original-order combine graph PASS\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
