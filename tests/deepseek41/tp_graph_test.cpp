#include "strata/deepseek41/kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <string>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
namespace C=strata::deepseek41::cuda;
namespace K=strata::kernels;
void ck(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Mapped {
    T* host=nullptr;
    explicit Mapped(size_t n) {ck(cudaHostAlloc(&host,n*sizeof(T),cudaHostAllocMapped|cudaHostAllocPortable));std::fill_n(host,n,T{});}
    T* device(int id) {ck(cudaSetDevice(id));T* p=nullptr;ck(cudaHostGetDevicePointer(&p,host,0));return p;}
    ~Mapped() {cudaFreeHost(host);}
};
struct Rank {
    int id;cudaStream_t stream=nullptr;float *input=nullptr,*output=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    explicit Rank(int d,size_t n):id(d) {ck(cudaSetDevice(id));ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaMalloc(&input,n*4));ck(cudaMalloc(&output,n*4));}
    void clear_graph() {ck(cudaSetDevice(id));if(exec) ck(cudaGraphExecDestroy(exec));if(graph) ck(cudaGraphDestroy(graph));exec=nullptr;graph=nullptr;}
    ~Rank() {cudaSetDevice(id);if(exec) cudaGraphExecDestroy(exec);if(graph) cudaGraphDestroy(graph);cudaFree(input);cudaFree(output);cudaStreamDestroy(stream);}
};
int main() {
    try {
        int count=0;ck(cudaGetDeviceCount(&count));if(count<2) throw std::runtime_error("native TP test requires both GPUs");
        constexpr int H=5120,Max=H*8;
        Mapped<float> parts(2*Max),result(2*Max);Mapped<uint32_t> ready(18),abort(1);
        std::array<Rank,2> ranks={Rank(0,Max),Rank(1,Max)};
        std::array<std::vector<float>,2> values={std::vector<float>(Max),std::vector<float>(Max)};
        std::mt19937 rng(5120);uint64_t compared=0;
        for(int tokens=1;tokens<=8;++tokens) {
            const int n=H*tokens;
            for(auto& r:ranks) {
                ck(cudaSetDevice(r.id));r.clear_graph();
                auto* p=parts.device(r.id);auto* seq=ready.device(r.id);auto* stop=abort.device(r.id);
                ck(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
                K::doorbell_publish(r.input,nullptr,nullptr,n,0,p+r.id*Max,nullptr,nullptr,seq+r.id*9,r.stream);
                C::tp_sum_wait(r.input,p+(1-r.id)*Max,seq+r.id*9,seq+(1-r.id)*9,stop,r.output,n,r.id,r.stream);
                ck(cudaMemcpyAsync(result.host+r.id*Max,r.output,n*4,cudaMemcpyDeviceToHost,r.stream));
                ck(cudaStreamEndCapture(r.stream,&r.graph));ck(cudaGraphInstantiate(&r.exec,r.graph,nullptr,nullptr,0));
            }
            for(int replay=0;replay<4;++replay) {
                for(int i=0;i<n;++i) for(int d=0;d<2;++d) {
                    uint32_t bits=(rng()&0x807fffffU)|((100+rng()%45)<<23);
                    if(i%31==0) bits=(rng()&0x80000000U);
                    if(i%31==1) bits=(rng()&0x807fffffU);
                    std::memcpy(&values[d][i],&bits,4);
                }
                for(auto& r:ranks) {
                    ck(cudaSetDevice(r.id));ck(cudaMemcpyAsync(r.input,values[r.id].data(),n*4,cudaMemcpyHostToDevice,r.stream));
                    ck(cudaGraphLaunch(r.exec,r.stream));
                }
                for(auto& r:ranks) {ck(cudaSetDevice(r.id));ck(cudaStreamSynchronize(r.stream));}
                for(int i=0;i<n;++i) {
                    float expected=0;expected+=values[0][i];expected+=values[1][i];
                    for(int d=0;d<2;++d) if(std::memcmp(&expected,result.host+d*Max+i,4))
                        throw std::runtime_error("native TP sum differs from ordered CPU sum at NT="+std::to_string(tokens)+" row="+std::to_string(i));
                }
                compared+=uint64_t(n)*2;
            }
        }
        // A missing peer must be releasable during an exception or shutdown.
        auto& r=ranks[0];cudaError_t status=cudaSuccess;const auto begin=std::chrono::steady_clock::now();
        std::thread launcher([&] {status=cudaSetDevice(0);if(status==cudaSuccess) status=cudaGraphLaunch(r.exec,r.stream);});
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        *static_cast<volatile uint32_t*>(abort.host)=1;std::atomic_thread_fence(std::memory_order_release);
        launcher.join();ck(status);ck(cudaSetDevice(0));ck(cudaStreamSynchronize(r.stream));
        if(std::chrono::steady_clock::now()-begin>std::chrono::seconds(2)) throw std::runtime_error("TP abort release exceeded deadline");
        std::cout<<"TP2 ordered CPU sum bitwise PASS, NT1..8, four changing graph replays, signed zero/subnormal, values="<<compared<<"; missing-peer abort PASS\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
