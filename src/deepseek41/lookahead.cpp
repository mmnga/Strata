#include "strata/deepseek41/lookahead.hpp"
#include "strata/deepseek41/cpu_kquant.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

namespace strata::deepseek41 {
struct Lookahead::Impl {
    ExpertCache& cache;
    int hidden, experts, layers, depth, top_k;
    uint16_t* routers = nullptr;
    float* bias = nullptr;
    std::vector<float> input;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    bool quit = false, busy = false, pending = false;
    int layer = 0;
    std::atomic<uint64_t> submitted{0}, skipped{0}, predictions{0};
    std::string error;
    Impl(const Model& m,ExpertCache& c,int d,int k) : cache(c),hidden(m.config().hidden),experts(m.config().experts),layers(m.config().layers),depth(d),top_k(k),input(hidden) {
        if (d < 1 || d > 2 || k < 1 || k > experts) throw std::invalid_argument("native gate lookahead depth/top-k");
        const auto checked = [](cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); };
        checked(cudaMallocHost(&routers,size_t(layers)*experts*hidden*2));
        checked(cudaMallocHost(&bias,size_t(layers)*experts*4));
        for (int l = 0; l < layers; ++l) {
            const auto p = "blk."+std::to_string(l)+".";
            const auto r = m.tensor(p+"ffn_gate_inp.weight"), b = m.tensor(p+"exp_probs_b.bias");
            if (r.tensor->type != 30 || b.tensor->type != 0) throw std::runtime_error("lookahead requires original BF16 router/F32 selection bias");
            std::memcpy(routers+size_t(l)*experts*hidden,r.data(),r.bytes()); std::memcpy(bias+size_t(l)*experts,b.data(),b.bytes());
        }
        worker = std::thread([this] {
            const auto cores = strata::kernels::cpu::physical_cores(false);
            const char* requested = std::getenv("STRATA_DS41_ROUTER_CORE");
            const int core = requested ? std::atoi(requested) : (cores.empty() ? -1 : cores.back());
            if (core >= 0) strata::kernels::cpu::pin_current_thread(core);
            std::vector<float> logits(experts), scores(experts), local(hidden);
            std::vector<int32_t> order(experts);
            for (;;) {
                int source;
                {
                    std::unique_lock<std::mutex> lock(mutex); cv.wait(lock,[&] { return quit || pending; });
                    if (quit) break;
                    source = layer; local = input; pending = false; busy = true;
                }
                try {
                    for (int l = source+1; l < std::min(layers,source+1+depth); ++l) {
                        cpu::bf16_rows_dot(routers+size_t(l)*experts*hidden,experts,hidden,local.data(),logits.data());
                        for (int e = 0; e < experts; ++e) { const float v = logits[e]; scores[e] = std::sqrt(std::max(v,0.0f)+std::log1p(std::exp(-std::abs(v))))+bias[size_t(l)*experts+e]; }
                        std::iota(order.begin(),order.end(),0);
                        std::partial_sort(order.begin(),order.begin()+top_k,order.end(),[&](int a,int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
                        cache.hint(l,std::vector<int32_t>(order.begin(),order.begin()+top_k)); ++predictions;
                    }
                } catch (const std::exception& e) { std::lock_guard<std::mutex> lock(mutex); error = e.what(); }
                { std::lock_guard<std::mutex> lock(mutex); busy = false; }
            }
        });
        std::fprintf(stderr,"strata-deepseek41: native next-layer gate lookahead depth=%d top_k=%d, pinned original BF16 routers\n",depth,top_k);
    }
    ~Impl() {
        { std::lock_guard<std::mutex> lock(mutex); quit = true; } cv.notify_one(); if (worker.joinable()) worker.join();
        cudaFreeHost(routers); cudaFreeHost(bias);
    }
};
Lookahead::Lookahead(const Model& m,ExpertCache& c,int d,int k) : p_(std::make_unique<Impl>(m,c,d,k)) {}
Lookahead::~Lookahead() = default;
void Lookahead::submit(int layer,const float* x) {
    std::lock_guard<std::mutex> lock(p_->mutex);
    if (!p_->error.empty()) throw std::runtime_error("native lookahead failed: "+p_->error);
    if (p_->pending || p_->busy || layer+1 >= p_->layers) { ++p_->skipped; return; }
    std::copy(x,x+p_->hidden,p_->input.begin()); p_->layer = layer; p_->pending = true; ++p_->submitted; p_->cv.notify_one();
}
void Lookahead::report() const {
    std::fprintf(stderr,"strata-deepseek41: native gate lookahead depth=%d submitted=%llu skipped=%llu predicted_layers=%llu\n",p_->depth,(unsigned long long)p_->submitted.load(),(unsigned long long)p_->skipped.load(),(unsigned long long)p_->predictions.load());
}
}  // namespace strata::deepseek41
