#include "strata/deepseek41/cpu_experts.hpp"
#include "strata/deepseek41/cpu_kquant.hpp"
#include "strata/deepseek41/flow_pool.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <sstream>
#include <set>
#include <array>
#include <numeric>
#include <cstdlib>
#include <immintrin.h>

namespace strata::deepseek41 {
namespace cpu { void q8k_quant_avx2(const float*,void*,int64_t); }
namespace host = strata::kernels::cpu;

struct CpuExperts::Impl {
    int participants;
    int host_core = -1;
    std::vector<std::thread> workers;
    std::unique_ptr<FlowPool> flow;
    // Match Strata's ExpertPool: waiting workers must not invalidate the
    // task counter or completion line while another worker is computing.
    alignas(64) std::atomic<uint64_t> epoch{0};
    alignas(64) std::atomic<int> next{0};
    alignas(64) std::atomic<int> parked{0};
    alignas(64) std::atomic<bool> stop{false};
    int tasks = 0;
    std::function<void(int)> fn;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<uint8_t> activation, quantized;
    std::vector<float> intermediate;
    explicit Impl(int n) : participants(n) {
        if (n < 1 || n > 24) throw std::invalid_argument("DeepSeek CPU math threads must be 1..24");
        ggml_cpu_init();
        auto cores = host::physical_cores(true); host_core = host::planned_host_core();
        if (const char* setting = std::getenv("STRATA_DS41_CPU_CORES")) {
            std::string text(setting); for (char& c : text) if (c == ',') c = ' ';
            std::istringstream input(text); std::vector<int> selected; std::set<int> seen; int core;
            while (input >> core) { if (core < 0 || !seen.insert(core).second) throw std::invalid_argument("invalid native CPU affinity list"); selected.push_back(core); }
            if (!input.eof() || selected.size() < size_t(n)) throw std::invalid_argument("native CPU affinity list is shorter than the math thread count");
            host_core = selected[0]; cores.assign(selected.begin()+1,selected.begin()+n);
        }
        const char* mode = std::getenv("STRATA_DS41_CPU_FLOW");
        if (mode && std::atoi(mode) != 0) {
            if (host_core < 0 || (n > 1 && cores.empty())) throw std::runtime_error("native flow requires CPU affinity");
            std::vector<int> cpus = {host_core};
            for (int i = 0; i < n-1; ++i) cpus.push_back(cores[size_t(i)%cores.size()]);
            flow = std::make_unique<FlowPool>(std::move(cpus));
            std::fprintf(stderr,"strata-deepseek41: native CPU flow math_threads=%d one epoch, original 5+3*(NT-1) partitions, no ready-queue/submit overlap\n",n);
            return;
        }
        for (int i = 0; i < n-1; ++i) workers.emplace_back([this,cores,i] {
            if (!cores.empty()) host::pin_current_thread(cores[size_t(i)%cores.size()]);
            uint64_t seen = 0;
            uint32_t spins = 0;
            auto idle = std::chrono::steady_clock::now();
            while (!stop.load(std::memory_order_relaxed)) {
                const auto current = epoch.load(std::memory_order_acquire);
                if (current == seen) {
                    _mm_pause();
                    if ((++spins & 1023u) != 0 || std::chrono::steady_clock::now()-idle < std::chrono::milliseconds(20)) continue;
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock,[&] { return stop.load() || epoch.load(std::memory_order_acquire) != seen; });
                    continue;
                }
                seen = current; drain(); parked.fetch_add(1,std::memory_order_release);
                idle = std::chrono::steady_clock::now();
            }
        });
    }
    ~Impl() {
        flow.reset();
        { std::lock_guard<std::mutex> lock(mu); stop.store(true,std::memory_order_release); }
        cv.notify_all(); for (auto& t : workers) t.join();
    }
    void drain() { for (;;) { const int task = next.fetch_add(1,std::memory_order_relaxed); if (task >= tasks) return; fn(task); } }
    void parallel(int n, std::function<void(int)> f) {
        fn = std::move(f); tasks = n; next.store(0,std::memory_order_relaxed); parked.store(0,std::memory_order_relaxed);
        { std::lock_guard<std::mutex> lock(mu); epoch.fetch_add(1,std::memory_order_release); }
        cv.notify_all(); drain();
        const auto begin = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (parked.load(std::memory_order_acquire) != participants-1) {
            _mm_pause();
            if ((++spins & 1023u) == 0 && std::chrono::steady_clock::now()-begin > std::chrono::seconds(60)) throw std::runtime_error("DeepSeek CPU pool stalled");
        }
    }
    void run_flow(const std::vector<ExpertBatch>& jobs,int h,int f,size_t act_bytes,size_t hidden_bytes,float* out) {
        struct Task { int job, first, last; bool down; };
        struct alignas(64) State { std::atomic<int> left{0}; std::atomic<bool> ready{false}; int parts=0; };
        std::array<State,64> states;
        std::vector<int> order(jobs.size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return jobs[a].count > jobs[b].count; });
        const auto cost = [&](int j) { return 5+3*(jobs[j].count-1); };
        int total_cost = 0; for (int j : order) total_cost += cost(j);
        for (int j : order) {
            auto& state = states[size_t(j)];
            state.parts = std::max(1,(3*participants*cost(j)+total_cost-1)/total_cost);
            state.left.store(state.parts,std::memory_order_relaxed);
        }
        std::vector<Task> plan;
        // Preserve the adopted flow ordering: every GU task can be claimed
        // before any consumer waits for its own expert's hidden activation.
        for (bool down : {false,true}) for (int j : order) {
            const int rows = down ? h : f, parts = states[size_t(j)].parts;
            for (int part = 0; part < parts; ++part) plan.push_back({j,rows*part/parts,rows*(part+1)/parts,down});
        }
        alignas(64) std::atomic<int> head{0};
        flow->run([&](int) {
            for (;;) {
                const int index = head.fetch_add(1,std::memory_order_relaxed);
                if (index >= int(plan.size())) break;
                const auto& task = plan[size_t(index)]; const auto& job = jobs[size_t(task.job)]; const auto& w = job.weights;
                auto& state = states[size_t(task.job)]; const void* act[8]; float* result[8];
                for (int t = 0; t < job.count; ++t) {
                    if (task.down) {
                        act[t] = quantized.data()+(size_t(task.job)*8+t)*hidden_bytes;
                        result[t] = out+size_t(job.destinations[t])*h;
                    } else {
                        act[t] = activation.data()+size_t(job.tokens[t])*act_bytes;
                        result[t] = intermediate.data()+(size_t(task.job)*8+t)*f;
                    }
                }
                if (!task.down) {
                    cpu::kq512_gu_rows(w.gate_type,w.gate,w.gate_row,0,h,act,job.count,result,task.first,task.last,w.limit,w.up);
                    if (state.left.fetch_sub(1,std::memory_order_acq_rel) == 1) {
                        for (int t = 0; t < job.count; ++t) cpu::q8k_quant_avx2(
                            intermediate.data()+(size_t(task.job)*8+t)*f,quantized.data()+(size_t(task.job)*8+t)*hidden_bytes,f);
                        state.ready.store(true,std::memory_order_release);
                    }
                } else {
                    while (!state.ready.load(std::memory_order_acquire)) _mm_pause();
                    cpu::kq512_rows(w.down_type,w.down,w.down_row,f,act,job.count,result,task.first,task.last);
                }
            }
        });
    }
    void run_multi(const std::vector<ExpertBatch>& jobs,const float* x,int tokens,float* out) {
        if (jobs.empty()) return;
        const int h = jobs[0].weights.hidden, f = jobs[0].weights.ff, tile = 32;
        if (h < 256 || f < 256 || h%256 || f%256 || jobs.size() > 64 || tokens < 1 || tokens > 8) throw std::invalid_argument("DeepSeek CPU expert geometry");
        for (const auto& job : jobs) {
            const auto& w = job.weights;
            if (job.count < 1 || job.count > 8) throw std::invalid_argument("DeepSeek CPU group size");
            for (int t = 0; t < job.count; ++t) if (job.tokens[t] < 0 || job.tokens[t] >= tokens || job.destinations[t] < 0) throw std::invalid_argument("DeepSeek CPU group entry");
            if (w.hidden != h || w.ff != f || w.gate_type != 12 || (w.down_type != 12 && w.down_type != 14) ||
            !w.gate || !w.up || !w.down || w.gate_row != ggml_row_size(GGML_TYPE_Q4_K,h) ||
            w.down_row != ggml_row_size(ggml_type(w.down_type),f)) throw std::invalid_argument("DeepSeek CPU expert layout");
        }
        const size_t act_bytes = ggml_row_size(GGML_TYPE_Q8_K,h), hidden_bytes = ggml_row_size(GGML_TYPE_Q8_K,f);
        activation.resize(act_bytes*size_t(tokens)); quantized.resize(hidden_bytes*jobs.size()*8); intermediate.resize(size_t(f)*jobs.size()*8);
        for (int t = 0; t < tokens; ++t) cpu::q8k_quant_avx2(x+size_t(t)*h,activation.data()+size_t(t)*act_bytes,h);
        if (flow) { run_flow(jobs,h,f,act_bytes,hidden_bytes,out); return; }
        const int gu_tiles = (f+tile-1)/tile, down_tiles = (h+tile-1)/tile;
        parallel(int(jobs.size())*gu_tiles,[&](int task) {
            const int e = task/gu_tiles, first = task%gu_tiles*tile, last = std::min(first+tile,f);
            const auto& job = jobs[e]; const auto& w = job.weights;
            const void* act[8]; float* ff[8];
            for (int t = 0; t < job.count; ++t) { act[t] = activation.data()+size_t(job.tokens[t])*act_bytes; ff[t] = intermediate.data()+(size_t(e)*8+t)*f; }
            cpu::kq512_gu_rows(w.gate_type,w.gate,w.gate_row,0,h,act,job.count,ff,first,last,w.limit,w.up);
        });
        parallel(int(jobs.size()),[&](int e) {
            for (int t = 0; t < jobs[e].count; ++t) cpu::q8k_quant_avx2(intermediate.data()+(size_t(e)*8+t)*f,quantized.data()+(size_t(e)*8+t)*hidden_bytes,f);
        });
        parallel(int(jobs.size())*down_tiles,[&](int task) {
            const int e = task/down_tiles, first = task%down_tiles*tile, last = std::min(first+tile,h);
            const auto& job = jobs[e]; const auto& w = job.weights; const void* act[8]; float* result[8];
            for (int t = 0; t < job.count; ++t) { act[t] = quantized.data()+(size_t(e)*8+t)*hidden_bytes; result[t] = out+size_t(job.destinations[t])*h; }
            cpu::kq512_rows(w.down_type,w.down,w.down_row,f,act,job.count,result,first,last);
        });
    }
};
CpuExperts::CpuExperts(int threads) : p_(std::make_unique<Impl>(threads)) {}
CpuExperts::~CpuExperts() = default;
void CpuExperts::run(const std::vector<ExpertWeights>& jobs,const float* x,float* out) {
    std::vector<ExpertBatch> batches(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) { batches[i].weights = jobs[i]; batches[i].count = 1; batches[i].destinations[0] = int(i); }
    p_->run_multi(batches,x,1,out);
}
void CpuExperts::run_multi(const std::vector<ExpertBatch>& jobs,const float* x,int tokens,float* out) { p_->run_multi(jobs,x,tokens,out); }
int CpuExperts::threads() const { return p_->participants; }
int CpuExperts::host_core() const { return p_->host_core; }

}  // namespace strata::deepseek41
