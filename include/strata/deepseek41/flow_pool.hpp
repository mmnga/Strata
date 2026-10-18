#pragma once
#include "strata/deepseek41/hardware_epoch_wait.hpp"

// Isolated Linux DeepSeek pool. Like Strata's ExpertPool, workers persist,
// pin once, spin briefly, then sleep, and the calling thread computes too.
// Each worker enters once per published epoch; completion is published only
// after its callback returns, so no old callback can touch a new batch.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

namespace strata::deepseek41 {

class FlowPool {
public:
    explicit FlowPool(std::vector<int> cpus, int spin_us = 20000)
        : cpus_(std::move(cpus)), spin_us_(spin_us), mwaitx_(native_mwaitx_enabled()) {
        if (cpus_.empty() || cpus_.size() > 48 || spin_us < 0)
            throw std::invalid_argument("invalid DeepSeek flow pool size/spin");
        cpu_set_t seen;
        CPU_ZERO(&seen);
        for (int cpu : cpus_) {
            if (cpu < 0 || cpu >= CPU_SETSIZE || CPU_ISSET(cpu, &seen))
                throw std::invalid_argument("invalid or duplicate DeepSeek flow pool CPU");
            CPU_SET(cpu, &seen);
        }
        try {
            for (int ith = 1; ith < size(); ++ith)
                threads_.emplace_back([this, ith] { worker(ith); });
        } catch (...) {
            close();
            throw;
        }
        while (initialized_.load(std::memory_order_acquire) != size() - 1) _mm_pause();
        if (initialization_error_.load()) {
            close();
            throw std::runtime_error("cannot pin DeepSeek flow pool worker");
        }
        if(mwaitx_)std::fprintf(stderr,"strata CPU pool MWAITX enabled: workers=%d bounded_timer_ticks=50000; original CV deadline/caller\n",size()-1);
    }
    ~FlowPool() { close(); }
    FlowPool(const FlowPool &) = delete;
    FlowPool & operator=(const FlowPool &) = delete;

    int size() const { return int(cpus_.size()); }

    // The single caller owns every phase. Callbacks must not throw or call
    // run recursively. The caller participates as worker0, for N total math
    // threads, rather than N workers plus a spinning host.
    void run(std::function<void(int)> task, std::function<void()> after_publish = {}) {
        if (!caller_pinned_) {
            if (pin(cpus_[0])) throw std::runtime_error("cannot pin DeepSeek flow pool caller");
            caller_pinned_ = true;
        }
        completed_.store(0, std::memory_order_relaxed);
        task_ = std::move(task);
        {
            // Pair with the sleeping worker's predicate/wait to prevent a
            // notification from falling between its test and sleep.
            std::lock_guard<std::mutex> lock(mutex_);
            epoch_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        std::exception_ptr publish_error;
        if (after_publish) {
            try { after_publish(); } catch (...) { publish_error = std::current_exception(); }
        }
        try { task_(0); } catch (...) { std::terminate(); }
        while (completed_.load(std::memory_order_acquire) != size() - 1) _mm_pause();
        // Every worker has returned from task_ before it increments completed.
        // It cannot enter again until it acquires a newer epoch.
        task_ = {};
        if (publish_error) std::rethrow_exception(publish_error);
    }

private:
    static int pin(int cpu) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    }

    void worker(int ith) {
        const int error = pin(cpus_[ith]);
        if (error) initialization_error_.store(error);
        initialized_.fetch_add(1, std::memory_order_release);
        if (error) return;
        uint64_t seen = 0;
        for (;;) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us_);
            while (!stop_.load(std::memory_order_acquire) && epoch_.load(std::memory_order_acquire) == seen) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_.wait(lock, [&] {
                        return stop_.load(std::memory_order_acquire) || epoch_.load(std::memory_order_acquire) != seen;
                    });
                    break;
                }
                if(mwaitx_)hardware_epoch_wait(epoch_,stop_,seen);
                else _mm_pause();
            }
            if (stop_.load(std::memory_order_acquire)) return;
            seen = epoch_.load(std::memory_order_acquire);
            try { task_(ith); } catch (...) { std::terminate(); }
            completed_.fetch_add(1, std::memory_order_release);
        }
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_.store(true, std::memory_order_release);
            if(mwaitx_)epoch_.fetch_add(1,std::memory_order_release);
        }
        wake_.notify_all();
        for (auto & thread : threads_) if (thread.joinable()) thread.join();
        threads_.clear();
    }

    std::vector<int> cpus_;
    int spin_us_;
    bool mwaitx_;
    bool caller_pinned_ = false;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void(int)> task_;
    alignas(64) std::atomic<uint64_t> epoch_{0};
    alignas(64) std::atomic<int> completed_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::atomic<int> initialized_{0}, initialization_error_{0};
};

} // namespace strata::deepseek41
