#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace strata::deepseek41 {
inline bool native_mwaitx_supported() {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    unsigned a,b,c,d;
    return __get_cpuid(0x80000001,&a,&b,&c,&d) && (c&(1u<<29));
#else
    return false;
#endif
}
inline bool native_mwaitx_enabled() {
    static const bool enabled=[] {
        const char * text=std::getenv("STRATA_CPU_FLOW_MWAITX");
        if(!text || std::strcmp(text,"0")==0)return false;
        if(std::strcmp(text,"1"))throw std::invalid_argument("STRATA_CPU_FLOW_MWAITX must be0 or1");
        if(!native_mwaitx_supported())throw std::runtime_error("MWAITX not supported on this CPU");
        return true;
    }();
    return enabled;
}
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
__attribute__((target("mwaitx")))
inline void hardware_epoch_wait(const std::atomic<uint64_t> & epoch,const std::atomic<bool> & stop,uint64_t seen) {
    _mm_monitorx(&epoch,0,0);
    // Do not move the checked load before arming the cache-line monitor.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if(!stop.load(std::memory_order_acquire) && epoch.load(std::memory_order_acquire)==seen)
        _mm_mwaitx(2,0,50000); // GCC: ECX extensions, EAX hints, EBX timer ticks
}
#else
inline void hardware_epoch_wait(const std::atomic<uint64_t> &,const std::atomic<bool> &,uint64_t) {
    throw std::runtime_error("MWAITX unavailable in this build");
}
#endif
} // namespace strata::deepseek41
