// Reused from the local Strata GLM Flash implementation; isolated DeepSeek dispatch.
// Process-local, opt-in weight prefetch; no arithmetic or memory-policy changes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace strata::deepseek41::cpu {

struct NativeWeightPrefetch {
    bool enabled = false;
    int rows = 1;
};

inline const NativeWeightPrefetch& native_weight_prefetch_setting() {
    static const NativeWeightPrefetch setting = [] {
        NativeWeightPrefetch result;
        const char* enabled = std::getenv("STRATA_NATIVE_WEIGHT_PREFETCH");
        if (enabled) {
            if (std::strcmp(enabled,"0") == 0) result.enabled = false;
            else if (std::strcmp(enabled,"1") == 0) result.enabled = true;
            else throw std::invalid_argument("STRATA_NATIVE_WEIGHT_PREFETCH must be 0 or 1");
        }
        const char* rows = std::getenv("STRATA_NATIVE_WEIGHT_PREFETCH_ROWS");
        if (rows) {
            if (rows[0] < '1' || rows[0] > '4' || rows[1] != '\0')
                throw std::invalid_argument("STRATA_NATIVE_WEIGHT_PREFETCH_ROWS must be 1 through 4");
            result.rows = rows[0]-'0';
        }
#if !defined(__GNUC__) && !defined(__clang__) && !(defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
        result.enabled = false;
#endif
        return result;
    }();
    return setting;
}

inline void native_prefetch_weight_address(const uint8_t* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address,0,2); // T1 on x86: weights arrive ahead of demand.
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_prefetch(reinterpret_cast<const char*>(address),_MM_HINT_T1);
#else
    (void)address;
#endif
}

// Every issued address lies inside the requested row and [first,last). The
// first hint uses the valid row start; subsequent hints use cache-line starts
// inside it, including the last partial line when the row is not aligned.
inline void native_prefetch_weight_row(const uint8_t* source, size_t row_bytes,
                                       int row, int first, int last) noexcept {
    if (row < first || row >= last || row < 0 || row_bytes == 0) return;
    const uint8_t* begin = source+size_t(row)*row_bytes;
    native_prefetch_weight_address(begin);
    const size_t next_line = 64-(reinterpret_cast<uintptr_t>(begin)&63);
    for (size_t offset=next_line; offset<row_bytes; offset+=64)
        native_prefetch_weight_address(begin+offset);
}

} // namespace strata::deepseek41::cpu
