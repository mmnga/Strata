#pragma once
#include <cstdint>
#include <type_traits>

namespace strata::deepseek41 {
// One immutable-per-launch packet replaces five small descriptor transfers.
// Native grouped kernels read only groups/starts and the active entries.
struct FfnPlan {
    static constexpr int Capacity=64;
    unsigned long long pointers[Capacity];
    int32_t starts[Capacity+1], destinations[Capacity], tokens[Capacity];
    int32_t groups;
    uint64_t gpu_mask;
};
static_assert(std::is_standard_layout_v<FfnPlan> && std::is_trivially_copyable_v<FfnPlan>);
} // namespace strata::deepseek41
