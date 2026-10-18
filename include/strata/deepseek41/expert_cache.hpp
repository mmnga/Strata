#pragma once
#include "strata/deepseek41/model.hpp"
#include "strata/deepseek41/resident.hpp"
#include <memory>
#include <vector>

namespace strata::deepseek41 {
// Reuses Strata's native ExpertCache slot storage. Copies always preserve the
// original three GGUF slices. Pending fills are invisible, and readers lease
// slots until their compute stream has completed.
class ExpertCache {
public:
    ExpertCache(const Model&,const ResidentWeights&,int devices,const std::vector<int>& mib,int draft_reserve_mib = 7680);
    ~ExpertCache();
    std::vector<const uint8_t*> acquire(int layer,const int32_t* ids,int count);
    void release(int layer,const int32_t* ids,int count);
    void observe(int layer,const int32_t* ids,int count);
    void hint(int layer,const std::vector<int32_t>& ids);
    void report() const;
    size_t gate_bytes(int layer) const;
    size_t down_bytes(int layer) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace strata::deepseek41
