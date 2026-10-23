#pragma once
#include "strata/deepseek41/model.hpp"
#include <memory>

namespace strata::deepseek41 {

// Owned tensors use matching 1 GiB (or opt-in 64 MiB) registrations and copy
// boundaries. CUDA host copies must not
// straddle two independent registrations, even when both ranges are pinned.
void copy_registered_slice(void* destination,const uint8_t* base,size_t offset,size_t bytes,void* stream);

// All CPU expert weights and original Engram tables are locked in RAM before
// inference. No demand-paging fallback. Dense weights live on the GPU instead.
class ResidentWeights {
public:
    ResidentWeights();
    ~ResidentWeights();
    void load(const Model& model);
    const uint8_t* data(const std::string& name) const;
    uint64_t bytes() const;
    void verify() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::deepseek41
