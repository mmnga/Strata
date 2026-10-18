#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::deepseek41 {

struct ExpertWeights {
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    int gate_type = 12, down_type = 12, hidden = 0, ff = 0;
    size_t gate_row = 0, down_row = 0;
    float limit = 10;
};
struct ExpertBatch {
    ExpertWeights weights;
    int count = 0;
    int tokens[8] = {}, destinations[8] = {};
};

// Dynamic-size expert pool for DeepSeek (5120/2304). Strata's Qwen pool uses
// fixed 2560/640 scratch. Both use all participating cores on each row phase.
class CpuExperts {
public:
    explicit CpuExperts(int threads = 24);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;
    // One activation, up to 64 experts. Output is [expert][hidden], unweighted.
    void run(const std::vector<ExpertWeights>& jobs, const float* x, float* out);
    void run_multi(const std::vector<ExpertBatch>& jobs,const float* x,int tokens,float* out);
    int threads() const;
    int host_core() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::deepseek41
