#pragma once
#include "strata/deepseek41/cpu_experts.hpp"
#include <memory>
#include <vector>

namespace strata::deepseek41 {
// Bases point to whole original, registered layer tensors, not expert slices:
// copies must split at the resident owner's 1 GiB registration boundaries.
struct StreamLayerWeights {
    ExpertWeights base;
    int experts = 0;
};

// Selected cold experts use two independent GPU streams while the same caller
// and CPU pool compute the remaining jobs. No cache heat or tensor conversion.
class StreamedExperts {
public:
    StreamedExperts(std::vector<StreamLayerWeights> layers,int devices,int jobs_per_gpu = 3);
    ~StreamedExperts();
    StreamedExperts(const StreamedExperts&) = delete;
    StreamedExperts& operator=(const StreamedExperts&) = delete;
    std::vector<ExpertBatch> submit(int layer,int owner,const std::vector<ExpertBatch>& jobs,
                                    const float* pinned_input,int tokens);
    // Complete every selected job before its output is consumed or input reused.
    void finish(float* parts);
    void report() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::deepseek41
