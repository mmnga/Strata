#pragma once
#include "strata/deepseek41/model.hpp"
#include "strata/deepseek41/expert_cache.hpp"
#include <memory>
namespace strata::deepseek41 {
class Lookahead {
public:
    Lookahead(const Model&,ExpertCache&,int depth = 2,int top_k = 12);
    ~Lookahead();
    void submit(int current_layer,const float* normalized);
    void report() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
