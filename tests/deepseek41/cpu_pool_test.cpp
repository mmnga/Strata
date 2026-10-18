#include "strata/deepseek41/cpu_experts.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace ds = strata::deepseek41;
std::vector<uint8_t> matrix(ggml_type type,int in,int out,std::mt19937& rng) {
    std::normal_distribution<float> normal(0,0.13f); std::vector<float> row(in);
    const size_t bytes = ggml_row_size(type,in); std::vector<uint8_t> w(bytes*size_t(out));
    for (int r = 0; r < out; ++r) {
        for (auto& v : row) v = normal(rng);
        ggml_get_type_traits(type)->from_float_ref(row.data(),w.data()+bytes*size_t(r),in);
    }
    return w;
}
std::vector<float> reference(const ds::ExpertWeights& w,const std::vector<float>& x) {
    const auto* gu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K); const auto* down = ggml_get_type_traits_cpu(ggml_type(w.down_type));
    std::vector<uint8_t> q(ggml_row_size(gu->vec_dot_type,w.hidden)), hq(ggml_row_size(down->vec_dot_type,w.ff));
    std::vector<float> hidden(w.ff), out(w.hidden);
    ggml_get_type_traits_cpu(gu->vec_dot_type)->from_float(x.data(),q.data(),w.hidden);
    for (int r = 0; r < w.ff; ++r) {
        float gate, up;
        gu->vec_dot(w.hidden,&gate,0,w.gate+size_t(r)*w.gate_row,0,q.data(),0,1);
        gu->vec_dot(w.hidden,&up,0,w.up+size_t(r)*w.gate_row,0,q.data(),0,1);
        gate = std::fmin(gate,w.limit); up = std::fmax(-w.limit,std::fmin(up,w.limit));
        hidden[r] = (gate/(1+std::exp(-gate)))*up;
    }
    ggml_get_type_traits_cpu(down->vec_dot_type)->from_float(hidden.data(),hq.data(),w.ff);
    for (int r = 0; r < w.hidden; ++r) down->vec_dot(w.ff,&out[r],0,w.down+size_t(r)*w.down_row,0,hq.data(),0,1);
    return out;
}
int main() {
    try {
        ggml_cpu_init(); std::mt19937 rng(193);
        constexpr int H = 5120, F = 2304;
        const auto gate = matrix(GGML_TYPE_Q4_K,H,F,rng), up = matrix(GGML_TYPE_Q4_K,H,F,rng);
        std::vector<float> x(H); std::normal_distribution<float> normal(0,1); for (auto& v : x) v = normal(rng);
        for (ggml_type type : {GGML_TYPE_Q4_K,GGML_TYPE_Q6_K}) {
            const auto down = matrix(type,F,H,rng);
            ds::ExpertWeights w; w.hidden = H; w.ff = F; w.gate = gate.data(); w.up = up.data(); w.down = down.data();
            w.down_type = int(type); w.gate_row = ggml_row_size(GGML_TYPE_Q4_K,H); w.down_row = ggml_row_size(type,F); w.limit = 10;
            const auto expected = reference(w,x);
            for (int threads : {1,24}) {
                ds::CpuExperts pool(threads);
                // Repeated different batch lengths catch stale worker claims or
                // phase buffers leaking between gate/up, quantize and down.
                for (int n : {1,6,3,6,1}) {
                    std::vector<ds::ExpertWeights> jobs(size_t(n),w); std::vector<float> actual(size_t(n)*H);
                    pool.run(jobs,x.data(),actual.data());
                    for (int j = 0; j < n; ++j) for (int i = 0; i < H; ++i) {
                        const float v = actual[size_t(j)*H+i];
                        if (!std::isfinite(v) || std::abs(v-expected[i]) > 2e-5f*(1+std::abs(expected[i]))) throw std::runtime_error("CPU row pool output mismatch");
                    }
                }
                std::vector<float> multi_input(size_t(H)*3), multi_out(size_t(H)*3);
                for (int t = 0; t < 3; ++t) for (int i = 0; i < H; ++i) multi_input[size_t(t)*H+i] = x[i]*float(t+1)/2;
                ds::ExpertBatch batch; batch.weights = w; batch.count = 3;
                for (int t = 0; t < 3; ++t) { batch.tokens[t] = t; batch.destinations[t] = 2-t; }
                pool.run_multi({batch},multi_input.data(),3,multi_out.data());
                for (int t = 0; t < 3; ++t) {
                    const auto ref = reference(w,std::vector<float>(multi_input.begin()+t*H,multi_input.begin()+(t+1)*H));
                    for (int i = 0; i < H; ++i) if (std::abs(multi_out[size_t(2-t)*H+i]-ref[i]) > 2e-5f*(1+std::abs(ref[i]))) throw std::runtime_error("multi-token CPU group/dispatch mismatch");
                }
                std::cout << "type=" << int(type) << " participants=" << threads << " phase/batch parity PASS\n";
            }
        }
        std::cout << "Native DeepSeek dynamic CPU expert pool PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
