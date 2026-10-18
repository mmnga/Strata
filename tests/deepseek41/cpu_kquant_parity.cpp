#include "strata/deepseek41/cpu_kquant.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    ggml_cpu_init();
    std::printf("CPU oracle: avx512=%d vnni=%d dpwssd=%d\n", strata::deepseek41::cpu::kq512_available(),
        strata::deepseek41::cpu::kq512_vnni_available(), strata::deepseek41::cpu::kq512_dpwssd_enabled());
    std::mt19937 random(42);
    std::normal_distribution<float> normal(0, 0.1f);
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        for (const int columns : {5120, 2304}) {
            constexpr int rows = 32;
            const size_t row_bytes = ggml_row_size(type, columns);
            std::vector<float> weights(rows * columns);
            for (auto & w : weights) w = normal(random);
            std::vector<uint8_t> quantized(rows * row_bytes);
            ggml_get_type_traits(type)->from_float_ref(weights.data(), quantized.data(), weights.size());
            const auto * traits = ggml_get_type_traits_cpu(type);
            const auto * act_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
            const size_t act_bytes = ggml_row_size(traits->vec_dot_type, columns);
            for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
                std::vector<float> input(tokens * columns), actual(tokens * rows), expected(tokens * rows);
                std::vector<uint8_t> activation(tokens * act_bytes);
                const void * inputs[8];
                float * outputs[8];
                for (auto & x : input) x = normal(random);
                for (int t = 0; t < tokens; ++t) {
                    inputs[t] = activation.data() + t * act_bytes;
                    outputs[t] = actual.data() + t * rows;
                    act_traits->from_float(input.data() + t * columns, activation.data() + t * act_bytes, columns);
                    for (int r = 0; r < rows; ++r)
                        traits->vec_dot(columns, &expected[t * rows + r], 0,
                            quantized.data() + r * row_bytes, 0, inputs[t], 0, 1);
                }
                strata::deepseek41::cpu::kq512_rows(type, quantized.data(), row_bytes, columns,
                    inputs, tokens, outputs, 0, rows);
                double error_squared = 0, reference_squared = 0, maximum = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i])) return 1;
                    const double error = double(actual[i]) - expected[i];
                    error_squared += error * error;
                    reference_squared += double(expected[i]) * expected[i];
                    maximum = std::max(maximum, std::abs(error));
                }
                const double relative = std::sqrt(error_squared / reference_squared);
                std::printf("{\"type\":\"%s\",\"input\":%d,\"tokens\":%d,\"max_abs\":%.9g,\"relative_rms\":%.9g}\n",
                    ggml_type_name(type), columns, tokens, maximum, relative);
                if (maximum > 1e-5 || relative > 1e-5) return 1;
            }
        }
    }
}
