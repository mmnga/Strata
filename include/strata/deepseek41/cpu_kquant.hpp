// Reused from the local Strata GLM Flash implementation; isolated DeepSeek dispatch.
// include/strata/deepseek41/cpu_kquant.hpp - AVX-2 multi-token dot products for Unsloth UD-Q4_K_XL's expert formats
// (Q4_K/Q5_K gate/up and Q4_K/Q5_K/Q6_K down against Q8_K; Q5_1 against Q8_1, Q8_0 against Q8_0), bit-exact against
// ggml-cpu's AVX2 vec_dot for every token.  See src/kernels/cpu/kq_avx2.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::deepseek41::cpu {

// Existing Strata AVX2 quantizer; caller checks ISA, n is a multiple of256.
void q8k_quant_avx2(const float * x, void * y, int64_t n);

/// Q4_K (12), Q5_K (13), Q6_K (14), Q5_1 (7), Q8_0 (8).
bool kq256_supported(int ggml_type) noexcept;
/// AVX-512 register allocation and wider integer work, preserving the eight-lane FP arithmetic.
bool kq512_available() noexcept;
/// Native VNNI availability; singleton Q4_K kernels preserve the original eight FP dot lanes.
bool kq512_vnni_available() noexcept;
/// Opt-in wrapping integer MAC fusion; requires AVX512-VNNI and preserves all FP operations.
bool kq512_dpwssd_enabled() noexcept;
/// Strict opt-in coefficient conversion for Q4 VNNI; the default retains ggml's table.
bool kq512_q4_vnni_f16c_enabled();
/// Convert a block's two FP16 scales; exceptional halves retain ggml's exact table values.
void kq512_half_pair_f16c(const void* fp16_pair, float* fp32_pair);
/// Match Q4's coefficients {activation_d*d, -activation_d*dmin} without changing operand order.
void kq512_q4_coefficients_f16c(const void* fp16_pair, float activation_d, float* coefficients);
void kq512_q4_vnni_rows(const uint8_t* weights, size_t row_bytes, int columns, const void* activation,
                       float* output, int first, int last);
void kq512_q4_vnni_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int columns, const void* activation,
                          float* ff, int first, int last, float limit = 0, const uint8_t* separate_up = nullptr);
// Singleton Q5_K down rows in adjacent pairs, sharing Q8 loads while keeping
// each row's original eight FP lanes and block accumulation order.
void kq512_q5_rows2(const uint8_t* weights, size_t row_bytes, int columns, const void* activation,
                    float* output, int first, int last);
/// A single activation shared by fused gate/up matrices, with exact per-matrix FP lanes.
void kq512_gu1_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act,
                    float* ff, int r0, int r1, float limit = 0, const uint8_t* separate_up = nullptr);
/// Two adjacent rows per matrix share activation loads; Gate and Up keep separate dot calls and FP lanes.
void kq512_gu2_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act,
                    float* ff, int r0, int r1, float limit = 0, const uint8_t* separate_up = nullptr);
void kq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1, float limit = 0, const uint8_t* separate_up = nullptr);
void kq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); optional GLM clamp and separate up rows.
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1, float limit = 0, const uint8_t* separate_up = nullptr);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// out[r] = w_r . x for BF16 rows (bits), fp32 x: the routing-aware prefetch's router (an estimate only).
void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out);
/// The same for `nt` <= 8 tokens (x: nt rows of cols, cols % 8 == 0), out[t * rows + r].
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out);

}  // namespace strata::deepseek41::cpu
