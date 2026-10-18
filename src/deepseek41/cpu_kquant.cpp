// Reused from the local Strata GLM Flash implementation; isolated DeepSeek dispatch.
// src/kernels/cpu/kq_avx2.cpp - K-quant expert rows for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// These follow the __AVX2__ branches of ggml-cpu/arch/x86/quants.c: Q4_K, Q5_K and Q6_K against Q8_K,
// Q5_1 against Q8_1 and Q8_0 against Q8_0. The functions below move the loop over tokens inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (kq_flash_test checks it directly against ggml), whatever the number of tokens. Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "strata/deepseek41/cpu_kquant.hpp"
#include "strata/deepseek41/cpu_weight_prefetch.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
// ggml-cpu initializes this table in ggml_cpu_init(); native_fmt and the direct
// kernel tests initialize it before entering any expert row kernel.
extern "C" float ggml_table_f32_f16[1 << 16];
#endif

namespace strata::deepseek41::cpu {
namespace {

inline bool split_dpwssd() {
    static const bool enabled=[] {const char * p=std::getenv("STRATA_CPU_DPWSSD_SPLIT");return p && std::strcmp(p,"1")==0;}();
    return enabled;
}

inline bool paired_half() {
    static const bool enabled=[] {const char * p=std::getenv("STRATA_CPU_HALF_PAIR");return p && std::strcmp(p,"1")==0;}();
    return enabled;
}


constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) constexpr uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
// Fused gate/up needs the same shuffle in each 256-bit half. Keep the masks
// already duplicated so each shuffle can consume one 64-byte memory operand.
alignas(64) constexpr auto k_gu1_shuffle = [] {
    std::array<std::array<uint8_t,64>,8> masks{};
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 64; ++k) masks[j][k] = k_shuffle[j*32+k%32];
    return masks;
}();
// A repeated uint32 scale for VNNI's four-byte integer lanes. The original K
// scale shuffle repeats a uint16 instead; its remaining two bytes are zeroed.
alignas(32) constexpr auto k_vnni_scale_shuffle = [] {
    std::array<std::array<uint8_t,32>,8> masks{};
    for (int group=0; group<8; ++group)
        for (int byte=0; byte<32; ++byte)
            masks[group][byte] = byte%4 < 2 ? uint8_t(2*group+byte%4) : uint8_t(0x80);
    return masks;
}();
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, __AVX2__)
void q5k_dot(const block_q5_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(15);
    __m256 acc[MAXT];
    float summs[MAXT]{};
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
        utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[2] = uaux; utmp[0] &= 0x3f3f3f3f;
        const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(ms, 1), sc = _mm256_castsi256_si128(ms);
        const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc), sc, 1);
        const __m256i high = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i q[8], s[8];
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256((const __m256i*)x[i].qs + j);
            const __m256i lo = _mm256_and_si256(bits, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            // Variable shifts use the vector count form (j is not an immediate).
            const __m128i c0 = _mm_cvtsi32_si128(2*j), c1 = _mm_cvtsi32_si128(2*j+1);
            q[2*j] = _mm256_add_epi8(lo, _mm256_slli_epi16(_mm256_and_si256(
                _mm256_srl_epi16(high, c0), _mm256_set1_epi8(1)), 4));
            q[2*j+1] = _mm256_add_epi8(hi, _mm256_slli_epi16(_mm256_and_si256(
                _mm256_srl_epi16(high, c1), _mm256_set1_epi8(1)), 4));
            s[2*j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j));
            s[2*j+1] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));
        }
        for (int t = 0; t < nt; ++t) {
            const auto* y = (const block_q8_K*)act[t] + i;
            const __m256i sums = _mm256_loadu_si256((const __m256i*)y->bsums);
            const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, pairs);
            const __m128i total = _mm_hadd_epi32(_mm_hadd_epi32(prod, _mm_setzero_si128()), _mm_setzero_si128());
            summs[t] += (-y->d * xdmin) * float(_mm_cvtsi128_si32(total));
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j) {
                const __m256i a = _mm256_madd_epi16(s[2*j], _mm256_maddubs_epi16(q[2*j],
                    _mm256_loadu_si256((const __m256i*)y->qs + 2*j)));
                const __m256i b = _mm256_madd_epi16(s[2*j+1], _mm256_maddubs_epi16(q[2*j+1],
                    _mm256_loadu_si256((const __m256i*)y->qs + 2*j+1)));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(a, b));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q6_K x Q8_K: unpack once, preserving ggml's integer sums and float accumulation.
void q6k_dot(const block_q6_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m256i scale16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x[i].scales));
        __m256i q[8], s[8];
        for (int j = 0; j < 2; ++j) {
            const __m256i a = _mm256_loadu_si256((const __m256i*)x[i].ql + 2*j);
            const __m256i b = _mm256_loadu_si256((const __m256i*)x[i].ql + 2*j+1);
            const __m256i high = _mm256_loadu_si256((const __m256i*)x[i].qh + j);
            q[4*j] = _mm256_or_si256(_mm256_and_si256(a, m15),
                _mm256_slli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(3)), 4));
            q[4*j+1] = _mm256_or_si256(_mm256_and_si256(b, m15),
                _mm256_slli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(12)), 2));
            q[4*j+2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a, 4), m15),
                _mm256_and_si256(high, _mm256_set1_epi8(48)));
            q[4*j+3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m15),
                _mm256_srli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(-64)), 2));
        }
        for (int j = 0; j < 8; ++j)
            s[j] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(x[i].scales[2*j])),
                                          _mm_set1_epi16(x[i].scales[2*j+1]), 1);
        for (int t = 0; t < nt; ++t) {
            const auto* y = (const block_q8_K*)act[t] + i;
            const __m256i sub = _mm256_slli_epi32(_mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i*)y->bsums), scale16), 5);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 8; ++j)
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s[j], _mm256_maddubs_epi16(q[j],
                    _mm256_loadu_si256((const __m256i*)y->qs + j))));
            sumi = _mm256_sub_epi32(sumi, sub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 13: q5k_dot((const block_q5_K*) row, n / QK_K, act, nt, res); break;
        case 14: q6k_dot((const block_q6_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

// The verify widths are known once per row group. Specializing them lets the
// compiler retain the unpacked weights and per-token accumulators in AVX-512's
// 32 vector registers, without changing the original eight-lane FP reduction.
// Q6_K also pairs adjacent groups in 512-bit integer operations, then folds back
// to those eight lanes before applying the original FP accumulation.
#if defined(__GNUC__) || defined(__clang__)
inline float half_at_ggml_table(const void* block, int index) {
    uint16_t bits;
    std::memcpy(&bits,(const uint8_t*)block+2*index,sizeof(bits));
    return ggml_table_f32_f16[bits];
}

inline __m128 half_pair_f16c(const void* block) {
    uint32_t bits;
    std::memcpy(&bits,block,sizeof(bits));
    const uint16_t low = uint16_t(bits), high = uint16_t(bits >> 16);
    // Finite binary16 -> binary32 is exact, including subnormal and signed
    // zero values. Inf/NaN retain the table's payloads and exception behavior.
    if ((low & 0x7c00) == 0x7c00 || (high & 0x7c00) == 0x7c00)
        return _mm_setr_ps(ggml_table_f32_f16[low],ggml_table_f32_f16[high],0.f,0.f);
    return _mm_cvtph_ps(_mm_cvtsi32_si128(int(bits)));
}

inline __m128 q4_coefficients_f16c(const void* block, float activation_d) {
    return _mm_mul_ps(_mm_setr_ps(activation_d,-activation_d,0.f,0.f),half_pair_f16c(block));
}

// Each VNNI lane sums the same four products as maddubs followed by maddwd.
// The original shuffle repeats one scale for both 16-bit pair sums, so applying
// that scale to the four-product sum is integer-identical. A nibble<=15, signed
// Q8<=128 in magnitude and scale<=63 bound each group by 483840, all eight
// groups by 3870720: neither original saturation nor int32 overflow is possible.
template<bool F16C_PAIR>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
float q4k_dot512_vnni(const block_q4_K* x, const block_q8_K* y, int nb) {
    const __m256i nibble_mask = _mm256_set1_epi8(15), zero = _mm256_setzero_si256();
    __m256 accumulator = _mm256_setzero_ps();
    __m128 min_accumulator = _mm_setzero_ps();
    for (int block=0; block<nb; ++block) {
        uint32_t unpacked[4];
        std::memcpy(unpacked,x[block].scales,12);
        unpacked[3] = ((unpacked[2] >> 4) & 0x0f0f0f0f) | (((unpacked[1] >> 6) & 0x03030303) << 4);
        const uint32_t auxiliary = unpacked[1] & 0x3f3f3f3f;
        unpacked[1] = (unpacked[2] & 0x0f0f0f0f) | (((unpacked[0] >> 6) & 0x03030303) << 4);
        unpacked[2] = auxiliary; unpacked[0] &= 0x3f3f3f3f;
        const __m256i min_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(unpacked[3],unpacked[2],unpacked[1],unpacked[0]));
        const __m256i bsums = _mm256_loadu_si256((const __m256i*)y[block].bsums);
        const __m128i pair_sums = _mm_hadd_epi16(_mm256_castsi256_si128(bsums),_mm256_extracti128_si256(bsums,1));
        const __m128i min_products = _mm_madd_epi16(_mm256_extracti128_si256(min_scales,1),pair_sums);
        __m128 coefficients;
        float dmin;
        if constexpr (F16C_PAIR) {
            // Match the scalar coefficient operand order, including the sign
            // applied to the activation scale before multiplying dmin.
            coefficients = q4_coefficients_f16c(&x[block],y[block].d);
            dmin = _mm_cvtss_f32(_mm_shuffle_ps(coefficients,coefficients,_MM_SHUFFLE(1,1,1,1)));
        } else {
            dmin = -y[block].d * half_at_ggml_table(&x[block],1);
        }
        min_accumulator = _mm_fmadd_ps(_mm_set1_ps(dmin),_mm_cvtepi32_ps(min_products),min_accumulator);
        const __m128i scales_low = _mm256_castsi256_si128(min_scales);
        const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(scales_low),scales_low,1);
        __m256i integer_sum = _mm256_setzero_si256();
        for (int group=0; group<4; ++group) {
            const __m256i bits = _mm256_loadu_si256((const __m256i*)x[block].qs+group);
            const __m256i low = _mm256_and_si256(bits,nibble_mask);
            const __m256i high = _mm256_and_si256(_mm256_srli_epi16(bits,4),nibble_mask);
            const __m256i low_scale = _mm256_shuffle_epi8(scales,_mm256_load_si256((const __m256i*)k_vnni_scale_shuffle[2*group].data()));
            const __m256i high_scale = _mm256_shuffle_epi8(scales,_mm256_load_si256((const __m256i*)k_vnni_scale_shuffle[2*group+1].data()));
            const __m256i low_dot = _mm256_dpbusd_epi32(zero,low,_mm256_loadu_si256((const __m256i*)y[block].qs+2*group));
            const __m256i high_dot = _mm256_dpbusd_epi32(zero,high,_mm256_loadu_si256((const __m256i*)y[block].qs+2*group+1));
            integer_sum = _mm256_add_epi32(integer_sum,_mm256_add_epi32(_mm256_mullo_epi32(low_scale,low_dot),
                                                                       _mm256_mullo_epi32(high_scale,high_dot)));
        }
        float d;
        if constexpr (F16C_PAIR) d = _mm_cvtss_f32(coefficients);
        else d = y[block].d * half_at_ggml_table(&x[block],0);
        accumulator = _mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(integer_sum),accumulator);
    }
    min_accumulator = _mm_add_ps(min_accumulator,_mm_movehl_ps(min_accumulator,min_accumulator));
    min_accumulator = _mm_add_ss(min_accumulator,_mm_movehdup_ps(min_accumulator));
    return hsum_float_8(accumulator) + _mm_cvtss_f32(min_accumulator);
}

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
template<int NT>
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
void q4k_dot512(const block_q4_K* x, int nb, const void* const* act, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NT];
    __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// Strata IQ madd_add integer fusion; the original FP operations stay in order.
template<int NT, bool Split = false, bool PairHalf = false>
__attribute__((target("avx512f,avx512vnni,avx512bw,avx512vl,fma,f16c")))
void q4k_dot512_dpwssd(const block_q4_K* x, int nb, const void* const* act, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NT];
    __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        float xd, xdmin;
        if constexpr (PairHalf) {
            uint32_t pair;
            std::memcpy(&pair, &x[i], sizeof(pair));
            const __m128 values = _mm_cvtph_ps(_mm_cvtsi32_si128(int(pair)));
            xd = _mm_cvtss_f32(values);
            xdmin = _mm_cvtss_f32(_mm_movehdup_ps(values));
        } else {
            xd = half_at(&x[i], 0); xdmin = half_at(&x[i], 1);
        }
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            [[maybe_unused]] __m256i high_sum = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                sumi = _mm256_dpwssd_epi32(sumi, scale_l[j], p16l);
                if constexpr (Split) high_sum = _mm256_dpwssd_epi32(high_sum, scale_h[j], p16h);
                else sumi = _mm256_dpwssd_epi32(sumi, scale_h[j], p16h);
            }
            if constexpr (Split) sumi = _mm256_add_epi32(sumi,high_sum);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, __AVX2__)
template<int NT>
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
void q5k_dot512(const block_q5_K* x, int nb, const void* const* act, float* res) {
    const __m256i m4 = _mm256_set1_epi8(15);
    __m256 acc[NT];
    float summs[NT]{};
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
        utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[2] = uaux; utmp[0] &= 0x3f3f3f3f;
        const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(ms, 1), sc = _mm256_castsi256_si128(ms);
        const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc), sc, 1);
        const __m256i high = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i q[8], s[8];
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256((const __m256i*)x[i].qs + j);
            const __m256i lo = _mm256_and_si256(bits, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            // Variable shifts use the vector count form (j is not an immediate).
            const __m128i c0 = _mm_cvtsi32_si128(2*j), c1 = _mm_cvtsi32_si128(2*j+1);
            q[2*j] = _mm256_add_epi8(lo, _mm256_slli_epi16(_mm256_and_si256(
                _mm256_srl_epi16(high, c0), _mm256_set1_epi8(1)), 4));
            q[2*j+1] = _mm256_add_epi8(hi, _mm256_slli_epi16(_mm256_and_si256(
                _mm256_srl_epi16(high, c1), _mm256_set1_epi8(1)), 4));
            s[2*j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j));
            s[2*j+1] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));
        }
        for (int t = 0; t < NT; ++t) {
            const auto* y = (const block_q8_K*)act[t] + i;
            const __m256i sums = _mm256_loadu_si256((const __m256i*)y->bsums);
            const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, pairs);
            const __m128i total = _mm_hadd_epi32(_mm_hadd_epi32(prod, _mm_setzero_si128()), _mm_setzero_si128());
            summs[t] += (-y->d * xdmin) * float(_mm_cvtsi128_si32(total));
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j) {
                const __m256i a = _mm256_madd_epi16(s[2*j], _mm256_maddubs_epi16(q[2*j],
                    _mm256_loadu_si256((const __m256i*)y->qs + 2*j)));
                const __m256i b = _mm256_madd_epi16(s[2*j+1], _mm256_maddubs_epi16(q[2*j+1],
                    _mm256_loadu_si256((const __m256i*)y->qs + 2*j+1)));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(a, b));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q6_K x Q8_K: unpack once, preserving ggml's integer sums and float accumulation.
template<int NT>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))
void q6k_dot512(const block_q6_K* x, int nb, const void* const* act, float* res) {
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m256i scale16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x[i].scales));
        __m512i q[4], scale[4];
        const __m512i mask_lo = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi8(3),_mm512_set1_epi8(12));
        const __m512i shift_lo = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi16(4),_mm512_set1_epi16(2));
        const __m512i mask_hi = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi8(48),_mm512_set1_epi8(-64));
        const __m512i shift_hi = _mm512_mask_blend_epi32(0xff00,_mm512_setzero_si512(),_mm512_set1_epi16(2));
        for (int j = 0; j < 2; ++j) {
            const __m512i bits = _mm512_loadu_si512(x[i].ql+64*j);
            const __m512i high = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)x[i].qh+j));
            q[2*j] = _mm512_or_si512(_mm512_and_si512(bits,_mm512_set1_epi8(15)),
                _mm512_sllv_epi16(_mm512_and_si512(high,mask_lo),shift_lo));
            q[2*j+1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits,4),_mm512_set1_epi8(15)),
                _mm512_srlv_epi16(_mm512_and_si512(high,mask_hi),shift_hi));
        }
        for (int j = 0; j < 4; ++j) {
            const int s0 = uint16_t(x[i].scales[4*j]) * 0x10001u;
            const int s1 = uint16_t(x[i].scales[4*j+1]) * 0x10001u;
            const int s2 = uint16_t(x[i].scales[4*j+2]) * 0x10001u;
            const int s3 = uint16_t(x[i].scales[4*j+3]) * 0x10001u;
            scale[j] = _mm512_set_epi32(s3,s3,s3,s3,s2,s2,s2,s2,s1,s1,s1,s1,s0,s0,s0,s0);
        }
        for (int t = 0; t < NT; ++t) {
            const auto* y = (const block_q8_K*)act[t] + i;
            const __m256i sub = _mm256_slli_epi32(_mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i*)y->bsums), scale16), 5);
            __m512i sumi = _mm512_setzero_si512();
            for (int j = 0; j < 4; ++j)
                sumi = _mm512_add_epi32(sumi,_mm512_madd_epi16(scale[j],_mm512_maddubs_epi16(q[j],
                    _mm512_loadu_si512((const __m512i*)y->qs+j))));
            const __m256i folded = _mm256_sub_epi32(_mm256_add_epi32(_mm512_castsi512_si256(sumi),
                _mm512_extracti64x4_epi64(sumi,1)),sub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(folded), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

alignas(64) constexpr auto k_q6_scale_words = [] {
    std::array<std::array<uint16_t,32>,4> result{};
    for (int group=0; group<4; ++group)
        for (int lane=0; lane<32; ++lane) result[group][lane] = uint16_t(4*group+lane/8);
    return result;
}();

// Strata IQ madd_add integer fusion; the original FP operations stay in order.
template<int NT, bool PermuteScales = false, bool Split = false>
__attribute__((target("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma,f16c")))
void q6k_dot512_dpwssd(const block_q6_K* x, int nb, const void* const* act, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m256i scale16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x[i].scales));
        __m512i q[4], scale[4];
        const __m512i mask_lo = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi8(3),_mm512_set1_epi8(12));
        const __m512i shift_lo = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi16(4),_mm512_set1_epi16(2));
        const __m512i mask_hi = _mm512_mask_blend_epi32(0xff00,_mm512_set1_epi8(48),_mm512_set1_epi8(-64));
        const __m512i shift_hi = _mm512_mask_blend_epi32(0xff00,_mm512_setzero_si512(),_mm512_set1_epi16(2));
        for (int j = 0; j < 2; ++j) {
            const __m512i bits = _mm512_loadu_si512(x[i].ql+64*j);
            const __m512i high = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)x[i].qh+j));
            q[2*j] = _mm512_or_si512(_mm512_and_si512(bits,_mm512_set1_epi8(15)),
                _mm512_sllv_epi16(_mm512_and_si512(high,mask_lo),shift_lo));
            q[2*j+1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits,4),_mm512_set1_epi8(15)),
                _mm512_srlv_epi16(_mm512_and_si512(high,mask_hi),shift_hi));
        }
        for (int j = 0; j < 4; ++j) {
            if constexpr (PermuteScales) {
                scale[j] = _mm512_permutexvar_epi16(_mm512_load_si512(k_q6_scale_words[j].data()),
                    _mm512_broadcast_i64x4(scale16));
            } else {
                const int s0 = uint16_t(x[i].scales[4*j]) * 0x10001u;
                const int s1 = uint16_t(x[i].scales[4*j+1]) * 0x10001u;
                const int s2 = uint16_t(x[i].scales[4*j+2]) * 0x10001u;
                const int s3 = uint16_t(x[i].scales[4*j+3]) * 0x10001u;
                scale[j] = _mm512_set_epi32(s3,s3,s3,s3,s2,s2,s2,s2,s1,s1,s1,s1,s0,s0,s0,s0);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const auto* y = (const block_q8_K*)act[t] + i;
            const __m256i sub = _mm256_slli_epi32(_mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i*)y->bsums), scale16), 5);
            __m512i sumi = _mm512_setzero_si512();
            if constexpr (Split) {
                __m512i even = _mm512_dpwssd_epi32(_mm512_setzero_si512(),scale[0],
                    _mm512_maddubs_epi16(q[0],_mm512_loadu_si512((const __m512i*)y->qs+0)));
                __m512i odd = _mm512_dpwssd_epi32(_mm512_setzero_si512(),scale[1],
                    _mm512_maddubs_epi16(q[1],_mm512_loadu_si512((const __m512i*)y->qs+1)));
                even = _mm512_dpwssd_epi32(even,scale[2],
                    _mm512_maddubs_epi16(q[2],_mm512_loadu_si512((const __m512i*)y->qs+2)));
                odd = _mm512_dpwssd_epi32(odd,scale[3],
                    _mm512_maddubs_epi16(q[3],_mm512_loadu_si512((const __m512i*)y->qs+3)));
                sumi = _mm512_add_epi32(even,odd);
            } else for (int j = 0; j < 4; ++j)
                sumi = _mm512_dpwssd_epi32(sumi,scale[j],_mm512_maddubs_epi16(q[j],
                    _mm512_loadu_si512((const __m512i*)y->qs+j)));
            const __m256i folded = _mm256_sub_epi32(_mm256_add_epi32(_mm512_castsi512_si256(sumi),
                _mm512_extracti64x4_epi64(sumi,1)),sub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(folded), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

// A lone CPU expert still computes two matrices against the same activation.
// Give gate and up independent eight-lane accumulators in the two halves of a
// ZMM register. Sharing the activation loads does not change either dot product.
inline __m256i kq45_scales(const uint8_t* scales) {
    uint32_t utmp[4];
    std::memcpy(utmp,scales,12);
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
    const uint32_t aux = utmp[1] & 0x3f3f3f3f;
    utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
    utmp[2] = aux; utmp[0] &= 0x3f3f3f3f;
    return _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3],utmp[2],utmp[1],utmp[0]));
}
template<typename Block, bool Fifth>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))
void kq45_gu1_dot(const Block* gate, const Block* up, int nb, const block_q8_K* y, float& g, float& u) {
    __m512 acc = _mm512_setzero_ps();
    __m256 acc_m = _mm256_setzero_ps();
    float sum_g = 0.f, sum_u = 0.f;
    const __m512i m15 = _mm512_set1_epi8(15);
    for (int i = 0; i < nb; ++i) {
        const float gd = y[i].d * half_at(&gate[i],0), ud = y[i].d * half_at(&up[i],0);
        const float gm = -y[i].d * half_at(&gate[i],1), um = -y[i].d * half_at(&up[i],1);
        const __m256i gs = kq45_scales(gate[i].scales), us = kq45_scales(up[i].scales);
        const __m128i gl = _mm256_castsi256_si128(gs), ul = _mm256_castsi256_si128(us);
        const __m256i gr = _mm256_inserti128_si256(_mm256_castsi128_si256(gl),gl,1);
        const __m256i ur = _mm256_inserti128_si256(_mm256_castsi128_si256(ul),ul,1);
        const __m512i scales = _mm512_inserti64x4(_mm512_castsi256_si512(gr),ur,1);
        const __m256i mins = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(gs,1)),
                                                   _mm256_extracti128_si256(us,1),1);
        const __m256i ys = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(ys),_mm256_extracti128_si256(ys,1));
        const __m256i pr = _mm256_madd_epi16(mins,_mm256_inserti128_si256(_mm256_castsi128_si256(pairs),pairs,1));
        if constexpr (Fifth) {
            const __m256i totals = _mm256_hadd_epi32(_mm256_hadd_epi32(pr,_mm256_setzero_si256()),_mm256_setzero_si256());
            sum_g += gm * float(_mm_cvtsi128_si32(_mm256_castsi256_si128(totals)));
            sum_u += um * float(_mm_cvtsi128_si32(_mm256_extracti128_si256(totals,1)));
        } else {
            const __m256 dm = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(gm)),_mm_set1_ps(um),1);
            acc_m = _mm256_fmadd_ps(dm,_mm256_cvtepi32_ps(pr),acc_m);
        }
        __m512i high = _mm512_setzero_si512();
        if constexpr (Fifth)
            high = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)gate[i].qh)),
                                      _mm256_loadu_si256((const __m256i*)up[i].qh),1);
        __m512i sumi = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i bits = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)gate[i].qs+j)),
                                                    _mm256_loadu_si256((const __m256i*)up[i].qs+j),1);
            __m512i lo = _mm512_and_si512(bits,m15);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(bits,4),m15);
            if constexpr (Fifth) {
                lo = _mm512_add_epi8(lo,_mm512_slli_epi16(_mm512_and_si512(_mm512_srl_epi16(high,_mm_cvtsi32_si128(2*j)),_mm512_set1_epi8(1)),4));
                hi = _mm512_add_epi8(hi,_mm512_slli_epi16(_mm512_and_si512(_mm512_srl_epi16(high,_mm_cvtsi32_si128(2*j+1)),_mm512_set1_epi8(1)),4));
            }
            const __m512i sl = _mm512_shuffle_epi8(scales,_mm512_loadu_si512(k_gu1_shuffle[2*j].data()));
            const __m512i sh = _mm512_shuffle_epi8(scales,_mm512_loadu_si512(k_gu1_shuffle[2*j+1].data()));
            const __m512i al = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)y[i].qs+2*j));
            const __m512i ah = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)y[i].qs+2*j+1));
            const __m512i p0 = _mm512_madd_epi16(sl,_mm512_maddubs_epi16(lo,al));
            const __m512i p1 = _mm512_madd_epi16(sh,_mm512_maddubs_epi16(hi,ah));
            sumi = _mm512_add_epi32(sumi,_mm512_add_epi32(p0,p1));
        }
        const __m512 d = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(gd)),_mm256_set1_ps(ud),1);
        acc = _mm512_fmadd_ps(d,_mm512_cvtepi32_ps(sumi),acc);
    }
    if constexpr (!Fifth) {
        __m128 mg = _mm256_castps256_ps128(acc_m), mu = _mm256_extractf128_ps(acc_m,1);
        mg = _mm_add_ps(mg,_mm_movehl_ps(mg,mg)); mg = _mm_add_ss(mg,_mm_movehdup_ps(mg));
        mu = _mm_add_ps(mu,_mm_movehl_ps(mu,mu)); mu = _mm_add_ss(mu,_mm_movehdup_ps(mu));
        sum_g = _mm_cvtss_f32(mg); sum_u = _mm_cvtss_f32(mu);
    }
    g = hsum_float_8(_mm512_castps512_ps256(acc)) + sum_g;
    u = hsum_float_8(_mm512_extractf32x8_ps(acc,1)) + sum_u;
}

// Strata IQ madd_add integer fusion; the original FP operations stay in order.
template<typename Block, bool Fifth, bool Split = false, bool PairHalf = false>
__attribute__((target("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma,f16c")))
void kq45_gu1_dot_dpwssd(const Block* gate, const Block* up, int nb, const block_q8_K* y, float& g, float& u) {
    __m512 acc = _mm512_setzero_ps();
    __m256 acc_m = _mm256_setzero_ps();
    float sum_g = 0.f, sum_u = 0.f;
    const __m512i m15 = _mm512_set1_epi8(15);
    for (int i = 0; i < nb; ++i) {
        float gd, ud, gm, um;
        if constexpr (PairHalf) {
            uint32_t gate_pair, up_pair;
            std::memcpy(&gate_pair, &gate[i], sizeof(gate_pair));
            std::memcpy(&up_pair, &up[i], sizeof(up_pair));
            const __m128 values = _mm_cvtph_ps(_mm_set_epi32(0,0,int(up_pair),int(gate_pair)));
            // Four independent original products, with identical sign operands.
            const __m128 products = _mm_mul_ps(values,_mm_setr_ps(y[i].d,-y[i].d,y[i].d,-y[i].d));
            gd = _mm_cvtss_f32(products);
            ud = _mm_cvtss_f32(_mm_shuffle_ps(products,products,_MM_SHUFFLE(2,2,2,2)));
            gm = _mm_cvtss_f32(_mm_movehdup_ps(products));
            um = _mm_cvtss_f32(_mm_shuffle_ps(products,products,_MM_SHUFFLE(3,3,3,3)));
        } else {
            gd = y[i].d * half_at(&gate[i],0); ud = y[i].d * half_at(&up[i],0);
            gm = -y[i].d * half_at(&gate[i],1); um = -y[i].d * half_at(&up[i],1);
        }
        const __m256i gs = kq45_scales(gate[i].scales), us = kq45_scales(up[i].scales);
        const __m128i gl = _mm256_castsi256_si128(gs), ul = _mm256_castsi256_si128(us);
        const __m256i gr = _mm256_inserti128_si256(_mm256_castsi128_si256(gl),gl,1);
        const __m256i ur = _mm256_inserti128_si256(_mm256_castsi128_si256(ul),ul,1);
        const __m512i scales = _mm512_inserti64x4(_mm512_castsi256_si512(gr),ur,1);
        const __m256i mins = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(gs,1)),
                                                   _mm256_extracti128_si256(us,1),1);
        const __m256i ys = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(ys),_mm256_extracti128_si256(ys,1));
        const __m256i pr = _mm256_madd_epi16(mins,_mm256_inserti128_si256(_mm256_castsi128_si256(pairs),pairs,1));
        if constexpr (Fifth) {
            const __m256i totals = _mm256_hadd_epi32(_mm256_hadd_epi32(pr,_mm256_setzero_si256()),_mm256_setzero_si256());
            sum_g += gm * float(_mm_cvtsi128_si32(_mm256_castsi256_si128(totals)));
            sum_u += um * float(_mm_cvtsi128_si32(_mm256_extracti128_si256(totals,1)));
        } else {
            const __m256 dm = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(gm)),_mm_set1_ps(um),1);
            acc_m = _mm256_fmadd_ps(dm,_mm256_cvtepi32_ps(pr),acc_m);
        }
        __m512i high = _mm512_setzero_si512();
        if constexpr (Fifth)
            high = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)gate[i].qh)),
                                      _mm256_loadu_si256((const __m256i*)up[i].qh),1);
        __m512i sumi = _mm512_setzero_si512();
        [[maybe_unused]] __m512i high_sum = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i bits = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)gate[i].qs+j)),
                                                    _mm256_loadu_si256((const __m256i*)up[i].qs+j),1);
            __m512i lo = _mm512_and_si512(bits,m15);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(bits,4),m15);
            if constexpr (Fifth) {
                lo = _mm512_add_epi8(lo,_mm512_slli_epi16(_mm512_and_si512(_mm512_srl_epi16(high,_mm_cvtsi32_si128(2*j)),_mm512_set1_epi8(1)),4));
                hi = _mm512_add_epi8(hi,_mm512_slli_epi16(_mm512_and_si512(_mm512_srl_epi16(high,_mm_cvtsi32_si128(2*j+1)),_mm512_set1_epi8(1)),4));
            }
            const __m512i sl = _mm512_shuffle_epi8(scales,_mm512_loadu_si512(k_gu1_shuffle[2*j].data()));
            const __m512i sh = _mm512_shuffle_epi8(scales,_mm512_loadu_si512(k_gu1_shuffle[2*j+1].data()));
            const __m512i al = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)y[i].qs+2*j));
            const __m512i ah = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)y[i].qs+2*j+1));
            sumi = _mm512_dpwssd_epi32(sumi,sl,_mm512_maddubs_epi16(lo,al));
            if constexpr (Split) high_sum = _mm512_dpwssd_epi32(high_sum,sh,_mm512_maddubs_epi16(hi,ah));
            else sumi = _mm512_dpwssd_epi32(sumi,sh,_mm512_maddubs_epi16(hi,ah));
        }
        const __m512 d = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(gd)),_mm256_set1_ps(ud),1);
        if constexpr (Split) sumi = _mm512_add_epi32(sumi,high_sum);
        acc = _mm512_fmadd_ps(d,_mm512_cvtepi32_ps(sumi),acc);
    }
    if constexpr (!Fifth) {
        __m128 mg = _mm256_castps256_ps128(acc_m), mu = _mm256_extractf128_ps(acc_m,1);
        mg = _mm_add_ps(mg,_mm_movehl_ps(mg,mg)); mg = _mm_add_ss(mg,_mm_movehdup_ps(mg));
        mu = _mm_add_ps(mu,_mm_movehl_ps(mu,mu)); mu = _mm_add_ss(mu,_mm_movehdup_ps(mu));
        sum_g = _mm_cvtss_f32(mg); sum_u = _mm_cvtss_f32(mu);
    }
    g = hsum_float_8(_mm512_castps512_ps256(acc)) + sum_g;
    u = hsum_float_8(_mm512_extractf32x8_ps(acc,1)) + sum_u;
}

template<int NT>
__attribute__((target("avx512f,avx512vnni,avx512bw,avx512vl,avx512dq,fma,f16c")))
void q4k_gu_multi(const block_q4_K* gate, const block_q4_K* up, int nb, const void* const* act, float* g, float* u) {
    __m512 acc[NT]; __m256 acc_m[NT];
    for (int t=0; t<NT; ++t) { acc[t]=_mm512_setzero_ps();acc_m[t]=_mm256_setzero_ps(); }
    for (int i=0; i<nb; ++i) {
        const float gd=half_at(gate+i,0),ud=half_at(up+i,0),gm=half_at(gate+i,1),um=half_at(up+i,1);
        const __m256i gs=kq45_scales(gate[i].scales),us=kq45_scales(up[i].scales);
        const __m128i gl=_mm256_castsi256_si128(gs),ul=_mm256_castsi256_si128(us);
        const __m256i gr=_mm256_inserti128_si256(_mm256_castsi128_si256(gl),gl,1);
        const __m256i ur=_mm256_inserti128_si256(_mm256_castsi128_si256(ul),ul,1);
        const __m512i scales=_mm512_inserti64x4(_mm512_castsi256_si512(gr),ur,1);
        const __m256i mins=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(gs,1)),
            _mm256_extracti128_si256(us,1),1);
        __m512i sumi[NT];
        for (int t=0; t<NT; ++t) {
            sumi[t]=_mm512_setzero_si512();
            const auto * y=static_cast<const block_q8_K*>(act[t])+i;
            const __m256i ys=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y->bsums));
            const __m128i pairs=_mm_hadd_epi16(_mm256_castsi256_si128(ys),_mm256_extracti128_si256(ys,1));
            const __m256i pr=_mm256_madd_epi16(mins,_mm256_inserti128_si256(_mm256_castsi128_si256(pairs),pairs,1));
            const __m256 dm=_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(-y->d*gm)),_mm_set1_ps(-y->d*um),1);
            acc_m[t]=_mm256_fmadd_ps(dm,_mm256_cvtepi32_ps(pr),acc_m[t]);
        }
        for (int j=0; j<4; ++j) {
            const __m512i bits=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gate[i].qs)+j)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(up[i].qs)+j),1);
            const __m512i lo=_mm512_and_si512(bits,_mm512_set1_epi8(15));
            const __m512i hi=_mm512_and_si512(_mm512_srli_epi16(bits,4),_mm512_set1_epi8(15));
            const __m512i sl=_mm512_shuffle_epi8(scales,_mm512_load_si512(k_gu1_shuffle[2*j].data()));
            const __m512i sh=_mm512_shuffle_epi8(scales,_mm512_load_si512(k_gu1_shuffle[2*j+1].data()));
            for (int t=0; t<NT; ++t) {
                const auto * y=static_cast<const block_q8_K*>(act[t])+i;
                const __m512i al=_mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y->qs)+2*j));
                const __m512i ah=_mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y->qs)+2*j+1));
                sumi[t]=_mm512_dpwssd_epi32(sumi[t],sl,_mm512_maddubs_epi16(lo,al));
                sumi[t]=_mm512_dpwssd_epi32(sumi[t],sh,_mm512_maddubs_epi16(hi,ah));
            }
        }
        for (int t=0; t<NT; ++t) {
            const auto * y=static_cast<const block_q8_K*>(act[t])+i;
            const __m512 d=_mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(y->d*gd)),_mm256_set1_ps(y->d*ud),1);
            acc[t]=_mm512_fmadd_ps(d,_mm512_cvtepi32_ps(sumi[t]),acc[t]);
        }
    }
    for (int t=0; t<NT; ++t) {
        __m128 mg=_mm256_castps256_ps128(acc_m[t]),mu=_mm256_extractf128_ps(acc_m[t],1);
        mg=_mm_add_ps(mg,_mm_movehl_ps(mg,mg));mg=_mm_add_ss(mg,_mm_movehdup_ps(mg));
        mu=_mm_add_ps(mu,_mm_movehl_ps(mu,mu));mu=_mm_add_ss(mu,_mm_movehdup_ps(mu));
        g[t]=hsum_float_8(_mm512_castps512_ps256(acc[t]))+_mm_cvtss_f32(mg);
        u[t]=hsum_float_8(_mm512_extractf32x8_ps(acc[t],1))+_mm_cvtss_f32(mu);
    }
}

template<int NT>
void dot_rows512_width(int type, const uint8_t* row, int n, const void* const* act, float* res) {
    if (kq512_dpwssd_enabled()) {
        if (type == 12) {
            if (paired_half()) q4k_dot512_dpwssd<NT,false,true>((const block_q4_K*)row,n/QK_K,act,res);
            else if (split_dpwssd()) q4k_dot512_dpwssd<NT,true>((const block_q4_K*)row,n/QK_K,act,res);
            else q4k_dot512_dpwssd<NT>((const block_q4_K*)row,n/QK_K,act,res);
            return;
        }
        if (type == 14) {
            static const bool permute = std::getenv("STRATA_CPU_Q6_PERMUTE") && std::atoi(std::getenv("STRATA_CPU_Q6_PERMUTE")) == 1;
            if (split_dpwssd()) {
                if (permute) q6k_dot512_dpwssd<NT,true,true>((const block_q6_K*)row,n/QK_K,act,res);
                else q6k_dot512_dpwssd<NT,false,true>((const block_q6_K*)row,n/QK_K,act,res);
            } else {
                if (permute) q6k_dot512_dpwssd<NT,true>((const block_q6_K*)row,n/QK_K,act,res);
                else q6k_dot512_dpwssd<NT>((const block_q6_K*)row,n/QK_K,act,res);
            }
            return;
        }
    }
    switch (type) {
        case 12: q4k_dot512<NT>((const block_q4_K*)row, n / QK_K, act, res); break;
        case 13: q5k_dot512<NT>((const block_q5_K*)row, n / QK_K, act, res); break;
        case 14: q6k_dot512<NT>((const block_q6_K*)row, n / QK_K, act, res); break;
    }
}

template<int Bit>
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
inline __m256i q5_high_bit(__m256i high) {
    const __m256i selected = _mm256_and_si256(high, _mm256_set1_epi8(1 << Bit));
    if constexpr (Bit < 4) return _mm256_slli_epi16(selected, 4 - Bit);
    else if constexpr (Bit > 4) return _mm256_srli_epi16(selected, Bit - 4);
    else return selected;
}

// Adjacent rows share Q8 loads, while each keeps the original eight dot lanes
// and four min lanes. Stream one group at a time so decoded groups do not all
// remain live along with the two independent rows' accumulators.
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
void q4k_dot512_rows2(const block_q4_K* x0, const block_q4_K* x1, const block_q8_K* y,
                      int nb, float& out0, float& out1) {
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m128 min0 = _mm_setzero_ps(), min1 = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const __m256i ms0 = kq45_scales(x0[i].scales), ms1 = kq45_scales(x1[i].scales);
        const __m256i bsums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(bsums),_mm256_extracti128_si256(bsums,1));
        const __m128i prod0 = _mm_madd_epi16(_mm256_extracti128_si256(ms0,1),pairs);
        const __m128i prod1 = _mm_madd_epi16(_mm256_extracti128_si256(ms1,1),pairs);
        min0 = _mm_fmadd_ps(_mm_set1_ps(-y[i].d * half_at(&x0[i],1)),_mm_cvtepi32_ps(prod0),min0);
        min1 = _mm_fmadd_ps(_mm_set1_ps(-y[i].d * half_at(&x1[i],1)),_mm_cvtepi32_ps(prod1),min1);
        const __m128i s0 = _mm256_castsi256_si128(ms0), s1 = _mm256_castsi256_si128(ms1);
        const __m256i scales0 = _mm256_inserti128_si256(_mm256_castsi128_si256(s0),s0,1);
        const __m256i scales1 = _mm256_inserti128_si256(_mm256_castsi128_si256(s1),s1,1);
        __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
#define Q4_PAIR_GROUP(J) { \
            const __m256i al = _mm256_loadu_si256((const __m256i*)y[i].qs + 2*(J)); \
            const __m256i ah = _mm256_loadu_si256((const __m256i*)y[i].qs + 2*(J)+1); \
            const __m256i bits0 = _mm256_loadu_si256((const __m256i*)x0[i].qs + (J)); \
            const __m256i bits1 = _mm256_loadu_si256((const __m256i*)x1[i].qs + (J)); \
            const __m256i lo0 = _mm256_and_si256(bits0,m15); \
            const __m256i lo1 = _mm256_and_si256(bits1,m15); \
            const __m256i hi0 = _mm256_and_si256(_mm256_srli_epi16(bits0,4),m15); \
            const __m256i hi1 = _mm256_and_si256(_mm256_srli_epi16(bits1,4),m15); \
            const __m256i sl0 = _mm256_shuffle_epi8(scales0,get_scale_shuffle_k4(2*(J))); \
            const __m256i sh0 = _mm256_shuffle_epi8(scales0,get_scale_shuffle_k4(2*(J)+1)); \
            const __m256i sl1 = _mm256_shuffle_epi8(scales1,get_scale_shuffle_k4(2*(J))); \
            const __m256i sh1 = _mm256_shuffle_epi8(scales1,get_scale_shuffle_k4(2*(J)+1)); \
            sum0 = _mm256_add_epi32(sum0,_mm256_add_epi32(_mm256_madd_epi16(sl0,_mm256_maddubs_epi16(lo0,al)), \
                                                          _mm256_madd_epi16(sh0,_mm256_maddubs_epi16(hi0,ah)))); \
            sum1 = _mm256_add_epi32(sum1,_mm256_add_epi32(_mm256_madd_epi16(sl1,_mm256_maddubs_epi16(lo1,al)), \
                                                          _mm256_madd_epi16(sh1,_mm256_maddubs_epi16(hi1,ah)))); \
        }
        Q4_PAIR_GROUP(0) Q4_PAIR_GROUP(1) Q4_PAIR_GROUP(2) Q4_PAIR_GROUP(3)
#undef Q4_PAIR_GROUP
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * half_at(&x0[i],0)),_mm256_cvtepi32_ps(sum0),acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * half_at(&x1[i],0)),_mm256_cvtepi32_ps(sum1),acc1);
    }
    min0 = _mm_add_ps(min0,_mm_movehl_ps(min0,min0));
    min0 = _mm_add_ss(min0,_mm_movehdup_ps(min0));
    min1 = _mm_add_ps(min1,_mm_movehl_ps(min1,min1));
    min1 = _mm_add_ss(min1,_mm_movehdup_ps(min1));
    out0 = hsum_float_8(acc0) + _mm_cvtss_f32(min0);
    out1 = hsum_float_8(acc1) + _mm_cvtss_f32(min1);
}

// Unlike packing two rows into one ZMM, two YMM accumulators need no lane
// insertion or fold. A Q8 block's loads and pair sums are shared by both rows.
__attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
void q5k_dot512_rows2(const block_q5_K* x0, const block_q5_K* x1, const block_q8_K* y,
                      int nb, float& out0, float& out1) {
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float min0 = 0, min1 = 0;
    for (int i = 0; i < nb; ++i) {
        uint32_t a[4], b[4];
        auto unpack = [] (const uint8_t* source, uint32_t* u) {
            std::memcpy(u, source, 12);
            u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
            const uint32_t old = u[1] & 0x3f3f3f3f;
            u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
            u[2] = old; u[0] &= 0x3f3f3f3f;
        };
        unpack(x0[i].scales,a); unpack(x1[i].scales,b);
        const __m256i ms0 = _mm256_cvtepu8_epi16(_mm_set_epi32(a[3],a[2],a[1],a[0]));
        const __m256i ms1 = _mm256_cvtepu8_epi16(_mm_set_epi32(b[3],b[2],b[1],b[0]));
        const __m256i bsums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(bsums),_mm256_extracti128_si256(bsums,1));
        const __m128i prod0 = _mm_madd_epi16(_mm256_extracti128_si256(ms0,1),pairs);
        const __m128i prod1 = _mm_madd_epi16(_mm256_extracti128_si256(ms1,1),pairs);
        const __m128i zero = _mm_setzero_si128();
        const __m128i total0 = _mm_hadd_epi32(_mm_hadd_epi32(prod0,zero),zero);
        const __m128i total1 = _mm_hadd_epi32(_mm_hadd_epi32(prod1,zero),zero);
        min0 += (-y[i].d * half_at(&x0[i],1)) * float(_mm_cvtsi128_si32(total0));
        min1 += (-y[i].d * half_at(&x1[i],1)) * float(_mm_cvtsi128_si32(total1));
        const __m128i s0 = _mm256_castsi256_si128(ms0), s1 = _mm256_castsi256_si128(ms1);
        const __m256i scales0 = _mm256_inserti128_si256(_mm256_castsi128_si256(s0),s0,1);
        const __m256i scales1 = _mm256_inserti128_si256(_mm256_castsi128_si256(s1),s1,1);
        const __m256i high0 = _mm256_loadu_si256((const __m256i*)x0[i].qh);
        const __m256i high1 = _mm256_loadu_si256((const __m256i*)x1[i].qh);
        __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
        // Explicit groups make shifts and shuffle masks constants. The fifth
        // bit and low nibble are disjoint, so OR equals addition exactly and
        // allows AVX-512's ternary logical instruction to combine the masks.
#define Q5_PAIR_GROUP(J) { \
            const __m256i al = _mm256_loadu_si256((const __m256i*)y[i].qs + 2*(J)); \
            const __m256i ah = _mm256_loadu_si256((const __m256i*)y[i].qs + 2*(J)+1); \
            const __m256i bits0 = _mm256_loadu_si256((const __m256i*)x0[i].qs + (J)); \
            const __m256i bits1 = _mm256_loadu_si256((const __m256i*)x1[i].qs + (J)); \
            const __m256i lo0 = _mm256_or_si256(_mm256_and_si256(bits0,m15),q5_high_bit<2*(J)>(high0)); \
            const __m256i lo1 = _mm256_or_si256(_mm256_and_si256(bits1,m15),q5_high_bit<2*(J)>(high1)); \
            const __m256i hi0 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits0,4),m15),q5_high_bit<2*(J)+1>(high0)); \
            const __m256i hi1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1,4),m15),q5_high_bit<2*(J)+1>(high1)); \
            const __m256i sl0 = _mm256_shuffle_epi8(scales0,get_scale_shuffle_k4(2*(J))); \
            const __m256i sh0 = _mm256_shuffle_epi8(scales0,get_scale_shuffle_k4(2*(J)+1)); \
            const __m256i sl1 = _mm256_shuffle_epi8(scales1,get_scale_shuffle_k4(2*(J))); \
            const __m256i sh1 = _mm256_shuffle_epi8(scales1,get_scale_shuffle_k4(2*(J)+1)); \
            sum0 = _mm256_add_epi32(sum0,_mm256_add_epi32(_mm256_madd_epi16(sl0,_mm256_maddubs_epi16(lo0,al)), \
                                                          _mm256_madd_epi16(sh0,_mm256_maddubs_epi16(hi0,ah)))); \
            sum1 = _mm256_add_epi32(sum1,_mm256_add_epi32(_mm256_madd_epi16(sl1,_mm256_maddubs_epi16(lo1,al)), \
                                                          _mm256_madd_epi16(sh1,_mm256_maddubs_epi16(hi1,ah)))); \
        }
        Q5_PAIR_GROUP(0) Q5_PAIR_GROUP(1) Q5_PAIR_GROUP(2) Q5_PAIR_GROUP(3)
#undef Q5_PAIR_GROUP
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * half_at(&x0[i],0)),_mm256_cvtepi32_ps(sum0),acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d * half_at(&x1[i],0)),_mm256_cvtepi32_ps(sum1),acc1);
    }
    out0 = hsum_float_8(acc0) + min0;
    out1 = hsum_float_8(acc1) + min1;
}
void dot_rows512(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (nt) {
#define KQ_WIDTH(N) case N: dot_rows512_width<N>(type,row,n,act,res); break;
        KQ_WIDTH(1) KQ_WIDTH(2) KQ_WIDTH(3) KQ_WIDTH(4)
        KQ_WIDTH(5) KQ_WIDTH(6) KQ_WIDTH(7) KQ_WIDTH(8)
#undef KQ_WIDTH
    }
}
#endif

}  // namespace

bool kq256_supported(int type) noexcept { return type == 12 || type == 13 || type == 14 || type == 7 || type == 8; }

bool kq512_available() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

bool kq512_vnni_available() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return kq512_available() && __builtin_cpu_supports("avx512vnni");
#else
    return false;
#endif
}

bool kq512_dpwssd_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_CPU_DPWSSD");
        return value && std::strcmp(value,"1") == 0 && kq512_vnni_available();
    }();
    return enabled;
}

static bool single_q4_vnni() {
    static const bool enabled = [] {
        const char * value = std::getenv("STRATA_CPU_VNNI");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && kq512_vnni_available();
}

bool kq512_q4_vnni_f16c_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_KQ512_Q4_VNNI_F16C");
        if (!value || std::strcmp(value,"0") == 0) return false;
        if (std::strcmp(value,"1") != 0)
            throw std::invalid_argument("STRATA_KQ512_Q4_VNNI_F16C must be 0 or 1");
        return true;
    }();
    return enabled;
}

void kq512_half_pair_f16c(const void* fp16_pair, float* fp32_pair) {
#if defined(__GNUC__) || defined(__clang__)
    float converted[4];
    _mm_storeu_ps(converted,half_pair_f16c(fp16_pair));
    std::memcpy(fp32_pair,converted,2*sizeof(float));
#else
    fp32_pair[0] = half_at(fp16_pair,0);
    fp32_pair[1] = half_at(fp16_pair,1);
#endif
}

void kq512_q4_coefficients_f16c(const void* fp16_pair, float activation_d, float* coefficients) {
#if defined(__GNUC__) || defined(__clang__)
    float converted[4];
    _mm_storeu_ps(converted,q4_coefficients_f16c(fp16_pair,activation_d));
    std::memcpy(coefficients,converted,2*sizeof(float));
#else
    coefficients[0] = activation_d * half_at(fp16_pair,0);
    coefficients[1] = -activation_d * half_at(fp16_pair,1);
#endif
}

void kq512_q4_vnni_rows(const uint8_t* weights, size_t row_bytes, int columns, const void* activation,
                       float* output, int first, int last) {
#if defined(__GNUC__) || defined(__clang__)
    if (kq512_vnni_available()) {
        const auto rows = [&]<bool F16C_PAIR>() {
            for (int row=first; row<last; ++row)
                output[row] = q4k_dot512_vnni<F16C_PAIR>((const block_q4_K*)(weights+size_t(row)*row_bytes),
                                                        (const block_q8_K*)activation,columns/QK_K);
        };
        if (kq512_q4_vnni_f16c_enabled()) rows.template operator()<true>();
        else rows.template operator()<false>();
        return;
    }
#endif
    const void* act[1] = {activation}; float* out[1] = {output};
    kq256_rows(12,weights,row_bytes,columns,act,1,out,first,last);
}

void kq512_q4_vnni_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int columns, const void* activation,
                          float* ff, int first, int last, float limit, const uint8_t* separate_up) {
#if defined(__GNUC__) || defined(__clang__)
    if (kq512_vnni_available()) {
        const uint8_t* up = separate_up ? separate_up : blob+up_off;
        const auto rows = [&]<bool F16C_PAIR>() {
            for (int row=first; row<last; ++row) {
                float g = q4k_dot512_vnni<F16C_PAIR>((const block_q4_K*)(blob+size_t(row)*gu_row),
                                                    (const block_q8_K*)activation,columns/QK_K);
                float u = q4k_dot512_vnni<F16C_PAIR>((const block_q4_K*)(up+size_t(row)*gu_row),
                                                    (const block_q8_K*)activation,columns/QK_K);
                if (limit > 0) { g = std::fmin(g,limit); u = std::fmax(-limit,std::fmin(u,limit)); }
                ff[row] = (g/(1.f+std::exp(-g)))*u;
            }
        };
        if (kq512_q4_vnni_f16c_enabled()) rows.template operator()<true>();
        else rows.template operator()<false>();
        return;
    }
#endif
    const void* act[1] = {activation}; float* out[1] = {ff};
    kq256_gu_rows(12,blob,gu_row,up_off,columns,act,1,out,first,last,limit,separate_up);
}

void kq512_q5_rows2(const uint8_t* weights, size_t row_bytes, int columns, const void* activation,
                    float* output, int first, int last) {
#if defined(__GNUC__) || defined(__clang__)
    if (kq512_available()) {
        const auto& prefetch = native_weight_prefetch_setting();
        const int ahead = 2*prefetch.rows;
        if (prefetch.enabled) {
            const int count = std::min(ahead,last-first);
            for (int row=first; row<first+count; ++row)
                native_prefetch_weight_row(weights,row_bytes,row,first,last);
        }
        int row = first;
        for (; row + 1 < last; row += 2) {
            if (prefetch.enabled && ahead < last-row) {
                native_prefetch_weight_row(weights,row_bytes,row+ahead,first,last);
                if (ahead+1 < last-row)
                    native_prefetch_weight_row(weights,row_bytes,row+ahead+1,first,last);
            }
            q5k_dot512_rows2((const block_q5_K*)(weights + size_t(row)*row_bytes),
                            (const block_q5_K*)(weights + size_t(row+1)*row_bytes),
                            (const block_q8_K*)activation,columns/QK_K,output[row],output[row+1]);
        }
        if (row < last) {
            const void* act[1] = {activation};
            q5k_dot512<1>((const block_q5_K*)(weights + size_t(row)*row_bytes),columns/QK_K,act,output+row);
        }
        return;
    }
#endif
    const void* act[1] = {activation}; float* out[1] = {output};
    kq256_rows(13,weights,row_bytes,columns,act,1,out,first,last);
}

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // each row is read once for all `nt` (<= 8) tokens: the router (2.6 MB per layer) streams once per prediction
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int c = 0; c + 8 <= cols; c += 8) {
            const __m256 wf = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
            for (int t = 0; t < nt; ++t) acc[t] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + (size_t) t * cols + c), acc[t]);
        }
        for (int t = 0; t < nt; ++t) out[(size_t) t * rows + r] = hsum_float_8(acc[t]);   // cols % 8 == 0 (2560)
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, (separate_up ? separate_up : blob + up_off) + (size_t) r * gu_row, n, act + t0, m, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) {
                if (limit > 0) { g[t] = std::fmin(g[t], limit); u[t] = std::fmax(-limit, std::fmin(u[t], limit)); }
                ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

void kq512_gu1_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act,
                    float* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
#if defined(__GNUC__) || defined(__clang__)
    if ((type == 12 || type == 13) && kq512_available()) {
        const uint8_t* up = separate_up ? separate_up : blob + up_off;
        const bool dpwssd = kq512_dpwssd_enabled();
        const auto& prefetch = native_weight_prefetch_setting();
        const auto hint = [&](int row) {
            native_prefetch_weight_row(blob,gu_row,row,r0,r1);
            native_prefetch_weight_row(up,gu_row,row,r0,r1);
        };
        if (prefetch.enabled)
            for (int row = r0; row < r1 && row-r0 < prefetch.rows; ++row) hint(row);
        for (int r = r0; r < r1; ++r) {
            if (prefetch.enabled && prefetch.rows < r1-r) hint(r+prefetch.rows);
            float g, u;
            if (type == 12 && dpwssd && paired_half()) kq45_gu1_dot_dpwssd<block_q4_K,false,false,true>((const block_q4_K*)(blob+r*gu_row),
                (const block_q4_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            else if (type == 12 && dpwssd && split_dpwssd()) kq45_gu1_dot_dpwssd<block_q4_K,false,true>((const block_q4_K*)(blob+r*gu_row),
                (const block_q4_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            else if (type == 12 && dpwssd) kq45_gu1_dot_dpwssd<block_q4_K,false>((const block_q4_K*)(blob+r*gu_row),
                (const block_q4_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            else if (type == 13 && dpwssd) kq45_gu1_dot_dpwssd<block_q5_K,true>((const block_q5_K*)(blob+r*gu_row),
                (const block_q5_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            else if (type == 12) kq45_gu1_dot<block_q4_K,false>((const block_q4_K*)(blob+r*gu_row),
                (const block_q4_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            else kq45_gu1_dot<block_q5_K,true>((const block_q5_K*)(blob+r*gu_row),
                (const block_q5_K*)(up+r*gu_row),n/QK_K,(const block_q8_K*)act,g,u);
            if (limit > 0) { g = std::fmin(g,limit); u = std::fmax(-limit,std::fmin(u,limit)); }
            ff[r] = (g / (1.f + std::exp(-g))) * u;
        }
        return;
    }
#endif
    const void* ap[1] = {act}; float* out[1] = {ff};
    kq256_gu_rows(type,blob,gu_row,up_off,n,ap,1,out,r0,r1,limit,separate_up);
}

void kq512_gu2_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act,
                    float* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    int r = r0;
#if defined(__GNUC__) || defined(__clang__)
    if ((type == 12 || type == 13) && kq512_available()) {
        const uint8_t* up = separate_up ? separate_up : blob + up_off;
        for (; r + 1 < r1; r += 2) {
            float g0, g1, u0, u1;
            // Gate and Up are separate pair calls. Four dots never need live
            // accumulators at once, and no ZMM lane insertion/fold is needed.
            if (type == 12) {
                q4k_dot512_rows2((const block_q4_K*)(blob + size_t(r)*gu_row),
                    (const block_q4_K*)(blob + size_t(r+1)*gu_row),(const block_q8_K*)act,n/QK_K,g0,g1);
                q4k_dot512_rows2((const block_q4_K*)(up + size_t(r)*gu_row),
                    (const block_q4_K*)(up + size_t(r+1)*gu_row),(const block_q8_K*)act,n/QK_K,u0,u1);
            } else {
                q5k_dot512_rows2((const block_q5_K*)(blob + size_t(r)*gu_row),
                    (const block_q5_K*)(blob + size_t(r+1)*gu_row),(const block_q8_K*)act,n/QK_K,g0,g1);
                q5k_dot512_rows2((const block_q5_K*)(up + size_t(r)*gu_row),
                    (const block_q5_K*)(up + size_t(r+1)*gu_row),(const block_q8_K*)act,n/QK_K,u0,u1);
            }
            if (limit > 0) {
                g0 = std::fmin(g0,limit); u0 = std::fmax(-limit,std::fmin(u0,limit));
                g1 = std::fmin(g1,limit); u1 = std::fmax(-limit,std::fmin(u1,limit));
            }
            ff[r] = (g0 / (1.f + std::exp(-g0))) * u0;
            ff[r+1] = (g1 / (1.f + std::exp(-g1))) * u1;
        }
    }
#endif
    // The unpaired tail and unsupported CPUs keep the existing unpaired dot.
    if (r < r1) {
        const void* ap[1] = {act}; float* out[1] = {ff};
        kq256_gu_rows(type,blob,gu_row,up_off,n,ap,1,out,r,r1,limit,separate_up);
    }
}

void kq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1, float limit, const uint8_t* separate_up) {
    static const bool multi = std::getenv("STRATA_CPU_GU_MULTI") && std::atoi(std::getenv("STRATA_CPU_GU_MULTI")) == 1;
#if defined(__GNUC__) || defined(__clang__)
    if (multi && type == 12 && (nt == 2 || nt == 3) && kq512_dpwssd_enabled()) {
        const uint8_t * up=separate_up ? separate_up : blob+up_off;
        float g[3],u[3];
        for (int row=r0; row<r1; ++row) {
            const auto * gate_row=reinterpret_cast<const block_q4_K*>(blob+size_t(row)*gu_row);
            const auto * up_row=reinterpret_cast<const block_q4_K*>(up+size_t(row)*gu_row);
            if (nt == 2) q4k_gu_multi<2>(gate_row,up_row,n/QK_K,act,g,u);
            else q4k_gu_multi<3>(gate_row,up_row,n/QK_K,act,g,u);
            for (int t=0; t<nt; ++t) {
                if (limit > 0) { g[t]=std::fmin(g[t],limit);u[t]=std::fmax(-limit,std::fmin(u[t],limit)); }
                ff[t][row]=(g[t]/(1.f+std::exp(-g[t])))*u[t];
            }
        }
        return;
    }
#endif
    static const bool singleton_gu2 = std::getenv("STRATA_CPU_GU2") && std::atoi(std::getenv("STRATA_CPU_GU2")) != 0;
    if ((type == 12 || type == 13) && nt == 1 && singleton_gu2) {
        kq512_gu2_rows(type, blob, gu_row, up_off, n, act[0], ff[0], r0, r1, limit, separate_up);
        return;
    }
    static const bool singleton_gu1 = std::getenv("STRATA_CPU_GU1") && std::atoi(std::getenv("STRATA_CPU_GU1")) != 0;
    if ((type == 12 || type == 13) && nt == 1 && singleton_gu1) {
        kq512_gu1_rows(type, blob, gu_row, up_off, n, act[0], ff[0], r0, r1, limit, separate_up);
        return;
    }
    if (type == 12 && nt == 1 && single_q4_vnni()) {
        kq512_q4_vnni_gu_rows(blob, gu_row, up_off, n, act[0], ff[0], r0, r1, limit, separate_up);
        return;
    }
    if (type < 12 || type > 14 || !kq512_available()) {
        kq256_gu_rows(type,blob,gu_row,up_off,n,act,nt,ff,r0,r1,limit,separate_up);
        return;
    }
#if defined(__GNUC__) || defined(__clang__)
    const uint8_t* up = separate_up ? separate_up : blob + up_off;
    const auto& prefetch = native_weight_prefetch_setting();
    const auto hint = [&](int row) {
        native_prefetch_weight_row(blob,gu_row,row,r0,r1);
        native_prefetch_weight_row(up,gu_row,row,r0,r1);
    };
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        if (prefetch.enabled)
            for (int row = r0; row < r1 && row-r0 < prefetch.rows; ++row) hint(row);
        for (int r = r0; r < r1; ++r) {
            if (prefetch.enabled && prefetch.rows < r1-r) hint(r+prefetch.rows);
            dot_rows512(type, blob + (size_t)r * gu_row, n, act + t0, m, g);
            dot_rows512(type, up + (size_t)r * gu_row, n, act + t0, m, u);
            for (int t = 0; t < m; ++t) {
                if (limit > 0) { g[t] = std::fmin(g[t], limit); u[t] = std::fmax(-limit, std::fmin(u[t], limit)); }
                ff[t0+t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
    }
#else
    kq256_gu_rows(type,blob,gu_row,up_off,n,act,nt,ff,r0,r1,limit,separate_up);
#endif
}
void kq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1) {
    if (type == 12 && nt == 1 && single_q4_vnni()) {
        kq512_q4_vnni_rows(w, row_bytes, n, act[0], out[0], r0, r1);
        return;
    }
    if (type < 12 || type > 14 || !kq512_available()) {
        kq256_rows(type,w,row_bytes,n,act,nt,out,r0,r1);
        return;
    }
#if defined(__GNUC__) || defined(__clang__)
    const auto& prefetch = native_weight_prefetch_setting();
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        if (prefetch.enabled)
            for (int row = r0; row < r1 && row-r0 < prefetch.rows; ++row)
                native_prefetch_weight_row(w,row_bytes,row,r0,r1);
        for (int r = r0; r < r1; ++r) {
            if (prefetch.enabled && prefetch.rows < r1-r)
                native_prefetch_weight_row(w,row_bytes,r+prefetch.rows,r0,r1);
            dot_rows512(type,w+(size_t)r*row_bytes,n,act+t0,m,s);
            for (int t = 0; t < m; ++t) out[t0+t][r] = s[t];
        }
    }
#else
    kq256_rows(type,w,row_bytes,n,act,nt,out,r0,r1);
#endif
}
}  // namespace strata::deepseek41::cpu
