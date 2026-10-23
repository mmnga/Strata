#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace strata::deepseek41::cuda {
namespace {
__device__ __constant__ int8_t fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
template<bool Alu> __device__ float weight(const uint8_t* row,int i) {
    const uint8_t* block = row+size_t(i/32)*17;
    const uint8_t bits = block[1+i%16]; const int q = (bits >> ((i%32)/16*4))&15;
    // Mirrors GGML_E8M0_TO_FP32_HALF, including its two subnormal exponents.
    const uint32_t scale = block[0] < 2 ? 0x00200000u << block[0] : uint32_t(block[0]-1) << 23;
    if constexpr (Alu) {
        // The original signed integers, including positive zero for nibble 8.
        // Avoid a lane-dependent constant-memory table lookup.
        const int magnitude = q&7;
        const int value = magnitude < 4 ? magnitude : (2+(magnitude&1)) << ((magnitude>>1)-1);
        return float((q&8) ? -value : value)*__uint_as_float(scale);
    } else return float(fp4[q])*__uint_as_float(scale);
}
__device__ float reduce(float x) { for (int n = 16; n; n >>= 1) x += __shfl_down_sync(0xffffffffu,x,n); return x; }
template<bool Alu> __global__ void gu_kernel(const uint8_t* gate,const uint8_t* up,const float* x,const int32_t* ids,
                          float* hidden,int ni,int nf,int k,float limit) {
    const int row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x%32, entry = blockIdx.y;
    if (row >= nf) return;
    const int token = entry/k, expert = ids[entry];
    const size_t offset = (size_t(expert)*nf+row)*size_t(ni/32)*17;
    gate += offset; up += offset; x += size_t(token)*ni;
    float g = 0, u = 0;
    for (int i = lane; i < ni; i += 128) {
        const float v0=x[i],v1=x[i+32],v2=x[i+64],v3=x[i+96];
        const float g0=weight<Alu>(gate,i),u0=weight<Alu>(up,i);
        const float g1=weight<Alu>(gate,i+32),u1=weight<Alu>(up,i+32);
        const float g2=weight<Alu>(gate,i+64),u2=weight<Alu>(up,i+64);
        const float g3=weight<Alu>(gate,i+96),u3=weight<Alu>(up,i+96);
        g=fmaf(g0,v0,g);u=fmaf(u0,v0,u);
        g=fmaf(g1,v1,g);u=fmaf(u1,v1,u);
        g=fmaf(g2,v2,g);u=fmaf(u2,v2,u);
        g=fmaf(g3,v3,g);u=fmaf(u3,v3,u);
    }
    g = reduce(g); u = reduce(u);
    if (!lane) { g = fminf(g,limit); u = fmaxf(-limit,fminf(u,limit)); hidden[size_t(entry)*nf+row] = (g/(1+expf(-g)))*u; }
}
template<bool Alu> __global__ void down_kernel(const uint8_t* w,const float* hidden,const int32_t* ids,float* out,int nf,int no) {
    const int row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x%32, entry = blockIdx.y;
    if (row >= no) return;
    w += (size_t(ids[entry])*no+row)*size_t(nf/32)*17; hidden += size_t(entry)*nf;
    float v = 0; for (int i = lane; i < nf; i += 128) {
        const float h0=hidden[i],h1=hidden[i+32],h2=hidden[i+64],h3=hidden[i+96];
        const float w0=weight<Alu>(w,i),w1=weight<Alu>(w,i+32),w2=weight<Alu>(w,i+64),w3=weight<Alu>(w,i+96);
        v=fmaf(w0,h0,v);v=fmaf(w1,h1,v);v=fmaf(w2,h2,v);v=fmaf(w3,h3,v);
    }
    v = reduce(v); if (!lane) out[size_t(entry)*no+row] = v;
}
cudaStream_t checked(void* s,int ni,int no,int t) {
    if (!s || ni < 32 || ni%32 || no < 1 || t < 1 || t > 8) throw std::invalid_argument("DSpark MXFP4 geometry");
    return static_cast<cudaStream_t>(s);
}
}
void mxfp4_pipe_gu(const void* g,const void* u,const float* x,const int32_t* ids,float* h,int ni,int nf,int k,int t,float limit,void* s,bool alu) {
    if (ni%128) throw std::invalid_argument("DSpark pipelined GU width");
    if (k < 1 || k > 64) throw std::invalid_argument("DSpark MXFP4 routing");
    const auto stream = checked(s,ni,nf,t);
    if (alu) gu_kernel<true><<<dim3((nf+3)/4,k*t),128,0,stream>>>(static_cast<const uint8_t*>(g),static_cast<const uint8_t*>(u),x,ids,h,ni,nf,k,limit);
    else gu_kernel<false><<<dim3((nf+3)/4,k*t),128,0,stream>>>(static_cast<const uint8_t*>(g),static_cast<const uint8_t*>(u),x,ids,h,ni,nf,k,limit);
}
void mxfp4_pipe_down(const void* w,const float* h,const int32_t* ids,float* out,int nf,int no,int k,int t,void* s,bool alu) {
    if (nf%128) throw std::invalid_argument("DSpark pipelined Down width");
    if (k < 1 || k > 64) throw std::invalid_argument("DSpark MXFP4 routing");
    const auto stream = checked(s,nf,no,t);
    if (alu) down_kernel<true><<<dim3((no+3)/4,k*t),128,0,stream>>>(static_cast<const uint8_t*>(w),h,ids,out,nf,no);
    else down_kernel<false><<<dim3((no+3)/4,k*t),128,0,stream>>>(static_cast<const uint8_t*>(w),h,ids,out,nf,no);
}
}  // namespace strata::deepseek41::cuda
