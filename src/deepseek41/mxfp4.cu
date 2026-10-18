#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace strata::deepseek41::cuda {
namespace {
__device__ __constant__ int8_t fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
__device__ float weight(const uint8_t* row,int i) {
    const uint8_t* block = row+size_t(i/32)*17;
    const uint8_t bits = block[1+i%16]; const int q = (bits >> ((i%32)/16*4))&15;
    // Mirrors GGML_E8M0_TO_FP32_HALF, including its two subnormal exponents.
    const uint32_t scale = block[0] < 2 ? 0x00200000u << block[0] : uint32_t(block[0]-1) << 23;
    return float(fp4[q])*__uint_as_float(scale);
}
__device__ float reduce(float x) { for (int n = 16; n; n >>= 1) x += __shfl_down_sync(0xffffffffu,x,n); return x; }
__global__ void mmv(const uint8_t* w,const float* x,float* y,int ni,int no) {
    const int row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x%32, t = blockIdx.y;
    if (row >= no) return;
    w += size_t(row)*size_t(ni/32)*17; x += size_t(t)*ni;
    float v = 0; for (int i = lane; i < ni; i += 32) v = fmaf(weight(w,i),x[i],v);
    v = reduce(v); if (!lane) y[size_t(t)*no+row] = v;
}
__global__ void gu_kernel(const uint8_t* gate,const uint8_t* up,const float* x,const int32_t* ids,
                          float* hidden,int ni,int nf,int k,float limit) {
    const int row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x%32, entry = blockIdx.y;
    if (row >= nf) return;
    const int token = entry/k, expert = ids[entry];
    const size_t offset = (size_t(expert)*nf+row)*size_t(ni/32)*17;
    gate += offset; up += offset; x += size_t(token)*ni;
    float g = 0, u = 0;
    for (int i = lane; i < ni; i += 32) { const float v = x[i]; g = fmaf(weight(gate,i),v,g); u = fmaf(weight(up,i),v,u); }
    g = reduce(g); u = reduce(u);
    if (!lane) { g = fminf(g,limit); u = fmaxf(-limit,fminf(u,limit)); hidden[size_t(entry)*nf+row] = (g/(1+expf(-g)))*u; }
}
__global__ void down_kernel(const uint8_t* w,const float* hidden,const int32_t* ids,float* out,int nf,int no) {
    const int row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x%32, entry = blockIdx.y;
    if (row >= no) return;
    w += (size_t(ids[entry])*no+row)*size_t(nf/32)*17; hidden += size_t(entry)*nf;
    float v = 0; for (int i = lane; i < nf; i += 32) v = fmaf(weight(w,i),hidden[i],v);
    v = reduce(v); if (!lane) out[size_t(entry)*no+row] = v;
}
cudaStream_t checked(void* s,int ni,int no,int t) {
    if (!s || ni < 32 || ni%32 || no < 1 || t < 1 || t > 8) throw std::invalid_argument("DSpark MXFP4 geometry");
    return static_cast<cudaStream_t>(s);
}
}
void mxfp4_mmv(const void* w,const float* x,float* y,int ni,int no,int t,void* s) {
    mmv<<<dim3((no+3)/4,t),128,0,checked(s,ni,no,t)>>>(static_cast<const uint8_t*>(w),x,y,ni,no);
}
void mxfp4_gu(const void* g,const void* u,const float* x,const int32_t* ids,float* h,int ni,int nf,int k,int t,float limit,void* s) {
    if (k < 1 || k > 64) throw std::invalid_argument("DSpark MXFP4 routing");
    gu_kernel<<<dim3((nf+3)/4,k*t),128,0,checked(s,ni,nf,t)>>>(static_cast<const uint8_t*>(g),static_cast<const uint8_t*>(u),x,ids,h,ni,nf,k,limit);
}
void mxfp4_down(const void* w,const float* h,const int32_t* ids,float* out,int nf,int no,int k,int t,void* s) {
    if (k < 1 || k > 64) throw std::invalid_argument("DSpark MXFP4 routing");
    down_kernel<<<dim3((no+3)/4,k*t),128,0,checked(s,nf,no,t)>>>(static_cast<const uint8_t*>(w),h,ids,out,nf,no);
}
}  // namespace strata::deepseek41::cuda
