#include "strata/deepseek41/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <climits>
#include <stdexcept>

namespace strata::deepseek41::cuda {
namespace {
constexpr int threads = 256;
cudaStream_t stream_cast(void* stream) {
    if (!stream) throw std::invalid_argument("DeepSeek CUDA requires an explicit stream");
    return static_cast<cudaStream_t>(stream);
}
__device__ float sum(float v) {
    __shared__ float partial[8];
    for (int n = 16; n; n >>= 1) v += __shfl_down_sync(0xffffffffu, v, n);
    if (!(threadIdx.x & 31)) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
    if (threadIdx.x < 32) for (int n = 16; n; n >>= 1) v += __shfl_down_sync(0xffffffffu, v, n);
    if (!threadIdx.x) partial[0] = v;
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}
__device__ float sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
__device__ float maximum(float v) {
    __shared__ float partial[8];
    for (int n = 16; n; n >>= 1) v = fmaxf(v,__shfl_down_sync(0xffffffffu,v,n));
    if (!(threadIdx.x & 31)) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? partial[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32) for (int n = 16; n; n >>= 1) v = fmaxf(v,__shfl_down_sync(0xffffffffu,v,n));
    if (!threadIdx.x) partial[0] = v;
    __syncthreads();
    const float result = partial[0];
    __syncthreads();
    return result;
}
__global__ void rms_kernel(const float* x, const float* w, float* y, int width, float eps) {
    x += size_t(blockIdx.x)*width; y += size_t(blockIdx.x)*width;
    float s = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) s += x[i]*x[i];
    s = rsqrtf(sum(s)/width+eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) y[i] = (x[i]*s)*(w ? w[i] : 1.0f);
}
__global__ void init_kernel(const float* x, float* r, float* pre, int hidden) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < hidden*4; i += blockDim.x) r[size_t(t)*hidden*4+i] = x[size_t(t)*hidden+i%hidden];
    if (threadIdx.x < 4) pre[t*4+threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}
__global__ void coeff_kernel(const float* mix, const float* scale, const float* base,
                             float* pre, float* post, float* comb, float eps, int iterations) {
    if (threadIdx.x) return;
    const int t = blockIdx.x;
    mix += t*24; pre += t*4; post += t*4; comb += t*16;
    for (int h = 0; h < 4; ++h) {
        pre[h] = sigmoid(fmaf(mix[h],scale[0],base[h]))+eps;
        post[h] = 2*sigmoid(fmaf(mix[4+h],scale[1],base[4+h]));
    }
    float c[16];
    for (int src = 0; src < 4; ++src) {
        float maximum = -INFINITY, total = 0;
        for (int dst = 0; dst < 4; ++dst) {
            const int i = dst+4*src;
            c[i] = fmaf(mix[8+i],scale[2],base[8+i]); maximum = fmaxf(maximum,c[i]);
        }
        for (int dst = 0; dst < 4; ++dst) { const int i = dst+4*src; c[i] = expf(c[i]-maximum); total += c[i]; }
        const float inv = 1.0f/total;
        for (int dst = 0; dst < 4; ++dst) c[dst+4*src] = c[dst+4*src]*inv+eps;
    }
    for (int it = 0; it < iterations; ++it) {
        if (it) for (int src = 0; src < 4; ++src) {
            float total = eps;
            for (int dst = 0; dst < 4; ++dst) total += c[dst+4*src];
            const float inv = 1.0f/total;
            for (int dst = 0; dst < 4; ++dst) c[dst+4*src] *= inv;
        }
        for (int dst = 0; dst < 4; ++dst) {
            float total = eps;
            for (int src = 0; src < 4; ++src) total += c[dst+4*src];
            const float inv = 1.0f/total;
            for (int src = 0; src < 4; ++src) c[dst+4*src] *= inv;
        }
    }
    for (int i = 0; i < 16; ++i) comb[i] = c[i];
}
__global__ void pre_kernel(const float* r, const float* pre, float* y, int hidden) {
    const int t = blockIdx.y, i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= hidden) return;
    r += size_t(t)*hidden*4+i; pre += t*4;
    float v = 0;
    for (int h = 0; h < 4; ++h) v = fmaf(r[h*hidden],pre[h],v);
    y[size_t(t)*hidden+i] = v;
}
__global__ void post_kernel(const float* x, const float* r, const float* post, const float* c, float* y, int hidden) {
    const int t = blockIdx.y, i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= hidden) return;
    x += size_t(t)*hidden; r += size_t(t)*hidden*4+i; y += size_t(t)*hidden*4+i;
    post += t*4; c += t*16;
    // Reading all four source values before writing makes in-place residual safe.
    float src[4]; for (int h = 0; h < 4; ++h) src[h] = r[h*hidden];
    for (int dst = 0; dst < 4; ++dst) {
        float v = __fmul_rn(x[i],post[dst]);
        for (int h = 0; h < 4; ++h) v = fmaf(src[h],c[dst+4*h],v);
        y[dst*hidden] = v;
    }
}
__global__ void engram_kernel(float* r, const float* kv, const float* qw, const float* kw,
                              float* gates, int hidden, float eps) {
    const int h = blockIdx.x%4, t = blockIdx.x/4;
    r += (size_t(t)*4+h)*hidden; kv += size_t(t)*hidden*5;
    const float* key = kv+size_t(h)*hidden;
    qw += size_t(h)*hidden; kw += size_t(h)*hidden;
    float xx = 0, kk = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) { xx += r[i]*r[i]; kk += key[i]*key[i]; }
    const float xn = rsqrtf(sum(xx)/hidden+eps), kn = rsqrtf(sum(kk)/hidden+eps);
    float dot = 0;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dot += ((r[i]*xn)*qw[i])*((key[i]*kn)*kw[i]);
    const float score = sum(dot)*rsqrtf(float(hidden));
    const float gate = sigmoid(copysignf(sqrtf(fmaxf(fabsf(score),1e-6f)),score));
    if (!threadIdx.x && gates) gates[t*4+h] = gate;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) r[i] += kv[hidden*4+i]*gate;
}
__global__ void route_kernel(const float* logits, const float* bias, int32_t* ids, float* weights,
                             int experts, int k, float scale) {
    __shared__ float probs[512], values[256], selected[64];
    __shared__ int indices[256], winner;
    const int lane = threadIdx.x, t = blockIdx.x;
    logits += size_t(t)*experts; ids += t*k; weights += t*k;
    for (int i = lane; i < experts; i += blockDim.x) {
        const float v = logits[i]; probs[i] = sqrtf(fmaxf(v,0.0f)+log1pf(expf(-fabsf(v))));
    }
    __syncthreads();
    for (int rank = 0; rank < k; ++rank) {
        float best = -INFINITY; int id = INT_MAX;
        for (int i = lane; i < experts; i += blockDim.x) {
            bool used = false; for (int j = 0; j < rank; ++j) used |= ids[j] == i;
            const float v = probs[i]+bias[i];
            if (!used && (v > best || (v == best && i < id))) { best = v; id = i; }
        }
        values[lane] = best; indices[lane] = id; __syncthreads();
        for (int n = 128; n; n >>= 1) {
            if (lane < n) {
                const float v = values[lane+n]; const int j = indices[lane+n];
                if (v > values[lane] || (v == values[lane] && j < indices[lane])) { values[lane] = v; indices[lane] = j; }
            }
            __syncthreads();
        }
        if (!lane) { winner = indices[0]; ids[rank] = winner; selected[rank] = probs[winner]; }
        __syncthreads();
    }
    if (!lane) {
        float total = 0; for (int i = 0; i < k; ++i) total += selected[i];
        const float inv = 1.0f/fmaxf(total,6.103515625e-5f);
        for (int i = 0; i < k; ++i) weights[i] = (selected[i]*inv)*scale;
    }
}
__global__ void swiglu_kernel(const float* g, const float* u, float* y, int n, float limit) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) { const float v = fminf(g[i],limit); y[i] = (v*sigmoid(v))*fminf(fmaxf(u[i],-limit),limit); }
}
__global__ void combine_kernel(const float* parts, const float* w, const float* shared, float* y, int h, int k) {
    const int t = blockIdx.y, i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= h) return;
    parts += size_t(t)*k*h+i; w += t*k;
    float v = 0; for (int j = 0; j < k; ++j) v += parts[size_t(j)*h]*w[j];
    y[size_t(t)*h+i] = v+(shared ? shared[size_t(t)*h+i] : 0.0f);
}
template<bool System=false>
__global__ void combine_sources_kernel(const float* device_parts,const float* host_parts,const uint64_t* gpu_mask,
                                       const float* w,const float* shared,float* y,int h,int k) {
    const int t=blockIdx.y, i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=h) return;
    const uint64_t mask=*gpu_mask; w+=t*k;
    float v=0;
    for(int j=0;j<k;++j) {
        const int row=t*k+j;
        float part;
        if constexpr(System) {
            const size_t at=size_t(row)*h+i;
            part=(mask & (uint64_t(1)<<row)) ? device_parts[at] :
                reinterpret_cast<const volatile float*>(host_parts)[at];
        } else {
            const float* parts=(mask & (uint64_t(1)<<row)) ? device_parts : host_parts;
            part=parts[size_t(row)*h+i];
        }
        v+=part*w[j];
    }
    y[size_t(t)*h+i]=v+(shared ? shared[size_t(t)*h+i] : 0.0f);
}
__global__ void dense_kernel(const float* w, const float* x, float* y, int n_in, int n_out) {
    const int row = blockIdx.x, t = blockIdx.y;
    w += size_t(row)*n_in; x += size_t(t)*n_in;
    float v = 0; for (int i = threadIdx.x; i < n_in; i += blockDim.x) v = fmaf(w[i],x[i],v);
    v = sum(v); if (!threadIdx.x) y[size_t(t)*n_out+row] = v;
}
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) out[i] = x[i]+y[i];
}
__global__ void strided_copy_kernel(const float* x,float* y,int width,int xs,int ys) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x, row = blockIdx.y;
    if (i < width) y[size_t(row)*ys+i] = x[size_t(row)*xs+i];
}
__global__ void gather_row_kernel(const float* rows,const int32_t* id,float* out,int width,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<width) out[i]=(*id>=0 && *id<count) ? rows[size_t(*id)*width+i] : 0.0f;
}
__global__ void draft_confidence_kernel(const float* hidden,const float* markov,const float* w,
                                      const float* bias,float* out,int h,int m) {
    float v=0;
    for (int i=threadIdx.x;i<h+m;i+=blockDim.x) v=fmaf(w[i],i<h ? hidden[i] : markov[i-h],v);
    v=sum(v);
    if (!threadIdx.x) *out=sigmoid(v+(bias ? *bias : 0.0f));
}
__global__ void mean_kernel(const float* r,float* y,int h,int stride) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x, t = blockIdx.y;
    if (i >= h) return;
    r += size_t(t)*h*4+i;
    float v = r[0]; for (int s = 1; s < 4; ++s) v += r[size_t(s)*h];
    y[size_t(t)*stride+i] = v*0.25f;
}
__global__ void rope_kernel(float* x, int d, int heads, const int32_t* pos, Rope c,
                            float low, float high, bool inverse) {
    const int pair = blockIdx.x*blockDim.x+threadIdx.x;
    if (pair >= heads*c.dim/2) return;
    const int h = pair/(c.dim/2), i = pair%(c.dim/2), t = blockIdx.y;
    const float theta_scale = powf(c.base,-2.0f/c.dim);
    const float extrap = float(pos[t])*powf(theta_scale,float(i));
    float theta = c.scale*extrap;
    if (c.yarn) {
        const float ramp = 1.0f-fminf(1.0f,fmaxf(0.0f,(i-low)/fmaxf(0.001f,high-low)));
        theta = theta*(1.0f-ramp)+extrap*ramp;
    }
    // DeepSeek's attention factor cancels YaRN's magnitude factor, so both
    // forward and inverse rotations have unit magnitude, also in long context.
    const float cs = cosf(theta), sn = sinf(theta)*(inverse ? -1.0f : 1.0f);
    x += (size_t(t)*heads+h)*d+d-c.dim+2*i;
    const float a = x[0], b = x[1]; x[0] = a*cs-b*sn; x[1] = a*sn+b*cs;
}
__global__ void store_kernel(const float* x, half* cache, int n, int row) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) cache[size_t(row)*n+i] = __float2half_rn(x[i]);
}
__global__ void pool_kernel(const float* v, const float* s, float* y, int n, int ratio) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    if (ratio == 1) { y[i] = v[i]; return; }
    const float m = fmaxf(s[i],s[n+i]), a = expf(s[i]-m), b = expf(s[n+i]-m);
    // Pool in projection precision; the cache converts to F16 only after norm+RoPE.
    y[i] = v[i]*(a/(a+b))+v[n+i]*(b/(a+b));
}
__global__ void store_positions_kernel(const float* x,half* cache,int width,const int32_t* positions,int capacity,int ratio) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x, t = blockIdx.y, pos = positions[t];
    if (i >= width || (ratio && (pos+1)%ratio)) return;
    const int row = ratio ? pos/ratio : pos%capacity;
    cache[size_t(row)*width+i] = __float2half_rn(x[size_t(t)*width+i]);
}
__global__ void projection_ring_kernel(const float* v,const float* s,float* rv,float* rs,int width,
                                       const int32_t* positions,int capacity) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x, t = blockIdx.y;
    if (i >= width) return;
    const size_t dst = size_t(positions[t]%capacity)*width+i, src = size_t(t)*width+i;
    rv[dst] = v[src]; if (s && rs) rs[dst] = s[src];
}
__global__ void pool_positions_kernel(const float* v,const float* s,float* y,int width,
                                      const int32_t* positions,int capacity,int ratio) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x, t = blockIdx.y, pos = positions[t];
    if (i >= width) return;
    y += size_t(t)*width;
    if ((pos+1)%ratio) { y[i] = 0; return; }
    const size_t first = size_t((pos+1-ratio)%capacity)*width+i;
    if (ratio == 1) { y[i] = v[first]; return; }
    const size_t second = size_t(pos%capacity)*width+i;
    const float m = fmaxf(s[first],s[second]), a = expf(s[first]-m), b = expf(s[second]-m);
    y[i] = v[first]*(a/(a+b))+v[second]*(b/(a+b));
}
template<bool DevicePositions,bool DraftBlock=false,bool Normalize=false,bool Emit=false,bool QueryPrefetch=false>
__global__ void attention_kernel(const float* q, const half* raw, const half* comp,
                                 const int32_t* ids, const float* sinks, float* out,
                                 int d, int window, int position, int selected,int capacity,int last,
                                 const int32_t* positions,int heads,int ratio,int max_selected,int ids_stride,
                                 float* score_out=nullptr,unsigned long long* row_out=nullptr,int output_stride=0) {
    extern __shared__ float scores[];
    if constexpr (DevicePositions) {
        const int t = blockIdx.y; position = positions[t];
        if constexpr (DraftBlock) last = positions[gridDim.y-1];
        else last = position;
        selected = ratio ? min((position+1)/ratio,max_selected) : 0;
        q += size_t(t)*heads*d; out += size_t(t)*heads*d;
        if (ids) ids += size_t(t)*ids_stride;
    }
    const int head = blockIdx.x, lane = threadIdx.x%32, warp = threadIdx.x/32;
    if constexpr (Emit) {
        const size_t slot=(DevicePositions ? size_t(blockIdx.y)*heads : 0)+head;
        score_out+=slot*output_stride;row_out+=slot*output_stride;
    }
    const int first = max(0,position-window+1), nr = last-first+1, n = nr+selected;
    q += size_t(head)*d; out += size_t(head)*d;
    float query_fragment[16];
    if constexpr (QueryPrefetch) {
#pragma unroll
        for(int k=0;k<16;++k) query_fragment[k]=q[lane+32*k];
    }
    for (int row = warp; row < n; row += 8) {
        const half* key = row < nr ? raw+size_t((first+row)%capacity)*d :
            comp+size_t(ids ? ids[row-nr] : row-nr)*d;
        float dot = 0;
        if constexpr (QueryPrefetch) {
            uint16_t fragment[16];
#pragma unroll
            for(int k=0;k<16;++k) fragment[k]=reinterpret_cast<const uint16_t*>(key)[lane+32*k];
#pragma unroll
            for(int k=0;k<16;++k) dot+=query_fragment[k]*__half2float(__ushort_as_half(fragment[k]));
        } else for (int i = lane; i < d; i += 32) dot += q[i]*__half2float(key[i]);
        for (int k = 16; k; k >>= 1) dot += __shfl_down_sync(0xffffffffu,dot,k);
        if (!lane) {
            scores[row] = dot*rsqrtf(float(d));
            if constexpr (Emit) row_out[row]=reinterpret_cast<unsigned long long>(key);
        }
    }
    __syncthreads();
    float m = sinks[head]; for (int row = threadIdx.x; row < n; row += blockDim.x) m = fmaxf(m,scores[row]);
    m = maximum(m);
    float denom = threadIdx.x == 0 ? expf(sinks[head]-m) : 0;
    for (int row = threadIdx.x; row < n; row += blockDim.x) { scores[row] = expf(scores[row]-m); denom += scores[row]; }
    denom = sum(denom);
    if constexpr (Emit) {
        for(int row=threadIdx.x;row<n;row+=blockDim.x) score_out[row]=scores[row]/denom;
        return;
    }
    if constexpr (Normalize) {
        // Every output channel uses the same rounded weight. Compute each
        // division once, with the original denominator and row order.
        for(int row=threadIdx.x;row<n;row+=blockDim.x) scores[row]=scores[row]/denom;
        __syncthreads();
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = 0;
        for (int row = 0; row < n; ++row) {
            const half* value = row < nr ? raw+size_t((first+row)%capacity)*d :
                comp+size_t(ids ? ids[row-nr] : row-nr)*d;
            const float weight=Normalize ? scores[row] : scores[row]/denom;
            v += __half2float(value[i])*weight;
        }
        out[i] = v;
    }
}
template<bool DevicePositions>
__global__ void index_kernel(const float* q, const float* w, const half* keys,
                             float* scores, int d, int heads,const int32_t* positions,int ratio,int stride) {
    __shared__ float dots[32];
    const int lane = threadIdx.x%32, warp = threadIdx.x/32, row = blockIdx.x;
    if constexpr (DevicePositions) {
        const int t = blockIdx.y;
        scores += size_t(t)*stride; q += size_t(t)*heads*d; w += size_t(t)*heads;
        if (row >= (positions[t]+1)/ratio) { if (!threadIdx.x) scores[row] = -INFINITY; return; }
    }
    keys += size_t(row)*d;
    for (int h = warp; h < heads; h += 8) {
        float dot = 0; for (int i = lane; i < d; i += 32) dot += q[size_t(h)*d+i]*__half2float(keys[i]);
        for (int k = 16; k; k >>= 1) dot += __shfl_down_sync(0xffffffffu,dot,k);
        if (!lane) dots[h] = fmaxf(dot,0.0f)*(w[h]*rsqrtf(float(d*heads)));
    }
    __syncthreads();
    if (!threadIdx.x) { float v = 0; for (int h = 0; h < heads; ++h) v += dots[h]; scores[row] = v; }
}
__global__ void topk_kernel(float* scores, int32_t* ids, int rows, int k) {
    __shared__ float values[256]; __shared__ int indices[256];
    const int tid = threadIdx.x;
    for (int rank = 0; rank < k; ++rank) {
        float best = -INFINITY; int id = INT_MAX;
        for (int i = tid; i < rows; i += blockDim.x) {
            const float v = scores[i]; if (v > best || (v == best && i < id)) { best = v; id = i; }
        }
        values[tid] = best; indices[tid] = id; __syncthreads();
        for (int n = 128; n; n >>= 1) {
            if (tid < n) {
                const int j = indices[tid+n]; const float v = values[tid+n];
                if (v > values[tid] || (v == values[tid] && j < indices[tid])) { values[tid] = v; indices[tid] = j; }
            }
            __syncthreads();
        }
        if (!tid) { ids[rank] = indices[0]; scores[indices[0]] = -INFINITY; }
        __syncthreads();
    }
}
__global__ void sequence_kernel(int32_t* ids,int n) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) ids[i] = i;
}
__device__ __forceinline__ uint32_t load_system(const uint32_t* pointer) {
#if __CUDA_ARCH__ >= 700
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
    return value;
#else
    return *reinterpret_cast<const volatile uint32_t*>(pointer);
#endif
}
__global__ void host_wait_kernel(const uint32_t* ready,const uint32_t* expected,const uint32_t* abort) {
    if(threadIdx.x) return;
    const uint32_t want=load_system(expected);
    while(load_system(ready)!=want && !load_system(abort)) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    __threadfence_system();
}
__global__ void tp_sum_wait_kernel(const float* local,const volatile float* peer,
    const volatile uint32_t* own_ready,const volatile uint32_t* peer_ready,
    const volatile uint32_t* abort,float* out,int n,int rank) {
    __shared__ int stopped;
    if(threadIdx.x==0) {
        const uint32_t want=*own_ready;
        while(*peer_ready<want && !*abort) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        stopped=*abort!=0;
        __threadfence_system();
    }
    __syncthreads();
    if(stopped) return;
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) {
        const float a=rank==0 ? local[i] : peer[i];
        const float b=rank==0 ? peer[i] : local[i];
        out[i]=__fadd_rn(__fadd_rn(0.0f,a),b);
    }
}

// Adapt Qwen's fixed 64-cell split-K decode design to the original D512 MLA
// ring and selected compressed rows. Query dot order stays lane+32*k, down
// shuffles16..1; only softmax/value aggregation is split into chunks.
constexpr int DsChunk=64,DsHeads=4,DsDim=512,DsStride=544;
__device__ float ds_warp_sum(float v) {
    for(int k=16;k;k>>=1) v+=__shfl_down_sync(0xffffffffu,v,k);
    return __shfl_sync(0xffffffffu,v,0);
}
__device__ float ds_warp_max(float v) {
    for(int k=16;k;k>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,k));
    return __shfl_sync(0xffffffffu,v,0);
}
__global__ void attention_chunk_kernel(const float* q,const half* raw,const half* comp,
    const int32_t* ids,float* scratch,int heads,int window,const int32_t* positions,
    int ratio,int selected_limit,int ids_stride,int capacity,int chunks) {
    __shared__ float sq[DsHeads][DsDim],sp[DsHeads][DsChunk];
    __shared__ const half* rows[DsChunk];
    __shared__ float maxima[DsHeads],totals[DsHeads];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const int token=blockIdx.z,head0=int(blockIdx.y)*DsHeads,chunk=blockIdx.x;
    const int pos=positions[token],first=max(0,pos-window+1),nr=pos-first+1;
    const int selected=ratio ? min((pos+1)/ratio,selected_limit) : 0;
    const int begin=chunk*DsChunk,here=min(DsChunk,nr+selected-begin);
    q+=size_t(token)*heads*DsDim+head0*DsDim;
    if(ids) ids+=size_t(token)*ids_stride;
    if(here<=0) {
        if(tid<DsHeads) {
            float* p=scratch+((size_t(token)*heads+head0+tid)*chunks+chunk)*DsStride;
            p[DsDim]=-INFINITY;p[DsDim+1]=0;
        }
        return;
    }
    for(int i=tid;i<DsHeads*DsDim;i+=256) sq[i/DsDim][i%DsDim]=q[i];
    if(tid<DsChunk) {
        const int row=begin+tid;
        rows[tid]=tid>=here ? nullptr : row<nr ? raw+size_t((first+row)%capacity)*DsDim :
            comp+size_t(ids ? ids[row-nr] : row-nr)*DsDim;
    }
    __syncthreads();
    for(int cell=warp;cell<DsChunk;cell+=8) {
        if(cell>=here) {
            if(lane<DsHeads) sp[lane][cell]=-INFINITY;
            continue;
        }
        float key[16];
#pragma unroll
        for(int k=0;k<16;++k) key[k]=__half2float(rows[cell][lane+32*k]);
#pragma unroll
        for(int h=0;h<DsHeads;++h) {
            float dot=0;
#pragma unroll
            for(int k=0;k<16;++k) dot+=sq[h][lane+32*k]*key[k];
            for(int k=16;k;k>>=1) dot+=__shfl_down_sync(0xffffffffu,dot,k);
            if(!lane) sp[h][cell]=dot*rsqrtf(float(DsDim));
        }
    }
    __syncthreads();
    if(warp<DsHeads) {
        const float m=ds_warp_max(fmaxf(sp[warp][lane],sp[warp][lane+32]));
        const float a=lane<here ? expf(sp[warp][lane]-m) : 0;
        const float b=lane+32<here ? expf(sp[warp][lane+32]-m) : 0;
        sp[warp][lane]=a;sp[warp][lane+32]=b;
        const float total=ds_warp_sum(a+b);
        if(!lane) {maxima[warp]=m;totals[warp]=total;}
    }
    __syncthreads();
    float acc[DsHeads][2]={};
    for(int cell=0;cell<here;++cell) {
        const float a=__half2float(rows[cell][tid]);
        const float b=__half2float(rows[cell][tid+256]);
#pragma unroll
        for(int h=0;h<DsHeads;++h) {
            const float weight=sp[h][cell];
            acc[h][0]+=a*weight;acc[h][1]+=b*weight;
        }
    }
#pragma unroll
    for(int h=0;h<DsHeads;++h) {
        float* p=scratch+((size_t(token)*heads+head0+h)*chunks+chunk)*DsStride;
        p[tid]=acc[h][0];p[tid+256]=acc[h][1];
        if(!tid) {p[DsDim]=maxima[h];p[DsDim+1]=totals[h];}
    }
}
__global__ void attention_chunk_merge_kernel(const float* scratch,const float* sinks,float* out,
    int heads,int window,const int32_t* positions,int ratio,int selected_limit,int chunks) {
    __shared__ float scales[10];
    const int head=blockIdx.x,token=blockIdx.y,tid=threadIdx.x;
    const int pos=positions[token],first=max(0,pos-window+1);
    const int n=pos-first+1+(ratio ? min((pos+1)/ratio,selected_limit) : 0);
    const int active=(n+DsChunk-1)/DsChunk;
    const float* partial=scratch+(size_t(token)*heads+head)*chunks*DsStride;
    float m=sinks[head];if(tid<active) m=fmaxf(m,partial[size_t(tid)*DsStride+DsDim]);
    m=maximum(m);
    const float scale=tid<active ? expf(partial[size_t(tid)*DsStride+DsDim]-m) : 0;
    float denom=tid==0 ? expf(sinks[head]-m) : 0;
    if(tid<active) denom+=scale*partial[size_t(tid)*DsStride+DsDim+1];
    denom=sum(denom);
    if(tid<active) scales[tid]=scale/denom;
    __syncthreads();
    for(int i=tid;i<DsDim;i+=256) {
        float v=0;for(int c=0;c<active;++c) v+=partial[size_t(c)*DsStride+i]*scales[c];
        out[(size_t(token)*heads+head)*DsDim+i]=v;
    }
}

// Parallelize independent rows without changing any row's original dot order.
// The following pass keeps the original whole-head 256-thread reductions.
__global__ void attention_score_grid_kernel(const float* q,const half* raw,const half* comp,
    const int32_t* ids,float* scores,unsigned long long* pointers,int d,int heads,int window,
    const int32_t* positions,int ratio,int max_selected,int ids_stride,int capacity,int stride) {
    const int token=blockIdx.y,head=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int pos=positions[token],first=max(0,pos-window+1),nr=pos-first+1;
    const int selected=ratio ? min((pos+1)/ratio,max_selected) : 0,n=nr+selected;
    const int base=int(blockIdx.z)*8;
    if(base>=n) return; // Uniform across the CTA, before its query barrier.
    __shared__ float query[512];
    const size_t slot=size_t(token)*heads+head;
    for(int i=threadIdx.x;i<d;i+=blockDim.x) query[i]=q[slot*d+i];
    __syncthreads();
    const int row=base+warp;
    if(row>=n) return; // Entire warp returns; all query loads already completed.
    if(ids) ids+=size_t(token)*ids_stride;
    const half* key=row<nr ? raw+size_t((first+row)%capacity)*d :
        comp+size_t(ids ? ids[row-nr] : row-nr)*d;
    float query_fragment[16];uint16_t fragment[16];
#pragma unroll
    for(int k=0;k<16;++k) query_fragment[k]=query[lane+32*k];
#pragma unroll
    for(int k=0;k<16;++k) fragment[k]=reinterpret_cast<const uint16_t*>(key)[lane+32*k];
    float dot=0;
#pragma unroll
    for(int k=0;k<16;++k) dot+=query_fragment[k]*__half2float(__ushort_as_half(fragment[k]));
    for(int k=16;k;k>>=1) dot+=__shfl_down_sync(0xffffffffu,dot,k);
    if(!lane) {
        scores[slot*stride+row]=dot*rsqrtf(float(d));
        pointers[slot*stride+row]=reinterpret_cast<unsigned long long>(key);
    }
}
__global__ void attention_score_softmax_kernel(float* weights,const float* sinks,int heads,int window,
    const int32_t* positions,int ratio,int max_selected,int stride) {
    extern __shared__ float scores[];
    const int head=blockIdx.x,token=blockIdx.y,pos=positions[token],first=max(0,pos-window+1);
    const int n=pos-first+1+(ratio ? min((pos+1)/ratio,max_selected) : 0);
    weights+=(size_t(token)*heads+head)*stride;
    for(int row=threadIdx.x;row<n;row+=blockDim.x) scores[row]=weights[row];
    __syncthreads();
    float m=sinks[head];for(int row=threadIdx.x;row<n;row+=blockDim.x) m=fmaxf(m,scores[row]);
    m=maximum(m);
    float denom=threadIdx.x==0 ? expf(sinks[head]-m) : 0;
    for(int row=threadIdx.x;row<n;row+=blockDim.x) {scores[row]=expf(scores[row]-m);denom+=scores[row];}
    denom=sum(denom);
    for(int row=threadIdx.x;row<n;row+=blockDim.x) weights[row]=scores[row]/denom;
}
__device__ __forceinline__ uint16_t global_value_bits(const uint16_t* pointer) {
    uint16_t value;
    asm("ld.global.ca.u16 %0, [%1];" : "=h"(value) : "l"(pointer));
    return value;
}
template<bool Prefetch=false,bool Global=false>
__global__ void attention_value_kernel(const float* weights,const unsigned long long* pointers,float* out,
    int d,int heads,int window,const int32_t* positions,int ratio,int selected_limit,int stride) {
    extern __shared__ __align__(16) unsigned char stage[];
    auto* probability=reinterpret_cast<float*>(stage);
    auto* row_address=reinterpret_cast<unsigned long long*>(stage+((size_t(stride)*4+7)&~size_t(7)));
    const int head=blockIdx.x,token=blockIdx.y,tid=threadIdx.x,dim=int(blockIdx.z)*64+tid;
    const int pos=positions[token],first=max(0,pos-window+1);
    const int n=pos-first+1+(ratio ? min((pos+1)/ratio,selected_limit) : 0);
    const size_t slot=(size_t(token)*heads+head)*stride;
    for(int row=tid;row<n;row+=64) {probability[row]=weights[slot+row];row_address[row]=pointers[slot+row];}
    __syncthreads();
    if(dim<d) {
        float v=0;int row=0;
        if constexpr (Prefetch) {
            for(;row+7<n;row+=8) {
                uint16_t raw[8];float weight[8];
#pragma unroll
                for(int j=0;j<8;++j) {
                    const auto* value=reinterpret_cast<const uint16_t*>(row_address[row+j]);
                    if constexpr (Global) raw[j]=global_value_bits(value+dim); else raw[j]=value[dim];
                    weight[j]=probability[row+j];
                }
#pragma unroll
                for(int j=0;j<8;++j) v+=__half2float(__ushort_as_half(raw[j]))*weight[j];
            }
        }
        for(;row<n;++row) {
            if constexpr (Global) {
                const auto* value=reinterpret_cast<const uint16_t*>(row_address[row]);
                v+=__half2float(__ushort_as_half(global_value_bits(value+dim)))*probability[row];
            } else {
                const auto* value=reinterpret_cast<const half*>(row_address[row]);
                v+=__half2float(value[dim])*probability[row];
            }
        }
        out[(size_t(token)*heads+head)*d+dim]=v;
    }
}

// Every head of a token has the same raw/sparse row list. Share only its
// pointer and exact half read; each head keeps its original probability and
// sequential row-ordered FMA recurrence. Two heads retain128 tiles at NT1.
__global__ void attention_value_head_pair_kernel(const float* weights,const unsigned long long* pointers,float* out,
    int d,int heads,int window,const int32_t* positions,int ratio,int selected_limit,int stride) {
    extern __shared__ __align__(16) unsigned char stage[];
    auto* p0=reinterpret_cast<float*>(stage);
    auto* p1=p0+stride;
    auto* rows=reinterpret_cast<unsigned long long*>(p1+stride);
    const int head=2*int(blockIdx.x),token=blockIdx.y,tid=threadIdx.x,dim=int(blockIdx.z)*64+tid;
    const int pos=positions[token],first=max(0,pos-window+1);
    const int n=pos-first+1+(ratio ? min((pos+1)/ratio,selected_limit) : 0);
    const size_t slot=(size_t(token)*heads+head)*stride;
    for(int row=tid;row<n;row+=64) {
        p0[row]=weights[slot+row];p1[row]=weights[slot+stride+row];rows[row]=pointers[slot+row];
    }
    __syncthreads();
    if(dim<d) {
        float v0=0,v1=0;
        for(int row=0;row<n;++row) {
            const auto* value=reinterpret_cast<const uint16_t*>(rows[row]);
            const float x=__half2float(__ushort_as_half(global_value_bits(value+dim)));
            v0+=x*p0[row];v1+=x*p1[row];
        }
        out[(size_t(token)*heads+head)*d+dim]=v0;
        out[(size_t(token)*heads+head+1)*d+dim]=v1;
    }
}

}  // namespace

void rms(const float* x,const float* w,float* y,int n,int rows,float eps,void* s) { rms_kernel<<<rows,threads,0,stream_cast(s)>>>(x,w,y,n,eps); }
void hc_init(const float* x,float* r,float* pre,int n,int t,void* s) { init_kernel<<<t,threads,0,stream_cast(s)>>>(x,r,pre,n); }
void hc_coeff(const float* m,const float* sc,const float* b,float* pre,float* post,float* c,int t,float eps,int it,void* s) {
    if (it < 1) throw std::invalid_argument("Sinkhorn needs at least one iteration");
    coeff_kernel<<<t,32,0,stream_cast(s)>>>(m,sc,b,pre,post,c,eps,it);
}
void hc_pre(const float* r,const float* pre,float* y,int n,int t,void* s) { pre_kernel<<<dim3((n+255)/256,t),threads,0,stream_cast(s)>>>(r,pre,y,n); }
void hc_post(const float* x,const float* r,const float* post,const float* c,float* y,int n,int t,void* s) { post_kernel<<<dim3((n+255)/256,t),threads,0,stream_cast(s)>>>(x,r,post,c,y,n); }
void engram_add(float* r,const float* kv,const float* qw,const float* kw,float* gates,int n,int t,float eps,void* s) { engram_kernel<<<t*4,threads,0,stream_cast(s)>>>(r,kv,qw,kw,gates,n,eps); }
void route(const float* l,const float* b,int32_t* ids,float* w,int e,int k,int t,float sc,void* s) {
    if (e < 1 || e > 512 || k < 1 || k > 64 || k > e) throw std::invalid_argument("DeepSeek routing geometry");
    route_kernel<<<t,threads,0,stream_cast(s)>>>(l,b,ids,w,e,k,sc);
}
void swiglu(const float* g,const float* u,float* y,int n,float limit,void* s) { swiglu_kernel<<<(n+255)/256,threads,0,stream_cast(s)>>>(g,u,y,n,limit); }
void combine(const float* p,const float* w,const float* sh,float* y,int h,int k,int t,void* s) { combine_kernel<<<dim3((h+255)/256,t),threads,0,stream_cast(s)>>>(p,w,sh,y,h,k); }
void combine_sources(const float* p,const float* cpu,const uint64_t* mask,const float* w,const float* sh,float* y,int h,int k,int t,void* s) {
    combine_sources_kernel<false><<<dim3((h+255)/256,t),threads,0,stream_cast(s)>>>(p,cpu,mask,w,sh,y,h,k);
}
void combine_sources_system(const float* p,const float* cpu,const uint64_t* mask,const float* w,const float* sh,float* y,int h,int k,int t,void* s) {
    combine_sources_kernel<true><<<dim3((h+255)/256,t),threads,0,stream_cast(s)>>>(p,cpu,mask,w,sh,y,h,k);
}
void wait_host(const uint32_t* ready,const uint32_t* expected,const uint32_t* abort,void* s) {
    if(!ready || !expected || !abort) throw std::invalid_argument("native mapped Host wait needs valid words");
    host_wait_kernel<<<1,32,0,stream_cast(s)>>>(ready,expected,abort);
}
void dense(const float* w,const float* x,float* y,int ni,int no,int t,void* s) { dense_kernel<<<dim3(no,t),threads,0,stream_cast(s)>>>(w,x,y,ni,no); }
void add(const float* x,const float* y,float* out,int n,void* s) { add_kernel<<<(n+255)/256,threads,0,stream_cast(s)>>>(x,y,out,n); }
void tp_sum_wait(const float* local,const float* peer,const uint32_t* own_ready,
    const uint32_t* peer_ready,const uint32_t* abort,float* out,int n,int rank,void* s) {
    if(n<1 || (rank!=0 && rank!=1)) throw std::invalid_argument("native TP2 sum geometry");
    tp_sum_wait_kernel<<<(n+255)/256,threads,0,stream_cast(s)>>>(local,peer,own_ready,peer_ready,abort,out,n,rank);
}
void strided_copy(const float* x,float* y,int w,int rows,int xs,int ys,void* s) { strided_copy_kernel<<<dim3((w+255)/256,rows),threads,0,stream_cast(s)>>>(x,y,w,xs,ys); }
void gather_row(const float* rows,const int32_t* id,float* out,int w,int count,void* s) {
    gather_row_kernel<<<(w+255)/256,threads,0,stream_cast(s)>>>(rows,id,out,w,count);
}
void draft_confidence(const float* h,const float* m,const float* w,const float* b,float* out,int nh,int nm,void* s) {
    draft_confidence_kernel<<<1,threads,0,stream_cast(s)>>>(h,m,w,b,out,nh,nm);
}
void hc_mean(const float* r,float* y,int h,int t,int stride,void* s) { mean_kernel<<<dim3((h+255)/256,t),threads,0,stream_cast(s)>>>(r,y,h,stride); }
void rope(float* x,int d,int heads,int t,const int32_t* pos,Rope c,bool inv,void* s) {
    if (c.dim < 2 || c.dim%2 || c.dim > d || c.base <= 1 || c.scale <= 0) throw std::invalid_argument("DeepSeek RoPE geometry");
    float low = 0, high = 0;
    if (c.yarn) {
        if (c.original_context <= 0 || c.beta_fast <= 0 || c.beta_slow <= 0) throw std::invalid_argument("DeepSeek YaRN configuration");
        auto corr = [&](float beta) { return c.dim*std::log(c.original_context/(beta*2.0f*3.14159265358979323846f))/(2*std::log(c.base)); };
        low = std::fmax(0.0f,std::floor(corr(c.beta_fast))); high = std::fmin(float(c.dim-1),std::ceil(corr(c.beta_slow)));
    }
    rope_kernel<<<dim3((heads*c.dim/2+255)/256,t),threads,0,stream_cast(s)>>>(x,d,heads,pos,c,low,high,inv);
}
void store_f16(const float* x,uint16_t* c,int n,int row,void* s) { store_kernel<<<(n+255)/256,threads,0,stream_cast(s)>>>(x,reinterpret_cast<half*>(c),n,row); }
void pool(const float* v,const float* scores,float* y,int n,int ratio,void* s) {
    if (ratio != 1 && ratio != 2) throw std::invalid_argument("DeepSeek pool ratio");
    pool_kernel<<<(n+255)/256,threads,0,stream_cast(s)>>>(v,scores,y,n,ratio);
}
void store_f16_positions(const float* x,uint16_t* cache,int w,int t,const int32_t* pos,int capacity,int ratio,void* s) {
    if (capacity < 1 || ratio < 0 || ratio > 2) throw std::invalid_argument("DeepSeek position store geometry");
    store_positions_kernel<<<dim3((w+255)/256,t),threads,0,stream_cast(s)>>>(x,reinterpret_cast<half*>(cache),w,pos,capacity,ratio);
}
void projection_ring(const float* v,const float* scores,float* rv,float* rs,int w,int t,const int32_t* pos,int capacity,void* s) {
    if (capacity < t) throw std::invalid_argument("DeepSeek projection ring capacity");
    projection_ring_kernel<<<dim3((w+255)/256,t),threads,0,stream_cast(s)>>>(v,scores,rv,rs,w,pos,capacity);
}
void pool_positions(const float* v,const float* scores,float* y,int w,int t,const int32_t* pos,int capacity,int ratio,void* s) {
    if (capacity < t+ratio-1 || (ratio != 1 && ratio != 2)) throw std::invalid_argument("DeepSeek position pool geometry");
    pool_positions_kernel<<<dim3((w+255)/256,t),threads,0,stream_cast(s)>>>(v,scores,y,w,pos,capacity,ratio);
}
void attention(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,float* out,
               int d,int heads,int window,int pos,int selected,void* s,int capacity) {
    if (pos < 0 || window < 1 || selected < 0 || (selected && !comp)) throw std::invalid_argument("DeepSeek attention state");
    attention_kernel<false><<<heads,threads,size_t(window+selected)*4,stream_cast(s)>>>(q,reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,pos,selected,capacity ? capacity : window,pos,nullptr,heads,0,0,0);
}
void draft_attention(const float* q,const uint16_t* raw,const float* sinks,float* out,int d,int heads,int window,int pos,int end,int capacity,void* s) {
    if (end < pos || capacity < window+end-pos) throw std::invalid_argument("DSpark noncausal window capacity");
    attention_kernel<false><<<heads,threads,size_t(window+end-pos)*4,stream_cast(s)>>>(q,reinterpret_cast<const half*>(raw),nullptr,nullptr,sinks,out,d,window,pos,0,capacity,end,nullptr,heads,0,0,0);
}
void draft_attention_positions(const float* q,const uint16_t* raw,const float* sinks,float* out,int d,int heads,int tokens,int window,const int32_t* pos,int capacity,void* s) {
    attention_kernel<true,true><<<dim3(heads,tokens),threads,size_t(window+tokens-1)*4,stream_cast(s)>>>(q,reinterpret_cast<const half*>(raw),nullptr,nullptr,sinks,out,d,window,0,0,capacity,0,pos,heads,0,0,0);
}
void index_scores(const float* q,const float* w,const uint16_t* keys,float* scores,int d,int heads,int rows,void* s) {
    if (heads > 32 || heads < 1) throw std::invalid_argument("DeepSeek indexer heads");
    if (rows) index_kernel<false><<<rows,threads,0,stream_cast(s)>>>(q,w,reinterpret_cast<const half*>(keys),scores,d,heads,nullptr,1,0);
}
void index_scores_positions(const float* q,const float* w,const uint16_t* keys,float* scores,int d,int heads,
                            int bucket,int tokens,const int32_t* pos,int ratio,int stride,void* s) {
    if (heads < 1 || heads > 32 || bucket < 1 || bucket > stride || (ratio != 1 && ratio != 2)) throw std::invalid_argument("DeepSeek bucket index geometry");
    index_kernel<true><<<dim3(bucket,tokens),threads,0,stream_cast(s)>>>(q,w,reinterpret_cast<const half*>(keys),scores,d,heads,pos,ratio,stride);
}
void attention_positions(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,
                         float* out,int d,int heads,int tokens,int window,const int32_t* pos,int ratio,int max_selected,
                         int ids_stride,int raw_capacity,void* s) {
    if (window < 1 || raw_capacity < window+tokens || ratio < 0 || ratio > 2 || max_selected < 0 || (ratio && !comp)) throw std::invalid_argument("DeepSeek position attention geometry");
    attention_kernel<true><<<dim3(heads,tokens),threads,size_t(window+max_selected)*4,stream_cast(s)>>>(q,reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,0,0,raw_capacity,0,pos,heads,ratio,max_selected,ids_stride);
}
void attention_positions_normalized(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,
                         float* out,int d,int heads,int tokens,int window,const int32_t* pos,int ratio,int max_selected,
                         int ids_stride,int raw_capacity,void* s) {
    if (window < 1 || raw_capacity < window+tokens || ratio < 0 || ratio > 2 || max_selected < 0 || (ratio && !comp)) throw std::invalid_argument("DeepSeek position attention geometry");
    attention_kernel<true,false,true><<<dim3(heads,tokens),threads,size_t(window+max_selected)*4,stream_cast(s)>>>(q,reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,0,0,raw_capacity,0,pos,heads,ratio,max_selected,ids_stride);
}
void attention_positions_chunked(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,float* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=DsDim || heads<1 || heads%DsHeads || tokens<1 || tokens>8 || window<1 || window>128 ||
       raw_capacity<window+tokens || ratio<0 || ratio>2 || max_selected<0 || max_selected>512 ||
       (ratio && !comp) || !scratch) throw std::invalid_argument("DeepSeek split attention geometry");
    const int chunks=(window+max_selected+DsChunk-1)/DsChunk;
    attention_chunk_kernel<<<dim3(chunks,heads/DsHeads,tokens),threads,0,stream_cast(s)>>>(q,
        reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,scratch,heads,window,
        pos,ratio,max_selected,ids_stride,raw_capacity,chunks);
    attention_chunk_merge_kernel<<<dim3(heads,tokens),threads,0,stream_cast(s)>>>(scratch,sinks,out,
        heads,window,pos,ratio,max_selected,chunks);
}
void attention_positions_values(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    attention_kernel<true,false,false,true><<<dim3(heads,tokens),threads,size_t(stride)*4,stream_cast(s)>>>(q,
        reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,0,0,
        raw_capacity,0,pos,heads,ratio,max_selected,ids_stride,weights,rows,stride);
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    attention_value_kernel<false><<<dim3(heads,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
        window,pos,ratio,max_selected,stride);
}
void attention_positions_values_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    attention_kernel<true,false,false,true><<<dim3(heads,tokens),threads,size_t(stride)*4,stream_cast(s)>>>(q,
        reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,0,0,
        raw_capacity,0,pos,heads,ratio,max_selected,ids_stride,weights,rows,stride);
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    attention_value_kernel<true><<<dim3(heads,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
        window,pos,ratio,max_selected,stride);
}
void attention_positions_query_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool value_prefetch) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    attention_kernel<true,false,false,true,true><<<dim3(heads,tokens),threads,size_t(stride)*4,stream_cast(s)>>>(q,
        reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,sinks,out,d,window,0,0,
        raw_capacity,0,pos,heads,ratio,max_selected,ids_stride,weights,rows,stride);
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    if(value_prefetch) attention_value_kernel<true,true><<<dim3(heads,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
        window,pos,ratio,max_selected,stride);
    else attention_value_kernel<false,true><<<dim3(heads,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
        window,pos,ratio,max_selected,stride);
}
void attention_positions_score_grid(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool pair_heads) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch || (pair_heads && heads%2))
        throw std::invalid_argument("DeepSeek score-grid geometry");
    const int stride=window+max_selected;
    const size_t count=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((count*4+255)&~size_t(255)));
    attention_score_grid_kernel<<<dim3(heads,tokens,(stride+7)/8),threads,0,stream_cast(s)>>>(q,
        reinterpret_cast<const half*>(raw),reinterpret_cast<const half*>(comp),ids,weights,rows,d,heads,window,
        pos,ratio,max_selected,ids_stride,raw_capacity,stride);
    attention_score_softmax_kernel<<<dim3(heads,tokens),threads,size_t(stride)*4,stream_cast(s)>>>(weights,sinks,
        heads,window,pos,ratio,max_selected,stride);
    if(pair_heads) {
        const size_t shared=size_t(stride)*(2*sizeof(float)+sizeof(unsigned long long));
        attention_value_head_pair_kernel<<<dim3(heads/2,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
            window,pos,ratio,max_selected,stride);
    } else {
        const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
        attention_value_kernel<false,true><<<dim3(heads,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,
            window,pos,ratio,max_selected,stride);
    }
}
void top_k(float* scores,int32_t* ids,int rows,int k,void* s) {
    if (k < 0 || k > rows) throw std::invalid_argument("DeepSeek indexer top-k");
    if (k) topk_kernel<<<1,threads,0,stream_cast(s)>>>(scores,ids,rows,k);
}
size_t top_k_scratch_bytes(int rows) {
    if (rows < 1) throw std::invalid_argument("DeepSeek radix top-k rows");
    size_t temporary = 0;
    const auto e = cub::DeviceRadixSort::SortPairsDescending(nullptr,temporary,static_cast<const float*>(nullptr),static_cast<float*>(nullptr),static_cast<const int32_t*>(nullptr),static_cast<int32_t*>(nullptr),rows);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    const size_t plane = (size_t(rows)*4+255)&~size_t(255);
    return plane*2+temporary;
}
void top_k_sorted(const float* scores,int32_t* ids,int rows,int k,void* scratch,size_t bytes,void* stream) {
    if (k < 0 || k > rows) throw std::invalid_argument("DeepSeek radix top-k geometry");
    if (!k) return;
    const size_t plane = (size_t(rows)*4+255)&~size_t(255);
    if (bytes <= plane*2) throw std::invalid_argument("DeepSeek radix top-k scratch");
    auto* src_ids = static_cast<int32_t*>(scratch);
    auto* sorted = reinterpret_cast<float*>(static_cast<uint8_t*>(scratch)+plane);
    void* temp = static_cast<uint8_t*>(scratch)+plane*2; size_t available = bytes-plane*2;
    sequence_kernel<<<(rows+255)/256,threads,0,stream_cast(stream)>>>(src_ids,rows);
    const auto e = cub::DeviceRadixSort::SortPairsDescending(temp,available,scores,sorted,src_ids,ids,rows,0,32,stream_cast(stream));
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

}  // namespace strata::deepseek41::cuda
