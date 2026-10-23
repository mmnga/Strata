#include "strata/deepseek41/attention_async.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
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
// Reuse Strata fused_gr's cp.async commit/wait pattern, preserving the value
// kernel's .ca cache operator and copying sixteen raw bytes without conversion.
__device__ __forceinline__ void cp16(void*smem,const void*gmem){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 unsigned address=unsigned(__cvta_generic_to_shared(smem));
 asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(address),"l"(gmem):"memory");
#else
 *reinterpret_cast<uint4*>(smem)=*reinterpret_cast<const uint4*>(gmem);
#endif
}
__device__ __forceinline__ void commit_copy(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 asm volatile("cp.async.commit_group;\n":::"memory");
#endif
}
template<int N>__device__ __forceinline__ void wait_copy(){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory");
#endif
}
__device__ __forceinline__ void load_tile(uint16_t*tile,const unsigned long long*rows,int first,int count,int dim){
 const int group=threadIdx.x/8,col=(threadIdx.x%8)*8;
 #pragma unroll
 for(int j=0;j<8;++j){const int row=first+group+j*8;if(row<count){const auto*source=reinterpret_cast<const uint16_t*>(rows[row]);cp16(tile+(group+j*8)*64+col,source+dim+col);}}
 commit_copy();
}
__global__ void attention_value_async_kernel(const float*weights,const unsigned long long*pointers,float*out,
 int d,int heads,int window,const int32_t*positions,int ratio,int selected_limit,int stride){
 extern __shared__ __align__(16) unsigned char stage[];
 auto*p0=reinterpret_cast<float*>(stage);auto*p1=p0+stride;
 auto*rows=reinterpret_cast<unsigned long long*>(p1+stride);
 auto*tile=reinterpret_cast<uint16_t*>(rows+stride);
 const int head=2*int(blockIdx.x),token=blockIdx.y,tid=threadIdx.x,dim=int(blockIdx.z)*64;
 const int pos=positions[token],first=max(0,pos-window+1);
 const int n=pos-first+1+(ratio?min((pos+1)/ratio,selected_limit):0);
 const size_t slot=(size_t(token)*heads+head)*stride;
 for(int row=tid;row<n;row+=64){p0[row]=weights[slot+row];p1[row]=weights[slot+stride+row];rows[row]=pointers[slot+row];}
 __syncthreads();
 load_tile(tile,rows,0,n,dim);
 float v0=0,v1=0;
 for(int start=0;start<n;start+=64){
  const int current=(start/64)&1,next=current^1;
  if(start+64<n){load_tile(tile+next*64*64,rows,start+64,n,dim);wait_copy<1>();}
  else wait_copy<0>();
  __syncthreads();
  const int count=min(64,n-start);
  for(int row=0;row<count;++row){const float x=__half2float(__ushort_as_half(tile[current*64*64+row*64+tid]));v0+=x*p0[start+row];v1+=x*p1[start+row];}
  __syncthreads(); // Every consumer finishes before this buffer is reused.
 }
 out[(size_t(token)*heads+head)*d+dim+tid]=v0;
 out[(size_t(token)*heads+head+1)*d+dim+tid]=v1;
}
}

void attention_positions_score_grid_async(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch || heads%2)
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
    const size_t shared=size_t(stride)*(2*sizeof(float)+sizeof(unsigned long long))+2*64*64*sizeof(uint16_t);
    attention_value_async_kernel<<<dim3(heads/2,tokens,8),64,shared,stream_cast(s)>>>(weights,rows,out,d,heads,window,pos,ratio,max_selected,stride);
}
}
