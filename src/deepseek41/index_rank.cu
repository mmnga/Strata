#include "strata/deepseek41/index_rank.hpp"
#include "strata/deepseek41/kernels.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
__global__ void rank_kernel(const float* scores,int32_t* ids,int rows,int stride){
    using Sort=cub::BlockRadixSort<uint32_t,128,4,int32_t>;
    __shared__ typename Sort::TempStorage storage;
    uint32_t keys[4];int32_t values[4];
    #pragma unroll
    for(int item=0;item<4;++item){
        const int i=threadIdx.x*4+item;values[item]=i;
        if(i<rows){uint32_t bits=__float_as_uint(scores[size_t(blockIdx.x)*stride+i]);
            // CUB equates signed zeros and otherwise sorts the full float bits.
            if(bits==0x80000000u)bits=0;
            keys[item]=bits^((bits&0x80000000u)?0xffffffffu:0x80000000u);
        }else keys[item]=0; // Valid zero-key items precede padding by stability.
    }
    Sort(storage).SortDescending(keys,values);
    #pragma unroll
    for(int item=0;item<4;++item){const int i=threadIdx.x*4+item;if(i<rows)ids[size_t(blockIdx.x)*stride+i]=values[item];}
}
}
void index_rank_batch(const float*scores,int32_t*ids,int rows,int k,int tokens,int stride,void*scratch,size_t bytes,void*stream){
    if(!stream||rows<1||k<0||k>rows||tokens<1||tokens>8||stride<rows)throw std::invalid_argument("batched index ranking geometry");
    if(!k)return;
    if(rows<=512)rank_kernel<<<tokens,128,0,static_cast<cudaStream_t>(stream)>>>(scores,ids,rows,stride);
    else for(int t=0;t<tokens;++t)top_k_sorted(scores+size_t(t)*stride,ids+size_t(t)*stride,rows,k,scratch,bytes,stream);
}
}
