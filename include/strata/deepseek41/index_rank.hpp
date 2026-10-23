#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::deepseek41::cuda {
void index_rank_batch(const float* scores,int32_t* ids,int rows,int top_k,int tokens,int stride,void* scratch,size_t bytes,void* stream);
}
