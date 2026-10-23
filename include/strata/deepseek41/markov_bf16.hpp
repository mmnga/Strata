#pragma once
#include "strata/deepseek41/dspark_head.hpp"
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::deepseek41::cuda {
void markov_bf16(const uint16_t*,const float*,float*,int,int,void*);
// Optional host capture observer for numerical graph introspection; no device work.
void conditional_draft_head_bf16(const DraftHeadView&,const uint16_t*,int,float,void*,void* const*,void(*)(void*,void*)=nullptr,void* =nullptr);
}
