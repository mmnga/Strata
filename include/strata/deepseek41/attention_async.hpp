#pragma once
#include <cstdint>
namespace strata::deepseek41::cuda {
void attention_positions_score_grid_async(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* positions,int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream);
}
