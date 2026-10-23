#pragma once
#include <cstdint>
#include <initializer_list>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strata::deepseek41 {
struct HeadPick {float score;int32_t id;};
static_assert(sizeof(HeadPick)==8);
void gather_head_picks(const float*,const int32_t*,HeadPick*,int,int,int,void*);
inline void merge_head_picks(const HeadPick*a,const HeadPick*b,int tokens,int vocab,int32_t*out){
 if(!a||!b||!out||tokens<1||tokens>8||vocab!=129280)throw std::invalid_argument("head pick merge geometry");
 for(int t=0;t<tokens;++t){float best=-std::numeric_limits<float>::infinity();int32_t id=0;
  for(const HeadPick*p:{a+t,b+t}){if(p->id<0||p->id>=vocab)throw std::runtime_error("head pick outside vocabulary");if(p->score>best||(p->score==best&&p->id<id)){best=p->score;id=p->id;}}
  out[t]=id;
 }
}
}
