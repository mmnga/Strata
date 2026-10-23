#include "strata/deepseek41/head_pick.hpp"
#include <cuda_runtime.h>
namespace strata::deepseek41 {
namespace {
__global__ void gather_picks_kernel(const float*logits,const int32_t*picks,HeadPick*out,int tokens,int rows,int offset){
 const int t=threadIdx.x;if(t>=tokens)return;const int id=picks[t];
 if(id>=0&&id<rows)out[t]={logits[size_t(t)*rows+id],id+offset};else out[t]={__int_as_float(0x7fc00000),-1};
}
}
void gather_head_picks(const float*x,const int32_t*p,HeadPick*out,int tokens,int rows,int offset,void*stream){
 if(!x||!p||!out||!stream||tokens<1||tokens>8||rows<1||offset<0||offset+rows>129280)throw std::invalid_argument("head pick gather geometry");
 gather_picks_kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(x,p,out,tokens,rows,offset);
}
}
