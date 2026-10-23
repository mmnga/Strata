#include "strata/deepseek41/norm_quant.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
namespace strata::deepseek41::cuda {
namespace {
struct Q81 {half2 ds;int8_t qs[32];};static_assert(sizeof(Q81)==36);
__device__ float norm_product(float x,float inv,float w){
 // This CU deliberately matches native_mmvq's fast-math quantizer contract.
 // Explicit non-FTZ PTX keeps BOTH original normalization multiplies IEEE.
 float first,result;
 asm volatile("mul.rn.f32 %0, %1, %2;":"=f"(first):"f"(x),"f"(inv));
 asm volatile("mul.rn.f32 %0, %1, %2;":"=f"(result):"f"(first),"f"(w));
 return result;
}
__device__ float warp_sum(float x){
 #pragma unroll
 for(int offset=16;offset>0;offset>>=1)x+=__shfl_xor_sync(0xffffffffu,x,offset,32);
 return x;
}
__device__ float warp_max(float x){
 #pragma unroll
 for(int offset=16;offset>0;offset>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,offset,32));
 return x;
}
__launch_bounds__(256,1)
__global__ void producer(const float*x,const float*inv,const float*w,float*normalized,Q81*q8,int*bm,int*pm,float*dl,int H,int total,int cp){
 const int i=blockIdx.x*256+threadIdx.x;if(i>=total)return;
 const int c=i/H,idx=i-c*H,lane=threadIdx.x&31;
 const float xi=norm_product(x[i],inv[c],w?w[idx]:1.f);normalized[i]=xi;
 const float amax=warp_max(fabsf(xi)),sum=warp_sum(xi);
 const float d=strata::kernels::q8_1_finite(amax/127.f);
 const int8_t q=strata::kernels::q8_1_quant(xi,d,amax);
 const half2 ds=strata::kernels::q8_1_ds(d,sum);
 q8[i/32].qs[i%32]=q;if(!lane)q8[i/32].ds=ds;
 if(cp){unsigned v=unsigned(uint8_t(q))<<(8*(lane&3));
  v|=__shfl_xor_sync(0xffffffffu,v,1);v|=__shfl_xor_sync(0xffffffffu,v,2);
  const int nb=H/32,b=idx/32,p=lane>>2;
  if(!(lane&3)){bm[(size_t(b)*8+p)*cp+c]=int(v);pm[(size_t(p)*nb+b)*cp+c]=int(v);}
  if(!lane)dl[size_t(b)*cp+c]=__low2float(ds);
 }
}
}
void norm_quant_q8(const float*x,const float*inv,const float*w,float*norm,void*q8,void*il,int H,int tokens,int cp,void*stream){
 if(H!=5120||tokens<1||tokens>8||!stream||(cp!=0&&cp!=2&&cp!=4&&cp!=8)||(cp&&cp<tokens)||(cp&&!il))throw std::invalid_argument("norm/Q81 geometry");
 const size_t plane=size_t(H/32)*8*cp;auto*bm=static_cast<int*>(il);auto*pm=cp?bm+plane:nullptr;auto*dl=cp?reinterpret_cast<float*>(pm+plane):nullptr;
 producer<<<(H*tokens+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(x,inv,w,norm,static_cast<Q81*>(q8),bm,pm,dl,H,H*tokens,cp);
}
}
