// DeepSeek-only composition of the existing HC RMS/Q81 producer and
// Q4KTraits row dot. The native MMVQ kernels and default calls stay intact.
namespace {
__device__ __forceinline__ float hc_ieee_add(float a,float b){float y;asm("add.rn.f32 %0,%1,%2;":"=f"(y):"f"(a),"f"(b));return y;}
__device__ __forceinline__ float hc_ieee_mul(float a,float b){float y;asm("mul.rn.f32 %0,%1,%2;":"=f"(y):"f"(a),"f"(b));return y;}
__device__ __forceinline__ float hc_ieee_fma(float a,float b,float c){float y;asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(y):"f"(a),"f"(b),"f"(c));return y;}
__launch_bounds__(1024,1)
__global__ void ds41_hc_consumer_kernel(const Q4KBlock* weights,const float* residual,
 float* normalized,Q81Block* quant,float* inverse,float* mix,int width,float eps){
 __shared__ float rms_partial[8],local_inverse,dot_partial[3][32];
 __shared__ __align__(16) unsigned char quant_raw[23040];
 auto* local_q=reinterpret_cast<Q81Block*>(quant_raw);
 const int tid=threadIdx.x,lane=tid&31,row=blockIdx.x,token=blockIdx.y;
 residual+=size_t(token)*width;
 float value=0;
 if(tid<256){
  for(int i=tid;i<width;i+=256)value=hc_ieee_fma(residual[i],residual[i],value);
  for(int n=16;n;n>>=1)value=hc_ieee_add(value,__shfl_down_sync(0xffffffffu,value,n,32));
  if(!lane)rms_partial[tid>>5]=value;
 }
 __syncthreads();
 if(tid<32){
  value=tid<8?rms_partial[tid]:0.f;
  for(int n=16;n;n>>=1)value=hc_ieee_add(value,__shfl_down_sync(0xffffffffu,value,n,32));
  if(!tid){float mean,inv;const float divisor=float(width);
   asm("div.rn.f32 %0,%1,%2;":"=f"(mean):"f"(value),"f"(divisor));
   mean=hc_ieee_add(mean,eps);asm("rsqrt.approx.f32 %0,%1;":"=f"(inv):"f"(mean));
   local_inverse=inv;if(!row)inverse[token]=inv;
  }
 }
 __syncthreads();
 const float inv=local_inverse;
 for(int i=tid;i<width;i+=1024){
  const float xi=hc_ieee_mul(hc_ieee_mul(residual[i],inv),1.f);
  const float maximum=warp_max(fabsf(xi)),sum=warp_sum(xi);
  const float d=q8_1_finite(maximum/127.f);
  const int8_t q=q8_1_quant(xi,d,maximum);
  local_q[i/32].qs[lane]=q;
  if(!lane)local_q[i/32].ds=q8_1_ds(d,sum);
  if(!row){normalized[size_t(token)*width+i]=xi;
   quant[size_t(token)*(width/32)+i/32].qs[lane]=q;
   if(!lane)quant[size_t(token)*(width/32)+i/32].ds=q8_1_ds(d,sum);
  }
 }
 __syncthreads();
 // Exactly the adopted N89 row dot: four original virtual warps, each
 // original FMA chain, ascending warp partials and the original XOR tree.
 using F=Q4KTraits;
 float tmp=0;
 if(tid<128){
  const int blocks_per_row=width/F::DIV,kqs=F::kqs(tid);
  for(int kbx=tid/F::T;kbx<blocks_per_row;kbx+=F::BPI){
   const auto w=F::load(weights+size_t(row)*blocks_per_row+kbx,kqs);
   tmp+=F::apply(w,local_q+kbx*F::KBY,kqs);
  }
  if(tid>=32)dot_partial[(tid>>5)-1][lane]=tmp;
 }
 __syncthreads();
 if(tid<32){
  #pragma unroll
  for(int i=0;i<3;++i)tmp+=dot_partial[i][lane];
  tmp=warp_sum(tmp);if(!tid)mix[size_t(token)*24+row]=tmp;
 }
}
}
void native_deepseek41_hc_consumer(const void* weights,const float* residual,
 float* normalized,void* quant,float* inverse,float* mix,int width,int tokens,float eps,void* stream){
 if(!g_multi_exact||width!=20480||tokens<1||tokens>8||normalized==residual||!std::isfinite(eps)||eps<=0)
  throw std::invalid_argument("original consumer-owned HC geometry or alias");
 validate_shape(width,tokens,QK);validate_pointer(weights);validate_pointer(residual);validate_pointer(normalized);
 validate_pointer(quant);validate_pointer(inverse);validate_pointer(mix);validate_stream(stream);
 ds41_hc_consumer_kernel<<<dim3(24,tokens),1024,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<const Q4KBlock*>(weights),residual,normalized,static_cast<Q81Block*>(quant),inverse,mix,width,eps);
 launch_check();
}
