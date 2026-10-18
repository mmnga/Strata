#include "strata/deepseek41/kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace k = strata::deepseek41::cuda;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool v, const char* name) { if (!v) throw std::runtime_error(name); }
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t size) : n(size) { ck(cudaMalloc(&p,n*sizeof(T))); }
    explicit Device(const std::vector<T>& x) : Device(x.size()) { set(x); }
    ~Device() { cudaFree(p); }
    void set(const std::vector<T>& x) { require(x.size() == n,"copy size"); ck(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(cudaStream_t s) { ck(cudaStreamSynchronize(s)); std::vector<T> x(n); ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return x; }
};
std::vector<float> values(size_t n, float scale = 1) {
    std::vector<float> x(n); for (size_t i = 0; i < n; ++i) x[i] = scale*std::sin(float(i)*1.723f+0.4f); return x;
}
void near(const std::vector<float>& got,const std::vector<float>& expected,const char* name,float tol = 2e-5f) {
    require(got.size() == expected.size(),"comparison size");
    float worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float error = std::abs(got[i]-expected[i])/(1+std::abs(expected[i])); worst = std::max(worst,error);
        if (!std::isfinite(got[i]) || error > tol) {
            std::cerr << name << " at " << i << ": " << got[i] << " vs " << expected[i] << " error=" << error << '\n';
            throw std::runtime_error(name);
        }
    }
    std::cout << name << " error=" << worst << '\n';
}
double sigmoid(double x) { return 1/(1+std::exp(-x)); }

void hyperconnections(cudaStream_t s) {
    constexpr int H = 17, T = 3;
    const auto x = values(H*T), r = values(H*4*T,2), mix = values(24*T,3), base = values(24,0.6f);
    const std::vector<float> scale = {0.7f,1.3f,0.9f};
    Device<float> dx(x), dr(r), dm(mix), db(base), ds(scale), pre(T*4), post(T*4), comb(T*16), out(T*4*H), collapsed(T*H);
    k::hc_coeff(dm.p,ds.p,db.p,pre.p,post.p,comb.p,T,1e-6f,20,s);
    std::vector<float> rp(T*4), po(T*4), co(T*16);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < 4; ++h) {
            rp[t*4+h] = float(sigmoid(double(mix[t*24+h])*scale[0]+base[h])+1e-6);
            po[t*4+h] = float(2*sigmoid(double(mix[t*24+4+h])*scale[1]+base[4+h]));
        }
        double c[16];
        for (int src = 0; src < 4; ++src) {
            double total = 0;
            for (int dst = 0; dst < 4; ++dst) { const int i = dst+src*4; c[i] = std::exp(double(mix[t*24+8+i])*scale[2]+base[8+i]); total += c[i]; }
            for (int dst = 0; dst < 4; ++dst) c[dst+src*4] = c[dst+src*4]/total+1e-6;
        }
        for (int it = 0; it < 20; ++it) {
            if (it) for (int src = 0; src < 4; ++src) {
                double sum = 1e-6; for (int dst = 0; dst < 4; ++dst) sum += c[dst+src*4];
                for (int dst = 0; dst < 4; ++dst) c[dst+src*4] /= sum;
            }
            for (int dst = 0; dst < 4; ++dst) {
                double sum = 1e-6; for (int src = 0; src < 4; ++src) sum += c[dst+src*4];
                for (int src = 0; src < 4; ++src) c[dst+src*4] /= sum;
            }
        }
        for (int i = 0; i < 16; ++i) co[t*16+i] = float(c[i]);
    }
    near(pre.get(s),rp,"HC pre coefficients"); near(post.get(s),po,"HC post coefficients"); near(comb.get(s),co,"HC Sinkhorn orientation/epsilon");
    k::hc_pre(dr.p,pre.p,collapsed.p,H,T,s);
    std::vector<float> cp(T*H), yp(T*4*H);
    for (int t = 0; t < T; ++t) for (int i = 0; i < H; ++i) {
        for (int h = 0; h < 4; ++h) cp[t*H+i] += r[(t*4+h)*H+i]*rp[t*4+h];
        for (int dst = 0; dst < 4; ++dst) {
            double v = x[t*H+i]*po[t*4+dst];
            for (int src = 0; src < 4; ++src) v += r[(t*4+src)*H+i]*co[t*16+dst+src*4];
            yp[(t*4+dst)*H+i] = float(v);
        }
    }
    near(collapsed.get(s),cp,"HC collapse");
    k::hc_post(dx.p,dr.p,post.p,comb.p,out.p,H,T,s); near(out.get(s),yp,"HC expand");
    k::hc_post(dx.p,dr.p,post.p,comb.p,dr.p,H,T,s); near(dr.get(s),yp,"HC in-place expand");
    Device<float> init(T*H*4), init_pre(T*4);
    k::hc_init(dx.p,init.p,init_pre.p,H,T,s);
    k::hc_pre(init.p,init_pre.p,collapsed.p,H,T,s); near(collapsed.get(s),x,"HC initial lagged mix");
    const auto big = values(5120*2); Device<float> norm(big); k::rms(norm.p,nullptr,norm.p,5120,2,1e-20f,s);
    auto normalized = big;
    for (int t = 0; t < 2; ++t) { double ss = 0; for (int i = 0; i < 5120; ++i) ss += double(big[t*5120+i])*big[t*5120+i]; for (int i = 0; i < 5120; ++i) normalized[t*5120+i] /= float(std::sqrt(ss/5120+1e-20)); }
    near(norm.get(s),normalized,"RMS 5120 in-place");
}
void engram_router_ffn(cudaStream_t s) {
    constexpr int H = 256, T = 2;
    auto r = values(H*4*T), kv = values(H*5*T,0.7f), qw = values(H*4,0.4f), kw = values(H*4,0.6f);
    for (int i = 0; i < H; ++i) r[i] = 0; // signed positive zero in the training gate
    std::vector<float> expected = r, gates(T*4);
    for (int t = 0; t < T; ++t) for (int h = 0; h < 4; ++h) {
        double xx = 0, kk = 0, dot = 0;
        for (int i = 0; i < H; ++i) { const float x = r[(t*4+h)*H+i], key = kv[(t*5+h)*H+i]; xx += double(x)*x; kk += double(key)*key; dot += double(x)*key*qw[h*H+i]*kw[h*H+i]; }
        const double score = dot/std::sqrt(xx/H+1e-20)/std::sqrt(kk/H+1e-20)/std::sqrt(double(H));
        const double gate = sigmoid(std::copysign(std::sqrt(std::max(std::abs(score),1e-6)),score)); gates[t*4+h] = float(gate);
        for (int i = 0; i < H; ++i) expected[(t*4+h)*H+i] += float(kv[(t*5+4)*H+i]*gate);
    }
    Device<float> dr(r), dkv(kv), dq(qw), dk(kw), dg(T*4);
    k::engram_add(dr.p,dkv.p,dq.p,dk.p,dg.p,H,T,1e-20f,s);
    near(dr.get(s),expected,"Engram grouped norm/value"); near(dg.get(s),gates,"Engram copysign gate");
    constexpr int E = 384, K = 6;
    auto logits = values(E*3,20), bias = values(E,0.1f);
    for (int i = 0; i < E; ++i) { logits[E+i] = -100; logits[E*2+i] = 100; bias[i] = i < 10 ? 0.5f : 0; }
    Device<float> dl(logits), db(bias), weights(3*K); Device<int32_t> ids(3*K);
    k::route(dl.p,db.p,ids.p,weights.p,E,K,3,1.5f,s);
    auto picked = ids.get(s); auto actual_weights = weights.get(s);
    for (int t = 0; t < 3; ++t) {
        std::vector<int> order(E); std::iota(order.begin(),order.end(),0); std::vector<double> p(E);
        for (int i = 0; i < E; ++i) { const double v = logits[t*E+i]; p[i] = std::sqrt(std::max(v,0.0)+std::log1p(std::exp(-std::abs(v)))); }
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return p[a]+bias[a] > p[b]+bias[b]; });
        double total = 0; for (int j = 0; j < K; ++j) total += p[order[j]];
        for (int j = 0; j < K; ++j) { require(picked[t*K+j] == order[j],"router id/tie"); require(std::abs(actual_weights[t*K+j]-p[order[j]]/std::max(total,6.103515625e-5)*1.5) < 2e-6,"router unbiased weights"); }
    }
    const std::vector<float> g = {30,-30,9,10,11,0}, u = {30,30,-30,-9,8,1};
    Device<float> gg(g), uu(u), yy(6); k::swiglu(gg.p,uu.p,yy.p,6,10,s);
    std::vector<float> ref(6); for (int i = 0; i < 6; ++i) { const double a = std::min(g[i],10.0f); ref[i] = float(a*sigmoid(a)*std::max(-10.0f,std::min(10.0f,u[i]))); }
    near(yy.get(s),ref,"DeepSeek pre-SiLU clamp");
}
void attention_index_rope(cudaStream_t s) {
    constexpr int D = 512, A = 3, W = 128, P = 130;
    auto query = values(D*A,0.8f), raw = values(D*W,0.2f), comp = values(D*7,0.3f);
    std::vector<half> hr(raw.size()), hc(comp.size());
    for (size_t i = 0; i < raw.size(); ++i) hr[i] = __float2half(raw[i]);
    for (size_t i = 0; i < comp.size(); ++i) hc[i] = __float2half(comp[i]);
    Device<float> q(query), sinks(std::vector<float>{0.3f,-2,100}), output(D*A);
    Device<half> r(hr), c(hc); Device<int32_t> ids(std::vector<int32_t>{4,1,5});
    k::attention(q.p,reinterpret_cast<uint16_t*>(r.p),reinterpret_cast<uint16_t*>(c.p),ids.p,sinks.p,output.p,D,A,W,P,3,s);
    std::vector<float> expected(D*A);
    for (int h = 0; h < A; ++h) {
        std::vector<double> score(W+3); double maximum = h == 0 ? 0.3 : h == 1 ? -2 : 100;
        for (int row = 0; row < W+3; ++row) {
            const half* src = row < W ? hr.data()+size_t((P-W+1+row)%W)*D : hc.data()+size_t(std::vector<int>{4,1,5}[row-W])*D;
            double v = 0; for (int i = 0; i < D; ++i) v += double(query[h*D+i])*__half2float(src[i]);
            score[row] = v/std::sqrt(double(D)); maximum = std::max(maximum,score[row]);
        }
        double denom = std::exp((h == 0 ? 0.3 : h == 1 ? -2 : 100)-maximum);
        for (auto& v : score) { v = std::exp(v-maximum); denom += v; }
        for (int i = 0; i < D; ++i) {
            double v = 0;
            for (int row = 0; row < W+3; ++row) { const half* src = row < W ? hr.data()+size_t((P-W+1+row)%W)*D : hc.data()+size_t(std::vector<int>{4,1,5}[row-W])*D; v += __half2float(src[i])*score[row]/denom; }
            expected[h*D+i] = float(v);
        }
    }
    near(output.get(s),expected,"SWA ring + sparse compressed + sink");
    auto score = values(1025); score[20] = score[40] = 2;
    Device<float> ds(score); Device<int32_t> ti(512); k::top_k(ds.p,ti.p,1025,512,s);
    auto picked = ti.get(s); std::vector<int> order(1025); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return score[a] > score[b]; });
    for (int j = 0; j < 512; ++j) require(picked[j] == order[j],"sparse top-k ties");
    Device<int32_t> sorted_ids(1025); Device<uint8_t> radix(k::top_k_scratch_bytes(1025)); ds.set(score);
    k::top_k_sorted(ds.p,sorted_ids.p,1025,512,radix.p,radix.n,s);
    const auto radix_ids = sorted_ids.get(s);
    for (int j = 0; j < 1025; ++j) require(radix_ids[j] == order[j],"radix top-k order/ties");
    {
        constexpr int capacity = 262144;
        Device<float> input(capacity); Device<int32_t> sorted(capacity); Device<uint8_t> workspace(k::top_k_scratch_bytes(capacity));
        for (int rows : {1,2,31,32,255,256,257,512,513,1025,65537,capacity}) {
            auto v = values(capacity); input.set(v);
            std::vector<int> ranked(size_t(rows),0); std::iota(ranked.begin(),ranked.end(),0);
            std::stable_sort(ranked.begin(),ranked.end(),[&](int a,int b) { return v[a] > v[b]; });
            std::cout << "radix rows=" << rows << std::endl;
            k::top_k_sorted(input.p,sorted.p,rows,std::min(rows,512),workspace.p,workspace.n,s);
            auto actual = sorted.get(s);
            for (int i = 0; i < std::min(rows,512); ++i) require(actual[size_t(i)] == ranked[size_t(i)],"variable rows/max radix scratch");
        }
    }
    auto iq = values(128*32,0.4f), iw = values(32,0.3f); std::vector<half> ik(7*128);
    for (size_t i = 0; i < ik.size(); ++i) ik[i] = __float2half(float(std::cos(i*0.27)));
    Device<float> diq(iq), diw(iw), scores(7); Device<half> dik(ik);
    k::index_scores(diq.p,diw.p,reinterpret_cast<uint16_t*>(dik.p),scores.p,128,32,7,s);
    std::vector<float> sr(7);
    for (int row = 0; row < 7; ++row) for (int h = 0; h < 32; ++h) { double v = 0; for (int i = 0; i < 128; ++i) v += double(iq[h*128+i])*__half2float(ik[row*128+i]); sr[row] += float(std::max(v,0.0)*iw[h]/64); }
    near(scores.get(s),sr,"indexer head ReLU/weight scaling");
    for (bool yarn : {false,true}) {
        k::Rope rc; rc.yarn = yarn; rc.base = yarn ? 160000 : 10000; rc.scale = yarn ? 1.0f/16 : 1;
        rc.original_context = 65536; rc.beta_fast = 32; rc.beta_slow = 1;
        Device<int32_t> pos(std::vector<int32_t>{262143}); q.set(query);
        k::rope(q.p,D,A,1,pos.p,rc,false,s); const auto rotated = q.get(s);
        require(rotated[0] == query[0],"RoPE modified non-rotary dimensions");
        require(std::abs(rotated[D-64]-query[D-64]) > 0.01f,"RoPE did not rotate tail");
        k::rope(q.p,D,A,1,pos.p,rc,true,s); near(q.get(s),query,yarn ? "YaRN 262143 inverse" : "RoPE 262143 inverse",2e-6f);
    }
    Device<float> pool_values(std::vector<float>{3,8,5,2}), pool_scores(std::vector<float>{100,-100,99,-99}), pooled(2);
    k::pool(pool_values.p,pool_scores.p,pooled.p,2,2,s);
    near(pooled.get(s),std::vector<float>{float((3*std::exp(1.0)+5)/(std::exp(1.0)+1)),float((8+2*std::exp(1.0))/(1+std::exp(1.0)))},"compressor stable channel pooling");
}
void dspark_math(cudaStream_t s) {
    constexpr int H = 64, F = 64, E = 3, K = 2, T = 3;
    const int table[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    auto matrix = [&](int rows,int salt) {
        std::vector<uint8_t> w(size_t(rows)*H/32*17);
        for (int r = 0; r < rows; ++r) for (int b = 0; b < H/32; ++b) {
            auto* p = w.data()+(size_t(r)*(H/32)+b)*17; p[0] = uint8_t(123+(r+b+salt)%5);
            for (int j = 0; j < 16; ++j) p[1+j] = uint8_t(((r+j+salt)%16)|(((r+3*j+salt)%16)<<4));
        }
        return w;
    };
    auto dot = [&](const std::vector<uint8_t>& w,int row,const float* x) {
        double v = 0;
        for (int i = 0; i < H; ++i) {
            const auto* b = w.data()+(size_t(row)*(H/32)+i/32)*17;
            const int q = (b[1+i%16] >> ((i%32)/16*4))&15;
            v += std::ldexp(double(table[q]),int(b[0])-128)*x[i];
        }
        return float(v);
    };
    const auto g = matrix(E*F,0), u = matrix(E*F,2), d = matrix(E*H,5); const auto x = values(T*H,0.3f);
    Device<uint8_t> dg(g), du(u), dd(d); Device<float> dx(x), out(T*F);
    k::mxfp4_mmv(dg.p,dx.p,out.p,H,F,T,s); std::vector<float> expected(T*F);
    for (int t = 0; t < T; ++t) for (int r = 0; r < F; ++r) expected[t*F+r] = dot(g,r,x.data()+t*H);
    near(out.get(s),expected,"DSpark original MXFP4 values");
    const std::vector<int32_t> picks = {2,0,1,2,0,1}; Device<int32_t> ids(picks); Device<float> hidden(T*K*F), routed(T*K*H);
    k::mxfp4_gu(dg.p,du.p,dx.p,ids.p,hidden.p,H,F,K,T,0.2f,s);
    std::vector<float> href(T*K*F), oref(T*K*H);
    for (int entry = 0; entry < T*K; ++entry) for (int r = 0; r < F; ++r) {
        const float gate = std::min(0.2f,dot(g,picks[entry]*F+r,x.data()+(entry/K)*H));
        const float up = std::clamp(dot(u,picks[entry]*F+r,x.data()+(entry/K)*H),-0.2f,0.2f);
        href[entry*F+r] = float(gate*sigmoid(gate)*up);
    }
    near(hidden.get(s),href,"DSpark MXFP4 expert ids/clamp");
    k::mxfp4_down(dd.p,hidden.p,ids.p,routed.p,F,H,K,T,s);
    for (int entry = 0; entry < T*K; ++entry) for (int r = 0; r < H; ++r) oref[entry*H+r] = dot(d,picks[entry]*H+r,href.data()+entry*F);
    near(routed.get(s),oref,"DSpark MXFP4 routed down");
    constexpr int W = 5, P = 7, End = 10, Capacity = 13;
    const auto q = values(32,0.7f), rv = values(Capacity*32,0.6f); std::vector<half> cache(rv.size());
    for (size_t i = 0; i < cache.size(); ++i) cache[i] = __float2half(rv[i]);
    Device<float> query(q), sink(std::vector<float>{0.4f}), attn(32); Device<half> raw(cache);
    k::draft_attention(query.p,reinterpret_cast<uint16_t*>(raw.p),sink.p,attn.p,32,1,W,P,End,Capacity,s);
    std::vector<double> score(End-(P-W+1)+1); double maximum = 0.4;
    for (size_t j = 0; j < score.size(); ++j) {
        const int row = P-W+1+int(j); double v = 0;
        for (int i = 0; i < 32; ++i) v += double(q[i])*__half2float(cache[(row%Capacity)*32+i]);
        score[j] = v/std::sqrt(32.0); maximum = std::max(maximum,score[j]);
    }
    double denom = std::exp(0.4-maximum); for (auto& v : score) { v = std::exp(v-maximum); denom += v; }
    std::vector<float> ref(32);
    for (int i = 0; i < 32; ++i) { double v = 0; for (size_t j = 0; j < score.size(); ++j) v += __half2float(cache[((P-W+1+int(j))%Capacity)*32+i])*score[j]/denom; ref[i] = float(v); }
    near(attn.get(s),ref,"DSpark noncausal noise window");
}
int main() {
    try {
        int n = 0; ck(cudaGetDeviceCount(&n)); require(n > 0,"no CUDA devices");
        for (int device = 0; device < n; ++device) {
            ck(cudaSetDevice(device)); cudaStream_t s; ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            std::cout << "device=" << device << '\n';
            hyperconnections(s); engram_router_ffn(s); attention_index_rope(s); dspark_math(s);
            ck(cudaGetLastError()); ck(cudaStreamDestroy(s));
        }
        std::cout << "Native DeepSeek V4.1 CUDA numerical tests PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
