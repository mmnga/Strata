#include "strata/deepseek41/engine.hpp"
#include "strata/deepseek41/model.hpp"
#include "strata/deepseek41/resident.hpp"
#include "strata/deepseek41/cpu_experts.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/expert_cache.hpp"
#include "strata/deepseek41/lookahead.hpp"
#include "strata/deepseek41/streamed_experts.hpp"
#include "strata/deepseek41/ffn_plan.hpp"
#include "strata/core/graph.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <string>
#include <vector>
#include <source_location>
#include <deque>
#include <cstdlib>
#include <atomic>
#include <tuple>
#include <poll.h>
#include <immintrin.h>
#include <unistd.h>

namespace strata::deepseek41 {
namespace {
namespace K = strata::kernels;
namespace C = strata::deepseek41::cuda;
using Clock = std::chrono::steady_clock;
constexpr int Batch = 8;
void ck(cudaError_t e,std::source_location at = std::source_location::current()) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(at.file_name())+":"+std::to_string(at.line())+": "+cudaGetErrorString(e));
}
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
std::string block(int l) { return "blk."+std::to_string(l)+"."; }
struct Buffer {
    int device;
    void* p = nullptr;
    size_t bytes;
    Buffer(int d,size_t n) : device(d),bytes(n) { ck(cudaSetDevice(d)); ck(cudaMalloc(&p,n)); }
    ~Buffer() { cudaSetDevice(device); cudaFree(p); }
    float* f(size_t off = 0) const { return static_cast<float*>(p)+off; }
    int32_t* i() const { return static_cast<int32_t*>(p); }
    uint16_t* h() const { return static_cast<uint16_t*>(p); }
};
template<class T> struct Host {
    T* p = nullptr; size_t n;
    explicit Host(size_t size,unsigned flags=0) : n(size) {
        if(flags) ck(cudaHostAlloc(&p,n*sizeof(T),flags));
        else ck(cudaMallocHost(&p,n*sizeof(T)));
    }
    ~Host() { cudaFreeHost(p); }
};
struct Matrix {
    std::unique_ptr<Buffer> storage;
    int type = 0, in = 0, out = 0;
};
struct Rank {
    int id, rank, count;
    cudaStream_t stream = nullptr;
    float* mapped_cpu_parts = nullptr;
    float* mapped_attention_parts = nullptr;
    uint32_t* mapped_tp_ready = nullptr;
    uint32_t* mapped_tp_abort = nullptr;
    float* mapped_activation = nullptr;
    int32_t* mapped_route_ids = nullptr;
    uint32_t* mapped_route_bell = nullptr;
    uint32_t* mapped_ffn_ready = nullptr;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,std::unique_ptr<Buffer>> buffers;
    std::map<std::tuple<int,int,int>,strata::core::CapturedGraph> graphs;
    bool graph_enabled = [] { const char* v = std::getenv("STRATA_DS41_GRAPHS"); return !v || std::atoi(v) != 0; }();
    bool wide_il = [] { const char* v=std::getenv("STRATA_DS41_WIDE_IL"); return v && std::atoi(v)!=0; }();
    std::vector<std::unique_ptr<Buffer>> raw, compressed, keys, projections, pool_scores, selected;
    explicit Rank(int d,int r,int n) : id(d),rank(r),count(n) { ck(cudaSetDevice(id)); ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
    ~Rank() { cudaSetDevice(id); cudaStreamSynchronize(stream); graphs.clear(); matrices.clear(); buffers.clear(); raw.clear(); compressed.clear(); keys.clear(); projections.clear(); pool_scores.clear(); selected.clear(); cudaStreamDestroy(stream); }
    template<class F> void launch(int layer,int phase,int tokens,F&& body) {
        if (!graph_enabled) { body(); return; }
        auto& graph = graphs[{layer,phase,tokens}]; std::string error;
        if (!graph.valid()) {
            if (!graph.begin(stream,error)) throw std::runtime_error(error);
            try { body(); ck(cudaGetLastError()); }
            catch (...) {
                cudaGraph_t abandoned = nullptr; cudaStreamEndCapture(stream,&abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                throw;
            }
            if (!graph.end(stream,error)) throw std::runtime_error(error);
        }
        if (!graph.launch(stream,error)) throw std::runtime_error(error);
    }
    Buffer& buffer(const std::string& name,size_t bytes) {
        auto& p = buffers[name]; if (!p) p = std::make_unique<Buffer>(id,bytes*Batch);
        if (p->bytes < bytes) throw std::runtime_error("DeepSeek scratch capacity: "+name);
        return *p;
    }
    float* f(const std::string& name,size_t count) { return buffer(name,count*4).f(); }
    float* weight(const std::string& name) { return matrices.at(name).storage->f(); }
    Matrix& w(const std::string& name) { return matrices.at(name); }
    void load(const std::string& name,TensorRef ref,const Config& cfg) {
        const auto& shape = ref.tensor->shape;
        Matrix mat; mat.type = int(ref.tensor->type); mat.in = int(shape[0]);
        uint64_t elements = 1; for (uint64_t d : shape) elements *= d;
        mat.out = int(elements/shape[0]);
        const bool query = name.find(".attn_q_b.weight") != std::string::npos && name.find("indexer.") == std::string::npos;
        const bool oa = name.find(".attn_output_a.weight") != std::string::npos;
        const bool ob = name.find(".attn_output_b.weight") != std::string::npos;
        std::vector<uint8_t> slice;
        const uint8_t* src = ref.data(); size_t bytes = ref.bytes();
        if (query || oa) { bytes /= size_t(count); mat.out /= count; src += bytes*rank; }
        if (ob) {
            const size_t row = bytes/size_t(mat.out), partial = row/size_t(count);
            slice.resize(bytes/size_t(count));
            for (int r = 0; r < mat.out; ++r) std::memcpy(slice.data()+size_t(r)*partial,src+size_t(r)*row+size_t(rank)*partial,partial);
            src = slice.data(); bytes = slice.size(); mat.in /= count;
        }
        if (mat.type == 30 || mat.type == 1) {
            std::vector<float> floats(size_t(mat.in)*mat.out);
            if (mat.type == 30) dequantize_bf16(src,floats.data(),int(floats.size())); else dequantize_f16(src,floats.data(),int(floats.size()));
            mat.type = 0; bytes = floats.size()*4; mat.storage = std::make_unique<Buffer>(id,bytes);
            ck(cudaMemcpy(mat.storage->p,floats.data(),bytes,cudaMemcpyHostToDevice));
        } else {
            if (mat.type != 0 && mat.type != 39 && !K::native_mmvq_supported(mat.type)) throw std::runtime_error("unsupported native matrix: "+name);
            mat.storage = std::make_unique<Buffer>(id,bytes); ck(cudaMemcpy(mat.storage->p,src,bytes,cudaMemcpyHostToDevice));
        }
        matrices.emplace(name,std::move(mat));
        (void)cfg;
    }
    void quant_mat(int type,const void* weights,const float* x,float* y,int in,int out,int tokens) {
        void* scratch = buffer("q8",K::native_q8_1_bytes(in,tokens)).p;
        if(wide_il && tokens>=5 && tokens<=8 && (type==12 || type==14)) {
            void* interleaved=buffer("q8_il",K::native_q8_1_il_bytes(in,tokens)).p;
            K::native_quantize_q8_1_il_wide(x,scratch,interleaved,in,tokens,stream);
            K::native_mmvq_il_wide(type,weights,interleaved,y,in,out,tokens,stream);
        } else if (tokens > 1 && tokens <= 4 && K::native_mmvq_il_supported(type,tokens,out)) {
            void* interleaved = buffer("q8_il",K::native_q8_1_il_bytes(in,tokens)).p;
            K::native_quantize_q8_1_il(x,scratch,interleaved,in,tokens,stream);
            K::native_mmvq_il(type,weights,scratch,interleaved,y,in,out,tokens,stream);
        }
        else {
            K::native_quantize_q8_1(x,scratch,in,tokens,stream);
            K::native_mmvq(type,weights,scratch,y,in,out,tokens,stream);
        }
    }
    void mat(const std::string& name,const float* x,float* y,int tokens = 1) {
        const auto& m = matrices.at(name);
        if (m.type == 0) C::dense(m.storage->f(),x,y,m.in,m.out,tokens,stream);
        else if (m.type == 39) C::mxfp4_mmv(m.storage->p,x,y,m.in,m.out,tokens,stream);
        else quant_mat(m.type,m.storage->p,x,y,m.in,m.out,tokens);
    }
    void norm(const std::string& name,const float* x,float* y,int width,float eps,int tokens = 1) { C::rms(x,weight(name),y,width,tokens,eps,stream); }
};

bool owner_ffn_weight(const std::string& name) {
    if(name.rfind("blk.",0)!=0) return false;
    const auto dot=name.find('.',4);
    if(dot==std::string::npos) return false;
    const auto sub=name.substr(dot+1);
    return sub.rfind("ffn_",0)==0 || sub.rfind("hc_ffn_",0)==0 || sub=="exp_probs_b.bias";
}

// Shared/routing projections are consumed only by the FFN owner even when
// HC and its normalizations remain mirrored on both attention ranks.
bool owner_ffn_projection(const std::string& name) {
    if(name.rfind("blk.",0)!=0) return false;
    const auto dot=name.find('.',4); if(dot==std::string::npos) return false;
    const auto sub=name.substr(dot+1);
    return sub=="ffn_gate_inp.weight" || sub=="ffn_gate_shexp.weight" ||
           sub=="ffn_up_shexp.weight" || sub=="ffn_down_shexp.weight";
}

class Engine {
public:
    Model model;
    Config c;
    ResidentWeights resident;
    CpuExperts cpu;
    std::unique_ptr<ExpertCache> expert_cache;
    std::unique_ptr<Lookahead> lookahead;
    std::unique_ptr<StreamedExperts> streamed;
    std::vector<std::unique_ptr<Rank>> ranks;
    bool direct_ffn = [] { const char* v=std::getenv("STRATA_DS41_DIRECT_FFN"); return v && std::atoi(v)!=0; }();
    bool overlap_ffn = [] { const char* v=std::getenv("STRATA_DS41_FFN_OVERLAP"); return v && std::atoi(v)!=0; }();
    bool grouped_oa = [] { const char* v=std::getenv("STRATA_DS41_GROUPED_OA"); return v && std::atoi(v)!=0; }();
    bool route_handoff = [] { const char* v=std::getenv("STRATA_DS41_ROUTE_HANDOFF"); return v && std::atoi(v)!=0; }();
    bool tp_graph = [] { const char* v=std::getenv("STRATA_DS41_TP_GRAPH"); return v && std::atoi(v)!=0; }();
    bool hc_pre_norm = [] { const char* v=std::getenv("STRATA_DS41_HC_PRE_NORM"); return v && std::atoi(v)!=0; }();
    bool ffn_post_fuse = [] { const char* v=std::getenv("STRATA_DS41_FFN_POST_FUSE"); return v && std::atoi(v)!=0; }();
    bool ffn_graph = [] { const char* v=std::getenv("STRATA_DS41_FFN_GRAPH"); return v && std::atoi(v)!=0; }();
    bool normalize_attention = [] { const char* v=std::getenv("STRATA_DS41_ATTENTION_WEIGHTS"); return v && std::atoi(v)!=0; }();
    bool chunked_attention = [] { const char* v=std::getenv("STRATA_DS41_ATTENTION_CHUNKS"); return v && std::atoi(v)!=0; }();
    bool parallel_values = [] { const char* v=std::getenv("STRATA_DS41_ATTENTION_VALUES"); return v && std::atoi(v)!=0; }();
    bool prefetch_values = [] { const char* v=std::getenv("STRATA_DS41_VALUE_PREFETCH"); return v && std::atoi(v)!=0; }();
    bool prefetch_query = [] { const char* v=std::getenv("STRATA_DS41_QUERY_PREFETCH"); return v && std::atoi(v)!=0; }();
    bool score_grid = [] { const char* v=std::getenv("STRATA_DS41_SCORE_GRID"); return v && std::atoi(v)!=0; }();
    bool value_head_pair = [] { const char* v=std::getenv("STRATA_DS41_VALUE_HEAD_PAIR"); return v && std::atoi(v)!=0; }();
    Host<float> attention_parts, transfer, cpu_parts, activation, hc_transfer, tp_sum;
    Host<uint32_t> tp_ready{size_t(c.layers)*18,cudaHostAllocMapped|cudaHostAllocPortable};
    Host<uint32_t> tp_abort{1,cudaHostAllocMapped|cudaHostAllocPortable};
    Host<uint32_t> route_bell{9,cudaHostAllocMapped|cudaHostAllocPortable};
    uint32_t route_sequence = 0;
    Host<FfnPlan> ffn_plan{1};
    Host<int32_t> route_ids, picked;
    Host<unsigned long long> group_ptr;
    Host<int32_t> group_start, group_dst, group_tok, group_count;
    std::vector<int32_t> history;
    std::vector<float> engram_input;
    int position = 0, context, last_rows = 1;
    bool speculative = false;
    bool attention_replay = [] { const char* v = std::getenv("STRATA_DS41_ATTENTION_GRAPH"); return v && std::atoi(v) != 0; }();
    bool hc_owner = [] { const char* v=std::getenv("STRATA_DS41_HC_OWNER"); return v && std::atoi(v)!=0; }();
    bool owner_weights = [] { const char* v=std::getenv("STRATA_DS41_OWNER_WEIGHTS"); return v && std::atoi(v)!=0; }();
    std::vector<std::vector<int32_t>> window_routes;
    strata::kernels::cpu::ThreadAffinity affinity;
    Engine(const std::string& path,int ctx,int threads,int gpu_count,int cache_mib,int depth) : model(path),c(model.config()),cpu(threads),
        attention_parts(size_t(c.hidden)*gpu_count*Batch,tp_graph ? cudaHostAllocMapped|cudaHostAllocPortable : 0),transfer(size_t(c.hidden)*Batch),cpu_parts(size_t(c.hidden)*c.top_k*Batch,direct_ffn ? cudaHostAllocMapped|cudaHostAllocPortable : 0),activation(size_t(c.hidden)*Batch,route_handoff ? cudaHostAllocMapped|cudaHostAllocPortable : 0),hc_transfer(size_t(c.hidden*4+4)*Batch),tp_sum(size_t(c.hidden)*Batch),route_ids(c.top_k*Batch,route_handoff ? cudaHostAllocMapped|cudaHostAllocPortable : 0),picked(Batch),
        group_ptr(c.top_k*Batch),group_start(c.top_k*Batch+1),group_dst(c.top_k*Batch),group_tok(c.top_k*Batch),group_count(1),context(ctx),window_routes(c.layers) {
        if (c.hc != 4 || c.index_heads > 32 || c.experts > 512 || c.heads%gpu_count || c.o_groups%gpu_count ||
            ctx < c.sliding_window || ctx > c.context_train) throw std::runtime_error("unsupported native DeepSeek geometry/context");
        if(direct_ffn && c.top_k*Batch>FfnPlan::Capacity) throw std::runtime_error("native FFN packet capacity");
        *ffn_plan.p=FfnPlan{};
        std::fill_n(route_bell.p,route_bell.n,0u);
        std::fill_n(tp_ready.p,tp_ready.n,0u); *tp_abort.p=0;
        if(hc_pre_norm && c.hidden!=5120) throw std::runtime_error("fused HC pre/RMS requires H5120");
        if(ffn_post_fuse && (!ffn_graph || !direct_ffn || hc_owner || tp_graph))
            throw std::runtime_error("fused FFN post requires original FFN graph, direct FFN, mirrored HC, CPU TP sum");
        if(ffn_graph && (gpu_count!=2 || tp_graph || !route_handoff || !direct_ffn || hc_owner))
            throw std::runtime_error("FFN graph requires TP2, route handoff, direct FFN, mirrored HC and TP graph disabled");
        if(tp_graph && (gpu_count!=2 || !route_handoff || !attention_replay || hc_owner))
            throw std::runtime_error("TP graph requires TP2, route handoff, attention graph and mirrored HC");
        if(chunked_attention && (c.head_dim!=512 || c.sliding_window!=128 || c.index_top_k!=512 || (c.heads/gpu_count)%4))
            throw std::runtime_error("chunked attention requires original D512/window128/top512/head groups4");
        if(parallel_values && (chunked_attention || c.head_dim!=512 || c.sliding_window!=128 || c.index_top_k!=512))
            throw std::runtime_error("parallel values require original D512/window128/top512 and split-K disabled");
        if(prefetch_values && !parallel_values) throw std::runtime_error("value prefetch requires parallel values");
        if(prefetch_query && !parallel_values) throw std::runtime_error("query prefetch requires parallel values");
        if(score_grid && (!parallel_values || !prefetch_query || prefetch_values || !attention_replay))
            throw std::runtime_error("score grid requires query/parallel values, attention graph and value prefetch disabled");
        if(value_head_pair && (!score_grid || (c.heads/gpu_count)%2))
            throw std::runtime_error("value head pairing requires score grid and even heads per rank");
        affinity = K::cpu::pin_current_thread(cpu.host_core());
        for (int r = 0; r < gpu_count; ++r) ranks.push_back(std::make_unique<Rank>(r,r,gpu_count));
        for (auto& rp : ranks) {
            auto& r = *rp; ck(cudaSetDevice(r.id));
            size_t omitted_bytes=0; int omitted=0;
            for (size_t s = 0; s < model.gguf().size(); ++s) for (const auto& t : model.gguf().shard(s).tensors()) {
                if (t.name.find("_exps.weight") != std::string::npos || t.name.find("engram_embd.weight") != std::string::npos) continue;
                // Output and embedding are shared from rank zero; every layer
                // mirrors HC/KV and splits heads/output groups across ranks.
                if (r.rank && t.name.rfind("blk.",0) != 0) continue;
                if((hc_owner && owner_ffn_weight(t.name)) || (owner_weights && owner_ffn_projection(t.name))) {
                    const int layer=std::stoi(t.name.substr(4));
                    if(size_t(layer)*ranks.size()/size_t(c.layers)!=size_t(r.rank)) {
                        if(owner_weights && owner_ffn_projection(t.name)) {
                            const auto ref=model.tensor(t.name); size_t bytes=ref.bytes();
                            if(t.type==30 || t.type==1) { size_t elements=1;for(auto d:t.shape)elements*=d;bytes=elements*4; }
                            omitted_bytes+=bytes; ++omitted;
                        }
                        continue;
                    }
                }
                r.load(t.name,model.tensor(t.name),c);
            }
            if(owner_weights) std::fprintf(stderr,"strata-deepseek41: rank=%d owner-only FFN projections omitted=%d GPU_MiB=%.6f\n",r.id,omitted,double(omitted_bytes)/(1<<20));
            allocate(r);
            if(grouped_oa) for(int l=0;l<c.layers;++l) {
                const auto& w=r.w(block(l)+"attn_output_a.weight");
                if(w.type!=12 && w.type!=14) throw std::runtime_error("grouped output-A requires original Q4_K/Q6_K");
            }
            if(tp_graph) {
                ck(cudaHostGetDevicePointer(&r.mapped_attention_parts,attention_parts.p,0));
                ck(cudaHostGetDevicePointer(&r.mapped_tp_ready,tp_ready.p,0));
                ck(cudaHostGetDevicePointer(&r.mapped_tp_abort,tp_abort.p,0));
            }
            if(ffn_graph) {
                // TP graph is disabled; reuse its first otherwise-unused Host word.
                ck(cudaHostGetDevicePointer(&r.mapped_ffn_ready,tp_ready.p,0));
                ck(cudaHostGetDevicePointer(&r.mapped_tp_abort,tp_abort.p,0));
            }
            if(route_handoff) {
                ck(cudaHostGetDevicePointer(&r.mapped_activation,activation.p,0));
                ck(cudaHostGetDevicePointer(&r.mapped_route_ids,route_ids.p,0));
                ck(cudaHostGetDevicePointer(&r.mapped_route_bell,route_bell.p,0));
            }
            if(direct_ffn) {
                ck(cudaHostGetDevicePointer(&r.mapped_cpu_parts,cpu_parts.p,0));
                r.buffer("ffn_plan",sizeof(FfnPlan));
            }
            std::fprintf(stderr,"strata-deepseek41: native rank=%d heads=%d output_groups=%d dense weights loaded\n",r.id,c.heads/gpu_count,c.o_groups/gpu_count);
        }
        resident.load(model);
        enable_cache(cache_mib,depth);
        std::fprintf(stderr,"strata-deepseek41: native Strata driver, TP=%d all %d layers, CPU math threads=%d, context=%d F16 KV\n",gpu_count,c.layers,cpu.threads(),context);
    }
    void abort_tp() { *static_cast<volatile uint32_t*>(tp_abort.p)=1; std::atomic_thread_fence(std::memory_order_release); }
    ~Engine() { abort_tp(); streamed.reset(); lookahead.reset(); expert_cache.reset(); ranks.clear(); K::cpu::restore_thread_affinity(affinity); }
    void enable_attention_values() {
        if(!parallel_values) return;
        for(auto& rp:ranks) {ck(cudaSetDevice(rp->id));rp->buffer("attention_values",size_t(c.heads/rp->count)*640*12);}
    }
    void enable_chunked_attention() {
        if(!chunked_attention) return;
        for(auto& rp:ranks) {
            ck(cudaSetDevice(rp->id));
            // Use the established post-cache reserve, like cold expert scratch.
            rp->buffer("attention_chunks",size_t(c.heads/rp->count)*544*((c.sliding_window+c.index_top_k+63)/64)*4);
        }
    }
    void enable_streamed() {
        const char* setting=std::getenv("STRATA_DS41_STREAM_MISSES");
        if(!setting || std::atoi(setting)==0) return;
        std::vector<StreamLayerWeights> layers;
        for(int l=0;l<c.layers;++l) {
            const auto p=block(l); const auto g=model.tensor(p+"ffn_gate_exps.weight"), d=model.tensor(p+"ffn_down_exps.weight");
            StreamLayerWeights layer; layer.experts=c.experts; auto& w=layer.base;
            w.gate=resident.data(p+"ffn_gate_exps.weight"); w.up=resident.data(p+"ffn_up_exps.weight"); w.down=resident.data(p+"ffn_down_exps.weight");
            w.hidden=c.hidden; w.ff=c.expert_ff; w.gate_type=int(g.tensor->type); w.down_type=int(d.tensor->type);
            w.gate_row=g.bytes()/size_t(c.experts)/size_t(c.expert_ff); w.down_row=d.bytes()/size_t(c.experts)/size_t(c.hidden); w.limit=float(c.clamp_expert[l]);
            layers.push_back(layer);
        }
        // The bounded scratch comes from the existing 1536 MiB reserve,
        // after the established cache slot budget has been allocated.
        streamed=std::make_unique<StreamedExperts>(std::move(layers),int(ranks.size()),3);
    }
    void enable_cache(int mib,int depth,bool draft_loaded = false) {
        if (!mib) return;
        if (expert_cache) throw std::runtime_error("native cache already initialized");
        for (int l = 0; l < c.layers; ++l) {
            const auto p = block(l);
            if (!K::native_expert_supported(int(model.tensor(p+"ffn_gate_exps.weight").tensor->type),
                    int(model.tensor(p+"ffn_down_exps.weight").tensor->type),c.hidden,c.expert_ff))
                throw std::runtime_error("native grouped expert format is not built: layer "+std::to_string(l));
        }
        expert_cache = std::make_unique<ExpertCache>(model,resident,int(ranks.size()),std::vector<int>(ranks.size(),mib),draft_loaded ? 0 : 7680);
        if (depth) lookahead = std::make_unique<Lookahead>(model,*expert_cache,depth,std::min(12,c.experts));
    }
    void allocate(Rank& r) {
        const size_t H = c.hidden, D = c.head_dim, A = c.heads/r.count, I = c.index_dim, F = c.expert_ff*c.shared_experts;
        r.f("residual",H*4); r.f("flat_norm",H*4); r.f("pre",4); r.f("attn_pre",4); r.f("post",4); r.f("comb",16); r.f("mix",24);
        r.f("x",H); r.f("norm",H); r.f("qr",c.q_rank); r.f("q",D*A); r.f("kv",D); r.f("attn",D*A);
        r.f("oa",size_t(c.o_rank)*c.o_groups/r.count); r.f("partial",H); r.f("attn_sum",H);
        r.f("group_in",c.heads/c.o_groups*D); r.f("group_out",c.o_rank); r.f("features",H*3);
        r.f("gate",F); r.f("up",F); r.f("hidden",F); r.f("shared",H); r.f("ffn",H); r.f("parts",H*c.top_k);
        r.f("router",c.experts); r.f("route_weights",c.top_k); r.buffer("route_ids",size_t(c.top_k)*4);
        r.f("expert_gate",c.expert_ff); r.f("expert_up",c.expert_ff); r.f("expert_hidden",c.expert_ff);
        r.buffer("expert_xq",K::native_q8_1_bytes(c.hidden)); r.buffer("expert_hq",K::native_q8_1_bytes(c.expert_ff));
        r.buffer("expert_group_ptr",size_t(c.top_k*Batch)*8);
        r.buffer("expert_group_start",size_t(c.top_k*Batch+1)*4);
        r.buffer("expert_group_dst",size_t(c.top_k*Batch)*4); r.buffer("expert_group_tok",size_t(c.top_k*Batch)*4);
        r.buffer("expert_group_count",4);
        r.buffer("expert_scratch",K::native_expert_scratch_bytes(c.top_k*Batch,c.expert_ff));
        r.f("index_q",size_t(c.index_heads)*I); r.f("index_w",c.index_heads); r.f("index_scores",context);
        r.buffer("index_sort",C::top_k_scratch_bytes(context));
        r.f("latent",D); r.f("index_key",I); r.f("engram_input",c.engram.buckets()*c.engram.head_dim); r.f("engram_kv",H*5);
        r.f("compress_values",D); r.f("compress_scores",D);
        r.buffer("token",4); r.buffer("position",4); r.buffer("compress_position_2",4);
        const int quant_width=grouped_oa ? std::max(int(H*4),int(A*D)) : int(H*4);
        r.buffer("q8",K::native_q8_1_bytes(quant_width));
        r.buffer("q8_il",K::native_q8_1_il_bytes(quant_width,4));
        if (!r.rank) { r.f("logits",c.vocab); r.buffer("pick",4); r.buffer("penalty_history",size_t(context)*4); }
        r.raw.resize(c.layers); r.compressed.resize(c.layers); r.keys.resize(c.layers); r.projections.resize(c.layers); r.pool_scores.resize(c.layers); r.selected.resize(c.layers);
        for (int l = 0; l < c.layers; ++l) {
            r.raw[l] = std::make_unique<Buffer>(r.id,size_t(c.sliding_window+Batch)*D*2);
            const auto& streams = model.streams()[l];
            if (streams.publishes_kv) {
                const int ratio = int(c.compress_ratios[l]);
                r.compressed[l] = std::make_unique<Buffer>(r.id,size_t((context+ratio-1)/ratio)*D*2);
                r.projections[l] = std::make_unique<Buffer>(r.id,D*Batch*2*4); r.pool_scores[l] = std::make_unique<Buffer>(r.id,D*Batch*2*4);
            }
            if (streams.publishes_keys) r.keys[l] = std::make_unique<Buffer>(r.id,size_t((context+int(c.compress_ratios[l])-1)/int(c.compress_ratios[l]))*I*2);
            if (streams.publishes_top_k) r.selected[l] = std::make_unique<Buffer>(r.id,size_t(context)*4*Batch);
        }
    }
    C::Rope rope_config(int layer) const {
        C::Rope cfg; cfg.dim = c.rope_dim;
        if (c.compress_ratios[layer]) { cfg.base = float(c.compress_rope_base); cfg.scale = float(1/c.rope_scale); cfg.original_context = c.rope_context_original; cfg.beta_fast = float(c.beta_fast); cfg.beta_slow = float(c.beta_slow); cfg.yarn = true; }
        else cfg.base = float(c.rope_base);
        return cfg;
    }
    void hc(Rank& r,int layer,const char* suffix,float* pre,int tokens = 1) {
        const auto prefix = block(layer)+"hc_"+suffix;
        C::rms(r.f("residual",c.hidden*4),nullptr,r.f("flat_norm",c.hidden*4),c.hidden*4,tokens,float(c.rms_eps),r.stream);
        r.mat(prefix+"_fn.weight",r.f("flat_norm",c.hidden*4),r.f("mix",24),tokens);
        C::hc_coeff(r.f("mix",24),r.weight(prefix+"_scale.weight"),r.weight(prefix+"_base.weight"),pre,r.f("post",4),r.f("comb",16),tokens,float(c.hc_eps),c.sinkhorn_iterations,r.stream);
    }
    void pre_norm(Rank& r,const std::string& name,const float* pre,int tokens) {
        const int H=c.hidden;
        if(hc_pre_norm) C::hc_pre_rms(r.f("residual",H*4),pre,r.weight(name),r.f("x",H),r.f("norm",H),H,tokens,float(c.rms_eps),r.stream);
        else { C::hc_pre(r.f("residual",H*4),pre,r.f("x",H),H,tokens,r.stream);
            r.norm(name,r.f("x",H),r.f("norm",H),H,float(c.rms_eps),tokens); }
    }
    void attention_prepare(Rank& r,int layer,int tokens) {
        const auto p = block(layer); const auto& streams = model.streams()[layer];
        const int H = c.hidden, D = c.head_dim, A = c.heads/r.count, ratio = int(c.compress_ratios[layer]);
        float* x = r.f("norm",H); float* qr = r.f("qr",c.q_rank);
        hc(r,layer,"attn",r.f("attn_pre",4),tokens);
        pre_norm(r,p+"attn_norm.weight",r.f("pre",4),tokens);
        r.mat(p+"attn_q_a.weight",x,qr,tokens); r.norm(p+"attn_q_a_norm.weight",qr,qr,c.q_rank,float(c.rms_eps),tokens);
        r.mat(p+"attn_q_b.weight",qr,r.f("q",D*A),tokens); r.mat(p+"attn_kv.weight",x,r.f("kv",D),tokens);
        r.norm(p+"attn_kv_a_norm.weight",r.f("kv",D),r.f("kv",D),D,float(c.rms_eps),tokens);
        C::rope(r.f("q",D*A),D,A,tokens,r.buffer("position",4).i(),rope_config(layer),false,r.stream);
        C::rope(r.f("kv",D),D,1,tokens,r.buffer("position",4).i(),rope_config(layer),false,r.stream);
        if (ratio && streams.publishes_kv) {
            r.mat(p+"attn_compressor_kv.weight",x,r.f("compress_values",D),tokens);
            if (ratio > 1) r.mat(p+"attn_compressor_gate.weight",x,r.f("compress_scores",D),tokens);
        }
        if (ratio && streams.publishes_top_k) {
            r.mat(p+"indexer.attn_q_b.weight",qr,r.f("index_q",c.index_heads*c.index_dim),tokens);
            C::rope(r.f("index_q",c.index_heads*c.index_dim),c.index_dim,c.index_heads,tokens,r.buffer("position",4).i(),rope_config(layer),false,r.stream);
            r.mat(p+"indexer.proj.weight",x,r.f("index_w",c.index_heads),tokens);
        }
    }
    void attention_output(Rank& r,int layer,int tokens) {
        const auto p = block(layer); const int H = c.hidden, D = c.head_dim, A = c.heads/r.count;
        C::rope(r.f("attn",D*A),D,A,tokens,r.buffer("position",4).i(),rope_config(layer),true,r.stream);
        const int groups = c.o_groups/r.count, group_dim = c.heads/c.o_groups*D;
        auto& oa = r.w(p+"attn_output_a.weight");
        const size_t group_bytes = K::native_mmvq_weight_bytes(oa.type,group_dim,c.o_rank);
        if(grouped_oa) {
            void* q=r.buffers.at("q8")->p; void* il=nullptr;
            if(tokens>1 && tokens<=4 && K::native_mmvq_il_supported(oa.type,tokens,c.o_rank)) {
                il=r.buffers.at("q8_il")->p;
                K::native_quantize_q8_1_il(r.f("attn",D*A),q,il,D*A,tokens,r.stream);
            } else K::native_quantize_q8_1(r.f("attn",D*A),q,D*A,tokens,r.stream);
            K::native_mmvq_grouped(oa.type,oa.storage->p,q,il,r.f("oa",c.o_rank*groups),group_dim,c.o_rank,groups,tokens,r.stream);
        } else for (int g = 0; g < groups; ++g) {
            const float* input = r.f("attn",D*A)+g*group_dim;
            float* result = r.f("oa",c.o_rank*groups)+g*c.o_rank;
            if (tokens > 1) C::strided_copy(input,r.f("group_in",group_dim),group_dim,tokens,D*A,group_dim,r.stream);
            r.quant_mat(oa.type,static_cast<uint8_t*>(oa.storage->p)+size_t(g)*group_bytes,tokens > 1 ? r.f("group_in",group_dim) : input,
                        tokens > 1 ? r.f("group_out",c.o_rank) : result,group_dim,c.o_rank,tokens);
            if (tokens > 1) C::strided_copy(r.f("group_out",c.o_rank),result,c.o_rank,tokens,c.o_rank,c.o_rank*groups,r.stream);
        }
        r.mat(p+"attn_output_b.weight",r.f("oa",c.o_rank*groups),r.f("partial",H),tokens);
        if(!tp_graph) ck(cudaMemcpyAsync(attention_parts.p+size_t(r.rank)*H*Batch,r.f("partial",H),size_t(H)*tokens*4,cudaMemcpyDeviceToHost,r.stream));
    }
    std::pair<int,int> attention_shape(int layer,int tokens) const {
        int bucket=1,level=0;
        if(c.compress_ratios[layer] && model.streams()[layer].publishes_top_k) {
            const int rows=(position+tokens)/int(c.compress_ratios[layer]);
            while(bucket<rows) {bucket*=2;++level;}
            bucket=std::min(bucket,context);
        }
        return {bucket,level};
    }
    void attention_body(Rank& r,int layer,int tokens,int bucket) {
        const auto p=block(layer); const auto& streams=model.streams()[layer];
        const int D=c.head_dim,A=c.heads/r.count,ratio=int(c.compress_ratios[layer]);
        const auto* pos=r.buffers.at("position")->i();
        const auto* comp_pos=r.buffers.at(ratio==2 ? "compress_position_2" : "position")->i();
        attention_prepare(r,layer,tokens);
        C::store_f16_positions(r.f("kv",D),r.raw[layer]->h(),D,tokens,pos,c.sliding_window+Batch,0,r.stream);
        if (ratio && streams.publishes_kv) {
            C::projection_ring(r.f("compress_values",D),ratio > 1 ? r.f("compress_scores",D) : nullptr,
                r.projections[layer]->f(),ratio > 1 ? r.pool_scores[layer]->f() : nullptr,D,tokens,pos,Batch*2,r.stream);
            C::pool_positions(r.projections[layer]->f(),r.pool_scores[layer]->f(),r.f("latent",D),D,tokens,pos,Batch*2,ratio,r.stream);
            r.norm(p+"attn_compressor_norm.weight",r.f("latent",D),r.f("latent",D),D,float(c.rms_eps),tokens);
            const int capacity = (context+ratio-1)/ratio;
            if (streams.publishes_keys) {
                r.mat(p+"indexer.attn_k.weight",r.f("latent",D),r.f("index_key",c.index_dim),tokens);
                r.norm(p+"indexer.k_norm.weight",r.f("index_key",c.index_dim),r.f("index_key",c.index_dim),c.index_dim,float(c.rms_eps),tokens);
                C::rope(r.f("index_key",c.index_dim),c.index_dim,1,tokens,comp_pos,rope_config(layer),false,r.stream);
                C::store_f16_positions(r.f("index_key",c.index_dim),r.keys[layer]->h(),c.index_dim,tokens,pos,capacity,ratio,r.stream);
            }
            C::rope(r.f("latent",D),D,1,tokens,comp_pos,rope_config(layer),false,r.stream);
            C::store_f16_positions(r.f("latent",D),r.compressed[layer]->h(),D,tokens,pos,capacity,ratio,r.stream);
        }
        if (ratio && streams.publishes_top_k) {
            C::index_scores_positions(r.f("index_q",c.index_heads*c.index_dim),r.f("index_w",c.index_heads),r.keys[streams.key_source]->h(),
                r.f("index_scores",context),c.index_dim,c.index_heads,bucket,tokens,pos,ratio,context,r.stream);
            auto& sort = *r.buffers.at("index_sort");
            for (int t = 0; t < tokens; ++t) C::top_k_sorted(r.f("index_scores",context)+size_t(t)*context,
                r.selected[layer]->i()+size_t(t)*context,bucket,std::min(bucket,c.index_top_k),sort.p,sort.bytes,r.stream);
        }
        auto attend=normalize_attention ? C::attention_positions_normalized : C::attention_positions;
        auto value_pass=prefetch_values ? C::attention_positions_values_prefetch : C::attention_positions_values;
        if(score_grid) C::attention_positions_score_grid(r.f("q",D*A),r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
            ratio ? r.selected[streams.top_k_source]->i() : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,r.f("attn",D*A),
            r.buffers.at("attention_values")->p,D,A,tokens,c.sliding_window,pos,ratio,ratio ? c.index_top_k : 0,context,c.sliding_window+Batch,r.stream,value_head_pair);
        else if(parallel_values && prefetch_query) C::attention_positions_query_prefetch(r.f("q",D*A),r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
            ratio ? r.selected[streams.top_k_source]->i() : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,r.f("attn",D*A),
            r.buffers.at("attention_values")->p,D,A,tokens,c.sliding_window,pos,ratio,ratio ? c.index_top_k : 0,context,c.sliding_window+Batch,r.stream,prefetch_values);
        else if(parallel_values) value_pass(r.f("q",D*A),r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
            ratio ? r.selected[streams.top_k_source]->i() : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,r.f("attn",D*A),
            r.buffers.at("attention_values")->p,D,A,tokens,c.sliding_window,pos,ratio,ratio ? c.index_top_k : 0,context,c.sliding_window+Batch,r.stream);
        else if(chunked_attention) C::attention_positions_chunked(r.f("q",D*A),r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
            ratio ? r.selected[streams.top_k_source]->i() : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,r.f("attn",D*A),
            r.buffers.at("attention_chunks")->f(),D,A,tokens,c.sliding_window,pos,ratio,ratio ? c.index_top_k : 0,context,c.sliding_window+Batch,r.stream);
        else attend(r.f("q",D*A),r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
            ratio ? r.selected[streams.top_k_source]->i() : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,r.f("attn",D*A),
            D,A,tokens,c.sliding_window,pos,ratio,ratio ? c.index_top_k : 0,context,c.sliding_window+Batch,r.stream);
        attention_output(r,layer,tokens);
    }
    void attention(Rank& r,int layer,int tokens = 1) {
        if(!attention_replay) {attention_legacy(r,layer,tokens);return;}
        const auto [bucket,level]=attention_shape(layer,tokens);
        r.launch(layer,8+level,tokens,[&] {attention_body(r,layer,tokens,bucket);});
    }
    void attention_ffn(Rank& r,int layer,int tokens) {
        const auto [bucket,level]=attention_shape(layer,tokens);
        const int H=c.hidden;
        auto* own=r.mapped_tp_ready+size_t(layer)*18+size_t(r.rank)*9;
        auto* peer=r.mapped_tp_ready+size_t(layer)*18+size_t(1-r.rank)*9;
        r.launch(layer,32+level,tokens,[&] {
            attention_body(r,layer,tokens,bucket);
            K::doorbell_publish(r.f("partial",H),nullptr,nullptr,size_t(H)*tokens,0,
                r.mapped_attention_parts+size_t(r.rank)*H*Batch,nullptr,nullptr,own,r.stream);
            C::tp_sum_wait(r.f("partial",H),r.mapped_attention_parts+size_t(1-r.rank)*H*Batch,
                own,peer,r.mapped_tp_abort,r.f("attn_sum",H),H*tokens,r.rank,r.stream);
            ffn_prepare(r,layer,tokens);
            if(size_t(layer)*ranks.size()/size_t(c.layers)==size_t(r.rank)) {
                route_prepare(r,layer,tokens);
                shared_expert(r,layer,tokens);
            }
        });
    }
    void attention_legacy(Rank& r,int layer,int tokens) {
        const auto p = block(layer); const auto& streams = model.streams()[layer];
        const int D = c.head_dim, A = c.heads/r.count, ratio = int(c.compress_ratios[layer]);
        const int32_t* comp_pos = r.buffer(ratio == 2 ? "compress_position_2" : "position",4).i();
        r.launch(layer,0,tokens,[&] { attention_prepare(r,layer,tokens); });
        for (int t = 0; t < tokens; ++t) C::store_f16(r.f("kv",D)+size_t(t)*D,r.raw[layer]->h(),D,(position+t)%(c.sliding_window+Batch),r.stream);
        if (ratio) {
            if (streams.publishes_kv) {
                for (int t = 0; t < tokens; ++t) {
                    const int pos = position+t, slot = pos%(Batch*2);
                    ck(cudaMemcpyAsync(r.projections[layer]->f(size_t(slot)*D),r.f("compress_values",D)+size_t(t)*D,size_t(D)*4,cudaMemcpyDeviceToDevice,r.stream));
                    if (ratio > 1) ck(cudaMemcpyAsync(r.pool_scores[layer]->f(size_t(slot)*D),r.f("compress_scores",D)+size_t(t)*D,size_t(D)*4,cudaMemcpyDeviceToDevice,r.stream));
                    if ((pos+1)%ratio) continue;
                    const int first_slot = (pos+1-ratio)%(Batch*2);
                    C::pool(r.projections[layer]->f(size_t(first_slot)*D),r.pool_scores[layer]->f(size_t(first_slot)*D),r.f("latent",D),D,ratio,r.stream);
                    r.norm(p+"attn_compressor_norm.weight",r.f("latent",D),r.f("latent",D),D,float(c.rms_eps));
                    if (streams.publishes_keys) {
                        r.mat(p+"indexer.attn_k.weight",r.f("latent",D),r.f("index_key",c.index_dim));
                        r.norm(p+"indexer.k_norm.weight",r.f("index_key",c.index_dim),r.f("index_key",c.index_dim),c.index_dim,float(c.rms_eps));
                        C::rope(r.f("index_key",c.index_dim),c.index_dim,1,1,comp_pos+t,rope_config(layer),false,r.stream);
                        C::store_f16(r.f("index_key",c.index_dim),r.keys[layer]->h(),c.index_dim,pos/ratio,r.stream);
                    }
                    C::rope(r.f("latent",D),D,1,1,comp_pos+t,rope_config(layer),false,r.stream);
                    C::store_f16(r.f("latent",D),r.compressed[layer]->h(),D,pos/ratio,r.stream);
                }
            }
            if (streams.publishes_top_k) {
                for (int t = 0; t < tokens; ++t) {
                    const int rows = (position+t+1)/ratio; if (!rows) continue;
                    C::index_scores(r.f("index_q",c.index_heads*c.index_dim)+size_t(t)*c.index_heads*c.index_dim,r.f("index_w",c.index_heads)+size_t(t)*c.index_heads,r.keys[streams.key_source]->h(),r.f("index_scores",context),c.index_dim,c.index_heads,rows,r.stream);
                    auto& sort = *r.buffers.at("index_sort");
                    C::top_k_sorted(r.f("index_scores",context),r.selected[layer]->i()+size_t(t)*context,rows,std::min(rows,c.index_top_k),sort.p,sort.bytes,r.stream);
                }
            }
        }
        for (int t = 0; t < tokens; ++t) {
            const int selected = ratio ? std::min((position+t+1)/ratio,c.index_top_k) : 0;
            C::attention(r.f("q",D*A)+size_t(t)*D*A,r.raw[layer]->h(),ratio ? r.compressed[streams.kv_source]->h() : nullptr,
                         ratio ? r.selected[streams.top_k_source]->i()+size_t(t)*context : nullptr,r.weight(p+"attn_sinks.weight")+size_t(r.rank)*A,
                         r.f("attn",D*A)+size_t(t)*D*A,D,A,c.sliding_window,position+t,selected,r.stream,c.sliding_window+Batch);
        }
        r.launch(layer,1,tokens,[&] { attention_output(r,layer,tokens); });
    }
    void sync() { for (auto& r : ranks) { ck(cudaSetDevice(r->id)); ck(cudaStreamSynchronize(r->stream)); } }
    void reset() { sync(); position = 0; history.clear(); }
    void engram(int layer,const int32_t* inputs,int tokens) {
        const int eg = model.streams()[layer].engram_index;
        if (eg < 0) return;
        const auto p = block(layer); const auto table = model.tensor(p+"engram_embd.weight");
        const size_t width = c.engram.buckets()*size_t(c.engram.head_dim);
        engram_input.resize(width*size_t(tokens));
        const size_t row_bytes = table.bytes()/table.tensor->shape[1];
        const uint8_t* data = resident.data(p+"engram_embd.weight");
        auto local_history = history;
        for (int t = 0; t < tokens; ++t) {
            const auto ids = engram_rows(c.engram,size_t(eg),inputs[t],local_history); local_history.push_back(inputs[t]);
            for (size_t j = 0; j < ids.size(); ++j) {
                if (table.tensor->type != 12 || c.engram.head_dim%256) throw std::runtime_error("Engram table must be original Q4_K");
                for (int b = 0; b < c.engram.head_dim/256; ++b) dequantize_q4_K(data+size_t(ids[j])*row_bytes+size_t(b)*144,engram_input.data()+size_t(t)*width+j*c.engram.head_dim+b*256);
            }
        }
        for (auto& rp : ranks) {
            auto& r = *rp; ck(cudaSetDevice(r.id));
            ck(cudaMemcpyAsync(r.f("engram_input",engram_input.size()),engram_input.data(),engram_input.size()*4,cudaMemcpyHostToDevice,r.stream));
            r.mat(p+"engram_wkv.weight",r.f("engram_input",engram_input.size()),r.f("engram_kv",c.hidden*5),tokens);
            C::engram_add(r.f("residual",c.hidden*4),r.f("engram_kv",c.hidden*5),r.weight(p+"engram_q.weight"),r.weight(p+"engram_k.weight"),nullptr,c.hidden,tokens,float(c.rms_eps),r.stream);
        }
        // The same host staging vector is reused at the next Engram layer.
        // Every intervening layer synchronizes its TP reduction first.
    }
    void ffn_prepare(Rank& r,int layer,int tokens) {
        const int H=c.hidden;
        C::hc_post(r.f("attn_sum",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,tokens,r.stream);
        hc(r,layer,"ffn",r.f("pre",4),tokens);
        pre_norm(r,block(layer)+"ffn_norm.weight",r.f("attn_pre",4),tokens);
    }
    void route_prepare(Rank& r,int layer,int tokens) {
        const int H=c.hidden,entries=c.top_k*tokens; const auto p=block(layer); float* x=r.f("norm",H);
        r.mat(p+"ffn_gate_inp.weight",x,r.f("router",c.experts),tokens);
        C::route(r.f("router",c.experts),r.weight(p+"exp_probs_b.bias"),r.buffer("route_ids",size_t(c.top_k)*4).i(),r.f("route_weights",c.top_k),c.experts,c.top_k,tokens,float(c.expert_scale),r.stream);
        if(route_handoff) K::doorbell_publish(x,r.buffers.at("route_ids")->i(),r.f("route_weights",c.top_k),
            size_t(H)*tokens,entries,r.mapped_activation,r.mapped_route_ids,nullptr,r.mapped_route_bell,r.stream);
        else {
            ck(cudaMemcpyAsync(route_ids.p,r.buffer("route_ids",size_t(c.top_k)*4).p,size_t(entries)*4,cudaMemcpyDeviceToHost,r.stream));
            ck(cudaMemcpyAsync(activation.p,x,size_t(H)*tokens*4,cudaMemcpyDeviceToHost,r.stream));
        }
    }
    void shared_expert(Rank& r,int layer,int tokens) {
        const int H=c.hidden,F=c.expert_ff*c.shared_experts; const auto p=block(layer);
        float* x=r.f("norm",H);
        r.mat(p+"ffn_gate_shexp.weight",x,r.f("gate",F),tokens); r.mat(p+"ffn_up_shexp.weight",x,r.f("up",F),tokens);
        C::swiglu(r.f("gate",F),r.f("up",F),r.f("hidden",F),F*tokens,float(c.clamp_shared[layer]),r.stream);
        r.mat(p+"ffn_down_shexp.weight",r.f("hidden",F),r.f("shared",H),tokens);
        K::native_quantize_q8_1(x,r.buffer("expert_xq",K::native_q8_1_bytes(H,tokens)).p,H,tokens,r.stream);
    }
    void ffn(int layer,int tokens = 1) {
        struct AbortOnException {
            uint32_t* word;
            int depth;
            explicit AbortOnException(uint32_t* p) : word(p),depth(p ? std::uncaught_exceptions() : 0) {}
            ~AbortOnException() {
                if(word && std::uncaught_exceptions()>depth) {
                    *static_cast<volatile uint32_t*>(word)=1;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                }
            }
        } abort_on_exception(ffn_graph ? tp_abort.p : nullptr);
        const int H = c.hidden;
        const auto p = block(layer);
        // Routed/shared experts run once, on the layer's owner rank. Both TP
        // ranks receive the combined FFN before applying identical HC updates.
        auto& r = *ranks[size_t(layer)*ranks.size()/size_t(c.layers)]; ck(cudaSetDevice(r.id));
        const int entries = c.top_k*tokens;
        auto shared = [&] {shared_expert(r,layer,tokens);};
        if(!tp_graph) r.launch(layer,3,tokens,[&] {route_prepare(r,layer,tokens);});
        if(route_handoff) {
            // Reuse Qwen's system-fenced mapped payload. Shared work can run
            // after publication while the host plans and computes its misses.
            // Query the driver while polling; a memory-only spin cannot flush
            // a queued graph on every supported CUDA driver.
            if(!tp_graph) r.launch(layer,4,tokens,shared);
            const uint32_t want=++route_sequence;
            const auto deadline=Clock::now()+std::chrono::seconds(60);
            for(;;) {
                ck(cudaSetDevice(r.id));
                const auto status=cudaStreamQuery(r.stream);
                if(status!=cudaErrorNotReady) ck(status);
                bool others_ready=true;
                if(!hc_owner && !overlap_ffn) for(auto& other:ranks) if(other.get()!=&r) {
                    ck(cudaSetDevice(other->id));
                    const auto state=cudaStreamQuery(other->stream);
                    if(state==cudaErrorNotReady) others_ready=false; else ck(state);
                }
                if(*static_cast<volatile uint32_t*>(route_bell.p)==want && others_ready) break;
                if(Clock::now()>=deadline) {if(tp_graph) abort_tp(); throw std::runtime_error("native route handoff timed out");}
                _mm_pause();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
        } else {
            ck(cudaStreamSynchronize(r.stream));
            // The optional overlap experiment has its own TP sum staging.
            if(!hc_owner && !overlap_ffn) sync();
        }
        ck(cudaSetDevice(r.id));
        window_routes[size_t(layer)].assign(route_ids.p,route_ids.p+entries);
        if (expert_cache && !speculative) expert_cache->observe(layer,route_ids.p,entries);
        if (lookahead) lookahead->submit(layer,activation.p+size_t(tokens-1)*H);
        auto cached = expert_cache ? expert_cache->acquire(layer,route_ids.p,entries) : std::vector<const uint8_t*>(size_t(entries),nullptr);
        ck(cudaSetDevice(r.id));
        // Default ordering remains the original route fence then shared launch.
        if(!route_handoff) r.launch(layer,4,tokens,shared);
        const auto g = model.tensor(p+"ffn_gate_exps.weight"), d = model.tensor(p+"ffn_down_exps.weight");
        const size_t gate_bytes = g.bytes()/size_t(c.experts), down_bytes = d.bytes()/size_t(c.experts);
        std::vector<ExpertBatch> jobs;
        std::map<int,size_t> group_of;
        std::map<int,std::vector<int>> gpu_groups;
        std::vector<int> misses;
        if(direct_ffn) { ffn_plan.p->groups=0; ffn_plan.p->gpu_mask=0; }
        for (int j = 0; j < entries; ++j) {
            const int e = route_ids.p[j]; if (e < 0 || e >= c.experts) throw std::runtime_error("native route out of range");
            if (cached[size_t(j)]) {
                if(direct_ffn) ffn_plan.p->gpu_mask|=uint64_t(1)<<j;
                gpu_groups[e].push_back(j);
                continue;
            }
            ExpertWeights w; w.hidden = H; w.ff = c.expert_ff; w.gate_type = int(g.tensor->type); w.down_type = int(d.tensor->type);
            w.gate_row = gate_bytes/size_t(c.expert_ff); w.down_row = down_bytes/size_t(H); w.limit = float(c.clamp_expert[layer]);
            w.gate = resident.data(p+"ffn_gate_exps.weight")+size_t(e)*gate_bytes;
            w.up = resident.data(p+"ffn_up_exps.weight")+size_t(e)*gate_bytes;
            w.down = resident.data(p+"ffn_down_exps.weight")+size_t(e)*down_bytes;
            auto found = group_of.find(e);
            if (found == group_of.end()) { const size_t at = jobs.size(); group_of.emplace(e,at); ExpertBatch batch; batch.weights = w; jobs.push_back(batch); found = group_of.find(e); }
            auto& batch = jobs[found->second];
            if (batch.count >= Batch) throw std::runtime_error("duplicate route within a token");
            batch.tokens[batch.count] = j/c.top_k; batch.destinations[batch.count] = j; ++batch.count;
            misses.push_back(j);
        }
        auto combine=[&] {
            if(direct_ffn) {
                if(ffn_post_fuse) {
                    C::combine_sources_post_system(r.f("parts",size_t(H)*c.top_k),r.mapped_cpu_parts,
                        &static_cast<FfnPlan*>(r.buffers.at("ffn_plan")->p)->gpu_mask,r.f("route_weights",c.top_k),
                        r.f("shared",H),r.f("ffn",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),H,c.top_k,tokens,r.stream);
                } else {
                    const auto fn=ffn_graph ? C::combine_sources_system : C::combine_sources;
                    fn(r.f("parts",size_t(H)*c.top_k),r.mapped_cpu_parts,
                    &static_cast<FfnPlan*>(r.buffers.at("ffn_plan")->p)->gpu_mask,r.f("route_weights",c.top_k),
                    r.f("shared",H),r.f("ffn",H),H,c.top_k,tokens,r.stream);
                }
            } else C::combine(r.f("parts",size_t(H)*c.top_k),r.f("route_weights",c.top_k),r.f("shared",H),r.f("ffn",H),H,c.top_k,tokens,r.stream);
        };
        if (!gpu_groups.empty()) {
            auto* pointers=direct_ffn ? ffn_plan.p->pointers : group_ptr.p;
            auto* starts=direct_ffn ? ffn_plan.p->starts : group_start.p;
            auto* destinations=direct_ffn ? ffn_plan.p->destinations : group_dst.p;
            auto* source_tokens=direct_ffn ? ffn_plan.p->tokens : group_tok.p;
            int ng = 0, ne = 0; starts[0] = 0;
            for (const auto& [expert,indices] : gpu_groups) {
                (void)expert;
                pointers[ng] = reinterpret_cast<unsigned long long>(cached[size_t(indices.front())]);
                for (int j : indices) { destinations[ne] = j; source_tokens[ne] = j/c.top_k; ++ne; }
                starts[++ng] = ne;
            }
            if(direct_ffn) ffn_plan.p->groups=ng;
            else {
            group_count.p[0] = ng;
            ck(cudaMemcpyAsync(r.buffers.at("expert_group_ptr")->p,group_ptr.p,size_t(ng)*8,cudaMemcpyHostToDevice,r.stream));
            ck(cudaMemcpyAsync(r.buffers.at("expert_group_start")->p,group_start.p,size_t(ng+1)*4,cudaMemcpyHostToDevice,r.stream));
            ck(cudaMemcpyAsync(r.buffers.at("expert_group_dst")->p,group_dst.p,size_t(ne)*4,cudaMemcpyHostToDevice,r.stream));
            ck(cudaMemcpyAsync(r.buffers.at("expert_group_tok")->p,group_tok.p,size_t(ne)*4,cudaMemcpyHostToDevice,r.stream));
            ck(cudaMemcpyAsync(r.buffers.at("expert_group_count")->p,group_count.p,4,cudaMemcpyHostToDevice,r.stream));
            }
            const auto layout = K::native_expert_layout(int(g.tensor->type),int(d.tensor->type),H,c.expert_ff);
            if(!ffn_graph) r.launch(layer,5,tokens,[&] {
                if(direct_ffn) {
                    auto* plan=static_cast<FfnPlan*>(r.buffers.at("ffn_plan")->p);
                    ck(cudaMemcpyAsync(plan,ffn_plan.p,sizeof(FfnPlan),cudaMemcpyHostToDevice,r.stream));
                    K::native_expert_grouped_clamp(layout,plan->pointers,plan->starts,&plan->groups,
                        plan->destinations,plan->tokens,entries,entries,r.buffers.at("expert_xq")->p,
                        r.buffers.at("expert_scratch")->p,r.f("parts",size_t(H)*c.top_k),r.stream,float(c.clamp_expert[layer]));
                } else {
                K::native_expert_grouped_clamp(layout,static_cast<const unsigned long long*>(r.buffers.at("expert_group_ptr")->p),
                    r.buffers.at("expert_group_start")->i(),r.buffers.at("expert_group_count")->i(),
                    r.buffers.at("expert_group_dst")->i(),r.buffers.at("expert_group_tok")->i(),entries,entries,
                    r.buffers.at("expert_xq")->p,r.buffers.at("expert_scratch")->p,r.f("parts",size_t(H)*c.top_k),
                    r.stream,float(c.clamp_expert[layer]));
                }
            });
        } else if(direct_ffn && !ffn_graph) {
            // No grouped kernel is needed, but the combine still needs mask=0.
            ck(cudaMemcpyAsync(r.buffers.at("ffn_plan")->p,ffn_plan.p,sizeof(FfnPlan),cudaMemcpyHostToDevice,r.stream));
        }
        if(ffn_graph) {
            const bool any_cached=!gpu_groups.empty();
            const auto layout=K::native_expert_layout(int(g.tensor->type),int(d.tensor->type),H,c.expert_ff);
            r.launch(layer,any_cached ? 96 : 97,tokens,[&] {
                auto* plan=static_cast<FfnPlan*>(r.buffers.at("ffn_plan")->p);
                ck(cudaMemcpyAsync(plan,ffn_plan.p,sizeof(FfnPlan),cudaMemcpyHostToDevice,r.stream));
                if(any_cached) K::native_expert_grouped_clamp(layout,plan->pointers,plan->starts,&plan->groups,
                    plan->destinations,plan->tokens,entries,entries,r.buffers.at("expert_xq")->p,
                    r.buffers.at("expert_scratch")->p,r.f("parts",size_t(H)*c.top_k),r.stream,float(c.clamp_expert[layer]));
                C::wait_host(r.mapped_ffn_ready,r.mapped_route_bell,r.mapped_tp_abort,r.stream);
                combine();
                ck(cudaMemcpyAsync(transfer.p,r.f("ffn",H),size_t(H)*tokens*4,cudaMemcpyDeviceToHost,r.stream));
            });
        }
        const auto cpu_jobs=streamed ? streamed->submit(layer,r.id,jobs,activation.p,tokens) : jobs;
        cpu.run_multi(cpu_jobs,activation.p,tokens,cpu_parts.p);
        if(streamed) streamed->finish(cpu_parts.p);
        ck(cudaSetDevice(r.id));
        if(!direct_ffn) for (int j : misses) ck(cudaMemcpyAsync(r.f("parts",size_t(H)*c.top_k)+size_t(j)*H,cpu_parts.p+size_t(j)*H,size_t(H)*4,cudaMemcpyHostToDevice,r.stream));
        else std::atomic_thread_fence(std::memory_order_release);
        if(ffn_graph) {
            // Publish only after CPU and cold outputs are complete. Preserve
            // the original owner stream fence before buffer/cache reuse.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *static_cast<volatile uint32_t*>(tp_ready.p)=route_sequence;
        } else if(hc_owner) {
            r.launch(layer,6,tokens,[&] {
                combine();
                C::hc_post(r.f("ffn",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,tokens,r.stream);
                ck(cudaMemcpyAsync(hc_transfer.p,r.f("residual",H*4),size_t(H)*4*tokens*4,cudaMemcpyDeviceToHost,r.stream));
                ck(cudaMemcpyAsync(hc_transfer.p+size_t(H)*4*Batch,r.f("pre",4),size_t(tokens)*4*4,cudaMemcpyDeviceToHost,r.stream));
            });
        } else if(direct_ffn) {
            r.launch(layer,7,tokens,[&] {
                combine();
                ck(cudaMemcpyAsync(transfer.p,r.f("ffn",H),size_t(H)*tokens*4,cudaMemcpyDeviceToHost,r.stream));
            });
        } else {
            combine();
            ck(cudaMemcpyAsync(transfer.p,r.f("ffn",H),size_t(H)*tokens*4,cudaMemcpyDeviceToHost,r.stream));
        }
        ck(cudaStreamSynchronize(r.stream));
        if (expert_cache) { expert_cache->release(layer,route_ids.p,entries); if (!speculative) expert_cache->hint(layer,std::vector<int32_t>(route_ids.p,route_ids.p+entries)); }
        for (auto& other : ranks) {
            ck(cudaSetDevice(other->id));
            if(hc_owner) {
                if(other.get()==&r) continue;
                ck(cudaMemcpyAsync(other->f("residual",H*4),hc_transfer.p,size_t(H)*4*tokens*4,cudaMemcpyHostToDevice,other->stream));
                ck(cudaMemcpyAsync(other->f("pre",4),hc_transfer.p+size_t(H)*4*Batch,size_t(tokens)*4*4,cudaMemcpyHostToDevice,other->stream));
            } else {
                if (other.get() != &r) ck(cudaMemcpyAsync(other->f("ffn",H),transfer.p,size_t(H)*tokens*4,cudaMemcpyHostToDevice,other->stream));
                if(!ffn_post_fuse || other.get()!=&r)
                    C::hc_post(other->f("ffn",H),other->f("residual",H*4),other->f("post",4),other->f("comb",16),other->f("residual",H*4),H,tokens,other->stream);
            }
        }
    }
    void process(const int32_t* inputs,int tokens,bool output = true,bool verify = false) {
        if((tp_graph || ffn_graph) && *tp_abort.p) throw std::runtime_error("native mapped graph aborted");
        if (tokens < 1 || tokens > Batch || position+tokens > context) throw std::runtime_error("native token/context out of range");
        for (int t = 0; t < tokens; ++t) if (inputs[t] < 0 || inputs[t] >= c.vocab) throw std::runtime_error("invalid native input token");
        speculative = verify;
        const int H = c.hidden;
        int32_t pos[Batch], comp_pos[Batch];
        for (int t = 0; t < tokens; ++t) { pos[t] = position+t; comp_pos[t] = position+t-1; }
        auto& first = *ranks[0]; ck(cudaSetDevice(first.id));
        ck(cudaMemcpyAsync(first.buffer("token",4).p,inputs,size_t(tokens)*4,cudaMemcpyHostToDevice,first.stream));
        const auto& embedding = first.w("token_embd.weight");
        K::iq_embed_rows(embedding.type,embedding.storage->p,size_t(embedding.storage->bytes)/size_t(c.vocab),first.buffer("token",4).i(),tokens,H,first.f("x",H),first.stream);
        ck(cudaMemcpyAsync(transfer.p,first.f("x",H),size_t(H)*tokens*4,cudaMemcpyDeviceToHost,first.stream)); ck(cudaStreamSynchronize(first.stream));
        for (auto& rp : ranks) {
            auto& r = *rp; ck(cudaSetDevice(r.id));
            ck(cudaMemcpyAsync(r.buffer("position",4).p,pos,size_t(tokens)*4,cudaMemcpyHostToDevice,r.stream));
            ck(cudaMemcpyAsync(r.buffer("compress_position_2",4).p,comp_pos,size_t(tokens)*4,cudaMemcpyHostToDevice,r.stream));
            if (r.rank) ck(cudaMemcpyAsync(r.f("x",H),transfer.p,size_t(H)*tokens*4,cudaMemcpyHostToDevice,r.stream));
            C::hc_init(r.f("x",H),r.f("residual",H*4),r.f("pre",4),H,tokens,r.stream);
        }
        for (int layer = 0; layer < c.layers; ++layer) {
            for (auto& rp : ranks) if (layer >= c.layers-3) {
                auto& r = *rp; ck(cudaSetDevice(r.id));
                C::hc_mean(r.f("residual",H*4),r.f("features",H*3)+size_t(layer-(c.layers-3))*H,H,tokens,H*3,r.stream);
            }
            engram(layer,inputs,tokens);
            if(tp_graph) {
                for(auto& rp:ranks) {ck(cudaSetDevice(rp->id));attention_ffn(*rp,layer,tokens);}
                ffn(layer,tokens); continue;
            }
            for (auto& rp : ranks) {
                auto& r = *rp; ck(cudaSetDevice(r.id));
                attention(r,layer,tokens);
            }
            sync();
            auto* sum=overlap_ffn ? tp_sum.p : transfer.p;
            for (int i = 0; i < H*tokens; ++i) { float value = 0; for (size_t rank = 0; rank < ranks.size(); ++rank) value += attention_parts.p[rank*H*Batch+i]; sum[i] = value; }
            for (auto& rp : ranks) {
                auto& r = *rp; ck(cudaSetDevice(r.id));
                if(hc_owner && size_t(layer)*ranks.size()/size_t(c.layers)!=size_t(r.rank)) continue;
                ck(cudaMemcpyAsync(r.f("attn_sum",H),sum,size_t(H)*tokens*4,cudaMemcpyHostToDevice,r.stream));
                r.launch(layer,2,tokens,[&] {ffn_prepare(r,layer,tokens);});
            }
            ffn(layer,tokens);
        }
        if (output) {
            ck(cudaSetDevice(first.id));
            C::hc_pre(first.f("residual",H*4),first.f("pre",4),first.f("x",H),H,tokens,first.stream);
            first.norm("output_norm.weight",first.f("x",H),first.f("norm",H),H,float(c.rms_eps),tokens);
            first.mat("output.weight",first.f("norm",H),first.f("logits",c.vocab),tokens);
        }
        sync(); history.insert(history.end(),inputs,inputs+tokens); position += tokens; last_rows = tokens;
    }
    void step(int32_t token,bool output = true) { process(&token,1,output); }
    void retain(int count) {
        if (count < 1 || count > last_rows) throw std::runtime_error("invalid retained native window");
        if (expert_cache && speculative) for (int l = 0; l < c.layers; ++l) {
            const auto& ids = window_routes[size_t(l)]; expert_cache->observe(l,ids.data(),count*c.top_k);
            expert_cache->hint(l,std::vector<int32_t>(ids.begin(),ids.begin()+count*c.top_k));
        }
        position -= last_rows-count; history.resize(size_t(position)); last_rows = count; speculative = false;
    }
    std::vector<int32_t> greedy_rows() {
        auto& r = *ranks[0]; ck(cudaSetDevice(r.id)); K::SamplerParams params; params.greedy = true; params.temperature = 0;
        K::sample_tokens(r.f("logits",c.vocab),last_rows,c.vocab,nullptr,0,params,r.buffer("pick",4).i(),r.stream);
        ck(cudaMemcpyAsync(picked.p,r.buffer("pick",4).p,size_t(last_rows)*4,cudaMemcpyDeviceToHost,r.stream)); ck(cudaStreamSynchronize(r.stream));
        return std::vector<int32_t>(picked.p,picked.p+last_rows);
    }
    int sample(K::SamplerParams params = {}) {
        auto& r = *ranks[0]; ck(cudaSetDevice(r.id));
        const int n = std::min(int(history.size()),params.penalty_last_n);
        if (n) ck(cudaMemcpyAsync(r.buffer("penalty_history",size_t(context)*4).p,history.data()+history.size()-size_t(n),size_t(n)*4,cudaMemcpyHostToDevice,r.stream));
        params.counter = position;
        K::sample_tokens(r.f("logits",c.vocab)+size_t(last_rows-1)*c.vocab,1,c.vocab,n ? r.buffer("penalty_history",size_t(context)*4).i() : nullptr,n,params,r.buffer("pick",4).i(),r.stream);
        ck(cudaMemcpyAsync(picked.p,r.buffer("pick",4).p,4,cudaMemcpyDeviceToHost,r.stream)); ck(cudaStreamSynchronize(r.stream));
        return *picked.p;
    }
    void dump(const std::string& path) {
        auto& r = *ranks[0]; ck(cudaSetDevice(r.id)); std::vector<float> logits(c.vocab);
        ck(cudaMemcpy(logits.data(),r.f("logits",c.vocab)+size_t(last_rows-1)*c.vocab,logits.size()*4,cudaMemcpyDeviceToHost));
        std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<const char*>(logits.data()),std::streamsize(logits.size()*4));
        if (!out) throw std::runtime_error("logit output failed");
    }
    void check_batch(const std::vector<int32_t>& inputs) {
        if (inputs.size() < Batch || inputs.size() > size_t(context)) throw std::invalid_argument("CHECK needs 8 or more tokens within context");
        auto saved_lookahead = std::move(lookahead); auto saved_cache = std::move(expert_cache); auto saved_streamed=std::move(streamed);
        const size_t prefix = inputs.size()-Batch;
        auto prefill = [&] { reset(); for (size_t i = 0; i < prefix; i += Batch) process(inputs.data()+i,int(std::min(size_t(Batch),prefix-i)),false); };
        auto host_logits = [&] {
            auto& r = *ranks[0]; ck(cudaSetDevice(r.id)); std::vector<float> out(size_t(last_rows)*c.vocab);
            ck(cudaMemcpy(out.data(),r.f("logits",c.vocab),out.size()*4,cudaMemcpyDeviceToHost)); return out;
        };
        const bool candidate_attention = attention_replay; attention_replay = false;
        const bool candidate_hc_owner=hc_owner; hc_owner=false;
        const bool candidate_direct_ffn=direct_ffn; direct_ffn=false;
        const bool candidate_overlap_ffn=overlap_ffn; overlap_ffn=false;
        const bool candidate_grouped_oa=grouped_oa; grouped_oa=false;
        const bool candidate_route_handoff=route_handoff; route_handoff=false;
        const bool candidate_tp_graph=tp_graph; tp_graph=false;
        const bool candidate_ffn_graph=ffn_graph; ffn_graph=false;
        const bool candidate_ffn_post_fuse=ffn_post_fuse; ffn_post_fuse=false;
        const bool candidate_hc_pre_norm=hc_pre_norm; hc_pre_norm=false;
        const bool candidate_normalize_attention=normalize_attention; normalize_attention=false;
        const bool candidate_chunked_attention=chunked_attention; chunked_attention=false;
        const bool candidate_parallel_values=parallel_values; parallel_values=false;
        const bool candidate_prefetch_values=prefetch_values; prefetch_values=false;
        const bool candidate_prefetch_query=prefetch_query; prefetch_query=false;
        const bool candidate_score_grid=score_grid; score_grid=false;
        const bool candidate_value_head_pair=value_head_pair; value_head_pair=false;
        // The legacy oracle needs its duplicated FFN weights. Materialize only
        // these temporary matrices before timing, and never capture their
        // addresses into retained graphs. Production loads owner weights only.
        std::vector<std::pair<Rank*,std::string>> temporary_weights;
        std::vector<bool> saved_graphs;
        std::vector<bool> saved_wide_il;
        for(auto& rp:ranks) {
            auto& r=*rp; saved_graphs.push_back(r.graph_enabled); r.graph_enabled=false;
            saved_wide_il.push_back(r.wide_il); r.wide_il=false;
            ck(cudaSetDevice(r.id));
            for(size_t shard=0;shard<model.gguf().size();++shard) for(const auto& tensor:model.gguf().shard(shard).tensors()) {
                if(!owner_ffn_weight(tensor.name) || tensor.name.find("_exps.weight")!=std::string::npos || r.matrices.contains(tensor.name)) continue;
                r.load(tensor.name,model.tensor(tensor.name),c); temporary_weights.emplace_back(&r,tensor.name);
            }
        }
        prefill(); std::vector<float> expected;
        for (int t = 0; t < Batch; ++t) { step(inputs[prefix+size_t(t)]); auto row = host_logits(); expected.insert(expected.end(),row.begin(),row.end()); }
        attention_replay = candidate_attention;
        hc_owner=candidate_hc_owner;
        direct_ffn=candidate_direct_ffn;
        overlap_ffn=candidate_overlap_ffn;
        grouped_oa=candidate_grouped_oa;
        route_handoff=candidate_route_handoff;
        tp_graph=candidate_tp_graph;
        ffn_graph=candidate_ffn_graph;
        ffn_post_fuse=candidate_ffn_post_fuse;
        hc_pre_norm=candidate_hc_pre_norm;
        normalize_attention=candidate_normalize_attention;
        chunked_attention=candidate_chunked_attention;
        parallel_values=candidate_parallel_values;
        prefetch_values=candidate_prefetch_values;
        prefetch_query=candidate_prefetch_query;
        score_grid=candidate_score_grid;
        value_head_pair=candidate_value_head_pair;
        for(size_t rank=0;rank<ranks.size();++rank) {
            ranks[rank]->graph_enabled=saved_graphs[rank];
            ranks[rank]->wide_il=saved_wide_il[rank];
        }
        prefill(); process(inputs.data()+prefix,Batch,true,true); const auto actual = host_logits();
        auto compare = [&](const float* a,const float* b,size_t n,const char* stage) {
            double worst = 0, ss = 0, ref = 0;
            for (size_t i = 0; i < n; ++i) { const double delta = double(a[i])-b[i]; worst = std::max(worst,std::abs(delta)); ss += delta*delta; ref += double(b[i])*b[i]; if (!std::isfinite(a[i])) throw std::runtime_error("non-finite native batch logits"); }
            const double relative = std::sqrt(ss/std::max(1e-30,ref));
            std::printf("CHECK %s max_abs=%.9g relative_rms=%.9g\n",stage,worst,relative);
            if (worst > 0.01 || relative > 1e-4) throw std::runtime_error("native serial/batch logit mismatch");
        };
        compare(actual.data(),expected.data(),actual.size(),"batch8");
        retain(3); step(inputs[prefix+3]); const auto rewound = host_logits();
        compare(rewound.data(),expected.data()+size_t(3)*c.vocab,c.vocab,"reject-tail-and-replay");
        for(const auto& [rank,name]:temporary_weights) { ck(cudaSetDevice(rank->id)); rank->matrices.erase(name); }
        expert_cache = std::move(saved_cache); lookahead = std::move(saved_lookahead); streamed=std::move(saved_streamed); reset();
        std::puts("CHECK PASS native batch/rollback");
    }
};

class Draft {
public:
    GgufFile file;
    Config c;
    Rank rank;
    int position = 0, block_size = 5, mask = 0, context_limit = 0;
    bool replay = [] { const char* v=std::getenv("STRATA_DS41_DSPARK_GRAPH"); return v && std::atoi(v)!=0; }();
    bool mxfp4_alu = [] { const char* v=std::getenv("STRATA_DS41_MXFP4_ALU"); return v && std::atoi(v)!=0; }();
    float min_confidence = [] { const char* v=std::getenv("STRATA_DS41_DSPARK_P_MIN"); return v ? std::stof(v) : 0.0f; }();
    Host<int32_t> proposals{Batch};
    Host<float> confidence{Batch};
    Draft(const std::string& path,Engine& target) : file(path),c(target.c),rank(target.ranks.back()->id,0,1) {
        auto integer = [&](const std::string& key) {
            const auto* value = file.get(key); if (!value || !value->is_num() || value->u > INT32_MAX) throw std::runtime_error("DSpark missing/invalid "+key); return int(value->u);
        };
        auto same = [&](const char* suffix,int expected) { if (integer(std::string("dflash.")+suffix) != expected) throw std::runtime_error(std::string("DSpark incompatible ")+suffix); };
        const auto* architecture = file.get("general.architecture");
        if (!architecture || architecture->s != "dflash" || file.find("output_hc_fn.weight")) throw std::runtime_error("expected V4.1 DSpark draft without an HC output head");
        same("embedding_length",c.hidden); same("block_count",3); same("attention.head_count",c.heads);
        same("attention.key_length",c.head_dim); same("attention.value_length",c.head_dim); same("attention.q_lora_rank",c.q_rank);
        same("attention.output_group_count",c.o_groups); same("attention.output_lora_rank",c.o_rank);
        same("hyper_connection.count",4); same("expert_gating_func",4); same("expert_feed_forward_length",c.expert_ff);
        same("expert_count",128); same("expert_used_count",3); same("expert_shared_count",1);
        const auto layer_ids = file.integer_array("dflash.target_layers");
        if (layer_ids != std::vector<uint64_t>({37,38,39}) || c.layers != 40) throw std::runtime_error("DSpark requires the original target input layers 37/38/39");
        if (file.integer_array("dflash.attention.compress_ratios") != std::vector<uint64_t>({0,0,0})) throw std::runtime_error("DSpark stages must use plain sliding attention");
        c.layers = 3; c.experts = 128; c.top_k = 3; c.shared_experts = 1;
        c.clamp_expert = file.real_array("dflash.swiglu_clamp_exp"); c.clamp_shared = file.real_array("dflash.swiglu_clamp_shexp");
        if (c.clamp_expert.size() != 3 || c.clamp_shared.size() != 3) throw std::runtime_error("DSpark clamp metadata");
        block_size = integer("dflash.block_size"); mask = integer("tokenizer.ggml.mask_token_id");
        context_limit = target.context;
        if (!std::isfinite(min_confidence) || min_confidence<0 || min_confidence>1) throw std::runtime_error("DSpark confidence threshold must be 0..1");
        if (block_size < 1 || block_size >= Batch || mask < 0 || mask >= c.vocab) throw std::runtime_error("DSpark noise block geometry");
        ck(cudaSetDevice(rank.id)); uint64_t bytes = 0;
        for (const auto& tensor : file.tensors()) {
            TensorRef ref{&file,&tensor};
            if (!ref.bytes() || tensor.offset > file.file_size()-file.data_start() || ref.bytes() > file.file_size()-file.data_start()-tensor.offset) throw std::runtime_error("truncated DSpark tensor");
            rank.load(tensor.name,ref,c); bytes += rank.w(tensor.name).storage->bytes;
        }
        for (int l = 0; l < 3; ++l) {
            const auto p = block(l);
            for (const char* name : {"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) if (rank.w(p+name).type != 39) throw std::runtime_error("DSpark experts must remain original MXFP4");
        }
        rank.load("target_embedding",target.model.tensor("token_embd.weight"),c);
        rank.load("target_output",target.model.tensor("output.weight"),c);
        if (min_confidence>0) {
            const auto& conf=rank.w("conf_proj.weight");
            if (conf.type!=0 || conf.in!=c.hidden+256 || conf.out!=1) throw std::runtime_error("original DSpark confidence geometry");
        }
        allocate();
        if (replay) {
            // Initialize the existing sampler's per-device capability cache
            // before the first graph capture, without an inference warm-up.
            ck(cudaMemsetAsync(rank.f("logits",c.vocab),0,size_t(c.vocab)*4,rank.stream));
            K::SamplerParams params; params.greedy=true; params.temperature=0;
            K::sample_tokens(rank.f("logits",c.vocab),1,c.vocab,nullptr,0,params,rank.buffer("pick",4).i(),rank.stream);
            ck(cudaStreamSynchronize(rank.stream));
        }
        std::fprintf(stderr,"strata-deepseek41: DSpark graph=%d confidence_min=%.2f original noise block=%d\n",replay ? 1 : 0,min_confidence,block_size);
        std::fprintf(stderr,"strata-deepseek41: native DSpark device=%d stages=3 block=%d mask=%d target_layers=37,38,39, original MXFP4, %.3f GiB draft weights\n",rank.id,block_size,mask,double(bytes)/(1ull<<30));
    }
    void allocate() {
        auto& r = rank; const int H = c.hidden, D = c.head_dim, A = c.heads, F = c.expert_ff;
        r.f("residual",H*4); r.f("flat_norm",H*4); r.f("pre",4); r.f("attn_pre",4); r.f("post",4); r.f("comb",16); r.f("mix",24);
        r.f("x",H); r.f("norm",H); r.f("qr",c.q_rank); r.f("q",D*A); r.f("kv",D); r.f("attn",D*A);
        r.f("oa",c.o_rank*c.o_groups); r.f("group_in",A/c.o_groups*D); r.f("group_out",c.o_rank); r.f("out",H);
        r.f("gate",F); r.f("up",F); r.f("hidden",F); r.f("shared",H); r.f("ffn",H);
        r.f("router",c.experts); r.f("route_weights",c.top_k); r.buffer("route_ids",size_t(c.top_k)*4);
        r.f("expert_hidden",F*c.top_k); r.f("parts",H*c.top_k);
        r.f("features",H*3); r.f("encoded",H); r.f("logits",c.vocab); r.f("markov",256); r.f("bias",c.vocab); r.f("confidence",1);
        r.buffer("token",4); r.buffer("position",4); r.buffer("pick",4); r.buffer("q8",K::native_q8_1_bytes(H*4));
        // The copied original target_output is Q6_K. Five-position noise
        // decoding can use wide IL before any legacy IL allocation exists.
        // Allocate outside capture; the same bounded reserve covers it.
        if(r.wide_il) r.buffer("q8_il",K::native_q8_1_il_bytes(H*4,Batch));
        r.raw.resize(3); for (auto& raw : r.raw) raw = std::make_unique<Buffer>(r.id,size_t(c.sliding_window+Batch)*D*2);
    }
    C::Rope rope_config() const { C::Rope rc; rc.dim = c.rope_dim; rc.base = float(c.rope_base); return rc; }
    void reset() { ck(cudaSetDevice(rank.id)); ck(cudaStreamSynchronize(rank.stream)); position = 0; }
    void inject(Engine& target,int start,int count) {
        if (start != position || count < 1 || count > Batch) throw std::runtime_error("DSpark committed feature position mismatch");
        auto& r = rank; auto& source = *target.ranks.back(); ck(cudaSetDevice(r.id));
        const int H = c.hidden, D = c.head_dim; int32_t positions[Batch]; for (int t = 0; t < count; ++t) positions[t] = start+t;
        ck(cudaMemcpyAsync(r.f("features",H*3),source.f("features",H*3),size_t(count)*H*3*4,cudaMemcpyDeviceToDevice,r.stream));
        ck(cudaMemcpyAsync(r.buffer("position",4).p,positions,size_t(count)*4,cudaMemcpyHostToDevice,r.stream));
        r.mat("fc.weight",r.f("features",H*3),r.f("encoded",H),count);
        r.norm("enc.output_norm.weight",r.f("encoded",H),r.f("encoded",H),H,float(c.rms_eps),count);
        for (int l = 0; l < 3; ++l) {
            const auto p = block(l); r.mat(p+"attn_kv.weight",r.f("encoded",H),r.f("kv",D),count);
            r.norm(p+"attn_kv_a_norm.weight",r.f("kv",D),r.f("kv",D),D,float(c.rms_eps),count);
            C::rope(r.f("kv",D),D,1,count,r.buffer("position",4).i(),rope_config(),false,r.stream);
            for (int t = 0; t < count; ++t) C::store_f16(r.f("kv",D)+size_t(t)*D,r.raw[l]->h(),D,(start+t)%(c.sliding_window+Batch),r.stream);
        }
        ck(cudaStreamSynchronize(r.stream)); position += count;
    }
    void hc(int l,const char* sub,float* pre,int n) {
        auto& r = rank; const auto p = block(l)+"hc_"+sub;
        C::rms(r.f("residual",c.hidden*4),nullptr,r.f("flat_norm",c.hidden*4),c.hidden*4,n,float(c.rms_eps),r.stream);
        r.mat(p+"_fn.weight",r.f("flat_norm",c.hidden*4),r.f("mix",24),n);
        C::hc_coeff(r.f("mix",24),r.weight(p+"_scale.weight"),r.weight(p+"_base.weight"),pre,r.f("post",4),r.f("comb",16),n,float(c.hc_eps),c.sinkhorn_iterations,r.stream);
    }
    void attention(int l,int n) {
        auto& r = rank; const auto p = block(l); const int H = c.hidden, D = c.head_dim, A = c.heads;
        r.mat(p+"attn_q_a.weight",r.f("norm",H),r.f("qr",c.q_rank),n);
        r.norm(p+"attn_q_a_norm.weight",r.f("qr",c.q_rank),r.f("qr",c.q_rank),c.q_rank,float(c.rms_eps),n);
        r.mat(p+"attn_q_b.weight",r.f("qr",c.q_rank),r.f("q",A*D),n);
        r.mat(p+"attn_kv.weight",r.f("norm",H),r.f("kv",D),n);
        r.norm(p+"attn_kv_a_norm.weight",r.f("kv",D),r.f("kv",D),D,float(c.rms_eps),n);
        C::rope(r.f("q",A*D),D,A,n,r.buffer("position",4).i(),rope_config(),false,r.stream);
        C::rope(r.f("kv",D),D,1,n,r.buffer("position",4).i(),rope_config(),false,r.stream);
        if (replay) {
            C::store_f16_positions(r.f("kv",D),r.raw[l]->h(),D,n,r.buffer("position",4).i(),c.sliding_window+Batch,0,r.stream);
            C::draft_attention_positions(r.f("q",A*D),r.raw[l]->h(),r.weight(p+"attn_sinks.weight"),r.f("attn",A*D),D,A,n,c.sliding_window,r.buffer("position",4).i(),c.sliding_window+Batch,r.stream);
        } else {
            for (int t = 0; t < n; ++t) C::store_f16(r.f("kv",D)+size_t(t)*D,r.raw[l]->h(),D,(position+t)%(c.sliding_window+Batch),r.stream);
            for (int t = 0; t < n; ++t) C::draft_attention(r.f("q",A*D)+size_t(t)*A*D,r.raw[l]->h(),r.weight(p+"attn_sinks.weight"),r.f("attn",A*D)+size_t(t)*A*D,D,A,c.sliding_window,position+t,position+n-1,c.sliding_window+Batch,r.stream);
        }
        C::rope(r.f("attn",A*D),D,A,n,r.buffer("position",4).i(),rope_config(),true,r.stream);
        const int width = A/c.o_groups*D; const auto& matrix = r.w(p+"attn_output_a.weight");
        const size_t group_bytes = K::native_mmvq_weight_bytes(matrix.type,width,c.o_rank);
        for (int g = 0; g < c.o_groups; ++g) {
            C::strided_copy(r.f("attn",A*D)+size_t(g)*width,r.f("group_in",width),width,n,A*D,width,r.stream);
            void* q = r.buffer("q8",K::native_q8_1_bytes(width,n)).p; K::native_quantize_q8_1(r.f("group_in",width),q,width,n,r.stream);
            K::native_mmvq(matrix.type,static_cast<const uint8_t*>(matrix.storage->p)+size_t(g)*group_bytes,q,r.f("group_out",c.o_rank),width,c.o_rank,n,r.stream);
            C::strided_copy(r.f("group_out",c.o_rank),r.f("oa",c.o_groups*c.o_rank)+size_t(g)*c.o_rank,c.o_rank,n,c.o_rank,c.o_groups*c.o_rank,r.stream);
        }
        r.mat(p+"attn_output_b.weight",r.f("oa",c.o_groups*c.o_rank),r.f("out",H),n);
    }
    void ffn(int l,int n) {
        auto& r = rank; const auto p = block(l); const int H = c.hidden, F = c.expert_ff;
        r.mat(p+"ffn_gate_inp.weight",r.f("norm",H),r.f("router",c.experts),n);
        C::route(r.f("router",c.experts),r.weight(p+"exp_probs_b.bias"),r.buffer("route_ids",size_t(c.top_k)*4).i(),r.f("route_weights",c.top_k),c.experts,c.top_k,n,float(c.expert_scale),r.stream);
        r.mat(p+"ffn_gate_shexp.weight",r.f("norm",H),r.f("gate",F),n); r.mat(p+"ffn_up_shexp.weight",r.f("norm",H),r.f("up",F),n);
        C::swiglu(r.f("gate",F),r.f("up",F),r.f("hidden",F),n*F,float(c.clamp_shared[l]),r.stream);
        r.mat(p+"ffn_down_shexp.weight",r.f("hidden",F),r.f("shared",H),n);
        auto* ids = r.buffer("route_ids",size_t(c.top_k)*4).i();
        C::mxfp4_gu(r.w(p+"ffn_gate_exps.weight").storage->p,r.w(p+"ffn_up_exps.weight").storage->p,r.f("norm",H),ids,r.f("expert_hidden",F*c.top_k),H,F,c.top_k,n,float(c.clamp_expert[l]),r.stream,mxfp4_alu);
        C::mxfp4_down(r.w(p+"ffn_down_exps.weight").storage->p,r.f("expert_hidden",F*c.top_k),ids,r.f("parts",H*c.top_k),F,H,c.top_k,n,r.stream,mxfp4_alu);
        C::combine(r.f("parts",H*c.top_k),r.f("route_weights",c.top_k),r.f("shared",H),r.f("ffn",H),H,c.top_k,n,r.stream);
    }
    void noise_decode(int n) {
        auto& r=rank; const int H=c.hidden;
        const auto& embedding = r.w("target_embedding");
        K::iq_embed_rows(embedding.type,embedding.storage->p,embedding.storage->bytes/size_t(c.vocab),r.buffer("token",4).i(),n,H,r.f("x",H),r.stream);
        C::hc_init(r.f("x",H),r.f("residual",H*4),r.f("pre",4),H,n,r.stream);
        for (int l = 0; l < 3; ++l) {
            hc(l,"attn",r.f("attn_pre",4),n); C::hc_pre(r.f("residual",H*4),r.f("pre",4),r.f("x",H),H,n,r.stream);
            r.norm(block(l)+"attn_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n); attention(l,n);
            C::hc_post(r.f("out",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,n,r.stream);
            hc(l,"ffn",r.f("pre",4),n); C::hc_pre(r.f("residual",H*4),r.f("attn_pre",4),r.f("x",H),H,n,r.stream);
            r.norm(block(l)+"ffn_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n); ffn(l,n);
            C::hc_post(r.f("ffn",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,n,r.stream);
        }
        C::hc_pre(r.f("residual",H*4),r.f("pre",4),r.f("x",H),H,n,r.stream);
        r.norm("output_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n);
        r.mat("target_output",r.f("norm",H),r.f("logits",c.vocab),n);
    }
    std::vector<int32_t> propose_legacy(int anchor,int count) {
        auto& r = rank; ck(cudaSetDevice(r.id)); const int H = c.hidden, n = std::min(count,block_size);
        int32_t ids[Batch], positions[Batch]; for (int t = 0; t < n; ++t) { ids[t] = t ? mask : anchor; positions[t] = position+t; }
        ck(cudaMemcpyAsync(r.buffer("token",4).p,ids,size_t(n)*4,cudaMemcpyHostToDevice,r.stream));
        ck(cudaMemcpyAsync(r.buffer("position",4).p,positions,size_t(n)*4,cudaMemcpyHostToDevice,r.stream));
        noise_decode(n);
        std::vector<int32_t> output; int previous = anchor; K::SamplerParams params; params.greedy = true; params.temperature = 0;
        for (int t = 0; t < n; ++t) {
            const auto& w1 = r.w("markov_w1.weight");
            C::strided_copy(w1.storage->f(size_t(previous)*256),r.f("markov",256),256,1,256,256,r.stream);
            r.mat("markov_w2.weight",r.f("markov",256),r.f("bias",c.vocab));
            float* logits = r.f("logits",c.vocab)+size_t(t)*c.vocab;
            C::add(logits,r.f("bias",c.vocab),logits,c.vocab,r.stream);
            K::sample_tokens(logits,1,c.vocab,nullptr,0,params,r.buffer("pick",4).i(),r.stream);
            ck(cudaMemcpyAsync(&previous,r.buffer("pick",4).p,4,cudaMemcpyDeviceToHost,r.stream)); ck(cudaStreamSynchronize(r.stream));
            if (previous < 0 || previous >= c.vocab) throw std::runtime_error("DSpark prediction outside vocabulary");
            output.push_back(previous);
        }
        return output;
    }
    std::vector<int32_t> propose(int anchor,int count) {
        if (!replay && min_confidence<=0) return propose_legacy(anchor,count);
        if (count<=0) return {};
        auto& r=rank; ck(cudaSetDevice(r.id)); const int H=c.hidden;
        // Preserve the trained five-position noncausal noise block; only the
        // emitted prefix is shortened by confidence or the request boundary.
        const int n=std::min(min_confidence>0 ? block_size : count,context_limit-position);
        int32_t ids[Batch], positions[Batch];
        for (int t=0;t<n;++t) { ids[t]=t ? mask : anchor; positions[t]=position+t; }
        ck(cudaMemcpyAsync(r.buffer("token",4).p,ids,size_t(n)*4,cudaMemcpyHostToDevice,r.stream));
        ck(cudaMemcpyAsync(r.buffer("position",4).p,positions,size_t(n)*4,cudaMemcpyHostToDevice,r.stream));
        auto body=[&] {
            noise_decode(n);
            const auto& w1=r.w("markov_w1.weight");
            K::SamplerParams params; params.greedy=true; params.temperature=0;
            for (int t=0;t<n;++t) {
                const auto* previous=t ? r.buffer("pick",4).i()+t-1 : r.buffer("token",4).i();
                C::gather_row(w1.storage->f(),previous,r.f("markov",256),256,c.vocab,r.stream);
                r.mat("markov_w2.weight",r.f("markov",256),r.f("bias",c.vocab));
                auto* logits=r.f("logits",c.vocab)+size_t(t)*c.vocab;
                C::add(logits,r.f("bias",c.vocab),logits,c.vocab,r.stream);
                if (min_confidence>0) {
                    const auto bias=r.matrices.find("conf_proj.bias");
                    C::draft_confidence(r.f("x",H)+size_t(t)*H,r.f("markov",256),r.weight("conf_proj.weight"),
                        bias==r.matrices.end() ? nullptr : bias->second.storage->f(),r.f("confidence",1)+t,H,256,r.stream);
                }
                K::sample_tokens(logits,1,c.vocab,nullptr,0,params,r.buffer("pick",4).i()+t,r.stream);
            }
        };
        if (replay) r.launch(-1,0,n,body); else body();
        ck(cudaMemcpyAsync(proposals.p,r.buffer("pick",4).p,size_t(n)*4,cudaMemcpyDeviceToHost,r.stream));
        if (min_confidence>0) ck(cudaMemcpyAsync(confidence.p,r.f("confidence",1),size_t(n)*4,cudaMemcpyDeviceToHost,r.stream));
        ck(cudaStreamSynchronize(r.stream));
        std::vector<int32_t> output;
        for (int t=0;t<std::min(count,n);++t) {
            const int id=proposals.p[t];
            if (id<0 || id>=c.vocab) throw std::runtime_error("DSpark prediction outside vocabulary");
            if (min_confidence>0 && !std::isfinite(confidence.p[t])) throw std::runtime_error("DSpark confidence is not finite");
            const bool uncertain=min_confidence>0 && confidence.p[t]<min_confidence;
            if (uncertain && !output.empty()) break;
            output.push_back(id); // Existing accepted policy: always offer first.
            if (uncertain) break;
        }
        return output;
    }

};

std::vector<int32_t> tokens(std::string value) {
    for (char& ch : value) if (ch == ',') ch = ' ';
    std::istringstream in(value); std::vector<int32_t> result; int64_t n;
    while (in >> n) { if (n < 0 || n > INT32_MAX) throw std::invalid_argument("invalid token ID"); result.push_back(int32_t(n)); }
    if (!in.eof()) throw std::invalid_argument("invalid token list");
    return result;
}
struct Control {
    std::string pending;
    std::deque<std::string> lines;
    bool stopped = false, eof = false;
    void pump(int timeout) {
        if (eof) return;
        pollfd fd{STDIN_FILENO,POLLIN,0}; const int result = poll(&fd,1,timeout);
        if (result < 0) { if (errno == EINTR) return; throw std::runtime_error("control poll failed"); }
        if (!result) return;
        char bytes[65536]; const auto n = read(STDIN_FILENO,bytes,sizeof(bytes));
        if (n == 0) { eof = true; return; }
        if (n < 0) { if (errno == EINTR) return; throw std::runtime_error("control read failed"); }
        pending.append(bytes,size_t(n));
        for (;;) {
            const size_t at = pending.find('\n'); if (at == std::string::npos) break;
            std::string line = pending.substr(0,at); pending.erase(0,at+1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") stopped = true; else lines.push_back(std::move(line));
        }
    }
    bool next(std::string& line) {
        while (lines.empty() && !eof) pump(-1);
        if (lines.empty()) return false;
        line = std::move(lines.front()); lines.pop_front(); return true;
    }
};
int eos(const Model& model) {
    const auto* value = model.gguf().meta().get("tokenizer.ggml.eos_token_id");
    return value && value->is_num() ? int(value->u) : -1;
}
struct Request {
    int maximum = 64;
    int draft_depth = -1;
    K::SamplerParams sampling;
    std::vector<int32_t> prompt;
    Request() { sampling.temperature = 0; sampling.greedy = true; }
};
Request request(const std::string& line) {
    std::istringstream in(line); std::string command, item; Request r;
    if (!(in >> command >> r.maximum) || command != "GEN" || r.maximum < 0) throw std::runtime_error("expected GEN <max_new> [keys] <id,id,...>");
    while (in >> item) {
        const auto eq = item.find('=');
        if (eq == std::string::npos) { r.prompt = tokens(item); if (in >> item) throw std::runtime_error("extra GEN input"); break; }
        const std::string key = item.substr(0,eq), value = item.substr(eq+1);
        if (key == "temperature") { r.sampling.temperature = std::stof(value); r.sampling.greedy = r.sampling.temperature <= 0; }
        else if (key == "top_p") r.sampling.top_p = std::stof(value);
        else if (key == "top_k") r.sampling.top_k = std::stoi(value);
        else if (key == "min_p") r.sampling.min_p = std::stof(value);
        else if (key == "seed") r.sampling.seed = std::stoull(value);
        else if (key == "penalty_last_n") r.sampling.penalty_last_n = std::stoi(value);
        else if (key == "penalty_repeat") r.sampling.penalty_repeat = std::stof(value);
        else if (key == "penalty_freq") r.sampling.penalty_freq = std::stof(value);
        else if (key == "penalty_present") r.sampling.penalty_present = std::stof(value);
        else if (key == "draft_depth") {
            r.draft_depth = std::stoi(value);
            if (r.draft_depth < 0 || r.draft_depth > 5) throw std::runtime_error("draft_depth must be 0..5");
        }
        else if (key != "cache_id" && key != "pin" && key != "cache_mode") throw std::runtime_error("unsupported GEN key: "+key);
    }
    if (r.prompt.empty()) throw std::runtime_error("empty native prompt");
    r.sampling.penalty_last_n = std::max(0,r.sampling.penalty_last_n);
    return r;
}
void generate(Engine& e,const Request& request,Control* control,const std::vector<int32_t>& forced = {},const std::string& dump = {},Draft* draft = nullptr,int max_drafts = 5) {
    if (request.prompt.size() > size_t(e.context)) throw std::runtime_error("prompt exceeds native context");
    const auto& sampling = request.sampling;
    if (request.draft_depth >= 0) max_drafts = request.draft_depth;
    const bool neutral = sampling.penalty_last_n <= 0 || (sampling.penalty_repeat == 1 && sampling.penalty_freq == 0 && sampling.penalty_present == 0);
    Draft* active = draft && max_drafts && sampling.greedy && neutral && forced.empty() ? draft : nullptr;
    e.reset(); const auto prompt_start = Clock::now();
    if (active) active->reset();
    size_t read = 0; bool stopped = false;
    while (read < request.prompt.size()) {
        const int count = int(std::min(size_t(Batch),request.prompt.size()-read));
        e.process(request.prompt.data()+read,count,read+size_t(count) == request.prompt.size());
        if (active) active->inject(e,int(read),count);
        read += size_t(count);
        if (control) { control->pump(0); stopped = control->stopped; }
        if (control && (read%8 == 0 || read == request.prompt.size())) {
            std::printf("PP %zu %zu %.0f %.1f\n",read,request.prompt.size(),elapsed(prompt_start),1000*read/std::max(1e-9,elapsed(prompt_start))); std::fflush(stdout);
        }
        if (stopped) break;
    }
    const double prompt_ms = elapsed(prompt_start);
    if (!dump.empty() && !stopped) e.dump(dump);
    const auto decode_start = Clock::now();
    int produced = 0, offered = 0, accepted = 0, verified_rows = 0;
    double draft_ms = 0, verify_ms = 0;
    const int maximum = std::min(request.maximum,e.context-int(request.prompt.size()));
    const char* finish = stopped ? "cancel" : "length";
    while (!stopped && produced < maximum) {
        const int token = size_t(produced) < forced.size() ? forced[size_t(produced)] : e.sample(request.sampling);
        if (active && token != eos(e.model) && maximum-produced > 1 && e.position+1 < e.context) {
            const auto d0 = Clock::now();
            const auto proposals = active->propose(token,std::min({max_drafts,maximum-produced-1,e.context-e.position-1}));
            draft_ms += elapsed(d0); offered += int(proposals.size());
            std::vector<int32_t> window = {token}; window.insert(window.end(),proposals.begin(),proposals.end());
            const int start = e.position; const auto v0 = Clock::now();
            e.process(window.data(),int(window.size()),true,true); const auto actual = e.greedy_rows();
            verify_ms += elapsed(v0); verified_rows += int(window.size());
            int keep = 1;
            while (size_t(keep) < window.size() && proposals[size_t(keep-1)] == actual[size_t(keep-1)]) ++keep;
            for (int j = 0; j < keep; ++j) if (window[size_t(j)] == eos(e.model)) { keep = j+1; finish = "stop"; break; }
            accepted += keep-1; e.retain(keep); active->inject(e,start,keep);
            for (int j = 0; j < keep; ++j) { std::printf("T %d\n",window[size_t(j)]); ++produced; }
            std::fflush(stdout);
            if (std::strcmp(finish,"stop") == 0) break;
            if (control) { control->pump(0); stopped = control->stopped; if (stopped) finish = "cancel"; }
            continue;
        }
        std::printf("T %d\n",token); std::fflush(stdout); ++produced;
        if (token == eos(e.model)) { finish = "stop"; break; }
        if (produced >= maximum || e.position >= e.context) break;
        e.step(token); ++verified_rows;
        if (control) { control->pump(0); stopped = control->stopped; if (stopped) finish = "cancel"; }
    }
    const double decode_ms = elapsed(decode_start);
    const int steps = std::max(0,produced-1);
    std::printf("DONE %d %zu %.1f %.1f %s %d %d 0 0 0 0 0 0.0 %zu 0\n",produced,request.prompt.size(),prompt_ms,decode_ms,finish,accepted,offered,read); std::fflush(stdout);
    std::fprintf(stderr,"strata-deepseek41: native prompt_tokens=%zu prompt_ms=%.3f decode_steps=%d decode_ms=%.3f tok/s=%.6f\n",read,prompt_ms,steps,decode_ms,decode_ms > 0 ? 1000*steps/decode_ms : 0);
    if (active) std::fprintf(stderr,"strata-deepseek41: native DSpark accepted=%d offered=%d verified_rows=%d draft_ms=%.3f verify_ms=%.3f depth=%d\n",accepted,offered,verified_rows,draft_ms,verify_ms,max_drafts);
    e.resident.verify();
    if (e.expert_cache) e.expert_cache->report();
    if (e.lookahead) e.lookahead->report();
    if (e.streamed) e.streamed->report();
}
}  // namespace

int run(int argc,char** argv) {
    try {
        std::setvbuf(stdout,nullptr,_IONBF,0);
        std::string path, prompt, dump, force, dspark = "auto";
        int context = 262144, threads = 24, maximum = 64, tp = 2, cache_mib = -1, depth = 2, max_drafts = 5; bool serve = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string { if (++i >= argc) throw std::invalid_argument(arg+" needs a value"); return argv[i]; };
            if (arg == "--model" || arg == "--native") path = next();
            else if (arg == "--tokens") prompt = next();
            else if (arg == "--n") maximum = std::stoi(next());
            else if (arg == "--ctx") context = std::stoi(next());
            else if (arg == "--threads") threads = std::stoi(next());
            else if (arg == "--tp") tp = std::stoi(next());
            else if (arg == "--expert-cache-mib") cache_mib = std::stoi(next());
            else if (arg == "--lookahead") depth = std::stoi(next());
            else if (arg == "--dspark") dspark = next();
            else if (arg == "--no-dspark") dspark.clear();
            else if (arg == "--draft-depth") max_drafts = std::stoi(next());
            else if (arg == "--dump-full-logits") dump = next();
            else if (arg == "--force") force = next();
            else if (arg == "--serve") serve = true;
            else if (arg == "generate") {}
            else if (arg == "--gpu") { const auto value = next(); setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID",1); setenv("CUDA_VISIBLE_DEVICES",value.c_str(),1); }
            else if (arg == "--version") { std::printf("strata %s native-deepseek41\n",STRATA_VERSION); return 0; }
            else if (arg == "--help" || arg == "-h") {
                std::printf("Native Strata DeepSeek V4.1: --native SHARD.gguf --tp 2 --threads 24 --ctx 262144\n  --tokens ID,ID,... --n 64 [--force IDS] [--dump-full-logits FILE]\n  --serve (Strata engine protocol)\nModel experts and original Engram tables are required to be locked in RAM.\n"); return 0;
            } else throw std::invalid_argument("unsupported native DeepSeek option: "+arg);
        }
        if (path.empty() || (!serve && prompt.empty()) || maximum < 0 || (tp != 1 && tp != 2) || depth < 0 || depth > 2 || max_drafts < 1 || max_drafts > 5) throw std::invalid_argument("--native and --tokens/--serve are required; TP=1/2, lookahead=0..2, draft-depth=1..5");
        if (dspark == "auto") {
            const auto main = std::filesystem::weakly_canonical(path);
            const auto candidate = main.parent_path().parent_path().parent_path()/"DeepSeek-V4.1-Flash-1017-DSpark-corrected-Q8_0.gguf";
            dspark = main.filename().string().rfind("DeepSeek-V4.1-Flash-1017-",0) == 0 && std::filesystem::exists(candidate) ? candidate.string() : "";
        }
        int devices = 0; ck(cudaGetDeviceCount(&devices)); if (devices < tp) throw std::runtime_error("not enough GPUs for native TP");
        K::native_mmvq_set_multi_exact(true);
        Engine engine(path,context,threads,tp,dspark.empty() ? cache_mib : 0,depth);
        std::unique_ptr<Draft> draft;
        if (!dspark.empty()) { draft = std::make_unique<Draft>(dspark,engine); engine.enable_cache(cache_mib,depth,true); }
        engine.enable_streamed();
        engine.enable_chunked_attention();
        engine.enable_attention_values();
        if (!serve) {
            Request r; r.maximum = maximum; r.prompt = tokens(prompt); generate(engine,r,nullptr,tokens(force),dump,draft.get(),max_drafts); return 0;
        }
        std::printf("INFO engine=%s arch=deepseek41 tp=%d threads=%d kv=F16 lookahead=%d dspark=%d draft_depth=%d native_graphs=%d attention_graph=%d hc_owner=%d cold_streams=%d direct_ffn=%d ffn_overlap=%d grouped_oa=%d route_handoff=%d tp_graph=%d attention_weights=%d attention_chunks=%d attention_values=%d value_prefetch=%d query_prefetch=%d\n",
                    STRATA_VERSION,tp,threads,depth,draft ? 1 : 0,max_drafts,engine.ranks.front()->graph_enabled ? 1 : 0,engine.attention_replay ? 1 : 0,engine.hc_owner ? 1 : 0,engine.streamed ? 1 : 0,engine.direct_ffn ? 1 : 0,engine.overlap_ffn ? 1 : 0,engine.grouped_oa ? 1 : 0,engine.route_handoff ? 1 : 0,engine.tp_graph ? 1 : 0,engine.normalize_attention ? 1 : 0,engine.chunked_attention ? 1 : 0,engine.parallel_values ? 1 : 0,engine.prefetch_values ? 1 : 0,engine.prefetch_query ? 1 : 0);
        Control control; std::printf("READY %d stop\n",context);
        std::string line;
        while (control.next(line)) {
            if (line == "QUIT" || line == "EXIT") break;
            if (line.empty() || line == "PING") { if (line == "PING") std::puts("PONG"); continue; }
            if (line.rfind("CHECK ",0) == 0) { engine.check_batch(tokens(line.substr(6))); continue; }
            try { control.stopped = false; generate(engine,request(line),&control,{},"",draft.get(),max_drafts); }
            catch (const std::exception& error) { std::printf("ERR %s\n",error.what()); std::fflush(stdout); }
        }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"strata-deepseek41: %s\n",e.what()); return 1; }
}
}  // namespace strata::deepseek41
