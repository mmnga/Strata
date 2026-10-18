#include "strata/deepseek41/expert_cache.hpp"
#include "strata/core/expert_cache.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
constexpr size_t MiB = 1ull<<20;
}
struct ExpertCache::Impl {
    struct Slot {
        int expert = -1, pins = 0;
        bool pending = false;
        cudaEvent_t ready = nullptr;
    };
    struct Layer {
        strata::core::ExpertCache storage;
        std::mutex mutex;
        int device = 0;
        size_t gu = 0, down = 0;
        const uint8_t *g = nullptr, *u = nullptr, *d = nullptr;
        std::vector<Slot> slots;
        std::vector<int> lookup;
        std::vector<uint64_t> heat, age;
        uint64_t clock = 0;
    };
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<cudaStream_t> streams;
    std::atomic<uint64_t> hits{0}, queries{0}, fills{0}, bytes{0};
    int experts, devices;
    Impl(const Model& m,const ResidentWeights& resident,int n,const std::vector<int>& budgets,int draft_reserve_mib) : experts(m.config().experts),devices(n) {
        const int count = m.config().layers;
        streams.resize(n);
        for (int d = 0; d < n; ++d) { ck(cudaSetDevice(d)); ck(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking)); }
        for (int l = 0; l < count; ++l) {
            auto layer = std::make_unique<Layer>(); auto& c = *layer; c.device = l*n/count;
            const auto p = "blk."+std::to_string(l)+".";
            c.gu = m.tensor(p+"ffn_gate_exps.weight").bytes()/size_t(experts); c.down = m.tensor(p+"ffn_down_exps.weight").bytes()/size_t(experts);
            c.g = resident.data(p+"ffn_gate_exps.weight"); c.u = resident.data(p+"ffn_up_exps.weight"); c.d = resident.data(p+"ffn_down_exps.weight");
            c.lookup.assign(experts,-1); c.heat.assign(experts,0); c.age.assign(experts,0); layers.push_back(std::move(layer));
        }
        for (int d = 0; d < n; ++d) {
            ck(cudaSetDevice(d)); size_t free = 0, total = 0; ck(cudaMemGetInfo(&free,&total));
            const size_t reserve = 1536*MiB+(d == n-1 ? size_t(draft_reserve_mib)*MiB : 0);
            size_t left = free > reserve ? free-reserve : 0;
            if (size_t(d) < budgets.size() && budgets[d] >= 0) left = std::min(left,size_t(budgets[d])*MiB);
            const size_t budget = left;
            bool added = true;
            while (added) {
                added = false;
                for (auto& layer : layers) {
                    auto& c = *layer; const size_t blob = c.gu*2+c.down;
                    if (c.device == d && c.slots.size() < size_t(experts) && blob <= left) { c.slots.emplace_back(); left -= blob; added = true; }
                }
            }
            size_t slots = 0;
            for (auto& layer : layers) {
                auto& c = *layer; if (c.device != d || c.slots.empty()) continue;
                std::string err;
                if (!c.storage.open(int64_t(c.slots.size()),1,experts,int64_t(c.gu*2+c.down),err)) throw std::runtime_error(err);
                for (auto& slot : c.slots) ck(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming));
                slots += c.slots.size();
            }
            std::fprintf(stderr,"strata-deepseek41: native expert cache device=%d slots=%zu MiB=%.1f reserve_MiB=%.1f\n",d,slots,double(budget-left)/MiB,double(reserve)/MiB);
        }
    }
    ~Impl() {
        for (size_t d = 0; d < streams.size(); ++d) { cudaSetDevice(int(d)); cudaStreamSynchronize(streams[d]); }
        for (auto& l : layers) {
            cudaSetDevice(l->device); for (auto& slot : l->slots) cudaEventDestroy(slot.ready); l->storage.close();
        }
        for (size_t d = 0; d < streams.size(); ++d) { cudaSetDevice(int(d)); cudaStreamDestroy(streams[d]); }
    }
    void hint(int layer,const std::vector<int32_t>& ids) {
        auto& l = *layers.at(size_t(layer)); std::lock_guard<std::mutex> lock(l.mutex);
        if (l.slots.empty()) return;
        ck(cudaSetDevice(l.device));
        for (int e : ids) {
            if (e < 0 || e >= experts || l.lookup[e] >= 0 || l.heat[e] < 512) continue;
            int victim = -1; uint64_t cold = UINT64_MAX, old = UINT64_MAX;
            for (size_t j = 0; j < l.slots.size(); ++j) {
                auto& slot = l.slots[j];
                if (slot.pending) { const auto state = cudaEventQuery(slot.ready); if (state == cudaSuccess) slot.pending = false; else if (state != cudaErrorNotReady) ck(state); }
                if (slot.pins || slot.pending) continue;
                if (slot.expert < 0) { victim = int(j); break; }
                const auto h = l.heat[slot.expert], age = l.age[slot.expert];
                if (h < cold || (h == cold && age < old)) { victim = int(j); cold = h; old = age; }
            }
            if (victim < 0) continue;
            auto& slot = l.slots[size_t(victim)];
            if (slot.expert >= 0 && l.heat[e] < cold+384) continue;
            if (slot.expert >= 0) l.lookup[slot.expert] = -1;
            slot.expert = e; slot.pending = true; l.lookup[e] = victim;
            uint8_t* dst = l.storage.device_slot(victim); auto stream = streams[size_t(l.device)];
            copy_registered_slice(dst,l.g,size_t(e)*l.gu,l.gu,stream);
            copy_registered_slice(dst+l.gu,l.u,size_t(e)*l.gu,l.gu,stream);
            copy_registered_slice(dst+l.gu*2,l.d,size_t(e)*l.down,l.down,stream);
            ck(cudaEventRecord(slot.ready,stream)); ++fills; bytes += l.gu*2+l.down;
        }
    }
};
ExpertCache::ExpertCache(const Model& m,const ResidentWeights& r,int n,const std::vector<int>& b,int reserve) : p_(std::make_unique<Impl>(m,r,n,b,reserve)) {}
ExpertCache::~ExpertCache() = default;
std::vector<const uint8_t*> ExpertCache::acquire(int layer,const int32_t* ids,int count) {
    auto& l = *p_->layers.at(size_t(layer)); std::lock_guard<std::mutex> lock(l.mutex); ck(cudaSetDevice(l.device));
    std::vector<const uint8_t*> out(size_t(count),nullptr);
    for (int j = 0; j < count; ++j) {
        const int e = ids[j]; if (e < 0 || e >= p_->experts) throw std::runtime_error("invalid cached expert ID");
        ++p_->queries; const int i = l.lookup[e]; if (i < 0) continue; auto& slot = l.slots[size_t(i)];
        if (slot.pending) { const auto state = cudaEventQuery(slot.ready); if (state == cudaSuccess) slot.pending = false; else if (state == cudaErrorNotReady) continue; else ck(state); }
        ++slot.pins; out[size_t(j)] = l.storage.device_slot(i); ++p_->hits;
    }
    return out;
}
void ExpertCache::release(int layer,const int32_t* ids,int count) {
    auto& l = *p_->layers.at(size_t(layer)); std::lock_guard<std::mutex> lock(l.mutex);
    for (int j = 0; j < count; ++j) { const int i = l.lookup.at(size_t(ids[j])); if (i >= 0 && l.slots[size_t(i)].pins) --l.slots[size_t(i)].pins; }
}
void ExpertCache::observe(int layer,const int32_t* ids,int count) {
    auto& l = *p_->layers.at(size_t(layer)); std::lock_guard<std::mutex> lock(l.mutex); ++l.clock;
    if (l.clock%128 == 0) for (auto& heat : l.heat) heat = heat*7/10;
    for (int j = 0; j < count; ++j) { const int e = ids[j]; if (e < 0 || e >= p_->experts) throw std::runtime_error("invalid observed expert ID"); l.heat[e] += 256; l.age[e] = l.clock; }
}
void ExpertCache::hint(int layer,const std::vector<int32_t>& ids) { p_->hint(layer,ids); }
size_t ExpertCache::gate_bytes(int layer) const { return p_->layers.at(size_t(layer))->gu; }
size_t ExpertCache::down_bytes(int layer) const { return p_->layers.at(size_t(layer))->down; }
void ExpertCache::report() const {
    std::fprintf(stderr,"strata-deepseek41: native cache hits=%llu/%llu (%.4f) fills=%llu raw_copy_GiB=%.3f\n",(unsigned long long)p_->hits.load(),(unsigned long long)p_->queries.load(),double(p_->hits.load())/std::max(1.0,double(p_->queries.load())),(unsigned long long)p_->fills.load(),double(p_->bytes.load())/(1ull<<30));
}
}  // namespace strata::deepseek41
