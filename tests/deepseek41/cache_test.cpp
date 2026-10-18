#include "strata/deepseek41/expert_cache.hpp"
#include "fixture.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(bool v,const char* text) { if (!v) throw std::runtime_error(text); }
int main() {
    const auto path = std::filesystem::temp_directory_path()/"strata-deepseek41-cache-test.gguf";
    try {
        deepseek_fixture::write(path);
        strata::deepseek41::Model model(path.string()); strata::deepseek41::ResidentWeights resident;
        resident.load(model);
        {
            strata::deepseek41::ExpertCache cache(model,resident,1,{1});
            const int32_t ids[] = {0,1,2,3}; cache.observe(0,ids,4); cache.observe(0,ids,4); cache.hint(0,{0,1,2,3});
            std::vector<const uint8_t*> leased;
            for (int attempt = 0; attempt < 100; ++attempt) {
                leased = cache.acquire(0,ids,4);
                if (std::all_of(leased.begin(),leased.end(),[](auto* p) { return p != nullptr; })) break;
                cache.release(0,ids,4); std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            check(std::all_of(leased.begin(),leased.end(),[](auto* p) { return p != nullptr; }),"pending slots never became ready");
            auto verify = [&](int e,const uint8_t* gpu) {
                const size_t gu = cache.gate_bytes(0), down = cache.down_bytes(0); std::vector<uint8_t> copy(gu*2+down);
                ck(cudaMemcpy(copy.data(),gpu,copy.size(),cudaMemcpyDeviceToHost));
                const auto g = resident.data("blk.0.ffn_gate_exps.weight"), u = resident.data("blk.0.ffn_up_exps.weight"), d = resident.data("blk.0.ffn_down_exps.weight");
                check(!std::memcmp(copy.data(),g+size_t(e)*gu,gu),"cached gate bytes differ");
                check(!std::memcmp(copy.data()+gu,u+size_t(e)*gu,gu),"cached up bytes differ");
                check(!std::memcmp(copy.data()+gu*2,d+size_t(e)*down,down),"cached down bytes differ");
            };
            for (int e = 0; e < 4; ++e) verify(e,leased[size_t(e)]);
            const int32_t incoming = 4; for (int j = 0; j < 4; ++j) cache.observe(0,&incoming,1);
            cache.hint(0,{incoming}); check(cache.acquire(0,&incoming,1)[0] == nullptr,"hint evicted a leased expert");
            cache.release(0,ids,1); cache.hint(0,{incoming});
            const uint8_t* replacement = nullptr;
            for (int attempt = 0; attempt < 100 && !replacement; ++attempt) { replacement = cache.acquire(0,&incoming,1)[0]; if (!replacement) std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            check(replacement != nullptr,"released slot was not replaced"); verify(4,replacement);
            check(cache.acquire(0,ids,1)[0] == nullptr,"old expert remained published after eviction");
            cache.release(0,&incoming,1); cache.release(0,ids+1,3); cache.report();
        }
        std::filesystem::remove(path);
        std::cout << "Native Strata cache byte identity, fill readiness and leased eviction PASS\n";
    } catch (const std::exception& e) { std::filesystem::remove(path); std::cerr << e.what() << '\n'; return 1; }
}
