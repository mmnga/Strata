#include "strata/deepseek41/window_heat.hpp"
#include <array>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
using strata::deepseek41::observe_window_heat;
void require(bool b,const char* text) {if(!b) throw std::runtime_error(text);}
int main() {
    try {
        constexpr int E=384;uint32_t rng=198713;
        auto next=[&] {rng=1664525*rng+1013904223;return rng;};
        std::vector<uint64_t> heat(E),age(E),ref_heat(E),ref_age(E);std::vector<uint32_t> frequency(E),ref_frequency(E);
        uint64_t clock=0,ref_clock=0;int checks=0;
        for(int generation=0;generation<4;++generation) {
            std::fill(heat.begin(),heat.end(),0);std::fill(age.begin(),age.end(),0);std::fill(frequency.begin(),frequency.end(),0);clock=0;
            if(generation==3) {clock=127;for(int e=0;e<E;++e) {heat[e]=next()%8192;age[e]=next()%128;frequency[e]=UINT32_MAX-(next()%7);}}
            ref_heat=heat;ref_age=age;ref_frequency=frequency;ref_clock=clock;
            for(int window=0;window<1000;++window) {
                std::array<int32_t,48> routes;routes.fill(-1);
                const int retained=next()%9,count=retained*6;
                for(int t=0;t<retained;++t) {std::set<int> selected;
                    for(int k=0;k<6;++k) {int e;do {e=(next()%3==0) ? int(next()%12) : int(next()%E);}while(!selected.insert(e).second);routes[t*6+k]=e;}}
                // Independent counts/set reference, not the production age test.
                ++ref_clock;
                if(ref_clock%128==0) for(auto& f:ref_frequency) f/=2;
                if(ref_clock%4==0) for(auto& h:ref_heat) h=h*7/10;
                std::array<int,E> counts{};for(int j=0;j<count;++j) ++counts[routes[j]];
                for(int e=0;e<E;++e) if(counts[e]) {ref_heat[e]+=256;ref_age[e]=ref_clock;
                    ref_frequency[e]=uint32_t(std::min<uint64_t>(UINT32_MAX,uint64_t(ref_frequency[e])+counts[e]));}
                observe_window_heat(heat,age,frequency,clock,routes.data(),count);
                require(clock==ref_clock && heat==ref_heat && age==ref_age && frequency==ref_frequency,"retained-window heat differs from unique/count oracle");++checks;
            }
        }
        // Invalid discarded tails are never read; invalid retained IDs cannot partially mutate training.
        auto saved_heat=heat,saved_age=age;auto saved_frequency=frequency;const auto saved_clock=clock;
        const int32_t bad[]={1,E};bool rejected=false;
        try {observe_window_heat(heat,age,frequency,clock,bad,2);}catch(const std::invalid_argument&) {rejected=true;}
        require(rejected && heat==saved_heat && age==saved_age && frequency==saved_frequency && clock==saved_clock,"invalid prefix changed metadata");
        rejected=false;try {observe_window_heat(heat,age,frequency,clock,nullptr,1);}catch(const std::invalid_argument&) {rejected=true;}require(rejected,"null retained routes accepted");
        std::cout<<checks<<" accepted windows /384 experts, retained0..8, repeated routes, 4/128 decays, saturated counts, resets and invalid-prefix immutability PASS\n";
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
