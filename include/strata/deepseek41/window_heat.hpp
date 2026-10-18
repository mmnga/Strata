#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::deepseek41 {
// Original accepted-window heat rule from expert_dispatch::accept_window:
// one credit/expert, separate retained occurrence counts, 4/128-window decay.
// Call under the existing layer mutex. Predictions never call this function.
inline void observe_window_heat(std::vector<uint64_t>& heat,std::vector<uint64_t>& age,
                                std::vector<uint32_t>& frequency,uint64_t& clock,
                                const int32_t* ids,int count) {
    if(heat.empty() || heat.size()!=age.size() || heat.size()!=frequency.size() ||
       count<0 || count>48 || (count && !ids)) throw std::invalid_argument("cache window heat layout");
    for(int j=0;j<count;++j) if(ids[j]<0 || size_t(ids[j])>=heat.size())
        throw std::invalid_argument("invalid retained cache expert");
    const auto stamp=++clock;
    if(stamp%128==0) for(auto& f:frequency) f/=2;
    if(stamp%4==0) for(auto& h:heat) h=h*7/10;
    for(int j=0;j<count;++j) {
        const int e=ids[j];
        if(age[e]!=stamp) heat[e]+=256;
        age[e]=stamp;
        if(frequency[e]<UINT32_MAX) ++frequency[e];
    }
}
} // namespace strata::deepseek41
