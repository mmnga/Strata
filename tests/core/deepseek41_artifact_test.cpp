#include "strata/deepseek41/model.hpp"
#include "gguf_fixture.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace ds = strata::deepseek41;
void check(bool ok, const char* s) { if (!ok) throw std::runtime_error(s); }
template<class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::runtime_error&) { caught = true; }
    check(caught,"malformed input accepted");
}
int main() {
    const auto path = std::filesystem::temp_directory_path()/"strata-deepseek41-arrays-test.gguf";
    try {
        std::vector<uint64_t> ids(129280), wide(129280);
        std::vector<float> reals(96);
        for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i; wide[i] = (uint64_t(1)<<63)+i; }
        for (size_t i = 0; i < reals.size(); ++i) reals[i] = float(i)/8;
        fixture::write(path,{fixture::u32s("ids",ids),fixture::u64s("wide",wide),fixture::f32s("real",reals)},{});
        {
            strata::GgufFile f(path.string());
            check(f.get("ids")->items.size() == 64,"metadata sample became unbounded");
            check(f.integer_array("ids") == ids,"full token map truncated");
            check(f.integer_array("wide") == wide,"uint64 values lost high bits");
            const auto actual = f.real_array("real");
            check(actual.size() == reals.size(),"real array truncated");
            for (size_t i = 0; i < reals.size(); ++i) check(actual[i] == reals[i],"real array changed");
            rejects([&] { f.integer_array("real"); }); rejects([&] { f.real_array("wide"); });
            rejects([&] { f.integer_array("missing"); });
        }
        ds::EngramConfig c;
        c.layers = {1,14}; c.heads = 2; c.head_dim = 256; c.max_ngram = 4; c.pad_id = 2;
        c.multipliers = {76632096046245,4839876093313,35959672319349,73987337458391,
                         67716810739261,51510806800915,30921347202721,82619226485591};
        c.primes = {101,103,107,109,113,127,101,103,107,109,113,127};
        c.offsets = {0,101,204,311,420,533,0,101,204,311,420,533};
        for (int i = 0; i < 130; ++i) c.token_map.push_back(uint64_t(i*3));
        c.token_map[129] = 99991;
        // Expected rows from an independent Python uint64 XOR/modulo oracle.
        check(ds::engram_rows(c,0,129,{}) == std::vector<int32_t>({54,119,233,384,532,618}),"BOS hash");
        const std::vector<int32_t> history = {17,5,3};
        check(ds::engram_rows(c,0,129,history) == std::vector<int32_t>({27,138,275,312,439,644}),"first table hash");
        const auto before = ds::engram_rows(c,1,129,history);
        check(before == std::vector<int32_t>({61,148,246,379,423,564}),"second table hash");
        auto branch = history; branch.push_back(129); ds::engram_rows(c,1,7,branch);
        check(ds::engram_rows(c,1,129,history) == before,"speculation mutated accepted history");
        check(ds::engram_rows(c,1,9,{17,-1,3}) == std::vector<int32_t>({52,145,223,336,521,616}),"look-back across sequence boundary");
        check(ds::engram_rows(c,1,-1,{129,128,127}) == std::vector<int32_t>({16,135,225,353,515,543}),"padding hash");
        rejects([&] { ds::engram_rows(c,2,1,{}); });
        c.primes[0] = 0; rejects([&] { ds::engram_rows(c,0,1,{}); });
        std::filesystem::remove(path);
        std::cout << "DeepSeek V4.1 full metadata arrays and Engram hashes PASS\n";
    } catch (const std::exception& e) {
        std::filesystem::remove(path); std::cerr << e.what() << '\n'; return 1;
    }
}
