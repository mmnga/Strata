#include "strata/deepseek41/model.hpp"
#include <cstdio>
#include <exception>
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"usage: strata-deepseek41-inspect SHARD.gguf\n");return 2; }
    try {
        const strata::deepseek41::Model m(argv[1]);const auto& c=m.config();
        std::printf("Strata native DeepSeek V4.1 artifact check (no inference)\n");
        std::printf("shards=%zu tensors=%zu layers=%d hidden=%d vocab=%d experts=%d top_k=%d\n",m.gguf().size(),m.tensor_count(),c.layers,c.hidden,c.vocab,c.experts,c.top_k);
        std::printf("routed_bytes=%llu engram_bytes=%llu dense_bytes=%llu full_token_map=%zu\n",(unsigned long long)m.routed_bytes(),(unsigned long long)m.engram_bytes(),(unsigned long long)m.dense_bytes(),c.engram.token_map.size());
        for(int i=0;i<c.layers;++i){const auto& s=m.streams()[i];std::printf("layer=%d ratio=%llu kv=%d key=%d topk=%d engram=%d\n",i,(unsigned long long)c.compress_ratios[i],s.kv_source,s.key_source,s.top_k_source,s.engram_index);}
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
