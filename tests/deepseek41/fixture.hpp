#pragma once
#include "../core/gguf_fixture.hpp"

namespace deepseek_fixture {
inline void write(const std::filesystem::path& path,bool executable = false) {
    const std::string p = "deepseek41.";
    std::vector<fixture::Kv> md = {fixture::str("general.architecture","deepseek41")};
    auto i = [&](const char* name,uint32_t n) { md.push_back(fixture::u32(p+name,n)); };
    auto f = [&](const char* name,float n) { md.push_back(fixture::f32(p+name,n)); };
    i("block_count",1); i("embedding_length",512); i("attention.head_count",1); i("attention.head_count_kv",1);
    i("attention.key_length",512); i("attention.value_length",512); i("rope.dimension_count",64);
    i("attention.q_lora_rank",256); i("attention.output_lora_rank",256); i("attention.output_group_count",1); i("attention.sliding_window",128);
    i("expert_count",8); i("expert_used_count",2); i("expert_feed_forward_length",256); i("expert_shared_count",1);
    f("expert_weights_scale",1.5f); md.push_back(fixture::boolean(p+"expert_weights_norm",true)); i("expert_gating_func",4);
    i("hyper_connection.count",4); i("hyper_connection.sinkhorn_iterations",20); f("hyper_connection.epsilon",1e-6f); f("attention.layer_norm_rms_epsilon",1e-20f);
    i("attention.indexer.head_count",1); i("attention.indexer.key_length",64); i("attention.indexer.top_k",8);
    i("context_length",2048); i("rope.scaling.original_context_length",1024); f("rope.freq_base",10000); f("attention.compress_rope_freq_base",160000);
    f("rope.scaling.factor",16); f("rope.scaling.yarn_beta_fast",32); f("rope.scaling.yarn_beta_slow",1); md.push_back(fixture::str(p+"rope.scaling.type","yarn"));
    md.push_back(fixture::u32s(p+"attention.compress_ratios",{0})); f("swiglu_clamp_exp",10); f("swiglu_clamp_shexp",10);
    i("engram.head_count",1); i("engram.key_length",256); i("engram.max_ngram_size",2); i("engram.pad_id",2);
    md.push_back(fixture::u32s(p+"engram.layer_ids",{0})); md.push_back(fixture::u64s(p+"engram.multipliers",{76632096046245,4839876093313}));
    md.push_back(fixture::u32s(p+"engram.primes",{3})); md.push_back(fixture::u32s(p+"engram.offsets",{0}));
    std::vector<uint64_t> map(16); std::vector<std::string> vocab(16);
    for (size_t n = 0; n < 16; ++n) { map[n] = n; vocab[n] = "token"+std::to_string(n); }
    md.push_back(fixture::u32s(p+"engram.token_map",map)); md.push_back(fixture::strs("tokenizer.ggml.tokens",vocab));
    std::vector<fixture::Tensor> ts;
    auto t = [&](const std::string& name,std::vector<uint64_t> shape,uint32_t type = 12) { fixture::Tensor tensor; tensor.name = name; tensor.shape = std::move(shape); tensor.type = type; tensor.seed = uint8_t(ts.size()+1); ts.push_back(std::move(tensor)); };
    t("token_embd.weight",{512,16}); t("output_norm.weight",{512},0); t("output.weight",{512,16},14);
    const std::string b = "blk.0.";
    t(b+"attn_norm.weight",{512},0); t(b+"attn_sinks.weight",{1},0); t(b+"attn_q_a.weight",{512,256}); t(b+"attn_q_a_norm.weight",{256},0);
    t(b+"attn_q_b.weight",{256,512}); t(b+"attn_kv.weight",{512,512}); t(b+"attn_kv_a_norm.weight",{512},0);
    t(b+"attn_output_a.weight",{512,256}); t(b+"attn_output_b.weight",{256,512});
    for (const char* sub : {"attn","ffn"}) { const auto h = b+"hc_"+sub; t(h+"_fn.weight",{2048,24}); t(h+"_base.weight",{24},0); t(h+"_scale.weight",{3},0); }
    t(b+"ffn_norm.weight",{512},0); t(b+"ffn_gate_inp.weight",{512,8},30); t(b+"exp_probs_b.bias",{8},0);
    t(b+"ffn_gate_exps.weight",{512,256,8}); t(b+"ffn_up_exps.weight",{512,256,8}); t(b+"ffn_down_exps.weight",{256,512,8});
    t(b+"ffn_gate_shexp.weight",{512,256}); t(b+"ffn_up_shexp.weight",{512,256}); t(b+"ffn_down_shexp.weight",{256,512});
    t(b+"engram_embd.weight",{256,3}); t(b+"engram_wkv.weight",{256,2560}); t(b+"engram_q.weight",{512,4},30); t(b+"engram_k.weight",{512,4},30);
    if (executable) {
        for (auto& value : md) {
            if (value.key == p+"block_count") value.u = 3;
            if (value.key == p+"attention.head_count" || value.key == p+"attention.output_group_count") value.u = 2;
            if (value.key == p+"attention.compress_ratios") value.arr = {1,1,1};
        }
        for (auto& tensor : ts) {
            if (tensor.name == b+"attn_sinks.weight") tensor.shape = {2};
            if (tensor.name == b+"attn_q_b.weight") tensor.shape = {256,1024};
            if (tensor.name == b+"attn_output_a.weight") tensor.shape = {512,512};
            if (tensor.name == b+"attn_output_b.weight") tensor.shape = {512,512};
        }
        t(b+"attn_compressor_kv.weight",{512,512}); t(b+"attn_compressor_norm.weight",{512},0);
        t(b+"indexer.attn_k.weight",{512,64}); t(b+"indexer.k_norm.weight",{64},0);
        t(b+"indexer.attn_q_b.weight",{256,64}); t(b+"indexer.proj.weight",{512,1});
        const auto original = ts;
        for (int layer = 1; layer < 3; ++layer) for (const auto& tensor : original) {
            if (tensor.name.rfind(b,0) != 0 || tensor.name.find("engram_") != std::string::npos) continue;
            auto copy = tensor; copy.name = "blk."+std::to_string(layer)+"."+tensor.name.substr(b.size()); ts.push_back(std::move(copy));
        }
    }
    const auto written = fixture::write(path,md,ts);
    if (executable) {
        // Keep each tensor's distinct pattern bytes, with zero scales making
        // every matrix finite and deterministic for a complete native forward.
        std::fstream out(path,std::ios::in|std::ios::out|std::ios::binary);
        for (size_t j = 0; j < ts.size(); ++j) {
            strata::TensorInfo info; info.shape = ts[j].shape; info.type = ts[j].type;
            const size_t bytes = strata::tensor_payload_bytes(info), start = size_t(written.data_start+written.offsets[j]);
            const uint32_t zero = 0;
            if (ts[j].type == 12) for (size_t off = 0; off < bytes; off += 144) { out.seekp(std::streamoff(start+off)); out.write(reinterpret_cast<const char*>(&zero),4); }
            else if (ts[j].type == 14) for (size_t off = 0; off < bytes; off += 210) { out.seekp(std::streamoff(start+off+208)); out.write(reinterpret_cast<const char*>(&zero),2); }
            else { std::vector<uint8_t> zeros(bytes); out.seekp(std::streamoff(start)); out.write(reinterpret_cast<const char*>(zeros.data()),std::streamsize(bytes)); }
        }
    }
}
}
