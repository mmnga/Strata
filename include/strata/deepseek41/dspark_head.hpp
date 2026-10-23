#pragma once
#include <cstdint>
namespace strata::deepseek41::cuda {
struct DraftHeadView {
    const float* hidden;
    const float* markov_rows;
    const float* markov_weight;
    const float* confidence_weight;
    const float* confidence_bias;
    const int32_t* anchor;
    float* logits;
    float* markov;
    float* bias;
    float* confidence;
    int32_t* picks;
    int hidden_width,vocab;
};
bool conditional_draft_available();
// Capture-only, fixed device buffers. Original first-always-offer policy;
// conditionals skip only Markov/sample work beyond the confidence prefix.
void conditional_draft_head(const DraftHeadView& view,int count,float threshold,
    void* stream,void* const* body_streams);
}
