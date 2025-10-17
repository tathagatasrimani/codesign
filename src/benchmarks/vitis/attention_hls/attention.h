#ifndef ATTENTION_H
#define ATTENTION_H

#include <math.h>

// Constant values can be updated as needed. Must be small for testing.
constexpr int MAX_BATCH = 2;
constexpr int MAX_HEADS = 2;
constexpr int MAX_SEQ_LEN = 16;
constexpr int D_HEAD = 8;
constexpr int MAX_DVALUE = D_HEAD;

void attn_init(int batch_size);

void attn_step(
    int batch_size,
    int num_heads,
    const float Q[MAX_BATCH][MAX_HEADS][D_HEAD],
    const float K_new[MAX_BATCH][MAX_HEADS][D_HEAD],
    const float V_new[MAX_BATCH][MAX_HEADS][MAX_DVALUE],
    float output[MAX_BATCH][MAX_HEADS][MAX_DVALUE]
);

#endif

