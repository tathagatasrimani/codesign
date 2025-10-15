#include "attention.h"
#include <math.h>
#include <string.h>

static float K_cache[MAX_BATCH][MAX_HEADS][MAX_SEQ_LEN][D_HEAD];
static float V_cache[MAX_BATCH][MAX_HEADS][MAX_SEQ_LEN][MAX_DVALUE];
static int cached_len[MAX_BATCH];
static int max_seq_len = MAX_SEQ_LEN;

void attn_init(int batch_size)
{
#pragma HLS INLINE off
    for (int b = 0; b < batch_size; b++) {
#pragma HLS PIPELINE
        cached_len[b] = 0;
    }
}

void attn_step(
    int batch_size,
    int num_heads,
    const float Q[MAX_BATCH][MAX_HEADS][D_HEAD],
    const float K_new[MAX_BATCH][MAX_HEADS][D_HEAD],
    const float V_new[MAX_BATCH][MAX_HEADS][MAX_DVALUE],
    float output[MAX_BATCH][MAX_HEADS][MAX_DVALUE]
) {
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=K_new offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=V_new offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=batch_size
#pragma HLS INTERFACE s_axilite port=num_heads
#pragma HLS INTERFACE s_axilite port=return

    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_BATCH
        int seq_len = cached_len[b] + 1;
        if (seq_len > max_seq_len) seq_len = max_seq_len;

        for (int h = 0; h < num_heads; h++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEADS
#pragma HLS PIPELINE

            for (int d = 0; d < D_HEAD; d++) {
#pragma HLS UNROLL factor=1
                K_cache[b][h][cached_len[b]][d] = K_new[b][h][d];
            }
            for (int d = 0; d < MAX_DVALUE; d++) {
#pragma HLS UNROLL factor=1
                V_cache[b][h][cached_len[b]][d] = V_new[b][h][d];
            }

            float scale = 1.0f / sqrtf((float)D_HEAD);
            float scores[MAX_SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=scores complete dim=1
            float exp_scores[MAX_SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=exp_scores complete dim=1

            float max_val = -1e30f;
            for (int j = 0; j < seq_len; j++) {
#pragma HLS PIPELINE
                float dot = 0.0f;
                for (int d = 0; d < D_HEAD; d++) {
#pragma HLS UNROLL factor=4
                    dot += Q[b][h][d] * K_cache[b][h][j][d];
                }
                float sc = dot * scale;
                scores[j] = sc;
                if (sc > max_val) max_val = sc;
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
#pragma HLS PIPELINE
                float v = scores[j] - max_val;
                float e = expf(v);
                exp_scores[j] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < seq_len; j++) {
#pragma HLS PIPELINE
                exp_scores[j] = exp_scores[j] * inv_sum;
            }

            for (int dv = 0; dv < MAX_DVALUE; dv++) {
#pragma HLS PIPELINE
                float acc = 0.0f;
                for (int j = 0; j < seq_len; j++) {
#pragma HLS UNROLL factor=1
                    acc += exp_scores[j] * V_cache[b][h][j][dv];
                }
                output[b][h][dv] = acc;
            }
        }

        if (cached_len[b] < max_seq_len - 1) {
            cached_len[b]++;
        } else {
            for (int h = 0; h < num_heads; h++) {
#pragma HLS PIPELINE
                for (int j = 0; j < max_seq_len - 1; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_SEQ_LEN
                    for (int d = 0; d < D_HEAD; d++) {
#pragma HLS UNROLL factor=2
                        K_cache[b][h][j][d] = K_cache[b][h][j+1][d];
                    }
                    for (int d = 0; d < MAX_DVALUE; d++) {
#pragma HLS UNROLL factor=2
                        V_cache[b][h][j][d] = V_cache[b][h][j+1][d];
                    }
                }
                for (int d = 0; d < D_HEAD; d++) {
#pragma HLS UNROLL factor=1
                    K_cache[b][h][max_seq_len-1][d] = K_new[b][h][d];
                }
                for (int d = 0; d < MAX_DVALUE; d++) {
#pragma HLS UNROLL factor=1
                    V_cache[b][h][max_seq_len-1][d] = V_new[b][h][d];
                }
            }
        }
    }
}

