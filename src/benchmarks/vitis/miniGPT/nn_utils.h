// nn_utils.h
#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <math.h>

// These can be overridden before including the header
#ifndef SEQ_LEN
#define SEQ_LEN 4
#endif

#ifndef D_MODEL
#define D_MODEL 8
#endif

#ifndef D_FF
#define D_FF 16
#endif

typedef float data_t;

// Activation
static inline data_t relu(data_t x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------
// Basic building blocks (all perfectly nested loops)
// ---------------------------------------------------------------------

// y = x @ W + b
// x: [B][IN], W: [IN][OUT], b: [OUT], y: [B][OUT]
static inline void linear_BxIN_INxOUT(
    int B, int IN, int OUT,
    data_t x[][D_MODEL],        // assume IN <= D_MODEL
    const data_t W[][D_MODEL],  // assume OUT <= D_MODEL
    const data_t b[],
    data_t y[][D_MODEL]
) {
    int i, j, k;
    for (i = 0; i < B; ++i) {
        for (j = 0; j < OUT; ++j) {
            data_t s = b[j];
            for (k = 0; k < IN; ++k) {
                s += x[i][k] * W[k][j];
            }
            y[i][j] = s;
        }
    }
}

// Softmax over last dimension of length N: in/out: [B][N]
static inline void softmax_BxN(
    int B, int N,
    data_t in[][SEQ_LEN],
    data_t out[][SEQ_LEN]
) {
    int i, j;
    for (i = 0; i < B; ++i) {
        // max
        data_t max_val = in[i][0];
        for (j = 1; j < N; ++j) {
            if (in[i][j] > max_val) {
                max_val = in[i][j];
            }
        }
        // exp & sum
        data_t sum_exp = 0.0f;
        for (j = 0; j < N; ++j) {
            data_t e = expf(in[i][j] - max_val);
            out[i][j] = e;
            sum_exp += e;
        }
        // normalize
        data_t inv_sum = 1.0f / sum_exp;
        for (j = 0; j < N; ++j) {
            out[i][j] = out[i][j] * inv_sum;
        }
    }
}

// Elementwise ReLU on [B][D]
static inline void relu_BxD(
    int B, int D,
    data_t in[][D_MODEL],
    data_t out[][D_MODEL]
) {
    int i, j;
    for (i = 0; i < B; ++i) {
        for (j = 0; j < D; ++j) {
            data_t v = in[i][j];
            out[i][j] = v > 0.0f ? v : 0.0f;
        }
    }
}

#endif // NN_UTILS_H
