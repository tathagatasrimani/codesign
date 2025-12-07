// transformer.c
// Minimal single-file Transformer encoder-like block (single head, single block)

#include <math.h>

#define SEQ_LEN 4
#define D_MODEL 8
#define D_FF    16

typedef float data_t;

static data_t relu(data_t x) {
    return x > 0.0f ? x : 0.0f;
}

void transformer_block(
    data_t x[SEQ_LEN][D_MODEL],
    const data_t Wq[D_MODEL][D_MODEL],
    const data_t Wk[D_MODEL][D_MODEL],
    const data_t Wv[D_MODEL][D_MODEL],
    const data_t Wo[D_MODEL][D_MODEL],
    const data_t W1[D_MODEL][D_FF],
    const data_t b1[D_FF],
    const data_t W2[D_FF][D_MODEL],
    const data_t b2[D_MODEL],
    data_t out[SEQ_LEN][D_MODEL]
) {
    int i, j, k;

    data_t Q[SEQ_LEN][D_MODEL];
    data_t K[SEQ_LEN][D_MODEL];
    data_t V[SEQ_LEN][D_MODEL];
    data_t scores[SEQ_LEN][SEQ_LEN];
    data_t attn[SEQ_LEN][SEQ_LEN];
    data_t context[SEQ_LEN][D_MODEL];
    data_t y1[SEQ_LEN][D_MODEL];
    data_t ff1[SEQ_LEN][D_FF];
    data_t ff2[SEQ_LEN][D_MODEL];

    // Projections: Q,K,V
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            data_t qs = 0.0f, ks = 0.0f, vs = 0.0f;
            for (k = 0; k < D_MODEL; ++k) {
                data_t xv = x[i][k];
                qs += xv * Wq[k][j];
                ks += xv * Wk[k][j];
                vs += xv * Wv[k][j];
            }
            Q[i][j] = qs;
            K[i][j] = ks;
            V[i][j] = vs;
        }
    }

    // Scaled dot-product scores
    const data_t scale = 1.0f / 2.82842712475f; // ~1/sqrt(8)
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < SEQ_LEN; ++j) {
            data_t dot = 0.0f;
            for (k = 0; k < D_MODEL; ++k) {
                dot += Q[i][k] * K[j][k];
            }
            scores[i][j] = dot * scale;
        }
    }

    // Softmax over sequence dimension (loops aligned i,j)
    for (i = 0; i < SEQ_LEN; ++i) {
        data_t max_val = scores[i][0];
        for (j = 1; j < SEQ_LEN; ++j) {
            if (scores[i][j] > max_val) {
                max_val = scores[i][j];
            }
        }

        data_t sum_exp = 0.0f;
        for (j = 0; j < SEQ_LEN; ++j) {
            data_t e = expf(scores[i][j] - max_val);
            attn[i][j] = e;
            sum_exp += e;
        }

        data_t inv_sum = 1.0f / sum_exp;
        for (j = 0; j < SEQ_LEN; ++j) {
            attn[i][j] = attn[i][j] * inv_sum;
        }
    }

    // Context = Attn @ V
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            data_t c = 0.0f;
            for (k = 0; k < SEQ_LEN; ++k) {
                c += attn[i][k] * V[k][j];
            }
            context[i][j] = c;
        }
    }

    // Output projection + residual: y1 = x + context @ Wo
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            data_t proj = 0.0f;
            for (k = 0; k < D_MODEL; ++k) {
                proj += context[i][k] * Wo[k][j];
            }
            y1[i][j] = x[i][j] + proj;
        }
    }

    // FFN layer 1: ReLU(y1 @ W1 + b1)
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_FF; ++j) {
            data_t s = b1[j];
            for (k = 0; k < D_MODEL; ++k) {
                s += y1[i][k] * W1[k][j];
            }
            ff1[i][j] = relu(s);
        }
    }

    // FFN layer 2: ff2 = ff1 @ W2 + b2
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            data_t s = b2[j];
            for (k = 0; k < D_FF; ++k) {
                s += ff1[i][k] * W2[k][j];
            }
            ff2[i][j] = s;
        }
    }

    // Final residual: out = y1 + ff2
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            out[i][j] = y1[i][j] + ff2[i][j];
        }
    }
}

// Deterministic init for stable IR / DSE
static void init_input(data_t x[SEQ_LEN][D_MODEL]) {
    int i, j;
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            x[i][j] = (data_t)(i * D_MODEL + j) * 0.01f;
        }
    }
}

static void init_matrix_square(data_t W[D_MODEL][D_MODEL]) {
    int i, j;
    for (i = 0; i < D_MODEL; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            W[i][j] = 0.01f * (data_t)(i + j + 1);
        }
    }
}

static void init_matrix_mixed(data_t W[D_MODEL][D_FF]) {
    int i, j;
    for (i = 0; i < D_MODEL; ++i) {
        for (j = 0; j < D_FF; ++j) {
            W[i][j] = 0.01f * (data_t)((i + 1) * (j + 1));
        }
    }
}

static void init_matrix_mixed_rev(data_t W[D_FF][D_MODEL]) {
    int i, j;
    for (i = 0; i < D_FF; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            W[i][j] = 0.01f * (data_t)(i + j + 1);
        }
    }
}

static void init_bias_ff1(data_t b[D_FF]) {
    int i;
    for (i = 0; i < D_FF; ++i) {
        b[i] = 0.0f;
    }
}

static void init_bias_ff2(data_t b[D_MODEL]) {
    int i;
    for (i = 0; i < D_MODEL; ++i) {
        b[i] = 0.0f;
    }
}

