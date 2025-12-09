// mini_chatgpt_kernel.c
// Pure compute kernel: single-layer decoder-only transformer (GPT-like).
// Only dependency: <math.h>. No stdio/stdlib/string/OMP, no dynamic alloc.

#include <math.h>

#define SEQ_LEN      8    // sequence length (tokens in context)
#define D_MODEL     16    // hidden size
#define D_FF        32    // feedforward size
#define VOCAB_SIZE  32    // vocab size
#define N_HEAD       2    // number of attention heads
#define D_HEAD      (D_MODEL / N_HEAD)

typedef float data_t;

// Simple ReLU
static inline data_t relu(data_t x) {
    return x > 0.0f ? x : 0.0f;
}

// Top-level kernel: one forward pass.
// Inputs:
//   token_ids[SEQ_LEN]                 : input token indices
//   tok_embedding[VOCAB_SIZE][D_MODEL] : token embedding table
//   Wq/Wk/Wv[D_MODEL][D_MODEL]         : attention projections
//   Wff1[D_MODEL][D_FF], bff1[D_FF]    : FFN1
//   Wff2[D_FF][D_MODEL], bff2[D_MODEL] : FFN2
//   Wout[D_MODEL][VOCAB_SIZE], bout[VOCAB_SIZE] : output projection
//
// Output:
//   logits[SEQ_LEN][VOCAB_SIZE]
void miniGPT(
    const data_t X[SEQ_LEN][D_MODEL],   // precomputed embeddings

    const data_t Wq[D_MODEL][D_MODEL],
    const data_t Wk[D_MODEL][D_MODEL],
    const data_t Wv[D_MODEL][D_MODEL],

    const data_t Wff1[D_MODEL][D_FF],
    const data_t bff1[D_FF],
    const data_t Wff2[D_FF][D_MODEL],
    const data_t bff2[D_MODEL],

    const data_t Wout[D_MODEL][VOCAB_SIZE],
    const data_t bout[VOCAB_SIZE],

    data_t logits[SEQ_LEN][VOCAB_SIZE]
) {
    int i, j, k, h;

    // Buffers
    data_t X[SEQ_LEN][D_MODEL];       // token embeddings
    data_t Q[SEQ_LEN][D_MODEL];
    data_t K[SEQ_LEN][D_MODEL];
    data_t V[SEQ_LEN][D_MODEL];

    data_t scores[SEQ_LEN][SEQ_LEN];  // reused per-head
    data_t attn[SEQ_LEN][SEQ_LEN];
    data_t context[SEQ_LEN][D_MODEL];

    data_t h1[SEQ_LEN][D_MODEL];      // after attention + residual
    data_t ff1[SEQ_LEN][D_FF];        // FF1 + ReLU
    data_t h2[SEQ_LEN][D_MODEL];      // after FF2 + residual

    // ---------------------------------------------------------------------
    // 1. Embedding lookup: perfectly nested i-j loop
    //    X[i][j] = tok_embedding[token_ids[i]][j]
    // ---------------------------------------------------------------------
    // for (i = 0; i < SEQ_LEN; ++i) {
    //     for (j = 0; j < D_MODEL; ++j) {
    //         X[i][j] = tok_embedding[token_ids[i]][j];
    //     }
    // }

    // ---------------------------------------------------------------------
    // 2. Q, K, V projections: X @ Wq, X @ Wk, X @ Wv
    // ---------------------------------------------------------------------

    // Init Q/K/V
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            Q[i][j] = 0.0f;
            K[i][j] = 0.0f;
            V[i][j] = 0.0f;
        }
    }

    // Q = X @ Wq
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            for (k = 0; k < D_MODEL; ++k) {
                Q[i][j] += X[i][k] * Wq[k][j];
            }
        }
    }

    // K = X @ Wk
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            for (k = 0; k < D_MODEL; ++k) {
                K[i][j] += X[i][k] * Wk[k][j];
            }
        }
    }

    // V = X @ Wv
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            for (k = 0; k < D_MODEL; ++k) {
                V[i][j] += X[i][k] * Wv[k][j];
            }
        }
    }

    // ---------------------------------------------------------------------
    // 3. Multi-head self-attention (naive split by heads)
    // ---------------------------------------------------------------------

    // Zero context
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            context[i][j] = 0.0f;
        }
    }

    const data_t scale = 1.0f / 4.0f; // ~1/sqrt(D_HEAD) for D_HEAD = 4

    for (h = 0; h < N_HEAD; ++h) {
        int offset = h * D_HEAD;

        // scores[i][j] = (Q_h[i] · K_h[j]) * scale
        for (i = 0; i < SEQ_LEN; ++i) {
            for (j = 0; j < SEQ_LEN; ++j) {
                data_t dot = 0.0f;
                for (k = 0; k < D_HEAD; ++k) {
                    dot += Q[i][offset + k] * K[j][offset + k];
                }
                scores[i][j] = dot * scale;
            }
        }

        // softmax over j for each i
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

        // context_h[i][:] = sum_t attn[i][t] * V_h[t][:]
        for (i = 0; i < SEQ_LEN; ++i) {
            for (j = 0; j < D_HEAD; ++j) {
                data_t c = 0.0f;
                for (k = 0; k < SEQ_LEN; ++k) {
                    c += attn[i][k] * V[k][offset + j];
                }
                context[i][offset + j] += c;
            }
        }
    }

    // ---------------------------------------------------------------------
    // 4. Attention residual: h1 = X + context
    // ---------------------------------------------------------------------
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            h1[i][j] = X[i][j] + context[i][j];
        }
    }

    // ---------------------------------------------------------------------
    // 5. FFN1: ff1 = ReLU(h1 @ Wff1 + bff1)
    // ---------------------------------------------------------------------
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_FF; ++j) {
            data_t s = bff1[j];
            for (k = 0; k < D_MODEL; ++k) {
                s += h1[i][k] * Wff1[k][j];
            }
            ff1[i][j] = relu(s);
        }
    }

    // ---------------------------------------------------------------------
    // 6. FFN2: tmp = ff1 @ Wff2 + bff2, then residual h2 = h1 + tmp
    // ---------------------------------------------------------------------
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < D_MODEL; ++j) {
            data_t s = bff2[j];
            for (k = 0; k < D_FF; ++k) {
                s += ff1[i][k] * Wff2[k][j];
            }
            h2[i][j] = h1[i][j] + s;
        }
    }

    // ---------------------------------------------------------------------
    // 7. Output projection: logits = h2 @ Wout + bout
    // ---------------------------------------------------------------------
    for (i = 0; i < SEQ_LEN; ++i) {
        for (j = 0; j < VOCAB_SIZE; ++j) {
            data_t s = bout[j];
            for (k = 0; k < D_MODEL; ++k) {
                s += h2[i][k] * Wout[k][j];
            }
            logits[i][j] = s;
        }
    }
}
