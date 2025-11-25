// tiny_llm.c
// Minimal single-file "LLM block" in C: one Transformer encoder block.
// Fixed sizes, no dynamic memory, HLS-friendly patterns.

#include <math.h>
#include <stdio.h>

#define SEQ_LEN 4
#define D_MODEL 8
#define D_FF 16

// Simple ReLU
static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Single-head self-attention + FFN transformer block
// x:      [SEQ_LEN][D_MODEL]   - input tokens
// Wq,Wk,Wv,Wo: [D_MODEL][D_MODEL] - attention weights
// W1:     [D_MODEL][D_FF], b1[D_FF]  - FFN first layer
// W2:     [D_FF][D_MODEL], b2[D_MODEL] - FFN second layer
// out:    [SEQ_LEN][D_MODEL]   - output tokens
void transformer_block(
    float x[SEQ_LEN][D_MODEL],
    const float Wq[D_MODEL][D_MODEL],
    const float Wk[D_MODEL][D_MODEL],
    const float Wv[D_MODEL][D_MODEL],
    const float Wo[D_MODEL][D_MODEL],
    const float W1[D_MODEL][D_FF],
    const float b1[D_FF],
    const float W2[D_FF][D_MODEL],
    const float b2[D_MODEL],
    float out[SEQ_LEN][D_MODEL]
) {
    float Q[SEQ_LEN][D_MODEL];
    float K[SEQ_LEN][D_MODEL];
    float V[SEQ_LEN][D_MODEL];
    float scores[SEQ_LEN][SEQ_LEN];
    float attn[SEQ_LEN][SEQ_LEN];
    float context[SEQ_LEN][D_MODEL];
    float y1[SEQ_LEN][D_MODEL];
    float ff1[SEQ_LEN][D_FF];
    float ff2[SEQ_LEN][D_MODEL];

    // 1. Linear projections: Q = X * Wq, K = X * Wk, V = X * Wv
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float v_sum = 0.0f;
            for (int j = 0; j < D_MODEL; ++j) {
                float x_val = x[t][j];
                q_sum += x_val * Wq[j][i];
                k_sum += x_val * Wk[j][i];
                v_sum += x_val * Wv[j][i];
            }
            Q[t][i] = q_sum;
            K[t][i] = k_sum;
            V[t][i] = v_sum;
        }
    }

    // 2. Scaled dot-product attention: scores = Q * K^T / sqrt(D_MODEL)
    const float scale = 1.0f / 2.82842712475f; // ~1/sqrt(8)
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int u = 0; u < SEQ_LEN; ++u) {
            float dot = 0.0f;
            for (int i = 0; i < D_MODEL; ++i) {
                dot += Q[t][i] * K[u][i];
            }
            scores[t][u] = dot * scale;
        }
    }

    // 3. Softmax over scores along the sequence dimension
    for (int t = 0; t < SEQ_LEN; ++t) {
        // find max for numerical stability
        float max_val = scores[t][0];
        for (int u = 1; u < SEQ_LEN; ++u) {
            if (scores[t][u] > max_val) {
                max_val = scores[t][u];
            }
        }
        float sum_exp = 0.0f;
        for (int u = 0; u < SEQ_LEN; ++u) {
            float e = expf(scores[t][u] - max_val);
            attn[t][u] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int u = 0; u < SEQ_LEN; ++u) {
            attn[t][u] *= inv_sum;
        }
    }

    // 4. Context = Attn * V
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            float c = 0.0f;
            for (int u = 0; u < SEQ_LEN; ++u) {
                c += attn[t][u] * V[u][i];
            }
            context[t][i] = c;
        }
    }

    // 5. Output projection Wo and residual: y1 = x + context*Wo
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            float proj = 0.0f;
            for (int j = 0; j < D_MODEL; ++j) {
                proj += context[t][j] * Wo[j][i];
            }
            y1[t][i] = x[t][i] + proj;  // residual connection
        }
    }

    // 6. Feed-forward: ff1 = ReLU(y1 * W1 + b1)
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int k = 0; k < D_FF; ++k) {
            float sum = b1[k];
            for (int j = 0; j < D_MODEL; ++j) {
                sum += y1[t][j] * W1[j][k];
            }
            ff1[t][k] = relu(sum);
        }
    }

    // 7. Feed-forward second layer: ff2 = ff1 * W2 + b2
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            float sum = b2[i];
            for (int k = 0; k < D_FF; ++k) {
                sum += ff1[t][k] * W2[k][i];
            }
            ff2[t][i] = sum;
        }
    }

    // 8. Final residual: out = y1 + ff2
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            out[t][i] = y1[t][i] + ff2[t][i];
        }
    }
}

// Simple deterministic initialization so you can get stable IR/DSE
static void init_input(float x[SEQ_LEN][D_MODEL]) {
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            x[t][i] = (float)(t * D_MODEL + i) * 0.01f;
        }
    }
}

static void init_matrix_square(float W[D_MODEL][D_MODEL]) {
    for (int i = 0; i < D_MODEL; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            W[i][j] = 0.01f * (float)(i + j + 1);
        }
    }
}

static void init_matrix_mixed(float W[D_MODEL][D_FF]) {
    for (int i = 0; i < D_MODEL; ++i) {
        for (int j = 0; j < D_FF; ++j) {
            W[i][j] = 0.01f * (float)((i + 1) * (j + 1));
        }
    }
}

static void init_matrix_mixed_rev(float W[D_FF][D_MODEL]) {
    for (int i = 0; i < D_FF; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            W[i][j] = 0.01f * (float)((i + j + 1));
        }
    }
}

static void init_bias_ff1(float b[D_FF]) {
    for (int i = 0; i < D_FF; ++i) {
        b[i] = 0.0f;
    }
}

static void init_bias_ff2(float b[D_MODEL]) {
    for (int i = 0; i < D_MODEL; ++i) {
        b[i] = 0.0f;
    }
}

int main() {
    float x[SEQ_LEN][D_MODEL];
    float out[SEQ_LEN][D_MODEL];

    float Wq[D_MODEL][D_MODEL];
    float Wk[D_MODEL][D_MODEL];
    float Wv[D_MODEL][D_MODEL];
    float Wo[D_MODEL][D_MODEL];
    float W1[D_MODEL][D_FF];
    float W2[D_FF][D_MODEL];
    float b1[D_FF];
    float b2[D_MODEL];

    init_input(x);
    init_matrix_square(Wq);
    init_matrix_square(Wk);
    init_matrix_square(Wv);
    init_matrix_square(Wo);
    init_matrix_mixed(W1);
    init_matrix_mixed_rev(W2);
    init_bias_ff1(b1);
    init_bias_ff2(b2);

    transformer_block(x, Wq, Wk, Wv, Wo, W1, b1, W2, b2, out);

    // Print a few outputs so it's not optimized away
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int i = 0; i < D_MODEL; ++i) {
            printf("%f ", out[t][i]);
        }
        printf("\n");
    }

    return 0;
}
