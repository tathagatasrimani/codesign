// transformer.c
// Simple Q/K/V projection with perfectly nested loops for ScaleHLS

#ifndef SEQ_LEN
#define SEQ_LEN  16    // sequence length (change as needed)
#endif

#ifndef D_MODEL
#define D_MODEL  32    // model dimension (change as needed)
#endif

typedef float data_t;

// Compute Q = X * Wq, K = X * Wk, V = X * Wv
// Trying all the completely nested loops for the computations.
void transformer_block_small(
    data_t X[SEQ_LEN][D_MODEL],
    data_t Wq[D_MODEL][D_MODEL],
    data_t Wk[D_MODEL][D_MODEL],
    data_t Wv[D_MODEL][D_MODEL],
    data_t Q[SEQ_LEN][D_MODEL],
    data_t K[SEQ_LEN][D_MODEL],
    data_t V[SEQ_LEN][D_MODEL]
) {
    int i, j, k;

    // Initialize outputs to zero (perfectly nested loops)
    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < D_MODEL; j++) {
            Q[i][j] = 0.0f;
            K[i][j] = 0.0f;
            V[i][j] = 0.0f;
        }
    }

    // Q = X * Wq
    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < D_MODEL; j++) {
            for (k = 0; k < D_MODEL; k++) {
                Q[i][j] += X[i][k] * Wq[k][j];
            }
        }
    }

    // K = X * Wk
    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < D_MODEL; j++) {
            for (k = 0; k < D_MODEL; k++) {
                K[i][j] += X[i][k] * Wk[k][j];
            }
        }
    }

    // V = X * Wv
    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < D_MODEL; j++) {
            for (k = 0; k < D_MODEL; k++) {
                V[i][j] += X[i][k] * Wv[k][j];
            }
        }
    }
}
