// file: llm_block.c
// Tiny LLM building block: Y = ReLU(X * W + B)
// X: [M x K], W: [K x N], B: [N], Y: [M x N]

#define M 1     // batch size
#define K 16    // input dim
#define N 16    // output dim

void linear_layer(float X[M][K],
                float W[K][N],
                float B[N],
                float Y[M][N]) {
#pragma scop
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = B[j];
      for (int k = 0; k < K; k++) {
        acc += X[i][k] * W[k][j];
      }
      // ReLU
      if (acc < 0.0f)
        acc = 0.0f;
      Y[i][j] = acc;
    }
  }
#pragma endscop
}
