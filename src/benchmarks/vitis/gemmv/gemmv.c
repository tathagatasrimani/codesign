#include "arith_ops.c"
#define N 32
void gemmv(float alpha, float beta, float y[N], float A[N][N],
               float x[N]) {
#pragma scop
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      y[i] += alpha * A[i][j] * x[j];
    }
  }
#pragma endscop
}
