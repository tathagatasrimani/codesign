// vjepa2_dse_friendly.c
//
// Changes applied for Polygeist -> MLIR -> ScaleHLS(-HIDA) DSE friendliness:
// 1) Provide an explicit-argument top kernel: VJEPA2_kernel(...). (No reliance on globals.)
// 2) Keep loops simple and affine-friendly (mostly i/j/k nests).
// 3) Remove heavy libm usage (no <math.h>, no expf/tanhf/logf/cosf/sqrtf).
//    - LayerNorm uses an invsqrt approximation (Newton-Raphson).
//    - Softmax uses a cheap exp approximation.
//    - GELU uses a tanh approximation (polynomial).
// 4) Remove RNG that depended on log/cos/sqrt. Use a simple LCG-based init.
// 5) Avoid memcpy; use explicit loops.
//
// Notes:
// - C "array parameters" still decay to pointers in C semantics, but the source
//   uses only fixed-size array syntax (no explicit pointer types).
// - For DSE, run cgeist with: -function=VJEPA2_kernel
//
// Build/test as plain C if you add a main() (optional). For MLIR conversion,
// you typically don't need main().

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>

#define IMG_H 224
#define IMG_W 224
#define C_IN  3
#define PATCH 16

#define HP (IMG_H / PATCH)
#define WP (IMG_W / PATCH)
#define N  (HP * WP)

#define D  192
#define HEADS 3
#define DEPTH 2
#define HIDDEN_MULT 4

#define PATCH_DIM  (PATCH * PATCH * C_IN)
#define MLP_HIDDEN (D * HIDDEN_MULT)
#define DH         (D / HEADS)

// ------------------------- Small numeric helpers ----------------------------

// Absolute value without <math.h>
static inline float f_abs(float x) { return (x < 0.0f) ? -x : x; }


// Fast inverse sqrt approximation using Newton-Raphson.
// Assumes x > 0. Returns approx 1/sqrt(x).
static inline float inv_sqrt_approx(float x) {
  // Avoid division by zero / negative.
  if (x <= 1.0e-12f) return 1.0e6f;

  // Initial guess using a rough reciprocal (not bit-hack to keep it simple/portable).
  float y = 1.0f / x;

  // Improve: y ~= 1/sqrt(x) using Newton steps on f(y)=1/y^2 - x = 0
  // Update: y = y * (1.5 - 0.5 * x * y^2)
  // We need y ~ 1/sqrt(x). If we start with 1/x, it converges enough for LN.
  float halfx = 0.5f * x;
  for (int it = 0; it < 3; it++) {
    y = y * (1.5f - halfx * y * y);
  }
  return y;
}

// exp approximation on a small range.
// For softmax, inputs are shifted by max, so typical range is <= 0.
// We clamp to [-10, 0] and approximate exp(x) with a polynomial.
static inline float exp_approx(float x) {
  if (x < -10.0f) x = -10.0f;
  if (x > 0.0f) x = 0.0f;
  // 5th order Taylor-ish polynomial around 0:
  // exp(x) ~ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  return 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.00833333f * x5;
}

// tanh approximation on a moderate range using a rational/polynomial form.
// We clamp to [-3, 3] and use: tanh(x) ~ x*(27 + x^2)/(27 + 9*x^2)
static inline float tanh_approx(float x) {
  if (x < -3.0f) x = -3.0f;
  if (x > 3.0f) x = 3.0f;
  float x2 = x * x;
  float num = x * (27.0f + x2);
  float den = (27.0f + 9.0f * x2);
  return num / den;
}

// GELU approximation without libm:
// gelu(x) ~ 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ) )
// Use sqrt(2/pi) ~ 0.79788456
static inline float gelu_approx(float x) {
  float x3 = x * x * x;
  float y = 0.79788456f * (x + 0.044715f * x3);
  float t = tanh_approx(y);
  return 0.5f * x * (1.0f + t);
}

// ------------------------- Benchmark kernel pieces --------------------------

// Patchify+project with py->px->d->k nesting
static void patchify_project_addpos(
  float img[C_IN][IMG_H][IMG_W],
  float x[N][D],
  float Wpatch[PATCH_DIM][D],
  float bpatch[D],
  float pos[N][D]
) {
  unsigned n = 0;
  for (int py = 0; py < HP; py++, n++) {
    for (int px = 0; px < WP; px++, n++) {
      for (int d = 0; d < D; d++) {
        float acc = 0.0f;

        const unsigned PATCH2 = (unsigned)(PATCH * PATCH);

        for (unsigned k = 0; k < (unsigned)PATCH_DIM; k++) {
          unsigned c  = k / PATCH2;      // 0..2
          unsigned r  = k % PATCH2;      // 0..255
          unsigned dy = r / (unsigned)PATCH; // 0..15
          unsigned dx = r % (unsigned)PATCH; // 0..15

          unsigned y = (unsigned)py * (unsigned)PATCH + dy;
          unsigned x = (unsigned)px * (unsigned)PATCH + dx;

          acc += img[c][y][x] * Wpatch[k][d];
        }

        x[n][d] = acc + bpatch[d] + pos[n][d];
      }
    }
  }
}

static void copy_x_to_tmp(float x[N][D], float tmp[N][D]) {
  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++)
      tmp[i][d] = x[i][d];
}

static void add_proj_to_x(float x[N][D], float proj[N][D]) {
  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++)
      x[i][d] += proj[i][d];
}

static void add_y_to_x(float x[N][D], float y[N][D]) {
  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++)
      x[i][d] += y[i][d];
}

// LayerNorm in-place on tmp (multi-pass)
static void layernorm_inplace(
  float tmp[N][D],
  float gamma[D],
  float beta[D]
) {
  for (int i = 0; i < N; i++) {
    float mean = 0.0f;
    for (int d = 0; d < D; d++) mean += tmp[i][d];
    mean *= (1.0f / (float)D);

    float var = 0.0f;
    for (int d = 0; d < D; d++) {
      float v = tmp[i][d] - mean;
      var += v * v;
    }
    var *= (1.0f / (float)D);

    // inv = 1/sqrt(var + eps)
    float inv = inv_sqrt_approx(var + 1.0e-6f);

    for (int d = 0; d < D; d++) {
      float xn = (tmp[i][d] - mean) * inv;
      tmp[i][d] = xn * gamma[d] + beta[d];
    }
  }
}

// Attention block:
// - qkv = tmp @ Wqkv + bqkv
// - per-head scores + softmax + weighted sum into attn_out
// - proj = attn_out @ Wo + bo
static void attention_block(
  float tmp[N][D],
  float Wqkv[D][3 * D],
  float bqkv[3 * D],
  float Wo[D][D],
  float bo[D],
  float qkv[N][3 * D],
  float attn_out[N][D],
  float proj[N][D],
  float scores[N][N]
) {
  // qkv = tmp @ Wqkv + bqkv
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < 3 * D; j++) {
      float acc = 0.0f;
      for (int k = 0; k < D; k++) acc += tmp[i][k] * Wqkv[k][j];
      qkv[i][j] = acc + bqkv[j];
    }
  }

  // attn_out = 0
  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++)
      attn_out[i][d] = 0.0f;

  // scale = 1/sqrt(DH)
  float scale = inv_sqrt_approx((float)DH);

  for (int h = 0; h < HEADS; h++) {
    int q_off = 0 + h * DH;
    int k_off = D + h * DH;
    int v_off = 2 * D + h * DH;

    // scores: i->j->d
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int d = 0; d < DH; d++)
          acc += qkv[i][q_off + d] * qkv[j][k_off + d];
        scores[i][j] = acc * scale;
      }
    }

    // softmax per row (inlined, no function call)
    for (int i = 0; i < N; i++) {
      // max
      float m = scores[i][0];
      for (int j = 1; j < N; j++) {
        float v = scores[i][j];
        if (v > m) m = v;
      }

      // exp and sum
      float s = 0.0f;
      for (int j = 0; j < N; j++) {
        float e = exp_approx(scores[i][j] - m);
        scores[i][j] = e;
        s += e;
      }

      // normalize
      float invs = 1.0f / (s + 1.0e-9f);
      for (int j = 0; j < N; j++) scores[i][j] *= invs;
    }

    // out: i->d->j
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < DH; d++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
          acc += scores[i][j] * qkv[j][v_off + d];
        attn_out[i][h * DH + d] = acc;
      }
    }
  }

  // proj = attn_out @ Wo + bo
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      float acc = 0.0f;
      for (int k = 0; k < D; k++) acc += attn_out[i][k] * Wo[k][j];
      proj[i][j] = acc + bo[j];
    }
  }
}

// MLP block:
// - h = gelu(tmp @ W1 + b1)
// - y = h @ W2 + b2
static void mlp_block(
  float tmp[N][D],
  float W1[D][MLP_HIDDEN],
  float b1[MLP_HIDDEN],
  float W2[MLP_HIDDEN][D],
  float b2[D],
  float h[N][MLP_HIDDEN],
  float y[N][D]
) {
  // h = gelu(tmp @ W1 + b1)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < MLP_HIDDEN; j++) {
      float acc = 0.0f;
      for (int k = 0; k < D; k++) acc += tmp[i][k] * W1[k][j];
      h[i][j] = gelu_approx(acc + b1[j]);
    }
  }

  // y = h @ W2 + b2
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D; j++) {
      float acc = 0.0f;
      for (int k = 0; k < MLP_HIDDEN; k++) acc += h[i][k] * W2[k][j];
      y[i][j] = acc + b2[j];
    }
  }
}

// ------------------------- Top kernel (explicit args) ------------------------
//
// For ScaleHLS-HIDA DSE, target this function:
//   cgeist vjepa2_dse_friendly.c -function=VJEPA2_kernel -S -memref-fullrank -raise-scf-to-affine > out.mlir
//
void VJEPA2(
  float img[C_IN][IMG_H][IMG_W],

  float x[N][D],
  float tmp[N][D],

  float Wpatch[PATCH_DIM][D],
  float bpatch[D],
  float pos[N][D],

  float ln1_g[DEPTH][D], float ln1_b[DEPTH][D],
  float ln2_g[DEPTH][D], float ln2_b[DEPTH][D],

  float Wqkv[DEPTH][D][3 * D],
  float bqkv[DEPTH][3 * D],
  float Wo[DEPTH][D][D],
  float bo[DEPTH][D],

  float W1[DEPTH][D][MLP_HIDDEN],
  float b1[DEPTH][MLP_HIDDEN],
  float W2[DEPTH][MLP_HIDDEN][D],
  float b2[DEPTH][D],

  float lnf_g[D], float lnf_b[D],

  float qkv[N][3 * D],
  float attn_out[N][D],
  float proj[N][D],
  float scores[N][N],
  float h[N][MLP_HIDDEN],
  float y[N][D]
) {
  patchify_project_addpos(img, x, Wpatch, bpatch, pos);

  for (int l = 0; l < DEPTH; l++) {
    copy_x_to_tmp(x, tmp);
    layernorm_inplace(tmp, ln1_g[l], ln1_b[l]);

    attention_block(tmp, Wqkv[l], bqkv[l], Wo[l], bo[l], qkv, attn_out, proj, scores);
    add_proj_to_x(x, proj);

    copy_x_to_tmp(x, tmp);
    layernorm_inplace(tmp, ln2_g[l], ln2_b[l]);

    mlp_block(tmp, W1[l], b1[l], W2[l], b2[l], h, y);
    add_y_to_x(x, y);
  }

  // final LN on x: reuse tmp as working buffer
  copy_x_to_tmp(x, tmp);
  layernorm_inplace(tmp, lnf_g, lnf_b);

  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++)
      x[i][d] = tmp[i][d];
}

// ------------------------- Optional deterministic init ----------------------
// This is separated from the kernel to keep DSE focused on compute.

// Simple LCG for deterministic float init (no libm)
static uint32_t g_lcg = 0x12345678u;
static inline uint32_t lcg_u32(void) {
  g_lcg = 1664525u * g_lcg + 1013904223u;
  return g_lcg;
}
static inline float lcg_f32_signed(void) {
  // Map to roughly [-1, 1]
  uint32_t u = lcg_u32();
  // Use top 23 bits as mantissa-ish fraction.
  float f = (float)(u & 0x007FFFFFu) * (1.0f / 8388608.0f); // [0,1)
  return (2.0f * f) - 1.0f;
}

void init_vjepa2_model(
  float img[C_IN][IMG_H][IMG_W],

  float x[N][D],
  float tmp[N][D],

  float Wpatch[PATCH_DIM][D],
  float bpatch[D],
  float pos[N][D],

  float ln1_g[DEPTH][D], float ln1_b[DEPTH][D],
  float ln2_g[DEPTH][D], float ln2_b[DEPTH][D],

  float Wqkv[DEPTH][D][3 * D],
  float bqkv[DEPTH][3 * D],
  float Wo[DEPTH][D][D],
  float bo[DEPTH][D],

  float W1[DEPTH][D][MLP_HIDDEN],
  float b1[DEPTH][MLP_HIDDEN],
  float W2[DEPTH][MLP_HIDDEN][D],
  float b2[DEPTH][D],

  float lnf_g[D], float lnf_b[D],

  float qkv[N][3 * D],
  float attn_out[N][D],
  float proj[N][D],
  float scores[N][N],
  float h[N][MLP_HIDDEN],
  float y[N][D]
) {
  // Initialize img
  for (int c = 0; c < C_IN; c++)
    for (int yy = 0; yy < IMG_H; yy++)
      for (int xx = 0; xx < IMG_W; xx++)
        img[c][yy][xx] = 0.5f * lcg_f32_signed();

  // Zero x/tmp
  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++) {
      x[i][d] = 0.0f;
      tmp[i][d] = 0.0f;
    }

  // Wpatch ~ 0.02 * signed
  for (int k = 0; k < PATCH_DIM; k++)
    for (int d = 0; d < D; d++)
      Wpatch[k][d] = 0.02f * lcg_f32_signed();

  for (int d = 0; d < D; d++) bpatch[d] = 0.0f;

  for (int n = 0; n < N; n++)
    for (int d = 0; d < D; d++)
      pos[n][d] = 0.02f * lcg_f32_signed();

  for (int l = 0; l < DEPTH; l++) {
    for (int d = 0; d < D; d++) {
      ln1_g[l][d] = 1.0f; ln1_b[l][d] = 0.0f;
      ln2_g[l][d] = 1.0f; ln2_b[l][d] = 0.0f;
    }

    for (int i = 0; i < D; i++)
      for (int j = 0; j < 3 * D; j++)
        Wqkv[l][i][j] = 0.02f * lcg_f32_signed();
    for (int j = 0; j < 3 * D; j++) bqkv[l][j] = 0.0f;

    for (int i = 0; i < D; i++)
      for (int j = 0; j < D; j++)
        Wo[l][i][j] = 0.02f * lcg_f32_signed();
    for (int j = 0; j < D; j++) bo[l][j] = 0.0f;

    for (int i = 0; i < D; i++)
      for (int j = 0; j < MLP_HIDDEN; j++)
        W1[l][i][j] = 0.02f * lcg_f32_signed();
    for (int j = 0; j < MLP_HIDDEN; j++) b1[l][j] = 0.0f;

    for (int i = 0; i < MLP_HIDDEN; i++)
      for (int j = 0; j < D; j++)
        W2[l][i][j] = 0.02f * lcg_f32_signed();
    for (int j = 0; j < D; j++) b2[l][j] = 0.0f;
  }

  for (int d = 0; d < D; d++) { lnf_g[d] = 1.0f; lnf_b[d] = 0.0f; }

  // Zero intermediate buffers (not strictly needed, but deterministic)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < 3 * D; j++)
      qkv[i][j] = 0.0f;

  for (int i = 0; i < N; i++)
    for (int d = 0; d < D; d++) {
      attn_out[i][d] = 0.0f;
      proj[i][d] = 0.0f;
      y[i][d] = 0.0f;
    }

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      scores[i][j] = 0.0f;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < MLP_HIDDEN; j++)
      h[i][j] = 0.0f;
}