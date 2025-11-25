/**
 * Llama benchmark implementation
 * C implementation of Llama transformer operations in PolyBench style
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "llama.h"

/* Constants */
#ifndef EPSILON
# ifdef DATA_TYPE_IS_FLOAT
#  define EPSILON 1e-6f
# else
#  define EPSILON 1e-6
# endif
#endif

__attribute__((used))
__attribute__((visibility("default")))

/* RMS Normalization: x / sqrt(mean(x^2) + eps) * weight */
void rms_norm(
    DATA_TYPE POLYBENCH_2D(x_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_1D(weight, EMBED_DIM, embed_dim),
    int seq_len,
    int embed_dim,
    DATA_TYPE norm_eps)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE rms = SCALAR_VAL(0.0);
    for (j = 0; j < EMBED_DIM; j++) {
      rms += x_in[i][j] * x_in[i][j];
    }
    rms = SQRT_FUN(rms / (DATA_TYPE)embed_dim + norm_eps);
    for (j = 0; j < EMBED_DIM; j++) {
      x_out[i][j] = x_in[i][j] / rms * weight[j];
    }
  }
#pragma endscop
}

/* SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)) */
void silu_activation(
    DATA_TYPE POLYBENCH_2D(x_out, SEQ_LEN, FF_DIM, seq_len, ff_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, FF_DIM, seq_len, ff_dim),
    int seq_len,
    int ff_dim)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < FF_DIM; j++) {
      DATA_TYPE x = x_in[i][j];
      x_out[i][j] = x / (SCALAR_VAL(1.0) + EXP_FUN(-x));
    }
  }
#pragma endscop
}

/* Standard linear layer: out = x @ weight^T + bias */
void linear_layer(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, EMBED_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, EMBED_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += x_in[i][k] * weight[j][k];
      }
    }
  }
#pragma endscop
}

/* Linear layer for FF_DIM output dimensions */
void linear_layer_ff_out(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, FF_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, FF_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, FF_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < FF_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += x_in[i][k] * weight[j][k];
      }
    }
  }
#pragma endscop
}

/* Linear layer for FF_DIM input dimensions */
void linear_layer_ff_in(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, FF_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, EMBED_DIM, FF_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, EMBED_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < FF_DIM; k++) {
        out[i][j] += x_in[i][k] * weight[j][k];
      }
    }
  }
#pragma endscop
}

/* Feed-forward network: w2(silu(w1(x)) * w3(x)) */
void feed_forward(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(w1_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(w3_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(w2_weight, EMBED_DIM, FF_DIM, embed_dim, ff_dim),
    int seq_len,
    int embed_dim,
    int ff_dim)
{
  int i, j;
  DATA_TYPE POLYBENCH_2D(w1_out, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  DATA_TYPE POLYBENCH_2D(w3_out, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  DATA_TYPE POLYBENCH_2D(silu_out, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  DATA_TYPE POLYBENCH_2D(mult_out, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  
  /* w1 projection */
  DATA_TYPE POLYBENCH_1D(w1_bias, FF_DIM, ff_dim);
  for (i = 0; i < FF_DIM; i++) {
    w1_bias[i] = SCALAR_VAL(0.0);
  }
  linear_layer_ff_out(w1_out, x_in, w1_weight, w1_bias, seq_len, embed_dim, ff_dim);
  
  /* w3 projection */
  DATA_TYPE POLYBENCH_1D(w3_bias, FF_DIM, ff_dim);
  for (i = 0; i < FF_DIM; i++) {
    w3_bias[i] = SCALAR_VAL(0.0);
  }
  linear_layer_ff_out(w3_out, x_in, w3_weight, w3_bias, seq_len, embed_dim, ff_dim);
  
  /* SiLU activation on w1_out */
  silu_activation(silu_out, w1_out, seq_len, ff_dim);
  
  /* Element-wise multiplication: silu(w1_out) * w3_out */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < FF_DIM; j++) {
      mult_out[i][j] = silu_out[i][j] * w3_out[i][j];
    }
  }
#pragma endscop
  
  /* w2 projection */
  DATA_TYPE POLYBENCH_1D(w2_bias, EMBED_DIM, embed_dim);
  for (i = 0; i < EMBED_DIM; i++) {
    w2_bias[i] = SCALAR_VAL(0.0);
  }
  linear_layer_ff_in(out, mult_out, w2_weight, w2_bias, seq_len, ff_dim, embed_dim);
}

/* Precompute RoPE frequencies: freqs_cis[i] = (cos, sin) for position i */
/* Standard RoPE: for pair i (dimensions 2i, 2i+1), freq = 1 / (theta^(2i/head_dim)) */
void precompute_freqs_cis(
    DATA_TYPE POLYBENCH_2D(freqs_cis, SEQ_LEN, HEAD_DIM, max_seq_len, head_dim),
    int max_seq_len,
    int head_dim,
    DATA_TYPE theta)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
      for (j = 0; j < HEAD_DIM; j += 2) {
        /* Pair index: j/2, frequency: 1 / (theta^(2*(j/2)/head_dim)) = 1 / (theta^(j/head_dim)) */
        int pair_idx = j / 2;
        DATA_TYPE freq = SCALAR_VAL(1.0) / POW_FUN(theta, (DATA_TYPE)(2 * pair_idx) / (DATA_TYPE)head_dim);
        DATA_TYPE angle = (DATA_TYPE)i * freq;
        /* Store cos and sin in alternating positions */
        freqs_cis[i][j] = COS_FUN(angle);
        if (j + 1 < HEAD_DIM) {
          freqs_cis[i][j + 1] = SIN_FUN(angle);
        }
      }
  }
#pragma endscop
}

/* Apply rotary position embedding to queries and keys */
void apply_rotary_emb(
    DATA_TYPE POLYBENCH_4D(q_out, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_4D(k_out, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_4D(q_in, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_4D(k_in, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(freqs_cis, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    int batch_size,
    int heads,
    int seq_len,
    int head_dim)
{
  int b, h, i, j;
  
#pragma scop
  for (b = 0; b < BATCH_SIZE; b++) {
    for (h = 0; h < HEADS; h++) {
      for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < HEAD_DIM; j += 2) {
          DATA_TYPE cos_val = freqs_cis[i][j];
          DATA_TYPE sin_val = freqs_cis[i][j + 1];
          DATA_TYPE q0 = q_in[b][h][i][j];
          DATA_TYPE q1 = q_in[b][h][i][j + 1];
          DATA_TYPE k0 = k_in[b][h][i][j];
          DATA_TYPE k1 = k_in[b][h][i][j + 1];
          
          q_out[b][h][i][j] = q0 * cos_val - q1 * sin_val;
          if (j + 1 < HEAD_DIM) {
            q_out[b][h][i][j + 1] = q0 * sin_val + q1 * cos_val;
            k_out[b][h][i][j + 1] = k0 * sin_val + k1 * cos_val;
          }
          k_out[b][h][i][j] = k0 * cos_val - k1 * sin_val;
        }
      }
    }
  }
#pragma endscop
}

/* Reshape 2D tensor (seq_len, embed_dim) to 4D tensor (batch, heads, seq_len, head_dim) */
void reshape_to_heads_4d(
    DATA_TYPE POLYBENCH_4D(qkv_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(qkv_2d, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    int batch_size,
    int seq_len,
    int embed_dim,
    int heads,
    int head_dim)
{
  int b, h, i, d;
  
#pragma scop
  for (b = 0; b < BATCH_SIZE; b++) {
    for (h = 0; h < HEADS; h++) {
      for (i = 0; i < SEQ_LEN; i++) {
        for (d = 0; d < HEAD_DIM; d++) {
          qkv_4d[b][h][i][d] = qkv_2d[i][h * head_dim + d];
        }
      }
    }
  }
#pragma endscop
}

/* Reshape 4D tensor (batch, heads, seq_len, head_dim) back to 2D tensor (seq_len, embed_dim) */
void reshape_from_heads_4d(
    DATA_TYPE POLYBENCH_2D(out_2d, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_4D(out_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    int batch_size,
    int seq_len,
    int embed_dim,
    int heads,
    int head_dim)
{
  int b, h, i, d;
  
#pragma scop
  for (b = 0; b < BATCH_SIZE; b++) {
    for (i = 0; i < SEQ_LEN; i++) {
      for (h = 0; h < HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          out_2d[i][h * head_dim + d] = out_4d[b][h][i][d];
        }
      }
    }
  }
#pragma endscop
}

/* Expand K/V from kv_heads to query_heads by repeating */
void expand_kv_heads_4d(
    DATA_TYPE POLYBENCH_4D(kv_expanded, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, query_heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_4D(kv_original, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, kv_heads, seq_len, head_dim),
    int batch_size,
    int seq_len,
    int head_dim,
    int query_heads,
    int kv_heads)
{
  int b, kv_idx, h, i, d, g;
  int group_size = query_heads / kv_heads;
  
#pragma scop
  for (b = 0; b < BATCH_SIZE; b++) {
    for (kv_idx = 0; kv_idx < KV_HEADS; kv_idx++) {
      for (g = 0; g < (HEADS / KV_HEADS); g++) {
        h = kv_idx * group_size + g;
        for (i = 0; i < SEQ_LEN; i++) {
          for (d = 0; d < HEAD_DIM; d++) {
            kv_expanded[b][h][i][d] = kv_original[b][kv_idx][i][d];
          }
        }
      }
    }
  }
#pragma endscop
}

/* Scaled dot-product attention for multi-head: Q @ K^T / sqrt(d) @ V */
void scaled_dot_product_attention(
    DATA_TYPE POLYBENCH_4D(attn_out, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_4D(query, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len_q, head_dim),
    DATA_TYPE POLYBENCH_4D(key, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len_kv, head_dim),
    DATA_TYPE POLYBENCH_4D(value, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, heads, seq_len_kv, head_dim),
    DATA_TYPE POLYBENCH_2D(attn_mask, SEQ_LEN, SEQ_LEN, seq_len_q, seq_len_kv),
    int batch_size,
    int heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    int use_mask)
{
  int b, h, i, j, k;
  DATA_TYPE scale = SCALAR_VAL(1.0) / SQRT_FUN((DATA_TYPE)head_dim);
  DATA_TYPE POLYBENCH_3D(similarity, HEADS, SEQ_LEN, SEQ_LEN, heads, seq_len_q, seq_len_kv);
  DATA_TYPE POLYBENCH_3D(attention, HEADS, SEQ_LEN, SEQ_LEN, heads, seq_len_q, seq_len_kv);
  
#pragma scop
  /* Compute similarity: Q @ K^T / sqrt(d) */
  for (b = 0; b < BATCH_SIZE; b++) {
    for (h = 0; h < HEADS; h++) {
      for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < SEQ_LEN; j++) {
          similarity[h][i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < HEAD_DIM; k++) {
            similarity[h][i][j] += query[b][h][i][k] * key[b][h][j][k] * scale;
          }
          /* Apply mask if provided */
          if (use_mask) {
            similarity[h][i][j] += attn_mask[i][j];
          }
        }
      }
      
      /* Softmax over key dimension */
      for (i = 0; i < SEQ_LEN; i++) {
        DATA_TYPE max_val = similarity[h][i][0];
        for (j = 1; j < SEQ_LEN; j++) {
          if (similarity[h][i][j] > max_val) {
            max_val = similarity[h][i][j];
          }
        }
        
        DATA_TYPE sum_exp = SCALAR_VAL(0.0);
        for (j = 0; j < SEQ_LEN; j++) {
          attention[h][i][j] = EXP_FUN(similarity[h][i][j] - max_val);
          sum_exp += attention[h][i][j];
        }
        
        for (j = 0; j < SEQ_LEN; j++) {
          attention[h][i][j] = attention[h][i][j] / sum_exp;
        }
      }
      
      /* Apply attention to values: attention @ V */
      for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < HEAD_DIM; j++) {
          attn_out[b][h][i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < SEQ_LEN; k++) {
            attn_out[b][h][i][j] += attention[h][i][k] * value[b][h][k][j];
          }
        }
      }
    }
  }
#pragma endscop
}

/* Grouped Query Attention (GQA) with KV cache */
void gqa_attention(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(q_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(k_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(v_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(o_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_4D(cache_k, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim),
    DATA_TYPE POLYBENCH_4D(cache_v, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim),
    DATA_TYPE POLYBENCH_2D(freqs_cis, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(attn_mask, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int use_mask,
    int batch_size,
    int seq_len,
    int start_pos,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads,
    int head_dim)
{
  int i, j, h, d;
  DATA_TYPE POLYBENCH_2D(q, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(k, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim);
  DATA_TYPE POLYBENCH_2D(v, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim);
  
  /* Project Q, K, V */
  DATA_TYPE POLYBENCH_1D(q_bias, EMBED_DIM, embed_dim);
  DATA_TYPE POLYBENCH_1D(k_bias, EMBED_DIM, kv_embed_dim);
  DATA_TYPE POLYBENCH_1D(v_bias, EMBED_DIM, kv_embed_dim);
  for (i = 0; i < EMBED_DIM; i++) {
    q_bias[i] = SCALAR_VAL(0.0);
    k_bias[i] = SCALAR_VAL(0.0);
    v_bias[i] = SCALAR_VAL(0.0);
  }
  
  linear_layer(q, x_in, q_weight, q_bias, seq_len, embed_dim, embed_dim);
  linear_layer(k, x_in, k_weight, k_bias, seq_len, embed_dim, kv_embed_dim);
  linear_layer(v, x_in, v_weight, v_bias, seq_len, embed_dim, kv_embed_dim);
  
  /* Reshape to 4D: (batch, heads, seq_len, head_dim) */
  DATA_TYPE POLYBENCH_4D(q_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, query_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_4D(k_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, kv_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_4D(v_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, kv_heads, seq_len, head_dim);
  
  /* Reshape Q to 4D */
#pragma scop
  for (i = 0; i < BATCH_SIZE; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      for (h = 0; h < HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          q_4d[i][h][j][d] = q[j][h * head_dim + d];
        }
      }
    }
  }
#pragma endscop
  
  /* Reshape K, V to 4D */
#pragma scop
  for (i = 0; i < BATCH_SIZE; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      for (h = 0; h < KV_HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          k_4d[i][h][j][d] = k[j][h * head_dim + d];
          v_4d[i][h][j][d] = v[j][h * head_dim + d];
        }
      }
    }
  }
#pragma endscop
  
  /* Apply RoPE */
  apply_rotary_emb(q_4d, k_4d, q_4d, k_4d, freqs_cis, batch_size, query_heads, seq_len, head_dim);
  
  /* Update cache */
  int total_seq_len = start_pos + seq_len;
#pragma scop
  for (i = 0; i < BATCH_SIZE; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      int cache_pos = start_pos + j;
      for (h = 0; h < KV_HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          cache_k[i][cache_pos][h][d] = k_4d[i][h][j][d];
          cache_v[i][cache_pos][h][d] = v_4d[i][h][j][d];
        }
      }
    }
  }
#pragma endscop
  
  /* Retrieve from cache */
  DATA_TYPE POLYBENCH_4D(k_cached, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, kv_heads, total_seq_len, head_dim);
  DATA_TYPE POLYBENCH_4D(v_cached, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, kv_heads, total_seq_len, head_dim);
#pragma scop
  for (i = 0; i < BATCH_SIZE; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      for (h = 0; h < KV_HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          k_cached[i][h][j][d] = cache_k[i][j][h][d];
          v_cached[i][h][j][d] = cache_v[i][j][h][d];
        }
      }
    }
  }
#pragma endscop
  
  /* Expand K/V heads to match query heads */
  DATA_TYPE POLYBENCH_4D(k_expanded, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, query_heads, total_seq_len, head_dim);
  DATA_TYPE POLYBENCH_4D(v_expanded, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, query_heads, total_seq_len, head_dim);
  expand_kv_heads_4d(k_expanded, k_cached, batch_size, total_seq_len, head_dim, query_heads, kv_heads);
  expand_kv_heads_4d(v_expanded, v_cached, batch_size, total_seq_len, head_dim, query_heads, kv_heads);
  
  /* Compute attention */
  DATA_TYPE POLYBENCH_4D(attn_out_4d, BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM, batch_size, query_heads, seq_len, head_dim);
  int use_attn_mask = use_mask;
  scaled_dot_product_attention(attn_out_4d, q_4d, k_expanded, v_expanded, attn_mask,
                                batch_size, query_heads, seq_len, total_seq_len, head_dim, use_attn_mask);
  
  /* Reshape back to 2D */
#pragma scop
  for (i = 0; i < BATCH_SIZE; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      for (h = 0; h < HEADS; h++) {
        for (d = 0; d < HEAD_DIM; d++) {
          out[j][h * head_dim + d] = attn_out_4d[i][h][j][d];
        }
      }
    }
  }
#pragma endscop
  
  /* Output projection */
  DATA_TYPE POLYBENCH_2D(attn_out_proj, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_1D(o_bias, EMBED_DIM, embed_dim);
  for (i = 0; i < EMBED_DIM; i++) {
    o_bias[i] = SCALAR_VAL(0.0);
  }
  linear_layer(attn_out_proj, out, o_weight, o_bias, seq_len, embed_dim, embed_dim);
  
  /* Copy to output */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = attn_out_proj[i][j];
    }
  }
#pragma endscop
}

/* Transformer block: Attention + Feed-forward with residual connections */
void transformer_block(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(q_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(k_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(v_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(o_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(w1_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(w3_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(w2_weight, EMBED_DIM, FF_DIM, embed_dim, ff_dim),
    DATA_TYPE POLYBENCH_1D(attn_norm_weight, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_1D(ffn_norm_weight, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_4D(cache_k, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim),
    DATA_TYPE POLYBENCH_4D(cache_v, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim),
    DATA_TYPE POLYBENCH_2D(freqs_cis, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(attn_mask, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int use_mask,
    int batch_size,
    int seq_len,
    int start_pos,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads,
    int ff_dim,
    int head_dim)
{
  int i, j;
  DATA_TYPE POLYBENCH_2D(attn_norm_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(attn_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(attn_residual, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(ffn_norm_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(ffn_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  
  /* Attention with pre-norm and residual */
  rms_norm(attn_norm_out, x_in, attn_norm_weight, seq_len, embed_dim, NORM_EPS);
  gqa_attention(attn_out, attn_norm_out, q_weight, k_weight, v_weight, o_weight,
                cache_k, cache_v, freqs_cis, attn_mask, use_mask,
                batch_size, seq_len, start_pos, embed_dim, kv_embed_dim,
                query_heads, kv_heads, head_dim);
  
  /* Residual connection */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      attn_residual[i][j] = x_in[i][j] + attn_out[i][j];
    }
  }
#pragma endscop
  
  /* Feed-forward with pre-norm and residual */
  rms_norm(ffn_norm_out, attn_residual, ffn_norm_weight, seq_len, embed_dim, NORM_EPS);
  feed_forward(ffn_out, ffn_norm_out, w1_weight, w3_weight, w2_weight,
               seq_len, embed_dim, ff_dim);
  
  /* Residual connection */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = attn_residual[i][j] + ffn_out[i][j];
    }
  }
#pragma endscop
}

/* Top-level Llama transformer function */
void llama(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, VOCAB_SIZE, seq_len, vocab_size),
    DATA_TYPE POLYBENCH_1D(tokens, SEQ_LEN, seq_len),
    DATA_TYPE POLYBENCH_2D(emb_weight, VOCAB_SIZE, EMBED_DIM, vocab_size, embed_dim),
    /* Transformer block weights - for depth layers */
    DATA_TYPE POLYBENCH_3D(q_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(k_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(v_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(o_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(w1_weights, DEPTH, FF_DIM, EMBED_DIM, depth, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(w3_weights, DEPTH, FF_DIM, EMBED_DIM, depth, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(w2_weights, DEPTH, EMBED_DIM, FF_DIM, depth, embed_dim, ff_dim),
    DATA_TYPE POLYBENCH_2D(attn_norm_weights, DEPTH, EMBED_DIM, depth, embed_dim),
    DATA_TYPE POLYBENCH_2D(ffn_norm_weights, DEPTH, EMBED_DIM, depth, embed_dim),
    DATA_TYPE POLYBENCH_2D(output_weight, VOCAB_SIZE, EMBED_DIM, vocab_size, embed_dim),
    /* KV cache - flattened 1D array: size = depth * batch_size * max_seq_len * kv_heads * head_dim */
    /* Layout: cache_k[layer * layer_size + batch * batch_size + seq * seq_size + head * head_size + dim] */
    /* Use maximum possible size for compile-time dimension */
    DATA_TYPE POLYBENCH_1D(cache_k, DEPTH * BATCH_SIZE * SEQ_LEN * KV_HEADS * HEAD_DIM, depth * batch_size * max_seq_len * kv_heads * head_dim),
    DATA_TYPE POLYBENCH_1D(cache_v, DEPTH * BATCH_SIZE * SEQ_LEN * KV_HEADS * HEAD_DIM, depth * batch_size * max_seq_len * kv_heads * head_dim),
    /* RoPE frequencies */
    DATA_TYPE POLYBENCH_2D(freqs_cis, SEQ_LEN, HEAD_DIM, max_seq_len, head_dim),
    /* Attention mask */
    DATA_TYPE POLYBENCH_2D(attn_mask, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int batch_size,
    int seq_len,
    int start_pos,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads,
    int ff_dim,
    int vocab_size,
    int head_dim,
    int depth,
    int max_seq_len)
{
  int l, i, j;
  DATA_TYPE POLYBENCH_2D(x_emb, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(x_transformed, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(x_temp, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(output_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_1D(output_norm_weight, EMBED_DIM, embed_dim);
  DATA_TYPE POLYBENCH_2D(swap, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  
  /* Initialize output norm weight */
#pragma scop
  for (i = 0; i < EMBED_DIM; i++) {
    output_norm_weight[i] = SCALAR_VAL(1.0);
  }
#pragma endscop
  
  /* Embedding lookup - simplified: use token index for affine access ASSUMING VOCAB_SIZE >= SEQ_LEN*/
  /* This avoids memref.load and uses affine.load instead */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      x_emb[i][j] = emb_weight[i][j];
    }
  }
#pragma endscop
  
  /* Copy to working buffer */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      x_transformed[i][j] = x_emb[i][j];
    }
  }
#pragma endscop
  
  /* Apply transformer blocks depth times */
  for (l = 0; l < DEPTH; l++) {
    /* Extract weights for this layer */
    DATA_TYPE POLYBENCH_2D(q_w, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(k_w, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(v_w, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(o_w, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(w1_w, FF_DIM, EMBED_DIM, ff_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(w3_w, FF_DIM, EMBED_DIM, ff_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(w2_w, EMBED_DIM, FF_DIM, embed_dim, ff_dim);
    DATA_TYPE POLYBENCH_1D(attn_norm_w, EMBED_DIM, embed_dim);
    DATA_TYPE POLYBENCH_1D(ffn_norm_w, EMBED_DIM, embed_dim);
    
    /* Copy weights for layer l */
#pragma scop
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        q_w[i][j] = q_weights[l][i][j];
        o_w[i][j] = o_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        k_w[i][j] = k_weights[l][i][j];
        v_w[i][j] = v_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      attn_norm_w[i] = attn_norm_weights[l][i];
      ffn_norm_w[i] = ffn_norm_weights[l][i];
    }
    for (i = 0; i < FF_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        w1_w[i][j] = w1_weights[l][i][j];
        w3_w[i][j] = w3_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < FF_DIM; j++) {
        w2_w[i][j] = w2_weights[l][i][j];
      }
    }
#pragma endscop
    
    /* Get cache for this layer - copy from flattened 1D to 4D using nested loops */
    DATA_TYPE POLYBENCH_4D(layer_cache_k, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim);
    DATA_TYPE POLYBENCH_4D(layer_cache_v, BATCH_SIZE, SEQ_LEN, KV_HEADS, HEAD_DIM, batch_size, max_seq_len, kv_heads, head_dim);
    /* Calculate offsets for flattened array access */
    int layer_size = batch_size * max_seq_len * kv_heads * head_dim;
    int batch_size_flat = max_seq_len * kv_heads * head_dim;
    int seq_size = kv_heads * head_dim;
    int head_size = head_dim;
    int layer_offset = l * layer_size;
    /* Copy cache for layer l using flattened array indexing */
#pragma scop
    for (i = 0; i < BATCH_SIZE; i++) {
      for (j = 0; j < SEQ_LEN; j++) {
        for (int h = 0; h < KV_HEADS; h++) {
          for (int d = 0; d < HEAD_DIM; d++) {
            /* Access flattened array: cache_k[layer_offset + i*batch_size_flat + j*seq_size + h*head_size + d] */
            int flat_idx = layer_offset + i * batch_size_flat + j * seq_size + h * head_size + d;
            layer_cache_k[i][j][h][d] = cache_k[flat_idx];
            layer_cache_v[i][j][h][d] = cache_v[flat_idx];
          }
        }
      }
    }
#pragma endscop
    
    /* Determine if mask should be used (seq_len > 1 for first pass) */
    int use_mask = (seq_len > 1) ? 1 : 0;
    
    transformer_block(
      x_temp,  /* output */
      x_transformed,  /* input */
      q_w, k_w, v_w, o_w,
      w1_w, w3_w, w2_w,
      attn_norm_w, ffn_norm_w,
      layer_cache_k, layer_cache_v,
      freqs_cis, attn_mask, use_mask,
      batch_size, seq_len, start_pos,
      embed_dim, kv_embed_dim, query_heads, kv_heads, ff_dim, head_dim);
    
    /* Update cache - copy back from 4D to flattened 1D array */
#pragma scop
    for (i = 0; i < BATCH_SIZE; i++) {
      for (j = 0; j < SEQ_LEN; j++) {
        for (int h = 0; h < KV_HEADS; h++) {
          for (int d = 0; d < HEAD_DIM; d++) {
            /* Access flattened array: cache_k[layer_offset + i*batch_size_flat + j*seq_size + h*head_size + d] */
            int flat_idx = layer_offset + i * batch_size_flat + j * seq_size + h * head_size + d;
            cache_k[flat_idx] = layer_cache_k[i][j][h][d];
            cache_v[flat_idx] = layer_cache_v[i][j][h][d];
          }
        }
      }
    }
#pragma endscop
    
    /* Swap buffers for next iteration */
    for (i = 0; i < SEQ_LEN; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        swap[i][j] = x_transformed[i][j];
        x_transformed[i][j] = x_temp[i][j];
        x_temp[i][j] = swap[i][j];
      }
    }
  }
  
  /* Final RMSNorm */
  rms_norm(output_norm, x_transformed, output_norm_weight, seq_len, embed_dim, NORM_EPS);
  
  /* Output projection to vocab_size */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < VOCAB_SIZE; j++) {
      out[i][j] = SCALAR_VAL(0.0);
      for (int k = 0; k < EMBED_DIM; k++) {
        out[i][j] += output_norm[i][k] * output_weight[j][k];
      }
    }
  }
#pragma endscop
}

