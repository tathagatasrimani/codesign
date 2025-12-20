/**
 * BitNet benchmark implementation
 * C implementation of BitNet transformer operations in PolyBench style
 */

#include "arith_ops.c"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bitnet.h"

/* Constants */
#ifndef EPSILON
# ifdef DATA_TYPE_IS_FLOAT
#  define EPSILON 1e-6f
#  define QUANT_SCALE 127.0f
#  define QUANT_MIN -128.0f
#  define QUANT_MAX 127.0f
# else
#  define EPSILON 1e-6
#  define QUANT_SCALE 127.0
#  define QUANT_MIN -128.0
#  define QUANT_MAX 127.0
# endif
#endif

__attribute__((used))
__attribute__((visibility("default")))

/* Simple RMS Normalization: x / sqrt(mean(x^2) + eps) * scale */
void simple_rms_norm(
    DATA_TYPE POLYBENCH_2D(x_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    int seq_len,
    int embed_dim)
{
  int i, j;
  DATA_TYPE scale = SCALAR_VAL(1.0) / SQRT_FUN((DATA_TYPE)embed_dim);

#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE rms = SCALAR_VAL(0.0);
    for (j = 0; j < EMBED_DIM; j++) {
      rms += x_in[i][j] * x_in[i][j];
    }
    rms = SQRT_FUN(rms / (DATA_TYPE)embed_dim + EPSILON);
    for (j = 0; j < EMBED_DIM; j++) {
      x_out[i][j] = x_in[i][j] / rms * scale;
    }
  }
#pragma endscop
}

/* Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta */
/* Simplified version: assumes gamma=1, beta=0, computes mean and var per token */
void layer_norm(
    DATA_TYPE POLYBENCH_2D(x_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    int seq_len,
    int embed_dim,
    DATA_TYPE eps)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE mean = SCALAR_VAL(0.0);
    DATA_TYPE var = SCALAR_VAL(0.0);
    
    /* Compute mean */
    for (j = 0; j < EMBED_DIM; j++) {
      mean += x_in[i][j];
    }
    mean = mean / (DATA_TYPE)embed_dim;
    
    /* Compute variance */
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE diff = x_in[i][j] - mean;
      var += diff * diff;
    }
    var = var / (DATA_TYPE)embed_dim;
    DATA_TYPE std = SQRT_FUN(var + eps);
    
    /* Normalize */
    for (j = 0; j < EMBED_DIM; j++) {
      x_out[i][j] = (x_in[i][j] - mean) / std;
    }
  }
#pragma endscop
}

/* Activation quantization: per-token 8-bit quantization */
void activation_quant(
    DATA_TYPE POLYBENCH_2D(x_quant, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    int seq_len,
    int embed_dim)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE max_val = SCALAR_VAL(0.0);
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE abs_val = FABS_FUN(x_in[i][j]);
      if (abs_val > max_val) {
        max_val = abs_val;
      }
    }
    max_val = FMAX_FUN(max_val, SCALAR_VAL(1e-5));
    DATA_TYPE scale = QUANT_SCALE / max_val;
    
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE quantized = ROUND_FUN(x_in[i][j] * scale);
      quantized = FMAX_FUN(FMIN_FUN(quantized, QUANT_MAX), QUANT_MIN);
      x_quant[i][j] = quantized / scale;
    }
  }
#pragma endscop
}

/* Weight quantization: sign(w) * mean(|w|) */
/* Note: mean is computed over entire weight matrix, not per row */
void weight_quant(
    DATA_TYPE POLYBENCH_2D(w_quant, EMBED_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(w_in, EMBED_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    int embed_dim_out,
    int embed_dim_in)
{
  int i, j;
  DATA_TYPE mean_abs = SCALAR_VAL(0.0);
  
#pragma scop
  /* Compute mean absolute value over entire weight matrix */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      mean_abs += FABS_FUN(w_in[i][j]);
    }
  }
  mean_abs = mean_abs / ((DATA_TYPE)embed_dim_out * (DATA_TYPE)embed_dim_in);
  
  /* Apply quantization: sign(w) * mean(|w|) */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE sign = (w_in[i][j] >= SCALAR_VAL(0.0)) ? SCALAR_VAL(1.0) : SCALAR_VAL(-1.0);
      w_quant[i][j] = sign * mean_abs;
    }
  }
#pragma endscop
}

/* BitLinear: Linear layer with RMSNorm, activation quant, and weight quant */
void bit_linear(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, EMBED_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, EMBED_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  DATA_TYPE POLYBENCH_2D(x_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(x_quant, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(w_quant, EMBED_DIM, EMBED_DIM, embed_dim_out, embed_dim_in);
  
  /* RMS Normalization */
  simple_rms_norm(x_norm, x_in, seq_len, embed_dim_in);
  
  /* Activation quantization */
  activation_quant(x_quant, x_norm, seq_len, embed_dim_in);
  
  /* Weight quantization */
  weight_quant(w_quant, weight, embed_dim_out, embed_dim_in);
  
  /* Linear transformation */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += x_quant[i][k] * w_quant[j][k];
      }
    }
  }
#pragma endscop
}

/* Reshape 2D tensor (seq_len, embed_dim) to 3D tensor (heads, seq_len, head_dim) */
void reshape_to_heads(
    DATA_TYPE POLYBENCH_3D(qkv_3d, HEADS, SEQ_LEN, HEAD_DIM, heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(qkv_2d, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    int seq_len,
    int embed_dim,
    int heads,
    int head_dim)
{
  int h, i, d;
  
#pragma scop
  for (h = 0; h < HEADS; h++) {
    for (i = 0; i < SEQ_LEN; i++) {
      for (d = 0; d < HEAD_DIM; d++) {
        /* Reshape: qkv_3d[h][i][d] = qkv_2d[i][h * head_dim + d] */
        qkv_3d[h][i][d] = qkv_2d[i][h * head_dim + d];
      }
    }
  }
#pragma endscop
}

/* Reshape 3D tensor (heads, seq_len, head_dim) back to 2D tensor (seq_len, embed_dim) */
void reshape_from_heads(
    DATA_TYPE POLYBENCH_2D(out_2d, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_3D(out_3d, HEADS, SEQ_LEN, HEAD_DIM, heads, seq_len, head_dim),
    int seq_len,
    int embed_dim,
    int heads,
    int head_dim)
{
  int h, i, d;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (h = 0; h < HEADS; h++) {
      for (d = 0; d < HEAD_DIM; d++) {
        /* Reshape: out_2d[i][h * head_dim + d] = out_3d[h][i][d] */
        out_2d[i][h * head_dim + d] = out_3d[h][i][d];
      }
    }
  }
#pragma endscop
}

/* Expand K/V from kv_heads to query_heads by repeating */
void expand_kv_heads(
    DATA_TYPE POLYBENCH_3D(kv_expanded, HEADS, SEQ_LEN, HEAD_DIM, query_heads, seq_len, head_dim),
    DATA_TYPE POLYBENCH_3D(kv_original, HEADS, SEQ_LEN, HEAD_DIM, kv_heads, seq_len, head_dim),
    int seq_len,
    int head_dim,
    int query_heads,
    int kv_heads)
{
  int kv_idx, h, i, d, g;
  int group_size = query_heads / kv_heads;
  
  /* Restructure loop to avoid division in index: iterate over kv_heads, repeat each group_size times */
#pragma scop
  for (kv_idx = 0; kv_idx < KV_HEADS; kv_idx++) {
    for (g = 0; g < (HEADS / KV_HEADS); g++) {
      h = kv_idx * group_size + g;  /* Compute query head index */
      for (i = 0; i < SEQ_LEN; i++) {
        for (d = 0; d < HEAD_DIM; d++) {
          kv_expanded[h][i][d] = kv_original[kv_idx][i][d];
        }
      }
    }
  }
#pragma endscop
}

/* Scaled dot-product attention for a single head: Q @ K^T / sqrt(d) @ V */
void scaled_dot_product_attention_head(
    DATA_TYPE POLYBENCH_2D(attn_out, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(query, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(key, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(value, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int head_dim)
{
  int i, j, k;
  DATA_TYPE scale = SCALAR_VAL(1.0) / SQRT_FUN((DATA_TYPE)head_dim);
  
#pragma scop
  /* Compute similarity: Q @ K^T */
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < SEQ_LEN; j++) {
      similarity[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < HEAD_DIM; k++) {
        similarity[i][j] += query[i][k] * key[j][k] * scale;
      }
    }
  }
  
  /* Softmax over sequence dimension */
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE max_val = similarity[i][0];
    for (j = 1; j < SEQ_LEN; j++) {
      if (similarity[i][j] > max_val) {
        max_val = similarity[i][j];
      }
    }
    
    DATA_TYPE sum_exp = SCALAR_VAL(0.0);
    for (j = 0; j < SEQ_LEN; j++) {
      attention[i][j] = EXP_FUN(similarity[i][j] - max_val);
      sum_exp += attention[i][j];
    }
    
    for (j = 0; j < SEQ_LEN; j++) {
      attention[i][j] = attention[i][j] / sum_exp;
    }
  }
  
  /* Apply attention to values: attention @ V */
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < HEAD_DIM; j++) {
      attn_out[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < SEQ_LEN; k++) {
        attn_out[i][j] += attention[i][k] * value[k][j];
      }
    }
  }
#pragma endscop
}

/* Grouped Query Attention (GQA) with proper multi-head reshaping */
/* query: (seq_len, embed_dim) - full embed_dim
 * key, value: (seq_len, kv_embed_dim) - smaller dimension for GQA
 */
void gqa_attention(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(query, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(key, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim),
    DATA_TYPE POLYBENCH_2D(value, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim),
    DATA_TYPE POLYBENCH_2D(similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads)
{
  int h, i, j;
  int head_dim = embed_dim / query_heads;
  
  /* Reshape Q, K, V from 2D to 3D (heads, seq_len, head_dim) */
  DATA_TYPE POLYBENCH_3D(q_3d, HEADS, SEQ_LEN, HEAD_DIM, query_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_3D(k_3d, HEADS, SEQ_LEN, HEAD_DIM, kv_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_3D(v_3d, HEADS, SEQ_LEN, HEAD_DIM, kv_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_3D(k_expanded, HEADS, SEQ_LEN, HEAD_DIM, query_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_3D(v_expanded, HEADS, SEQ_LEN, HEAD_DIM, query_heads, seq_len, head_dim);
  DATA_TYPE POLYBENCH_3D(attn_out_3d, HEADS, SEQ_LEN, HEAD_DIM, query_heads, seq_len, head_dim);
  
  /* Reshape query to 3D: (seq_len, embed_dim) -> (query_heads, seq_len, head_dim) */
  reshape_to_heads(q_3d, query, seq_len, embed_dim, query_heads, head_dim);
  
  /* Reshape key and value to 3D: (seq_len, kv_embed_dim) -> (kv_heads, seq_len, head_dim) */
  reshape_to_heads(k_3d, key, seq_len, kv_embed_dim, kv_heads, head_dim);
  reshape_to_heads(v_3d, value, seq_len, kv_embed_dim, kv_heads, head_dim);
  
  /* Expand K/V if query_heads > kv_heads (GQA) */
  if (query_heads > kv_heads) {
    expand_kv_heads(k_expanded, k_3d, seq_len, head_dim, query_heads, kv_heads);
    expand_kv_heads(v_expanded, v_3d, seq_len, head_dim, query_heads, kv_heads);
  } else {
    /* If equal, just copy */
    for (h = 0; h < HEADS; h++) {
      for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < HEAD_DIM; j++) {
          k_expanded[h][i][j] = k_3d[h][i][j];
          v_expanded[h][i][j] = v_3d[h][i][j];
        }
      }
    }
  }
  
  /* Compute attention for each head */
  /* Use temporary 2D arrays to avoid polygeist.subindex operations */
  DATA_TYPE POLYBENCH_2D(q_head, SEQ_LEN, HEAD_DIM, seq_len, head_dim);
  DATA_TYPE POLYBENCH_2D(k_head, SEQ_LEN, HEAD_DIM, seq_len, head_dim);
  DATA_TYPE POLYBENCH_2D(v_head, SEQ_LEN, HEAD_DIM, seq_len, head_dim);
  DATA_TYPE POLYBENCH_2D(attn_head, SEQ_LEN, HEAD_DIM, seq_len, head_dim);
  
  for (h = 0; h < HEADS; h++) {
    /* Extract head h data into temporary 2D arrays */
#pragma scop
    for (i = 0; i < SEQ_LEN; i++) {
      for (j = 0; j < HEAD_DIM; j++) {
        q_head[i][j] = q_3d[h][i][j];
        k_head[i][j] = k_expanded[h][i][j];
        v_head[i][j] = v_expanded[h][i][j];
      }
    }
#pragma endscop
    
    scaled_dot_product_attention_head(
      attn_head,  /* Output for this head */
      q_head,     /* Query for this head */
      k_head,     /* Key for this head (expanded if needed) */
      v_head,     /* Value for this head (expanded if needed) */
      similarity,
      attention,
      seq_len,
      head_dim);
    
    /* Copy result back to 3D array */
#pragma scop
    for (i = 0; i < SEQ_LEN; i++) {
      for (j = 0; j < HEAD_DIM; j++) {
        attn_out_3d[h][i][j] = attn_head[i][j];
      }
    }
#pragma endscop
  }
  
  /* Reshape output from 3D back to 2D: (query_heads, seq_len, head_dim) -> (seq_len, embed_dim) */
  reshape_from_heads(out, attn_out_3d, seq_len, embed_dim, query_heads, head_dim);
}

/* BitMGQA: Multi-head Grouped Query Attention with BitLinear projections */
void bit_mgqa(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(q_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(k_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(v_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(o_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_1D(q_bias, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_1D(k_bias, EMBED_DIM, kv_embed_dim),
    DATA_TYPE POLYBENCH_1D(v_bias, EMBED_DIM, kv_embed_dim),
    DATA_TYPE POLYBENCH_1D(o_bias, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_2D(similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int embed_dim,
    int query_heads,
    int kv_heads)
{
  int kv_embed_dim = embed_dim / query_heads * kv_heads;
  DATA_TYPE POLYBENCH_2D(q, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(k, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim);
  DATA_TYPE POLYBENCH_2D(v, SEQ_LEN, EMBED_DIM, seq_len, kv_embed_dim);
  DATA_TYPE POLYBENCH_2D(attn_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  
  /* Project Q, K, V using BitLinear */
  /* Q projects to full embed_dim, K/V project to smaller kv_embed_dim for GQA */
  bit_linear(q, x_in, q_weight, q_bias, seq_len, embed_dim, embed_dim);
  bit_linear(k, x_in, k_weight, k_bias, seq_len, embed_dim, kv_embed_dim);
  bit_linear(v, x_in, v_weight, v_bias, seq_len, embed_dim, kv_embed_dim);
  
  /* Compute grouped query attention with proper multi-head reshaping */
  gqa_attention(attn_out, q, k, v, similarity, attention, 
                seq_len, embed_dim, kv_embed_dim, query_heads, kv_heads);
  
  /* Apply layer norm before output projection (as in Python BitMGQA) */
  DATA_TYPE POLYBENCH_2D(attn_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  layer_norm(attn_norm, attn_out, seq_len, embed_dim, SCALAR_VAL(1e-5));
  
  /* Output projection */
  bit_linear(out, attn_norm, o_weight, o_bias, seq_len, embed_dim, embed_dim);
}

/* Helper: Layer normalization for FF_DIM dimensions */
static void layer_norm_ff(
    DATA_TYPE POLYBENCH_2D(x_out, SEQ_LEN, FF_DIM, seq_len, dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, FF_DIM, seq_len, dim),
    int seq_len,
    int dim,
    DATA_TYPE eps)
{
  int i, j;
  
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE mean = SCALAR_VAL(0.0);
    DATA_TYPE var = SCALAR_VAL(0.0);
    
    /* Compute mean */
    for (j = 0; j < FF_DIM; j++) {
      mean += x_in[i][j];
    }
    mean = mean / (DATA_TYPE)dim;
    
    /* Compute variance */
    for (j = 0; j < FF_DIM; j++) {
      DATA_TYPE diff = x_in[i][j] - mean;
      var += diff * diff;
    }
    var = var / (DATA_TYPE)dim;
    DATA_TYPE std = SQRT_FUN(var + eps);
    
    /* Normalize */
    for (j = 0; j < FF_DIM; j++) {
      x_out[i][j] = (x_in[i][j] - mean) / std;
    }
  }
#pragma endscop
}

/* Helper: Weight quantization for FF_DIM dimensions */
static void weight_quant_ff(
    DATA_TYPE POLYBENCH_2D(w_quant, FF_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(w_in, FF_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    int embed_dim_out,
    int embed_dim_in)
{
  int i, j;
  DATA_TYPE mean_abs = SCALAR_VAL(0.0);
  
#pragma scop
  /* Compute mean absolute value over entire weight matrix */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      mean_abs += FABS_FUN(w_in[i][j]);
    }
  }
  mean_abs = mean_abs / ((DATA_TYPE)embed_dim_out * (DATA_TYPE)embed_dim_in);
  
  /* Apply quantization: sign(w) * mean(|w|) */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE sign = (w_in[i][j] >= SCALAR_VAL(0.0)) ? SCALAR_VAL(1.0) : SCALAR_VAL(-1.0);
      w_quant[i][j] = sign * mean_abs;
    }
  }
#pragma endscop
}

/* Helper: Weight quantization for FF_DIM input dimensions */
static void weight_quant_ff_in(
    DATA_TYPE POLYBENCH_2D(w_quant, EMBED_DIM, FF_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(w_in, EMBED_DIM, FF_DIM, embed_dim_out, embed_dim_in),
    int embed_dim_out,
    int embed_dim_in)
{
  int i, j;
  DATA_TYPE mean_abs = SCALAR_VAL(0.0);
  
#pragma scop
  /* Compute mean absolute value over entire weight matrix */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      mean_abs += FABS_FUN(w_in[i][j]);
    }
  }
  mean_abs = mean_abs / ((DATA_TYPE)embed_dim_out * (DATA_TYPE)embed_dim_in);
  
  /* Apply quantization: sign(w) * mean(|w|) */
  for (i = 0; i < EMBED_DIM; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE sign = (w_in[i][j] >= SCALAR_VAL(0.0)) ? SCALAR_VAL(1.0) : SCALAR_VAL(-1.0);
      w_quant[i][j] = sign * mean_abs;
    }
  }
#pragma endscop
}

/* Helper: BitLinear for FF_DIM output dimensions */
static void bit_linear_ff_out(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, FF_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, FF_DIM, EMBED_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, FF_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  DATA_TYPE POLYBENCH_2D(x_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(x_quant, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(w_quant, FF_DIM, EMBED_DIM, embed_dim_out, embed_dim_in);
  
  /* RMS Normalization */
  simple_rms_norm(x_norm, x_in, seq_len, embed_dim_in);
  
  /* Activation quantization */
  activation_quant(x_quant, x_norm, seq_len, embed_dim_in);
  
  /* Weight quantization */
  weight_quant_ff(w_quant, weight, embed_dim_out, embed_dim_in);
  
  /* Linear transformation */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += x_quant[i][k] * w_quant[j][k];
      }
    }
  }
#pragma endscop
}

/* Helper: BitLinear for FF_DIM input dimensions */
static void bit_linear_ff_in(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim_out),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, FF_DIM, seq_len, embed_dim_in),
    DATA_TYPE POLYBENCH_2D(weight, EMBED_DIM, FF_DIM, embed_dim_out, embed_dim_in),
    DATA_TYPE POLYBENCH_1D(bias, EMBED_DIM, embed_dim_out),
    int seq_len,
    int embed_dim_in,
    int embed_dim_out)
{
  int i, j, k;
  DATA_TYPE POLYBENCH_2D(x_norm, SEQ_LEN, FF_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(x_quant, SEQ_LEN, FF_DIM, seq_len, embed_dim_in);
  DATA_TYPE POLYBENCH_2D(w_quant, EMBED_DIM, FF_DIM, embed_dim_out, embed_dim_in);
  
  /* RMS Normalization - need a version that works with FF_DIM */
  /* For now, inline the RMS norm computation */
  DATA_TYPE scale = SCALAR_VAL(1.0) / SQRT_FUN((DATA_TYPE)embed_dim_in);
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE rms = SCALAR_VAL(0.0);
    for (j = 0; j < EMBED_DIM; j++) {
      rms += x_in[i][j] * x_in[i][j];
    }
    rms = SQRT_FUN(rms / (DATA_TYPE)embed_dim_in + EPSILON);
    for (j = 0; j < EMBED_DIM; j++) {
      x_norm[i][j] = x_in[i][j] / rms * scale;
    }
  }
#pragma endscop
  
  /* Activation quantization - need a version that works with FF_DIM */
  /* Inline activation quant for FF_DIM */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    DATA_TYPE max_val = SCALAR_VAL(0.0);
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE abs_val = FABS_FUN(x_norm[i][j]);
      if (abs_val > max_val) {
        max_val = abs_val;
      }
    }
    max_val = FMAX_FUN(max_val, SCALAR_VAL(1e-5));
    DATA_TYPE quant_scale = QUANT_SCALE / max_val;
    
    for (j = 0; j < EMBED_DIM; j++) {
      DATA_TYPE quantized = ROUND_FUN(x_norm[i][j] * quant_scale);
      quantized = FMAX_FUN(FMIN_FUN(quantized, QUANT_MAX), QUANT_MIN);
      x_quant[i][j] = quantized / quant_scale;
    }
  }
#pragma endscop
  
  /* Weight quantization */
  weight_quant_ff_in(w_quant, weight, embed_dim_out, embed_dim_in);
  
  /* Linear transformation */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += x_quant[i][k] * w_quant[j][k];
      }
    }
  }
#pragma endscop
}

/* BitFeedForward: Feed-forward network with BitLinear */
void bit_feedforward(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(ff1_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(ff2_weight, EMBED_DIM, FF_DIM, embed_dim, ff_dim),
    DATA_TYPE POLYBENCH_1D(ff1_bias, FF_DIM, ff_dim),
    DATA_TYPE POLYBENCH_1D(ff2_bias, EMBED_DIM, embed_dim),
    int seq_len,
    int embed_dim,
    int ff_dim)
{
  int i, j;
  DATA_TYPE POLYBENCH_2D(ff1_out, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  DATA_TYPE POLYBENCH_2D(activated, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  
  /* First linear layer: embed_dim -> ff_dim */
  bit_linear_ff_out(ff1_out, x_in, ff1_weight, ff1_bias, seq_len, embed_dim, ff_dim);
  
  /* SiLU (Swish) activation: x * sigmoid(x) */
  /* sigmoid(x) = 1 / (1 + exp(-x)) */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < FF_DIM; j++) {
      DATA_TYPE x = ff1_out[i][j];
      /* SiLU: x * sigmoid(x) = x / (1 + exp(-x)) */
      DATA_TYPE silu = x / (SCALAR_VAL(1.0) + EXP_FUN(-x));
      activated[i][j] = silu;
    }
  }
#pragma endscop
  
  /* Post-activation layer norm (as in Python with post_act_ln=True) */
  DATA_TYPE POLYBENCH_2D(activated_norm, SEQ_LEN, FF_DIM, seq_len, ff_dim);
  layer_norm_ff(activated_norm, activated, seq_len, ff_dim, SCALAR_VAL(1e-5));
  
  /* Second linear layer: ff_dim -> embed_dim */
  bit_linear_ff_in(out, activated_norm, ff2_weight, ff2_bias, seq_len, ff_dim, embed_dim);
}

/* Transformer block: Attention + Feed-forward with residual connections */
void transformer_block(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    DATA_TYPE POLYBENCH_2D(q_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(k_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(v_weight, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(o_weight, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(ff1_weight, FF_DIM, EMBED_DIM, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_2D(ff2_weight, EMBED_DIM, FF_DIM, embed_dim, ff_dim),
    DATA_TYPE POLYBENCH_1D(q_bias, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_1D(k_bias, EMBED_DIM, kv_embed_dim),
    DATA_TYPE POLYBENCH_1D(v_bias, EMBED_DIM, kv_embed_dim),
    DATA_TYPE POLYBENCH_1D(o_bias, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_1D(ff1_bias, FF_DIM, ff_dim),
    DATA_TYPE POLYBENCH_1D(ff2_bias, EMBED_DIM, embed_dim),
    DATA_TYPE POLYBENCH_2D(similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads,
    int ff_dim)
{
  int i, j;
  DATA_TYPE POLYBENCH_2D(attn_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(ff_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(norm_out, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  
  /* Attention with residual */
  /* Note: kv_embed_dim is computed inside bit_mgqa, but we need it for weight dimensions */
  bit_mgqa(attn_out, x_in, q_weight, k_weight, v_weight, o_weight,
           q_bias, k_bias, v_bias, o_bias,
           similarity, attention,
           seq_len, embed_dim, query_heads, kv_heads);
  
  /* Residual connection */
  DATA_TYPE POLYBENCH_2D(attn_residual, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      attn_residual[i][j] = x_in[i][j] + attn_out[i][j];
    }
  }
#pragma endscop
  
  /* Layer norm after attention residual (as in Python Transformer) */
  layer_norm(norm_out, attn_residual, seq_len, embed_dim, SCALAR_VAL(1e-5));
  
  /* Feed-forward with residual */
  bit_feedforward(ff_out, norm_out, ff1_weight, ff2_weight,
                  ff1_bias, ff2_bias, seq_len, embed_dim, ff_dim);
  
  /* Residual connection for feed-forward */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      out[i][j] = norm_out[i][j] + ff_out[i][j];
    }
  }
#pragma endscop
}

/* Top-level BitNet function: matches Python BitNetTransformer.forward()
 * Structure: post_emb_norm -> transformer blocks (depth times) -> output_head
 * Note: For benchmark purposes, we assume embeddings are already provided as input
 */
void bitnet(
    DATA_TYPE POLYBENCH_2D(out, SEQ_LEN, VOCAB_SIZE, seq_len, vocab_size),
    DATA_TYPE POLYBENCH_2D(x_in, SEQ_LEN, EMBED_DIM, seq_len, embed_dim),
    /* Transformer block weights - for depth layers */
    DATA_TYPE POLYBENCH_3D(q_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(k_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(v_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, kv_embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(o_weights, DEPTH, EMBED_DIM, EMBED_DIM, depth, embed_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(ff1_weights, DEPTH, FF_DIM, EMBED_DIM, depth, ff_dim, embed_dim),
    DATA_TYPE POLYBENCH_3D(ff2_weights, DEPTH, EMBED_DIM, FF_DIM, depth, embed_dim, ff_dim),
    DATA_TYPE POLYBENCH_2D(q_biases, DEPTH, EMBED_DIM, depth, embed_dim),
    DATA_TYPE POLYBENCH_2D(k_biases, DEPTH, EMBED_DIM, depth, kv_embed_dim),
    DATA_TYPE POLYBENCH_2D(v_biases, DEPTH, EMBED_DIM, depth, kv_embed_dim),
    DATA_TYPE POLYBENCH_2D(o_biases, DEPTH, EMBED_DIM, depth, embed_dim),
    DATA_TYPE POLYBENCH_2D(ff1_biases, DEPTH, FF_DIM, depth, ff_dim),
    DATA_TYPE POLYBENCH_2D(ff2_biases, DEPTH, EMBED_DIM, depth, embed_dim),
    /* Output head weights */
    DATA_TYPE POLYBENCH_2D(output_weight, VOCAB_SIZE, EMBED_DIM, vocab_size, embed_dim),
    DATA_TYPE POLYBENCH_1D(output_bias, VOCAB_SIZE, vocab_size),
    /* Temporary buffers for attention computation */
    DATA_TYPE POLYBENCH_2D(similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int embed_dim,
    int kv_embed_dim,
    int query_heads,
    int kv_heads,
    int ff_dim,
    int vocab_size,
    int depth)
{
  int l, i, j, k;
  DATA_TYPE POLYBENCH_2D(x_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(x_transformed, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(x_temp, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(output_norm, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  DATA_TYPE POLYBENCH_2D(swap, SEQ_LEN, EMBED_DIM, seq_len, embed_dim);
  
  /* Post-embedding layer norm (as in Python BitNetTransformer.forward) */
  layer_norm(x_norm, x_in, seq_len, embed_dim, SCALAR_VAL(1e-5));
  
  /* Copy to working buffer */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < EMBED_DIM; j++) {
      x_transformed[i][j] = x_norm[i][j];
    }
  }
#pragma endscop
  
  /* Apply transformer blocks depth times (as in Python Transformer.forward) */
  for (l = 0; l < DEPTH; l++) {
    /* Extract weights for this layer into temporary 2D arrays to avoid polygeist.subindex */
    DATA_TYPE POLYBENCH_2D(q_w, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(k_w, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(v_w, EMBED_DIM, EMBED_DIM, kv_embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(o_w, EMBED_DIM, EMBED_DIM, embed_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(ff1_w, FF_DIM, EMBED_DIM, ff_dim, embed_dim);
    DATA_TYPE POLYBENCH_2D(ff2_w, EMBED_DIM, FF_DIM, embed_dim, ff_dim);
    DATA_TYPE POLYBENCH_1D(q_b, EMBED_DIM, embed_dim);
    DATA_TYPE POLYBENCH_1D(k_b, EMBED_DIM, kv_embed_dim);
    DATA_TYPE POLYBENCH_1D(v_b, EMBED_DIM, kv_embed_dim);
    DATA_TYPE POLYBENCH_1D(o_b, EMBED_DIM, embed_dim);
    DATA_TYPE POLYBENCH_1D(ff1_b, FF_DIM, ff_dim);
    DATA_TYPE POLYBENCH_1D(ff2_b, EMBED_DIM, embed_dim);
    
    /* Copy weights and biases for layer l */
#pragma scop
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        q_w[i][j] = q_weights[l][i][j];
      }
    }
    for (i = 0; i < (EMBED_DIM / HEADS * KV_HEADS); i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        k_w[i][j] = k_weights[l][i][j];
        v_w[i][j] = v_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        o_w[i][j] = o_weights[l][i][j];
      }
    }
    for (i = 0; i < FF_DIM; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        ff1_w[i][j] = ff1_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      for (j = 0; j < FF_DIM; j++) {
        ff2_w[i][j] = ff2_weights[l][i][j];
      }
    }
    for (i = 0; i < EMBED_DIM; i++) {
      q_b[i] = q_biases[l][i];
      o_b[i] = o_biases[l][i];
    }
    for (i = 0; i < (EMBED_DIM / HEADS * KV_HEADS); i++) {
      k_b[i] = k_biases[l][i];
      v_b[i] = v_biases[l][i];
    }
    for (i = 0; i < FF_DIM; i++) {
      ff1_b[i] = ff1_biases[l][i];
    }
    for (i = 0; i < EMBED_DIM; i++) {
      ff2_b[i] = ff2_biases[l][i];
    }
#pragma endscop
    
    /* Swap buffers: x_temp becomes input, x_transformed becomes output */
    transformer_block(
      x_temp,  /* output */
      x_transformed,  /* input */
      q_w, k_w, v_w, o_w,
      ff1_w, ff2_w,
      q_b, k_b, v_b, o_b,
      ff1_b, ff2_b,
      similarity, attention,
      seq_len, embed_dim, kv_embed_dim, query_heads, kv_heads, ff_dim);
    
    /* Swap buffers for next iteration */
    for (i = 0; i < SEQ_LEN; i++) {
      for (j = 0; j < EMBED_DIM; j++) {
        swap[i][j] = x_transformed[i][j];
        x_transformed[i][j] = x_temp[i][j];
        x_temp[i][j] = swap[i][j];
      }
    }
  }
  
  /* Output head: layer norm + linear projection to vocab_size (as in Python OutputHead) */
  layer_norm(output_norm, x_transformed, seq_len, embed_dim, SCALAR_VAL(1e-5));
  
  /* Linear projection to vocab_size */
#pragma scop
  for (i = 0; i < SEQ_LEN; i++) {
    for (j = 0; j < VOCAB_SIZE; j++) {
      out[i][j] = output_bias[j];
      for (k = 0; k < EMBED_DIM; k++) {
        out[i][j] += output_norm[i][k] * output_weight[j][k];
      }
    }
  }
#pragma endscop
}

