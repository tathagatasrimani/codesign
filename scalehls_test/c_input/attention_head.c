/**
 * Attention Head benchmark implementation
 * Scaled dot-product attention head wrapped in a top function with dummy computation
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "attention_head.h"

__attribute__((used))
__attribute__((visibility("default")))

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

/* Top-level function that performs dummy computation before calling attention head */
void attention_head(
    DATA_TYPE POLYBENCH_2D(output, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(query, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(key, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(value, SEQ_LEN, HEAD_DIM, seq_len, head_dim),
    DATA_TYPE POLYBENCH_2D(temp_similarity, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    DATA_TYPE POLYBENCH_2D(temp_attention, SEQ_LEN, SEQ_LEN, seq_len, seq_len),
    int seq_len,
    int head_dim)
{
  /* Call the attention head function */
  scaled_dot_product_attention_head(
    output,
    query,
    key,
    value,
    temp_similarity,
    temp_attention,
    seq_len,
    head_dim
  );
}

