/**
 * BitNet benchmark header file
 * C implementation of BitNet transformer operations
 */

#ifndef _BITNET_H
# define _BITNET_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(BATCH_SIZE) && !defined(SEQ_LEN) && !defined(EMBED_DIM) && !defined(HEADS) && !defined(KV_HEADS) && !defined(FF_DIM) && !defined(VOCAB_SIZE) && !defined(DEPTH)
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define BATCH_SIZE 1
#   define SEQ_LEN 64
#   define EMBED_DIM 128
#   define HEADS 4
#   define KV_HEADS 2
#   define FF_DIM 512
#   define VOCAB_SIZE 1000
#   define DEPTH 6
#  endif

#  ifdef SMALL_DATASET
#   define BATCH_SIZE 1
#   define SEQ_LEN 128
#   define EMBED_DIM 256
#   define HEADS 8
#   define KV_HEADS 4
#   define FF_DIM 1024
#   define VOCAB_SIZE 2000
#   define DEPTH 6
#  endif

#  ifdef MEDIUM_DATASET
#   define BATCH_SIZE 1
#   define SEQ_LEN 256
#   define EMBED_DIM 512
#   define HEADS 8
#   define KV_HEADS 4
#   define FF_DIM 2048
#   define VOCAB_SIZE 5000
#   define DEPTH 6
#  endif

#  ifdef LARGE_DATASET
#   define BATCH_SIZE 1
#   define SEQ_LEN 512
#   define EMBED_DIM 1024
#   define HEADS 8
#   define KV_HEADS 4
#   define FF_DIM 4096
#   define VOCAB_SIZE 10000
#   define DEPTH 6
#  endif

#  ifdef EXTRALARGE_DATASET
#   define BATCH_SIZE 1
#   define SEQ_LEN 1024
#   define EMBED_DIM 1024
#   define HEADS 8
#   define KV_HEADS 4
#   define FF_DIM 4096
#   define VOCAB_SIZE 20000
#   define DEPTH 6
#  endif

# endif /* !(BATCH_SIZE SEQ_LEN EMBED_DIM HEADS KV_HEADS FF_DIM VOCAB_SIZE DEPTH) */

# define _PB_BATCH_SIZE (BATCH_SIZE)
# define _PB_SEQ_LEN (SEQ_LEN)
# define _PB_EMBED_DIM (EMBED_DIM)
# define _PB_HEADS (HEADS)
# define _PB_KV_HEADS (KV_HEADS)
# define _PB_FF_DIM (FF_DIM)
# define _PB_VOCAB_SIZE (VOCAB_SIZE)
# define _PB_DEPTH (DEPTH)
# define _PB_HEAD_DIM (_PB_EMBED_DIM / _PB_HEADS)
/* Maximum head dimension for tensor declarations (use max across all dataset sizes) */
# ifndef HEAD_DIM
#  define HEAD_DIM 128  /* Max: 1024/8 = 128 for LARGE/EXTRALARGE */
# endif

/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_FLOAT
# endif

# ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
# endif

# ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
#  define FABS_FUN(x) fabsf(x)
#  define FMAX_FUN(x,y) ((x) > (y) ? (x) : (y))
#  define FMIN_FUN(x,y) ((x) < (y) ? (x) : (y))
#  define ROUND_FUN(x) ((x) >= SCALAR_VAL(0.0) ? (float)((int)((x) + SCALAR_VAL(0.5))) : (float)((int)((x) - SCALAR_VAL(0.5))))
# endif

# ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
#  define FABS_FUN(x) fabs(x)
#  define FMAX_FUN(x,y) ((x) > (y) ? (x) : (y))
#  define FMIN_FUN(x,y) ((x) < (y) ? (x) : (y))
#  define ROUND_FUN(x) ((x) >= SCALAR_VAL(0.0) ? (double)((long)((x) + SCALAR_VAL(0.5))) : (double)((long)((x) - SCALAR_VAL(0.5))))
# endif

#endif /* !_BITNET_H */

