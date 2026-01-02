/**
 * Attention Head benchmark header file
 * Scaled dot-product attention head benchmark
 */

#ifndef _ATTENTION_HEAD_H
# define _ATTENTION_HEAD_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(SEQ_LEN) && !defined(HEAD_DIM)
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define SEQ_LEN 64
#   define HEAD_DIM 64
#  endif

#  ifdef SMALL_DATASET
#   define SEQ_LEN 128
#   define HEAD_DIM 64
#  endif

#  ifdef MEDIUM_DATASET
#   define SEQ_LEN 256
#   define HEAD_DIM 128
#  endif

#  ifdef LARGE_DATASET
#   define SEQ_LEN 512
#   define HEAD_DIM 128
#  endif

#  ifdef EXTRALARGE_DATASET
#   define SEQ_LEN 1024
#   define HEAD_DIM 128
#  endif

# endif /* !(SEQ_LEN HEAD_DIM) */

# define _PB_SEQ_LEN (SEQ_LEN)
# define _PB_HEAD_DIM (HEAD_DIM)

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

#endif /* !_ATTENTION_HEAD_H */

