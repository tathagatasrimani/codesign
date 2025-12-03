/*
*   Byte-oriented AES-256 implementation.
*   All lookup tables replaced with 'on the fly' calculations.
*/
#include "support.h"

typedef struct {
  unsigned key[32];
  unsigned enckey[32];
  unsigned deckey[32];
} aes256_context;

void aes(unsigned key[32], unsigned enckey[32], unsigned deckey[32], unsigned k[32], unsigned buf[16]);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  aes256_context ctx;
  unsigned k[32];
  unsigned buf[16];
};

