// bench_io_prng.h
// Header with no stdlib/stdio dependencies, suitable for HLS kernels.

#ifndef BENCH_IO_PRNG_H
#define BENCH_IO_PRNG_H

#include <stdint.h>

///// File and section functions
// Implementations can use stdlib/stdio in a separate .c file.
// From the HLS/kernel side, these are just external functions.
char *readfile(int fd);
char *find_section_start(char *s, int n);

///// Array read functions
#define SECTION_TERMINATED -1
int parse_string(char *s, char *arr, int n); // n==-1 : %%-terminated
int parse_uint8_t_array(char *s, uint8_t *arr, int n);
int parse_uint16_t_array(char *s, uint16_t *arr, int n);
int parse_uint32_t_array(char *s, uint32_t *arr, int n);
int parse_uint64_t_array(char *s, uint64_t *arr, int n);
int parse_int8_t_array(char *s, int8_t *arr, int n);
int parse_int16_t_array(char *s, int16_t *arr, int n);
int parse_int32_t_array(char *s, int32_t *arr, int n);
int parse_int64_t_array(char *s, int64_t *arr, int n);
int parse_float_array(char *s, float *arr, int n);
int parse_double_array(char *s, double *arr, int n);

///// Array write functions
int write_string(int fd, char *arr, int n);
int write_uint8_t_array(int fd, uint8_t *arr, int n);
int write_uint16_t_array(int fd, uint16_t *arr, int n);
int write_uint32_t_array(int fd, uint32_t *arr, int n);
int write_uint64_t_array(int fd, uint64_t *arr, int n);
int write_int8_t_array(int fd, int8_t *arr, int n);
int write_int16_t_array(int fd, int16_t *arr, int n);
int write_int32_t_array(int fd, int32_t *arr, int n);
int write_int64_t_array(int fd, int64_t *arr, int n);
int write_float_array(int fd, float *arr, int n);
int write_double_array(int fd, double *arr, int n);

int write_section_header(int fd);

///// Per-benchmark hooks
void run_benchmark(void *vargs);
void input_to_data(int fd, void *vdata);
void data_to_input(int fd, void *vdata);
void output_to_data(int fd, void *vdata);
void data_to_output(int fd, void *vdata);
int check_data(void *vdata, void *vref);

extern int INPUT_SIZE;

///// TYPE macros
#define __STAC_EXPANDED(f_pfx,t,f_sfx) f_pfx##t##f_sfx
#define STAC(f_pfx,t,f_sfx) __STAC_EXPANDED(f_pfx,t,f_sfx)

/**** PRNG library. Based on https://github.com/rdadolf/prng *****/
#ifndef __PRNG_H__
#define __PRNG_H__

// Only stdint.h is needed for these types.

#define LAG1 (UINT16_C(24))
#define LAG2 (UINT16_C(55))
#define RAND_SSIZE ((UINT16_C(1))<<6)
#define RAND_SMASK (RAND_SSIZE-1)
#define RAND_EXHAUST_LIMIT LAG2
#define RAND_REFILL_COUNT ((LAG2*10)-RAND_EXHAUST_LIMIT)

struct prng_rand_t {
  uint64_t    s[RAND_SSIZE]; // Lags
  uint_fast16_t i;           // Location of the current lag
  uint_fast16_t c;           // Exhaustion count
};

#define PRNG_RAND_MAX UINT64_MAX

static inline uint64_t prng_rand(struct prng_rand_t *state) {
  uint_fast16_t i;
  uint_fast16_t r, new_rands = 0;

  if (!state->c) { // Randomness exhausted, run forward to refill
    new_rands += RAND_REFILL_COUNT + 1;
    state->c = RAND_EXHAUST_LIMIT - 1;
  } else {
    new_rands = 1;
    state->c--;
  }

  for (r = 0; r < new_rands; r++) {
    i = state->i;
    state->s[i & RAND_SMASK] =
      state->s[(i + RAND_SSIZE - LAG1) & RAND_SMASK] +
      state->s[(i + RAND_SSIZE - LAG2) & RAND_SMASK];
    state->i++;
  }
  return state->s[i & RAND_SMASK];
}

static inline void prng_srand(uint64_t seed, struct prng_rand_t *state) {
  uint_fast16_t i;

  state->c = RAND_EXHAUST_LIMIT;
  state->i = 0;

  state->s[0] = seed;
  for (i = 1; i < RAND_SSIZE; i++) {
    // Simple mixing of seed and index
    state->s[i] = i * (UINT64_C(2147483647)) + seed;
  }

  // Run forward 10,000 numbers to decorrelate initial state
  for (i = 0; i < 10000; i++) {
    prng_rand(state);
  }
}

// Clean up internal macros
#undef LAG1
#undef LAG2
#undef RAND_SSIZE
#undef RAND_SMASK
#undef RAND_EXHAUST_LIMIT
#undef RAND_REFILL_COUNT

#endif // __PRNG_H__

#endif // BENCH_IO_PRNG_H
