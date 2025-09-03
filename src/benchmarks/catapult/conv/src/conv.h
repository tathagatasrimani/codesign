#ifndef _GLOBAL_SIMPLE_H
#define _GLOBAL_SIMPLE_H

#include <ac_int.h>
#include <ac_channel.h>
#include <sstream>

template <size_t width, size_t EXTENT_0>
struct PackedInt {
  ac_int<width> value[EXTENT_0];

  std::string to_string() {
    std::stringstream ss;
    for (int i = 0; i < EXTENT_0; i++) {
      ss << value[i] << " ";
    }
    return ss.str();
  }  
};

template <size_t width, size_t EXTENT_0, size_t EXTENT_1>
struct PackedInt2D {
  PackedInt<width, EXTENT_0> value[EXTENT_1];
};

template<typename T, int N>
struct chanStruct{
  T data[N];
};

typedef ac_int<16, false> uint_16;
typedef ac_int<32, false> uint_32;
typedef ac_int<64, false> uint_64;
struct Params {
   uint_16 OY1;
   uint_16 OX1;
   uint_16 OY0;
   uint_16 OX0;

   uint_16 OC1;
   uint_16 IC1;
   uint_16 FX;
   uint_16 FY;
   uint_16 STRIDE;
};

#define ARRAY_DIMENSION 16
#define REPEAT(x) BOOST_PP_REPEAT(ARRAY_DIMENSION, x, 0)

#define INPUT_PRECISION 8
#define WEIGHT_PRECISION 8
#define OUTPUT_PRECISION (4*INPUT_PRECISION)

#define INPUT_BUFFER_SIZE  4096 // Input buffer size per IC0 per bank
#define WEIGHT_BUFFER_SIZE 8192 // Weight buffer size per OC0 per bank
#define ACCUMULATION_BUFFER_SIZE 256

typedef ac_int<INPUT_PRECISION,true> IDTYPE; 
typedef ac_int<WEIGHT_PRECISION,true> WDTYPE; 
typedef ac_int<OUTPUT_PRECISION,true> ODTYPE; 


// Max values for resnet-18
#define OY1_MAX 8
#define OY0_MAX 14
#define OX1_MAX 8
#define OX0_MAX 14
#define OC1_MAX 32
#define OC0_MAX 16
#define IC1_MAX 32
#define IC0_MAX 16
#define FX_MAX 7
#define FY_MAX 7
#define STRIDE_MAX 2

#define IX0_MAX ((OX0_MAX-1)*STRIDE_MAX+FX_MAX)
#define IY0_MAX ((OY0_MAX-1)*STRIDE_MAX+FY_MAX)

#endif

