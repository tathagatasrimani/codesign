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

#define PRECISION 16

typedef ac_int<PRECISION,true> DTYPE;
