#ifndef CUSTOM_CORES_H
#define CUSTOM_CORES_H

#pragma hls_design ccore
class BitAnd {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a & b;
    }
};

#pragma hls_design ccore
class BitOr {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a | b;
    }
};

#pragma hls_design ccore
class BitXOR {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a ^ b;
    }
};

#pragma hls_design ccore
class Add {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a + b;
    }
};

#pragma hls_design ccore
class Sub {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a - b;
    }
};

#pragma hls_design ccore
class Eq {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a == b;
    }
};

#pragma hls_design ccore
class NotEq {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a != b;
    }
};

#pragma hls_design ccore
class Lt {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a < b;
    }
};

#pragma hls_design ccore
class LtE {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a <= b;
    }
};

#pragma hls_design ccore
class Gt {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a > b;
    }
};

#pragma hls_design ccore
class Gte {
  public:
    #pragma hls_design interface
    bool CCS_BLOCK(run)(int a, int b) {
        return a >= b;
    }
};

// Usub and Uadd are same as Sub and Add
typedef Sub Usub;
typedef Add Uadd;

#pragma hls_design ccore
class Not {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a) {
        return ~a;
    }
};

#pragma hls_design ccore
class FloorDiv {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a / b;
    }
};

#pragma hls_design ccore
class LeftShift {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a << b;
    }
};

#pragma hls_design ccore
class RightShift {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a >> b;
    }
};

#pragma hls_design ccore
class Modulus {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a % b;
    }
};

#pragma hls_design ccore
class Multiplier {
  public:
    #pragma hls_design interface
    int CCS_BLOCK(run)(int a, int b) {
        return a * b;
    }
};

// Custom Register C-Core
#pragma hls_design ccore
class Register {
  private:
    int stored_value;

  public:
    Register() : stored_value(0) {}

    #pragma hls_design interface
    void CCS_BLOCK(write)(int input) {
        stored_value = input;
    }

    #pragma hls_design interface
    int CCS_BLOCK(read)() {
        return stored_value;
    }
};

#endif // CUSTOM_CORES_H
