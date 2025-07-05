#include <ac_int.h>
#include <ac_blackbox.h>

class bit_xor {
public:
    bit_xor() { }

    #pragma design interface ccore blackbox
    void run(ac_int<16> a, ac_int<16> b, ac_int<16> tag, ac_int<16> &z) {
        ac_blackbox()
            .entity("bit_xor")
            .verilog_files("bit_xor.v")
            .outputs("z")
            .area(40.1533125)
            .delay(0.6807)
            .end();
        z = a ^ b;
    }
};
