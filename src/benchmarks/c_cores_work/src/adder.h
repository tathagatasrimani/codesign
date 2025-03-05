#include <ac_int.h>
#include <ac_blackbox.h>

class adder {
public:
    adder() { }

    #pragma design interface ccore blackbox
    void run(ac_int<16> a, ac_int<16> b, ac_int<16> &z) {
        ac_blackbox()
            .entity("adder")
            .verilog_files("adder.v")
            .outputs("z")
            .area(102.5)
            .delay(0.50)
            .end();
        z = a + b;
    }
};
