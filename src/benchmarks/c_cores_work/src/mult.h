#include <ac_int.h>
#include <ac_blackbox.h>

class mult {
public:
    mult() { }

    #pragma design interface ccore blackbox
    void run(ac_int<16> a, ac_int<16> b, ac_int<16> &z) {
        ac_blackbox()
            .entity("mult")
            .verilog_files("mult.v")
            .outputs("z")
            .area(102.5)
            .delay(0.50)
            .end();
        z = a * b;
    }
};
