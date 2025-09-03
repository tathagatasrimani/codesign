#include <ac_int.h>
#include <ac_blackbox.h>

class add {
public:
    add() { }

    #pragma design interface ccore blackbox
    void run(ac_int<16> a, ac_int<16> b, ac_int<16> &z) {
        ac_blackbox()
            .entity("add")
            .verilog_files("add.v")
            .outputs("z")
            .area(5.5258125)
            .delay(0.7336)
            .end();
        z = a + b;
    }
};
