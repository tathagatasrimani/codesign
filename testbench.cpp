#include "Vconcat_sim_rtl.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdlib>
#include <ctime>

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);

    // Instantiate design
    Vconcat_sim_rtl *top = new Vconcat_sim_rtl;
    VerilatedVcdC *tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("wave.vcd");

    // Init
    top->arst_n = 0;
    top->clk = 0;
    top->a_chan_rsc_req_vz = 1;
    top->b_chan_rsc_req_vz = 1;
    top->c_chan_rsc_req_vz = 1;

    std::srand(std::time(nullptr));

    for (int t = 0; t < 100; t++) {
        // Rising edge
        top->clk = 1;

        if (t == 2) top->arst_n = 1; // release reset

        // Drive randomized inputs
        top->a_chan_rsc_q = rand() % 65536;
        top->b_chan_rsc_q = rand() % 65536;

        // Pulse handshake (valid always 1; release low)
        top->a_chan_rsc_req_vz = 1;
        top->a_chan_rsc_rls_lz = 0;
        top->b_chan_rsc_req_vz = 1;
        top->b_chan_rsc_rls_lz = 0;
        top->c_chan_rsc_req_vz = 1;
        top->c_chan_rsc_rls_lz = 0;

        top->eval();
        tfp->dump(t * 2);

        // Falling edge
        top->clk = 0;
        top->eval();
        tfp->dump(t * 2 + 1);
    }

    tfp->close();
    delete top;
    return 0;
}
