#include <string>
#include <fstream>
#include <iostream>
#include "mc_testbench.h"
#include <mc_reset.h>
#include <mc_transactors.h>
#include <mc_scverify.h>
#include <mc_stall_ctrl.h>
#include "ccs_ioport_trans_rsc_v1.h"
#include <mc_monitor.h>
#include <mc_simulator_extensions.h>
#include "mc_dut_wrapper.h"
#include "ccs_probes.cpp"
#include <mt19937ar.c>
#ifndef TO_QUOTED_STRING
#define TO_QUOTED_STRING(x) TO_QUOTED_STRING1(x)
#define TO_QUOTED_STRING1(x) #x
#endif
#ifndef TOP_HDL_ENTITY
#define TOP_HDL_ENTITY MatMult
#endif
// Hold time for the SCVerify testbench to account for the gate delay after downstream synthesis in pico second(s)
// Hold time value is obtained from 'top_gate_constraints.cpp', which is generated at the end of RTL synthesis
#ifdef CCS_DUT_GATE
extern double __scv_hold_time;
extern double __scv_hold_time_RSCID_1;
extern double __scv_hold_time_RSCID_2;
extern double __scv_hold_time_RSCID_3;
#else
double __scv_hold_time = 0.0; // default for non-gate simulation is zero
double __scv_hold_time_RSCID_1 = 0;
double __scv_hold_time_RSCID_2 = 0;
double __scv_hold_time_RSCID_3 = 0;
#endif

class scverify_top : public sc_module
{
public:
  sc_signal<sc_logic>                                                              rst;
  sc_signal<sc_logic>                                                              rst_n;
  sc_signal<sc_logic>                                                              SIG_SC_LOGIC_0;
  sc_signal<sc_logic>                                                              SIG_SC_LOGIC_1;
  sc_signal<sc_logic>                                                              TLS_design_is_idle;
  sc_signal<bool>                                                                  TLS_design_is_idle_reg;
  sc_clock                                                                         clk;
  mc_programmable_reset                                                            arst_n_driver;
  sc_signal<sc_logic>                                                              TLS_arst_n;
  sc_signal<sc_lv<400> >                                                           TLS_a_chan_rsc_dat;
  sc_signal<sc_logic>                                                              TLS_a_chan_rsc_vld;
  sc_signal<sc_logic>                                                              TLS_a_chan_rsc_rdy;
  sc_signal<sc_lv<400> >                                                           TLS_b_chan_rsc_dat;
  sc_signal<sc_logic>                                                              TLS_b_chan_rsc_vld;
  sc_signal<sc_logic>                                                              TLS_b_chan_rsc_rdy;
  sc_signal<sc_lv<400> >                                                           TLS_c_chan_rsc_dat;
  sc_signal<sc_logic>                                                              TLS_c_chan_rsc_vld;
  sc_signal<sc_logic>                                                              TLS_c_chan_rsc_rdy;
  ccs_DUT_wrapper                                                                  MatMult_INST;
  ccs_in_wait_trans_rsc_v1<1,400 >                                                 a_chan_rsc_INST;
  ccs_in_wait_trans_rsc_v1<1,400 >                                                 b_chan_rsc_INST;
  ccs_out_wait_trans_rsc_v1<1,400 >                                                c_chan_rsc_INST;
  tlm::tlm_fifo<mgc_sysc_ver_array1D<ac_int<16, true >,25> >                       TLS_in_fifo_a_chan_value_value;
  tlm::tlm_fifo<mc_wait_ctrl>                                                      TLS_in_wait_ctrl_fifo_a_chan_value_value;
  tlm::tlm_fifo<int>                                                               TLS_in_fifo_a_chan_value_value_sizecount;
  sc_signal<sc_logic>                                                              TLS_a_chan_rsc_trdone;
  mc_channel_input_transactor<mgc_sysc_ver_array1D<ac_int<16, true >,25>,16,true>  transactor_a_chan_value_value;
  tlm::tlm_fifo<mgc_sysc_ver_array1D<ac_int<16, true >,25> >                       TLS_in_fifo_b_chan_value_value;
  tlm::tlm_fifo<mc_wait_ctrl>                                                      TLS_in_wait_ctrl_fifo_b_chan_value_value;
  tlm::tlm_fifo<int>                                                               TLS_in_fifo_b_chan_value_value_sizecount;
  sc_signal<sc_logic>                                                              TLS_b_chan_rsc_trdone;
  mc_channel_input_transactor<mgc_sysc_ver_array1D<ac_int<16, true >,25>,16,true>  transactor_b_chan_value_value;
  tlm::tlm_fifo<mgc_sysc_ver_array1D<ac_int<16, true >,25> >                       TLS_out_fifo_c_chan_value_value;
  tlm::tlm_fifo<mc_wait_ctrl>                                                      TLS_out_wait_ctrl_fifo_c_chan_value_value;
  sc_signal<sc_logic>                                                              TLS_c_chan_rsc_trdone;
  mc_output_transactor<mgc_sysc_ver_array1D<ac_int<16, true >,25>,16,true>         transactor_c_chan_value_value;
  mc_testbench                                                                     testbench_INST;
  sc_signal<sc_logic>                                                              catapult_start;
  sc_signal<sc_logic>                                                              catapult_done;
  sc_signal<sc_logic>                                                              catapult_ready;
  sc_signal<sc_logic>                                                              in_sync;
  sc_signal<sc_logic>                                                              out_sync;
  sc_signal<sc_logic>                                                              inout_sync;
  sc_signal<unsigned>                                                              wait_for_init;
  sync_generator                                                                   sync_generator_INST;
  catapult_monitor                                                                 catapult_monitor_INST;
  ccs_probe_monitor                                                               *ccs_probe_monitor_INST;
  sc_event                                                                         generate_reset_event;
  sc_event                                                                         deadlock_event;
  sc_signal<sc_logic>                                                              deadlocked;
  sc_signal<sc_logic>                                                              maxsimtime;
  sc_event                                                                         max_sim_time_event;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_staller_inst_run_wen;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_a_chan_rsci_inst_MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst_a_chan_rsci_irdy_run_sct;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_a_chan_rsci_inst_MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst_a_chan_rsci_ivld;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_b_chan_rsci_inst_MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst_b_chan_rsci_irdy_run_sct;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_b_chan_rsci_inst_MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst_b_chan_rsci_ivld;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_c_chan_rsci_inst_MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst_c_chan_rsci_irdy;
  sc_signal<sc_logic>                                                              OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_c_chan_rsci_inst_MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst_c_chan_rsci_ivld_run_sct;
  sc_signal<sc_logic>                                                              OFS_a_chan_rsc_vld;
  sc_signal<sc_logic>                                                              OFS_b_chan_rsc_vld;
  sc_signal<sc_logic>                                                              OFS_c_chan_rsc_rdy;
  sc_signal<sc_logic>                                                              TLS_enable_stalls;
  sc_signal<unsigned short>                                                        TLS_stall_coverage;

  void TLS_arst_n_method();
  void drive_TLS_a_chan_rsc_trdone();
  void drive_TLS_b_chan_rsc_trdone();
  void drive_TLS_c_chan_rsc_trdone();
  void max_sim_time_notify();
  void start_of_simulation();
  void setup_debug();
  void debug(const char* varname, int flags, int count);
  void generate_reset();
  void install_observe_foreign_signals();
  void deadlock_watch();
  void deadlock_notify();

  // Constructor
  SC_HAS_PROCESS(scverify_top);
  scverify_top(const sc_module_name& name)
    : rst("rst")
    , rst_n("rst_n")
    , SIG_SC_LOGIC_0("SIG_SC_LOGIC_0")
    , SIG_SC_LOGIC_1("SIG_SC_LOGIC_1")
    , TLS_design_is_idle("TLS_design_is_idle")
    , TLS_design_is_idle_reg("TLS_design_is_idle_reg")
    , CCS_CLK_CTOR(clk, "clk", 5, SC_NS, 0.5, 0, SC_NS, false)
    , arst_n_driver("arst_n_driver", 10.000000, true)
    , TLS_arst_n("TLS_arst_n")
    , TLS_a_chan_rsc_dat("TLS_a_chan_rsc_dat")
    , TLS_a_chan_rsc_vld("TLS_a_chan_rsc_vld")
    , TLS_a_chan_rsc_rdy("TLS_a_chan_rsc_rdy")
    , TLS_b_chan_rsc_dat("TLS_b_chan_rsc_dat")
    , TLS_b_chan_rsc_vld("TLS_b_chan_rsc_vld")
    , TLS_b_chan_rsc_rdy("TLS_b_chan_rsc_rdy")
    , TLS_c_chan_rsc_dat("TLS_c_chan_rsc_dat")
    , TLS_c_chan_rsc_vld("TLS_c_chan_rsc_vld")
    , TLS_c_chan_rsc_rdy("TLS_c_chan_rsc_rdy")
    , MatMult_INST("rtl", TO_QUOTED_STRING(TOP_HDL_ENTITY))
    , a_chan_rsc_INST("a_chan_rsc", true)
    , b_chan_rsc_INST("b_chan_rsc", true)
    , c_chan_rsc_INST("c_chan_rsc", true)
    , TLS_in_fifo_a_chan_value_value("TLS_in_fifo_a_chan_value_value", -1)
    , TLS_in_wait_ctrl_fifo_a_chan_value_value("TLS_in_wait_ctrl_fifo_a_chan_value_value", -1)
    , TLS_in_fifo_a_chan_value_value_sizecount("TLS_in_fifo_a_chan_value_value_sizecount", 1)
    , TLS_a_chan_rsc_trdone("TLS_a_chan_rsc_trdone")
    , transactor_a_chan_value_value("transactor_a_chan_value_value", 0, 400, 0)
    , TLS_in_fifo_b_chan_value_value("TLS_in_fifo_b_chan_value_value", -1)
    , TLS_in_wait_ctrl_fifo_b_chan_value_value("TLS_in_wait_ctrl_fifo_b_chan_value_value", -1)
    , TLS_in_fifo_b_chan_value_value_sizecount("TLS_in_fifo_b_chan_value_value_sizecount", 1)
    , TLS_b_chan_rsc_trdone("TLS_b_chan_rsc_trdone")
    , transactor_b_chan_value_value("transactor_b_chan_value_value", 0, 400, 0)
    , TLS_out_fifo_c_chan_value_value("TLS_out_fifo_c_chan_value_value", -1)
    , TLS_out_wait_ctrl_fifo_c_chan_value_value("TLS_out_wait_ctrl_fifo_c_chan_value_value", -1)
    , TLS_c_chan_rsc_trdone("TLS_c_chan_rsc_trdone")
    , transactor_c_chan_value_value("transactor_c_chan_value_value", 0, 400, 0)
    , testbench_INST("user_tb")
    , catapult_start("catapult_start")
    , catapult_done("catapult_done")
    , catapult_ready("catapult_ready")
    , in_sync("in_sync")
    , out_sync("out_sync")
    , inout_sync("inout_sync")
    , wait_for_init("wait_for_init")
    , sync_generator_INST("sync_generator", true, false, false, false, 8, 8, 0)
    , catapult_monitor_INST("Monitor", clk, true, 8LL, 7LL)
    , ccs_probe_monitor_INST(NULL)
    , deadlocked("deadlocked")
    , maxsimtime("maxsimtime")
  {
    arst_n_driver.reset_out(TLS_arst_n);

    MatMult_INST.clk(clk);
    MatMult_INST.arst_n(TLS_arst_n);
    MatMult_INST.a_chan_rsc_dat(TLS_a_chan_rsc_dat);
    MatMult_INST.a_chan_rsc_vld(TLS_a_chan_rsc_vld);
    MatMult_INST.a_chan_rsc_rdy(TLS_a_chan_rsc_rdy);
    MatMult_INST.b_chan_rsc_dat(TLS_b_chan_rsc_dat);
    MatMult_INST.b_chan_rsc_vld(TLS_b_chan_rsc_vld);
    MatMult_INST.b_chan_rsc_rdy(TLS_b_chan_rsc_rdy);
    MatMult_INST.c_chan_rsc_dat(TLS_c_chan_rsc_dat);
    MatMult_INST.c_chan_rsc_vld(TLS_c_chan_rsc_vld);
    MatMult_INST.c_chan_rsc_rdy(TLS_c_chan_rsc_rdy);

    a_chan_rsc_INST.rdy(TLS_a_chan_rsc_rdy);
    a_chan_rsc_INST.vld(TLS_a_chan_rsc_vld);
    a_chan_rsc_INST.dat(TLS_a_chan_rsc_dat);
    a_chan_rsc_INST.clk(clk);
    a_chan_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_1)));

    b_chan_rsc_INST.rdy(TLS_b_chan_rsc_rdy);
    b_chan_rsc_INST.vld(TLS_b_chan_rsc_vld);
    b_chan_rsc_INST.dat(TLS_b_chan_rsc_dat);
    b_chan_rsc_INST.clk(clk);
    b_chan_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_2)));

    c_chan_rsc_INST.rdy(TLS_c_chan_rsc_rdy);
    c_chan_rsc_INST.vld(TLS_c_chan_rsc_vld);
    c_chan_rsc_INST.dat(TLS_c_chan_rsc_dat);
    c_chan_rsc_INST.clk(clk);
    c_chan_rsc_INST.add_attribute(*(new sc_attribute<double>("CLK_SKEW_DELAY", __scv_hold_time_RSCID_3)));

    transactor_a_chan_value_value.in_fifo(TLS_in_fifo_a_chan_value_value);
    transactor_a_chan_value_value.in_wait_ctrl_fifo(TLS_in_wait_ctrl_fifo_a_chan_value_value);
    transactor_a_chan_value_value.sizecount_fifo(TLS_in_fifo_a_chan_value_value_sizecount);
    transactor_a_chan_value_value.bind_clk(clk, true, rst);
    transactor_a_chan_value_value.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_a_chan_value_value.register_block(&a_chan_rsc_INST, a_chan_rsc_INST.basename(), TLS_a_chan_rsc_trdone,
        0, 0, 1);

    transactor_b_chan_value_value.in_fifo(TLS_in_fifo_b_chan_value_value);
    transactor_b_chan_value_value.in_wait_ctrl_fifo(TLS_in_wait_ctrl_fifo_b_chan_value_value);
    transactor_b_chan_value_value.sizecount_fifo(TLS_in_fifo_b_chan_value_value_sizecount);
    transactor_b_chan_value_value.bind_clk(clk, true, rst);
    transactor_b_chan_value_value.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_b_chan_value_value.register_block(&b_chan_rsc_INST, b_chan_rsc_INST.basename(), TLS_b_chan_rsc_trdone,
        0, 0, 1);

    transactor_c_chan_value_value.out_fifo(TLS_out_fifo_c_chan_value_value);
    transactor_c_chan_value_value.out_wait_ctrl_fifo(TLS_out_wait_ctrl_fifo_c_chan_value_value);
    transactor_c_chan_value_value.bind_clk(clk, true, rst);
    transactor_c_chan_value_value.add_attribute(*(new sc_attribute<int>("MC_TRANSACTOR_EVENT", 0 )));
    transactor_c_chan_value_value.register_block(&c_chan_rsc_INST, c_chan_rsc_INST.basename(), TLS_c_chan_rsc_trdone,
        0, 0, 1);

    testbench_INST.clk(clk);
    testbench_INST.ccs_a_chan_value_value(TLS_in_fifo_a_chan_value_value);
    testbench_INST.ccs_wait_ctrl_a_chan_value_value(TLS_in_wait_ctrl_fifo_a_chan_value_value);
    testbench_INST.ccs_sizecount_a_chan_value_value(TLS_in_fifo_a_chan_value_value_sizecount);
    testbench_INST.ccs_b_chan_value_value(TLS_in_fifo_b_chan_value_value);
    testbench_INST.ccs_wait_ctrl_b_chan_value_value(TLS_in_wait_ctrl_fifo_b_chan_value_value);
    testbench_INST.ccs_sizecount_b_chan_value_value(TLS_in_fifo_b_chan_value_value_sizecount);
    testbench_INST.ccs_c_chan_value_value(TLS_out_fifo_c_chan_value_value);
    testbench_INST.ccs_wait_ctrl_c_chan_value_value(TLS_out_wait_ctrl_fifo_c_chan_value_value);
    testbench_INST.design_is_idle(TLS_design_is_idle_reg);
    testbench_INST.enable_stalls(TLS_enable_stalls);
    testbench_INST.stall_coverage(TLS_stall_coverage);

    sync_generator_INST.clk(clk);
    sync_generator_INST.rst(rst);
    sync_generator_INST.in_sync(in_sync);
    sync_generator_INST.out_sync(out_sync);
    sync_generator_INST.inout_sync(inout_sync);
    sync_generator_INST.wait_for_init(wait_for_init);
    sync_generator_INST.catapult_start(catapult_start);
    sync_generator_INST.catapult_ready(catapult_ready);
    sync_generator_INST.catapult_done(catapult_done);

    catapult_monitor_INST.rst(rst);


    SC_METHOD(TLS_arst_n_method);
      sensitive_neg << TLS_arst_n;
      dont_initialize();

    SC_METHOD(drive_TLS_a_chan_rsc_trdone);
      sensitive << TLS_a_chan_rsc_rdy;
      sensitive << TLS_a_chan_rsc_vld;
      sensitive << rst;

    SC_METHOD(drive_TLS_b_chan_rsc_trdone);
      sensitive << TLS_b_chan_rsc_rdy;
      sensitive << TLS_b_chan_rsc_vld;
      sensitive << rst;

    SC_METHOD(drive_TLS_c_chan_rsc_trdone);
      sensitive << TLS_c_chan_rsc_vld;
      sensitive << TLS_c_chan_rsc_rdy;

    SC_METHOD(max_sim_time_notify);
      sensitive << max_sim_time_event;
      dont_initialize();

    SC_METHOD(generate_reset);
      sensitive << generate_reset_event;
      sensitive << testbench_INST.reset_request_event;

    SC_METHOD(deadlock_watch);
      sensitive << clk;
      sensitive << OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_staller_inst_run_wen;
      dont_initialize();

    SC_METHOD(deadlock_notify);
      sensitive << deadlock_event;
      dont_initialize();


    #if defined(CCS_SCVERIFY) && defined(CCS_DUT_RTL) && !defined(CCS_DUT_SYSC) && !defined(CCS_SYSC) && !defined(CCS_DUT_POWER)
        ccs_probe_monitor_INST = new ccs_probe_monitor("ccs_probe_monitor");
    ccs_probe_monitor_INST->clk(clk);
    ccs_probe_monitor_INST->rst(rst);
    #endif
    SIG_SC_LOGIC_0.write(SC_LOGIC_0);
    SIG_SC_LOGIC_1.write(SC_LOGIC_1);
    mt19937_init_genrand(19650218UL);
    install_observe_foreign_signals();
  }
};
void scverify_top::TLS_arst_n_method() {
  std::ostringstream msg;
  msg << "TLS_arst_n active @ " << sc_time_stamp();
  SC_REPORT_INFO("HW reset", msg.str().c_str());
  a_chan_rsc_INST.clear();
  b_chan_rsc_INST.clear();
  c_chan_rsc_INST.clear();
}

void scverify_top::drive_TLS_a_chan_rsc_trdone() {
  if (rst.read()==sc_dt::Log_1) { assert(TLS_a_chan_rsc_rdy.read()!= SC_LOGIC_1); }
  TLS_a_chan_rsc_trdone.write(TLS_a_chan_rsc_rdy.read() & TLS_a_chan_rsc_vld.read() & ~rst.read());
}

void scverify_top::drive_TLS_b_chan_rsc_trdone() {
  if (rst.read()==sc_dt::Log_1) { assert(TLS_b_chan_rsc_rdy.read()!= SC_LOGIC_1); }
  TLS_b_chan_rsc_trdone.write(TLS_b_chan_rsc_rdy.read() & TLS_b_chan_rsc_vld.read() & ~rst.read());
}

void scverify_top::drive_TLS_c_chan_rsc_trdone() {
  if (rst.read()==sc_dt::Log_1) { assert(TLS_c_chan_rsc_vld.read()!= SC_LOGIC_1); }
  TLS_c_chan_rsc_trdone.write(TLS_c_chan_rsc_vld.read() & TLS_c_chan_rsc_rdy.read());
}

void scverify_top::max_sim_time_notify() {
  testbench_INST.set_failed(true);
  testbench_INST.check_results();
  SC_REPORT_ERROR("System", "Specified maximum simulation time reached");
  sc_stop();
}

void scverify_top::start_of_simulation() {
  char *SCVerify_AUTOWAIT = getenv("SCVerify_AUTOWAIT");
  if (SCVerify_AUTOWAIT) {
    int l = atoi(SCVerify_AUTOWAIT);
    transactor_a_chan_value_value.set_auto_wait_limit(l);
    transactor_b_chan_value_value.set_auto_wait_limit(l);
    transactor_c_chan_value_value.set_auto_wait_limit(l);
  }
}

void scverify_top::setup_debug() {
#ifdef MC_DEFAULT_TRANSACTOR_LOG
  static int transactor_a_chan_value_value_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_b_chan_value_value_flags = MC_DEFAULT_TRANSACTOR_LOG;
  static int transactor_c_chan_value_value_flags = MC_DEFAULT_TRANSACTOR_LOG;
#else
  static int transactor_a_chan_value_value_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
  static int transactor_b_chan_value_value_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
  static int transactor_c_chan_value_value_flags = MC_TRANSACTOR_UNDERFLOW | MC_TRANSACTOR_WAIT;
#endif
  static int transactor_a_chan_value_value_count = -1;
  static int transactor_b_chan_value_value_count = -1;
  static int transactor_c_chan_value_value_count = -1;

  // At the breakpoint, modify the local variables
  // above to turn on/off different levels of transaction
  // logging for each variable. Available flags are:
  //   MC_TRANSACTOR_EMPTY       - log empty FIFOs (on by default)
  //   MC_TRANSACTOR_UNDERFLOW   - log FIFOs that run empty and then are loaded again (off)
  //   MC_TRANSACTOR_READ        - log all read events
  //   MC_TRANSACTOR_WRITE       - log all write events
  //   MC_TRANSACTOR_LOAD        - log all FIFO load events
  //   MC_TRANSACTOR_DUMP        - log all FIFO dump events
  //   MC_TRANSACTOR_STREAMCNT   - log all streamed port index counter events
  //   MC_TRANSACTOR_WAIT        - log user specified handshake waits
  //   MC_TRANSACTOR_SIZE        - log input FIFO size updates

  std::ifstream debug_cmds;
  debug_cmds.open("scverify.cmd",std::fstream::in);
  if (debug_cmds.is_open()) {
    std::cout << "Reading SCVerify debug commands from file 'scverify.cmd'" << std::endl;
    std::string line;
    while (getline(debug_cmds,line)) {
      std::size_t pos1 = line.find(" ");
      if (pos1 == std::string::npos) continue;
      std::size_t pos2 = line.find(" ", pos1+1);
      std::string varname = line.substr(0,pos1);
      std::string flags = line.substr(pos1+1,pos2-pos1-1);
      std::string count = line.substr(pos2+1);
      debug(varname.c_str(),std::atoi(flags.c_str()),std::atoi(count.c_str()));
    }
    debug_cmds.close();
  } else {
    debug("transactor_a_chan_value_value",transactor_a_chan_value_value_flags,transactor_a_chan_value_value_count);
    debug("transactor_b_chan_value_value",transactor_b_chan_value_value_flags,transactor_b_chan_value_value_count);
    debug("transactor_c_chan_value_value",transactor_c_chan_value_value_flags,transactor_c_chan_value_value_count);
  }
}

void scverify_top::debug(const char* varname, int flags, int count) {
  sc_module *xlator_p = 0;
  sc_attr_base *debug_attr_p = 0;
  if (strcmp(varname,"transactor_a_chan_value_value") == 0)
    xlator_p = &transactor_a_chan_value_value;
  if (strcmp(varname,"transactor_b_chan_value_value") == 0)
    xlator_p = &transactor_b_chan_value_value;
  if (strcmp(varname,"transactor_c_chan_value_value") == 0)
    xlator_p = &transactor_c_chan_value_value;
  if (xlator_p) {
    debug_attr_p = xlator_p->get_attribute("MC_TRANSACTOR_EVENT");
    if (!debug_attr_p) {
      debug_attr_p = new sc_attribute<int>("MC_TRANSACTOR_EVENT",flags);
      xlator_p->add_attribute(*debug_attr_p);
    }
    ((sc_attribute<int>*)debug_attr_p)->value = flags;
  }

  if (count>=0) {
    debug_attr_p = xlator_p->get_attribute("MC_TRANSACTOR_COUNT");
    if (!debug_attr_p) {
      debug_attr_p = new sc_attribute<int>("MC_TRANSACTOR_COUNT",count);
      xlator_p->add_attribute(*debug_attr_p);
    }
    ((sc_attribute<int>*)debug_attr_p)->value = count;
  }
}

// Process: SC_METHOD generate_reset
void scverify_top::generate_reset() {
  static bool activate_reset = true;
  static bool toggle_hw_reset = false;
  if (activate_reset || sc_time_stamp() == SC_ZERO_TIME) {
    setup_debug();
    activate_reset = false;
    rst.write(SC_LOGIC_1);
    arst_n_driver.reset_driver();
    generate_reset_event.notify(10.000000 , SC_NS);
  } else {
    if (toggle_hw_reset) {
      toggle_hw_reset = false;
      generate_reset_event.notify(10.000000 , SC_NS);
    } else {
      transactor_a_chan_value_value.reset_streams();
      transactor_b_chan_value_value.reset_streams();
      transactor_c_chan_value_value.reset_streams();
      rst.write(SC_LOGIC_0);
    }
    activate_reset = true;
  }
}

void scverify_top::install_observe_foreign_signals() {
#if !defined(CCS_DUT_SYSC) && defined(DEADLOCK_DETECTION)
#if defined(CCS_DUT_CYCLE) || defined(CCS_DUT_RTL)
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_staller_inst_run_wen, /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_staller_inst/run_wen);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_a_chan_rsci_inst_MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst_a_chan_rsci_irdy_run_sct,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_a_chan_rsci_inst/MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst/a_chan_rsci_irdy_run_sct);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_a_chan_rsci_inst_MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst_a_chan_rsci_ivld,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_a_chan_rsci_inst/MatMult_run_a_chan_rsci_a_chan_wait_ctrl_inst/a_chan_rsci_ivld);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_b_chan_rsci_inst_MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst_b_chan_rsci_irdy_run_sct,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_b_chan_rsci_inst/MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst/b_chan_rsci_irdy_run_sct);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_b_chan_rsci_inst_MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst_b_chan_rsci_ivld,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_b_chan_rsci_inst/MatMult_run_b_chan_rsci_b_chan_wait_ctrl_inst/b_chan_rsci_ivld);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_c_chan_rsci_inst_MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst_c_chan_rsci_irdy,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_c_chan_rsci_inst/MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst/c_chan_rsci_irdy);
  OBSERVE_FOREIGN_SIGNAL(OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_c_chan_rsci_inst_MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst_c_chan_rsci_ivld_run_sct,
      /scverify_top/rtl/MatMult_struct_inst/MatMult_run_inst/MatMult_run_c_chan_rsci_inst/MatMult_run_c_chan_rsci_c_chan_wait_ctrl_inst/c_chan_rsci_ivld_run_sct);
  OBSERVE_FOREIGN_SIGNAL(OFS_a_chan_rsc_vld, /scverify_top/rtl/a_chan_rsc_vld);
  OBSERVE_FOREIGN_SIGNAL(OFS_b_chan_rsc_vld, /scverify_top/rtl/b_chan_rsc_vld);
  OBSERVE_FOREIGN_SIGNAL(OFS_c_chan_rsc_rdy, /scverify_top/rtl/c_chan_rsc_rdy);
#endif
#endif
}

void scverify_top::deadlock_watch() {
#if !defined(CCS_DUT_SYSC) && defined(DEADLOCK_DETECTION)
#if defined(CCS_DUT_CYCLE) || defined(CCS_DUT_RTL)
#if defined(MTI_SYSTEMC) || defined(NCSC) || defined(VCS_SYSTEMC)
  if (!clk) {
    if (rst == SC_LOGIC_0 &&
      (OFS_MatMult_struct_inst_MatMult_run_inst_MatMult_run_staller_inst_run_wen.read() == SC_LOGIC_0)
      && (OFS_a_chan_rsc_vld.read() == SC_LOGIC_1)
      && (OFS_b_chan_rsc_vld.read() == SC_LOGIC_1)
      && (OFS_c_chan_rsc_rdy.read() == SC_LOGIC_1)
    ) {
      deadlocked.write(SC_LOGIC_1);
      deadlock_event.notify(275, SC_NS);
    } else {
      if (deadlocked.read() == SC_LOGIC_1)
        deadlock_event.cancel();
      deadlocked.write(SC_LOGIC_0);
    }
  }
#endif
#endif
#endif
}

void scverify_top::deadlock_notify() {
  if (deadlocked.read() == SC_LOGIC_1) {
    testbench_INST.check_results();
    SC_REPORT_ERROR("System", "Simulation deadlock detected");
    sc_stop();
  }
}

#if defined(MC_SIMULATOR_OSCI) || defined(MC_SIMULATOR_VCS)
int sc_main(int argc, char *argv[]) {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  scverify_top scverify_top("scverify_top");
  sc_start();
  return scverify_top.testbench_INST.failed();
}
#else
MC_MODULE_EXPORT(scverify_top);
#endif
