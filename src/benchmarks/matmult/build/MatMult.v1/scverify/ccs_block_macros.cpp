void mc_testbench_capture_IN( ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &a_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &b_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &c_chan) { mc_testbench::capture_IN(a_chan,b_chan,c_chan); }
void mc_testbench_capture_OUT( ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &a_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &b_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &c_chan) { mc_testbench::capture_OUT(a_chan,b_chan,c_chan); }
void mc_testbench_wait_for_idle_sync() {mc_testbench::wait_for_idle_sync(); }

class ccs_intercept
{
  public:
  void capture_THIS( void* _this ) {
    if ( dut == NULL ) dut = _this;
    cur_inst = _this;
    if ( capture_msg && dut != NULL && dut == cur_inst ) {
      std::cout << "SCVerify intercepting C++ function 'MatMult::run' for RTL block 'MatMult'" << std::endl;
      capture_msg = false;
    }
  }
  void capture_IN(  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &a_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &b_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &c_chan ) {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_capture_IN(a_chan,b_chan,c_chan);
  }
  void capture_OUT(  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &a_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &b_chan,  ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &c_chan ) {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_capture_OUT(a_chan,b_chan,c_chan);
  }
  void wait_for_idle_sync() {
    if ( dut != NULL && dut == cur_inst )
      mc_testbench_wait_for_idle_sync();
  }
  ccs_intercept(): dut(NULL), cur_inst(NULL), capture_msg(true){}
  private:
  void *dut, *cur_inst;
  bool capture_msg;
};

void MatMult::run(ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &a_chan,ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &b_chan,ac_channel<PackedInt2D<16UL, 5UL, 5UL > > &c_chan) {
  static ccs_intercept ccs_intercept_inst;
  void* ccs_intercept_this = this;
  ccs_intercept_inst.capture_THIS(ccs_intercept_this);
  ccs_intercept_inst.capture_IN(a_chan,b_chan,c_chan);
  ccs_real_run(a_chan,b_chan,c_chan);
  ccs_intercept_inst.capture_OUT(a_chan,b_chan,c_chan);
}
