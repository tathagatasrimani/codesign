#ifndef __INCLUDED_ram_sync_1r1w_trans_rsc_H__
#define __INCLUDED_ram_sync_1r1w_trans_rsc_H__
#include <mc_transactors.h>

template < 
  int DATA_WIDTH
  ,int ADDR_WIDTH
  ,int DEPTH
>
class ram_sync_1r1w_trans_rsc : public mc_wire_trans_rsc_base<DATA_WIDTH,DEPTH>
{
public:
  sc_in< bool >   clk;
  sc_in< sc_logic >   wen;
  sc_in< sc_lv<ADDR_WIDTH> >   wadr;
  sc_in< sc_lv<DATA_WIDTH> >   wdata;
  sc_in< sc_logic >   ren;
  sc_in< sc_lv<ADDR_WIDTH> >   radr;
  sc_out< sc_lv<DATA_WIDTH> >   rdata;

  typedef mc_wire_trans_rsc_base<DATA_WIDTH,DEPTH> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  SC_HAS_PROCESS( ram_sync_1r1w_trans_rsc );
  ram_sync_1r1w_trans_rsc(const sc_module_name& name, bool phase, double clk_skew_delay=0.0)
    : base(name, phase, clk_skew_delay)
    ,clk("clk")
    ,wen("wen")
    ,wadr("wadr")
    ,wdata("wdata")
    ,ren("ren")
    ,radr("radr")
    ,rdata("rdata")
    ,_is_connected_port_1(true)
    ,_is_connected_port_1_messaged(false)
  {
    SC_METHOD(at_active_clock_edge);
    sensitive << (phase ? clk.pos() : clk.neg());
    this->dont_initialize();

    MC_METHOD(clk_skew_delay);
    this->sensitive << this->_clk_skew_event;
    this->dont_initialize();
  }

  virtual void start_of_simulation() {
    if ((base::_holdtime == 0.0) && this->get_attribute("CLK_SKEW_DELAY")) {
      base::_holdtime = ((sc_attribute<double>*)(this->get_attribute("CLK_SKEW_DELAY")))->value;
    }
    if (base::_holdtime > 0) {
      std::ostringstream msg;
      msg << "ram_sync_1r1w_trans_rsc CLASS_STARTUP - CLK_SKEW_DELAY = "
        << base::_holdtime << " ps @ " << sc_time_stamp();
      SC_REPORT_INFO(this->name(), msg.str().c_str());
    }
    reset_memory();
  }

  virtual void inject_value(int addr, int idx_lhs, int mywidth, sc_lv_base& rhs, int idx_rhs) {
    this->set_value(addr, idx_lhs, mywidth, rhs, idx_rhs);
  }

  virtual void extract_value(int addr, int idx_rhs, int mywidth, sc_lv_base& lhs, int idx_lhs) {
    this->get_value(addr, idx_rhs, mywidth, lhs, idx_lhs);
  }

private:
  void at_active_clock_edge() {
    base::at_active_clk();
  }

  void clk_skew_delay() {
    this->exchange_value(0);
    if (wen.get_interface())
      _wen = wen.read();
    if (wadr.get_interface())
      _wadr = wadr.read();
    else {
      _is_connected_port_1 = false;
    }
    if (wdata.get_interface())
      _wdata = wdata.read();
    else {
      _is_connected_port_1 = false;
    }
    if (ren.get_interface())
      _ren = ren.read();
    if (radr.get_interface())
      _radr = radr.read();

    //  Write
    int _w_addr_port_1 = -1;
    if ( _is_connected_port_1 && (_wen==1)) {
      _w_addr_port_1 = get_addr(_wadr, "wadr");
      if (_w_addr_port_1 >= 0)
        inject_value(_w_addr_port_1, 0, DATA_WIDTH, _wdata, 0);
    }
    if( !_is_connected_port_1 && !_is_connected_port_1_messaged) {
      std::ostringstream msg;msg << "port_1 is not fully connected and writes on it will be ignored";
      SC_REPORT_WARNING(this->name(), msg.str().c_str());
      _is_connected_port_1_messaged = true;
    }

    //  Sync Read
    if ((_ren==1)) {
      const int addr = get_addr(_radr, "radr");
      if (addr >= 0)
      {
        if (addr==_w_addr_port_1) {
          sc_lv<DATA_WIDTH> dc; // X
          _rdata = dc;
        }
        else
          extract_value(addr, 0, DATA_WIDTH, _rdata, 0);
      }
      else { 
        sc_lv<DATA_WIDTH> dc; // X
        _rdata = dc;
      }
    }
    if (rdata.get_interface())
      rdata = _rdata;
    this->_value_changed.notify(SC_ZERO_TIME);
  }

  int get_addr(const sc_lv<ADDR_WIDTH>& addr, const char* pin_name) {
    if (addr.is_01()) {
      const int cur_addr = addr.to_uint();
      if (cur_addr < 0 || cur_addr >= DEPTH) {
#ifdef CCS_SYSC_DEBUG
        std::ostringstream msg;
        msg << "Invalid address '" << cur_addr << "' out of range [0:" << DEPTH-1 << "]";
        SC_REPORT_WARNING(pin_name, msg.str().c_str());
#endif
        return -1;
      } else {
        return cur_addr;
      }
    } else {
#ifdef CCS_SYSC_DEBUG
      std::ostringstream msg;
      msg << "Invalid Address '" << addr << "' contains 'X' or 'Z'";
      SC_REPORT_WARNING(pin_name, msg.str().c_str());
#endif
      return -1;
    }
  }

  void reset_memory() {
    this->zero_data();
    _wen = SC_LOGIC_X;
    _wadr = sc_lv<ADDR_WIDTH>();
    _wdata = sc_lv<DATA_WIDTH>();
    _ren = SC_LOGIC_X;
    _radr = sc_lv<ADDR_WIDTH>();
    _is_connected_port_1 = true;
    _is_connected_port_1_messaged = false;
  }

  sc_logic _wen;
  sc_lv<ADDR_WIDTH>  _wadr;
  sc_lv<DATA_WIDTH>  _wdata;
  sc_logic _ren;
  sc_lv<ADDR_WIDTH>  _radr;
  sc_lv<DATA_WIDTH>  _rdata;
  bool _is_connected_port_1;
  bool _is_connected_port_1_messaged;
};
#endif // ifndef __INCLUDED_ram_sync_1r1w_trans_rsc_H__


