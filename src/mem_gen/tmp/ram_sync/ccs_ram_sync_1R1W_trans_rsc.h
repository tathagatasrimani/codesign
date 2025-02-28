#ifndef __INCLUDED_ccs_ram_sync_1R1W_trans_rsc_H__
#define __INCLUDED_ccs_ram_sync_1R1W_trans_rsc_H__
#include <mc_transactors.h>

template < 
  int data_width
  ,int addr_width
  ,int depth
>
class ccs_ram_sync_1R1W_trans_rsc : public mc_wire_trans_rsc_base<data_width,depth>
{
public:
  sc_in< sc_lv<addr_width> >   radr;
  sc_in< sc_lv<addr_width> >   wadr;
  sc_in< sc_lv<data_width> >   d;
  sc_in< sc_logic >   we;
  sc_in< sc_logic >   re;
  sc_in< bool >   clk;
  sc_out< sc_lv<data_width> >   q;

  typedef mc_wire_trans_rsc_base<data_width,depth> base;
  MC_EXPOSE_NAMES_OF_BASE(base);

  SC_HAS_PROCESS( ccs_ram_sync_1R1W_trans_rsc );
  ccs_ram_sync_1R1W_trans_rsc(const sc_module_name& name, bool phase, double clk_skew_delay=0.0)
    : base(name, phase, clk_skew_delay)
    ,radr("radr")
    ,wadr("wadr")
    ,d("d")
    ,we("we")
    ,re("re")
    ,clk("clk")
    ,q("q")
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
      msg << "ccs_ram_sync_1R1W_trans_rsc CLASS_STARTUP - CLK_SKEW_DELAY = "
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
    if (radr.get_interface())
      _radr = radr.read();
    if (wadr.get_interface())
      _wadr = wadr.read();
    else {
      _is_connected_port_1 = false;
    }
    if (d.get_interface())
      _d = d.read();
    else {
      _is_connected_port_1 = false;
    }
    if (we.get_interface())
      _we = we.read();
    if (re.get_interface())
      _re = re.read();

    //  Write
    int _w_addr_port_1 = -1;
    if ( _is_connected_port_1 && (_we==1)) {
      _w_addr_port_1 = get_addr(_wadr, "wadr");
      if (_w_addr_port_1 >= 0)
        inject_value(_w_addr_port_1, 0, data_width, _d, 0);
    }
    if( !_is_connected_port_1 && !_is_connected_port_1_messaged) {
      std::ostringstream msg;msg << "port_1 is not fully connected and writes on it will be ignored";
      SC_REPORT_WARNING(this->name(), msg.str().c_str());
      _is_connected_port_1_messaged = true;
    }

    //  Sync Read
    if ((_re==1)) {
      const int addr = get_addr(_radr, "radr");
      if (addr >= 0)
      {
        if (addr==_w_addr_port_1) {
          sc_lv<data_width> dc; // X
          _q = dc;
        }
        else
          extract_value(addr, 0, data_width, _q, 0);
      }
      else { 
        sc_lv<data_width> dc; // X
        _q = dc;
      }
    }
    if (q.get_interface())
      q = _q;
    this->_value_changed.notify(SC_ZERO_TIME);
  }

  int get_addr(const sc_lv<addr_width>& addr, const char* pin_name) {
    if (addr.is_01()) {
      const int cur_addr = addr.to_uint();
      if (cur_addr < 0 || cur_addr >= depth) {
#ifdef CCS_SYSC_DEBUG
        std::ostringstream msg;
        msg << "Invalid address '" << cur_addr << "' out of range [0:" << depth-1 << "]";
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
    _radr = sc_lv<addr_width>();
    _wadr = sc_lv<addr_width>();
    _d = sc_lv<data_width>();
    _we = SC_LOGIC_X;
    _re = SC_LOGIC_X;
    _is_connected_port_1 = true;
    _is_connected_port_1_messaged = false;
  }

  sc_lv<addr_width>  _radr;
  sc_lv<addr_width>  _wadr;
  sc_lv<data_width>  _d;
  sc_logic _we;
  sc_logic _re;
  sc_lv<data_width>  _q;
  bool _is_connected_port_1;
  bool _is_connected_port_1_messaged;
};
#endif // ifndef __INCLUDED_ccs_ram_sync_1R1W_trans_rsc_H__


