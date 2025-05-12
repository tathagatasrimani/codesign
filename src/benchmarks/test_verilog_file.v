
`default_nettype none

module MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_wport_9_16_7_100_100_16_gen
    (
  d, wadr, we, wadr_d, d_d, we_d, port_1_w_ram_ir_internal_WMASK_B_d
);
  output [15:0] d;
  output [6:0] wadr;
  output we;
  input [6:0] wadr_d;
  input [15:0] d_d;
  input we_d;
  input port_1_w_ram_ir_internal_WMASK_B_d;



  // Interconnect Declarations for Component Instantiations 
  assign d = (d_d);
  assign wadr = (wadr_d);
  assign we = (port_1_w_ram_ir_internal_WMASK_B_d);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_8_16_7_100_100_16_gen
// ------------------------------------------------------------------


module MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_8_16_7_100_100_16_gen
    (
  q, radr, re, radr_d, re_d, q_d, port_0_r_ram_ir_internal_RMASK_B_d
);
  input [15:0] q;
  output [6:0] radr;
  output re;
  input [6:0] radr_d;
  input re_d;
  output [15:0] q_d;
  input port_0_r_ram_ir_internal_RMASK_B_d;



  // Interconnect Declarations for Component Instantiations 
  assign q_d = q;
  assign radr = (radr_d);
  assign re = (port_0_r_ram_ir_internal_RMASK_B_d);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_7_16_7_100_100_16_gen
// ------------------------------------------------------------------


module MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_7_16_7_100_100_16_gen
    (
  q, radr, re, radr_d, re_d, q_d, port_0_r_ram_ir_internal_RMASK_B_d
);
  input [15:0] q;
  output [6:0] radr;
  output re;
  input [6:0] radr_d;
  input re_d;
  output [15:0] q_d;
  input port_0_r_ram_ir_internal_RMASK_B_d;



  // Interconnect Declarations for Component Instantiations 
  assign q_d = q;
  assign radr = (radr_d);
  assign re = (port_0_r_ram_ir_internal_RMASK_B_d);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    MatMult_run_run_fsm
//  FSM Module
// ------------------------------------------------------------------

module MatMult_struct (
  clk, arst_n, a_chan_rsc_re, a_chan_rsc_radr, a_chan_rsc_q, a_chan_rsc_req_vz, a_chan_rsc_rls_lz,
      b_chan_rsc_re, b_chan_rsc_radr, b_chan_rsc_q, b_chan_rsc_req_vz, b_chan_rsc_rls_lz,
      c_chan_rsc_we, c_chan_rsc_wadr, c_chan_rsc_d, c_chan_rsc_req_vz, c_chan_rsc_rls_lz
);
    input clk;
    input arst_n;
    output a_chan_rsc_re;
    output [6:0] a_chan_rsc_radr;
    input [15:0] a_chan_rsc_q;
    input a_chan_rsc_req_vz;
    output a_chan_rsc_rls_lz;
    output b_chan_rsc_re;
    output [6:0] b_chan_rsc_radr;
    input [15:0] b_chan_rsc_q;
    input b_chan_rsc_req_vz;
    output b_chan_rsc_rls_lz;
    output c_chan_rsc_we;
    output [6:0] c_chan_rsc_wadr;
    output [15:0] c_chan_rsc_d;
    input c_chan_rsc_req_vz;
    output c_chan_rsc_rls_lz;


    // Interconnect Declarations
    wire [6:0] a_chan_rsci_radr_d;
    wire [15:0] a_chan_rsci_q_d;
    wire [6:0] b_chan_rsci_radr_d;
    wire [15:0] b_chan_rsci_q_d;
    wire [6:0] c_chan_rsci_wadr_d;
    wire [15:0] c_chan_rsci_d_d;
    wire a_chan_rsci_re_d_iff;
    wire b_chan_rsci_re_d_iff;
    wire c_chan_rsci_we_d_iff;

    // three memories
    MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_7_16_7_100_100_16_gen
        a_chan_rsci (
        .q(a_chan_rsc_q),
        .radr(a_chan_rsc_radr),
        .re(a_chan_rsc_re),
        .radr_d(a_chan_rsci_radr_d),
        .re_d(a_chan_rsci_re_d_iff),
        .q_d(a_chan_rsci_q_d),
        .port_0_r_ram_ir_internal_RMASK_B_d(a_chan_rsci_re_d_iff)
        );
    MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_rport_8_16_7_100_100_16_gen
        b_chan_rsci (
        .q(b_chan_rsc_q),
        .radr(b_chan_rsc_radr),
        .re(b_chan_rsc_re),
        .radr_d(b_chan_rsci_radr_d),
        .re_d(b_chan_rsci_re_d_iff),
        .q_d(b_chan_rsci_q_d),
        .port_0_r_ram_ir_internal_RMASK_B_d(b_chan_rsci_re_d_iff)
        );
    MatMult_ccs_ram_sync_1R1W_MatMult_run_a_value_value_rsc_ccs_ram_sync_1R1W_wport_9_16_7_100_100_16_gen
        c_chan_rsci (
        .d(c_chan_rsc_d),
        .wadr(c_chan_rsc_wadr),
        .we(c_chan_rsc_we),
        .wadr_d(c_chan_rsci_wadr_d),
        .d_d(c_chan_rsci_d_d),
        .we_d(c_chan_rsci_we_d_iff),
        .port_1_w_ram_ir_internal_WMASK_B_d(c_chan_rsci_we_d_iff)
        );

    reg [15:0] result;
    wire [15:0] add_result, mult_result;

    add  for_1_10_for_10_for_2_add_inst_run_rg (
      .a(result),
      .b(mult_result),
      .z(add_result)
    );

    mult  for_1_3_for_3_for_3_mul_inst_run_rg (
      .a(a_chan_rsci_q_d),
      .b(b_chan_rsci_q_d),
      .z(mult_result)
    );

    always @(posedge clk or negedge arst_n) begin
      if (!arst_n) begin
        result <= '0;
      end else begin
        result <= add_result;
      end
    end

    assign c_chan_rsci_d_d = result;

    
  
endmodule


module MatMult (
  clk, arst_n, a_chan_rsc_re, a_chan_rsc_radr, a_chan_rsc_q, a_chan_rsc_req_vz, a_chan_rsc_rls_lz,
      b_chan_rsc_re, b_chan_rsc_radr, b_chan_rsc_q, b_chan_rsc_req_vz, b_chan_rsc_rls_lz,
      c_chan_rsc_we, c_chan_rsc_wadr, c_chan_rsc_d, c_chan_rsc_req_vz, c_chan_rsc_rls_lz
);
  input clk;
  input arst_n;
  output a_chan_rsc_re;
  output [6:0] a_chan_rsc_radr;
  input [15:0] a_chan_rsc_q;
  input a_chan_rsc_req_vz;
  output a_chan_rsc_rls_lz;
  output b_chan_rsc_re;
  output [6:0] b_chan_rsc_radr;
  input [15:0] b_chan_rsc_q;
  input b_chan_rsc_req_vz;
  output b_chan_rsc_rls_lz;
  output c_chan_rsc_we;
  output [6:0] c_chan_rsc_wadr;
  output [15:0] c_chan_rsc_d;
  input c_chan_rsc_req_vz;
  output c_chan_rsc_rls_lz;


  // Interconnect Declarations for Component Instantiations 
  MatMult_struct MatMult_struct_inst (
      .clk(clk),
      .arst_n(arst_n),
      .a_chan_rsc_re(a_chan_rsc_re),
      .a_chan_rsc_radr(a_chan_rsc_radr),
      .a_chan_rsc_q(a_chan_rsc_q),
      .a_chan_rsc_req_vz(a_chan_rsc_req_vz),
      .a_chan_rsc_rls_lz(a_chan_rsc_rls_lz),
      .b_chan_rsc_re(b_chan_rsc_re),
      .b_chan_rsc_radr(b_chan_rsc_radr),
      .b_chan_rsc_q(b_chan_rsc_q),
      .b_chan_rsc_req_vz(b_chan_rsc_req_vz),
      .b_chan_rsc_rls_lz(b_chan_rsc_rls_lz),
      .c_chan_rsc_we(c_chan_rsc_we),
      .c_chan_rsc_wadr(c_chan_rsc_wadr),
      .c_chan_rsc_d(c_chan_rsc_d),
      .c_chan_rsc_req_vz(c_chan_rsc_req_vz),
      .c_chan_rsc_rls_lz(c_chan_rsc_rls_lz)
    );


endmodule






`default_nettype wire