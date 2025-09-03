graph [
  directed 1
  node [
    id 0
    label "add_inst_run_rg"
    type "adder"
  ]
  node [
    id 1
    label "mul_inst_run_rg"
    type "mult"
  ]
  edge [
    source 0
    target 0
    signal "add_inst_run_out_1"
  ]
  edge [
    source 0
    target 1
    signal "mul_inst_run_out_1"
  ]
  edge [
    source 1
    target 1
    signal "b_chan_rsci_q_d_mxwt"
  ]
]
