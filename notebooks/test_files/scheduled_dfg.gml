graph [
  directed 1
  node [
    id 0
    label "a_0;147"
    function "Regs"
    idx "147"
    cost 0.4
    scheduling_id 0
    allocation "Regs0"
    start_time 1.75
    end_time 2.15
    layer 1
    size 28
  ]
  node [
    id 1
    label "a_0;151"
    function "Regs"
    idx "151"
    cost 0.4
    scheduling_id 1
    allocation "Regs0"
    start_time 2.324
    end_time 2.724
    layer 3
    size 28
  ]
  node [
    id 2
    label "b_1;145"
    function "Regs"
    idx "145"
    cost 0.4
    scheduling_id 2
    allocation "Regs0"
    start_time 1.158
    end_time 1.558
    layer 0
    size 28
  ]
  node [
    id 3
    label "b_1;154"
    function "Regs"
    idx "154"
    cost 0.4
    scheduling_id 3
    allocation "Regs0"
    start_time 2.91
    end_time 3.31
    layer 3
    size 28
  ]
  node [
    id 4
    label "b_1;150"
    function "Regs"
    idx "150"
    cost 0.4
    scheduling_id 4
    allocation "Regs1"
    start_time 2.611
    end_time 3.011
    layer 3
    size 28
  ]
  node [
    id 5
    label "b_1;146"
    function "Regs"
    idx "146"
    cost 0.4
    scheduling_id 5
    allocation "Regs1"
    start_time 1.937
    end_time 2.337
    layer 1
    size 28
  ]
  node [
    id 6
    label "==;156"
    function "Eq"
    idx "156"
    cost 0.127304
    scheduling_id 6
    allocation "Eq0"
    start_time 3.844
    end_time 3.971
    layer 4
  ]
  node [
    id 7
    label "*;152"
    function "Mult"
    idx "152"
    cost 0.703026622
    scheduling_id 7
    allocation "Mult0"
    start_time 3.285
    end_time 3.988
    layer 4
  ]
  node [
    id 8
    label "+;148"
    function "Add"
    idx "148"
    cost 0.8
    scheduling_id 8
    allocation "Add0"
    start_time 2.529
    end_time 3.329
    layer 2
  ]
  node [
    id 9
    label "d_1;153"
    function "Regs"
    idx "153"
    cost 0.4
    scheduling_id 9
    allocation "Regs0"
    start_time 4.22
    end_time 4.62
    layer 5
    size 28
  ]
  node [
    id 10
    label "c_1;149"
    function "Regs"
    idx "159"
    cost 0.4
    scheduling_id 10
    allocation "Regs0"
    start_time 3.502
    end_time 3.902
    layer 3
    size 28
  ]
  node [
    id 11
    label "b_1;158"
    function "Regs"
    idx "158"
    cost 0.4
    scheduling_id 11
    allocation "Regs1"
    start_time 3.269
    end_time 3.669
    layer 3
  ]
  node [
    id 12
    label "+;160"
    function "Add"
    idx "160"
    cost 0.8
    scheduling_id 12
    allocation "Add0"
    start_time 3.958
    end_time 4.758
    layer 4
  ]
  node [
    id 13
    label "e_1;161"
    function "Regs"
    idx "161"
    cost 0.4
    scheduling_id 13
    allocation "Regs1"
    start_time 4.8
    end_time 5.2
    layer 5
    size 28
  ]
  node [
    id 14
    label "end"
    function "end"
    scheduling_id 14
    allocation ""
    start_time 5.2
    end_time 0.0
    layer 6
  ]
  node [
    id 15
    label "tmp_op_reg;162"
    function "Regs"
    cost 0.4
    scheduling_id 15
    allocation "Regs0"
    start_time 4.705
    end_time 5.105
    layer 5
    size 0
  ]
  node [
    id 16
    label "Buf0"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 16
    start_time 1.646
    end_time 1.646
  ]
  node [
    id 17
    label "Mem0"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 28
    scheduling_id 17
    start_time 0.8
    end_time 1.6
  ]
  node [
    id 18
    label "Buf1"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 18
    start_time 1.905
    end_time 1.905
  ]
  node [
    id 19
    label "Buf2"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 19
    start_time 0.971
    end_time 0.971
  ]
  node [
    id 20
    label "Mem2"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 28
    scheduling_id 20
    start_time -0.0
    end_time 0.8
  ]
  node [
    id 21
    label "Buf3"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 21
    start_time 2.11
    end_time 2.11
  ]
  node [
    id 22
    label "Buf4"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 22
    start_time 2.341
    end_time 2.341
  ]
  node [
    id 23
    label "Buf5"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 23
    start_time 1.747
    end_time 1.747
  ]
  node [
    id 24
    label "Buf6"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 24
    start_time 4.095
    end_time 4.095
  ]
  node [
    id 25
    label "Mem6"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 28
    scheduling_id 25
    start_time 3.2
    end_time 4.0
  ]
  node [
    id 26
    label "Buf7"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 26
    start_time 2.95
    end_time 2.95
  ]
  node [
    id 27
    label "Mem7"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 28
    scheduling_id 27
    start_time 1.6
    end_time 2.4
  ]
  node [
    id 28
    label "Buf9"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 28
    scheduling_id 28
    start_time 4.8
    end_time 4.8
  ]
  node [
    id 29
    label "Mem9"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 28
    scheduling_id 29
    start_time 4.0
    end_time 4.8
  ]
  node [
    id 30
    label "Buf10"
    function "Buf"
    allocation "Buf0"
    cost 0.0
    size 0
    scheduling_id 30
    start_time 3.642
    end_time 3.642
  ]
  node [
    id 31
    label "Mem10"
    function "MainMem"
    allocation "MainMem0"
    cost 0.8
    size 0
    scheduling_id 31
    start_time 2.4
    end_time 3.2
  ]
  edge [
    source 0
    target 8
    weight 0.4
  ]
  edge [
    source 1
    target 7
    weight 0.4
  ]
  edge [
    source 2
    target 3
    weight 0.4
  ]
  edge [
    source 2
    target 4
    weight 0.4
  ]
  edge [
    source 2
    target 5
    weight 0.4
  ]
  edge [
    source 3
    target 6
    weight 0.4
  ]
  edge [
    source 4
    target 7
    weight 0.4
  ]
  edge [
    source 5
    target 8
    weight 0.4
  ]
  edge [
    source 6
    target 15
    weight 0.4
  ]
  edge [
    source 7
    target 9
    weight 0.703026622
  ]
  edge [
    source 8
    target 10
    weight 0.8
  ]
  edge [
    source 9
    target 14
    weight 0.4
  ]
  edge [
    source 10
    target 12
    weight 0.4
  ]
  edge [
    source 11
    target 12
    weight 0.4
  ]
  edge [
    source 12
    target 13
    weight 0.8
  ]
  edge [
    source 13
    target 14
    weight 0.4
  ]
  edge [
    source 15
    target 14
    weight 0.127304
  ]
  edge [
    source 16
    target 0
    function "Mem"
    weight 0.0
  ]
  edge [
    source 17
    target 16
    function "Mem"
    weight 0.8
  ]
  edge [
    source 18
    target 1
    function "Mem"
    weight 0.0
  ]
  edge [
    source 19
    target 2
    function "Mem"
    weight 0.0
  ]
  edge [
    source 20
    target 19
    function "Mem"
    weight 0.8
  ]
  edge [
    source 21
    target 3
    function "Mem"
    weight 0.0
  ]
  edge [
    source 22
    target 4
    function "Mem"
    weight 0.0
  ]
  edge [
    source 23
    target 5
    function "Mem"
    weight 0.0
  ]
  edge [
    source 24
    target 9
    function "Mem"
    weight 0.0
  ]
  edge [
    source 25
    target 24
    function "Mem"
    weight 0.8
  ]
  edge [
    source 26
    target 10
    function "Mem"
    weight 0.0
  ]
  edge [
    source 27
    target 26
    function "Mem"
    weight 0.8
  ]
  edge [
    source 28
    target 13
    function "Mem"
    weight 0.0
  ]
  edge [
    source 29
    target 28
    function "Mem"
    weight 0.8
  ]
  edge [
    source 30
    target 15
    function "Mem"
    weight 0.0
  ]
  edge [
    source 31
    target 30
    function "Mem"
    weight 0.8
  ]
]
