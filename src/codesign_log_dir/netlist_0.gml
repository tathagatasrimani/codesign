graph [
  directed 1
  node [
    id 0
    label "Regs0"
    function "Regs"
    idx "44"
    cost 2
    allocation "a_1[i_1][k_1]"
    allocation "c_1[i_1][j_1]"
    allocation "c_1[i_1][j_1]"
    allocation "c_1[i_1][j_1]"
    allocation "c_1[i_1][j_1]"
    allocation "c_1[i_1][j_1]"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 1
    label "Mult0"
    function "Mult"
    idx "38"
    cost 0.98
    allocation "*"
    allocation "*"
    in_use 0
  ]
  node [
    id 2
    label "Regs1"
    function "Regs"
    idx "43"
    cost 2
    allocation "b_1[k_1][j_1]"
    allocation "d_1[i_1]"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 3
    label "Regs2"
    function "Regs"
    idx "39"
    cost 2
    allocation "_networkx_list_start"
    allocation "c_1[i_1][j_1]"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 4
    label "Add0"
    function "Add"
    idx "45"
    cost 0.94
    allocation "+"
    allocation "+"
    allocation "+"
    allocation "+"
    allocation "+"
    in_use 0
  ]
  node [
    id 5
    label "Buf0"
    function "Buf"
    size 1
    memory_module "<memory.Cache object at 0x7fe15927b670>"
    in_use 0
    allocation "[]"
  ]
  node [
    id 6
    label "Mem0"
    function "MainMem"
    size 4096
    memory_module "<memory.Memory object at 0x7fe15923b310>"
    in_use 0
    allocation "[]"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 4
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 1
    target 4
  ]
  edge [
    source 2
    target 1
  ]
  edge [
    source 2
    target 4
  ]
  edge [
    source 2
    target 5
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 3
    target 5
  ]
  edge [
    source 4
    target 0
  ]
  edge [
    source 5
    target 6
  ]
  edge [
    source 5
    target 0
  ]
  edge [
    source 5
    target 2
  ]
  edge [
    source 5
    target 3
  ]
  edge [
    source 6
    target 5
  ]
]
