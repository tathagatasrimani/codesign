graph [
  directed 1
  node [
    id 0
    label "Regs0"
    function "Regs"
    idx "28"
    cost 2
    allocation "c_1"
    allocation "c_1"
    allocation "c_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 1
    label "Add0"
    function "Add"
    idx "27"
    cost 0.94
    allocation "+"
    allocation "+"
    allocation "+"
    in_use 0
  ]
  node [
    id 2
    label "Regs1"
    function "Regs"
    idx "34"
    cost 2
    allocation "b_1"
    allocation "d_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 3
    label "Add1"
    function "Add"
    idx "31"
    cost 0.94
    allocation "+"
    allocation "+"
    in_use 0
  ]
  node [
    id 4
    label "Regs2"
    function "Regs"
    idx "24"
    cost 2
    allocation "_networkx_list_start"
    allocation "b_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 5
    label "Mult0"
    function "Mult"
    idx "33"
    cost 0.98
    allocation "*"
    allocation "*"
    allocation "*"
    in_use 0
  ]
  node [
    id 6
    label "Regs3"
    function "Regs"
    idx "23"
    cost 2
    allocation "_networkx_list_start"
    allocation "a_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 7
    label "Regs4"
    function "Regs"
    idx "32"
    cost 2
    allocation "d_1"
    allocation "d_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 8
    label "Mult1"
    function "Mult"
    idx "33"
    cost 0.98
    allocation "*"
    allocation "*"
    in_use 0
  ]
  node [
    id 9
    label "Regs5"
    function "Regs"
    idx "29"
    cost 2
    allocation "_networkx_list_start"
    allocation "a_1"
    in_use 0
    size 1
    var ""
  ]
  node [
    id 10
    label "Buf0"
    function "Buf"
    size 1
    memory_module "<memory.Cache object at 0x7fd03a19d160>"
    in_use 0
    allocation "[]"
  ]
  node [
    id 11
    label "Mem0"
    function "MainMem"
    size 512
    memory_module "<memory.Memory object at 0x7fd03a1a11f0>"
    in_use 0
    allocation "[]"
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 10
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 2
    target 3
  ]
  edge [
    source 2
    target 10
  ]
  edge [
    source 3
    target 8
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 4
    target 10
  ]
  edge [
    source 5
    target 1
  ]
  edge [
    source 5
    target 2
  ]
  edge [
    source 6
    target 5
  ]
  edge [
    source 6
    target 10
  ]
  edge [
    source 7
    target 8
  ]
  edge [
    source 7
    target 10
  ]
  edge [
    source 9
    target 3
  ]
  edge [
    source 9
    target 10
  ]
  edge [
    source 10
    target 11
  ]
  edge [
    source 10
    target 0
  ]
  edge [
    source 10
    target 2
  ]
  edge [
    source 10
    target 4
  ]
  edge [
    source 10
    target 6
  ]
  edge [
    source 10
    target 7
  ]
  edge [
    source 10
    target 9
  ]
  edge [
    source 11
    target 10
  ]
]
