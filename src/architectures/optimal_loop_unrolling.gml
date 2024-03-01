graph [
  directed 1
  node [
    id 0
    label "Regs0"
    function "Regs"
    idx "41"
    cost 2
    allocation 403
    in_use 0
    size 1
    var ""
  ]
  node [
    id 1
    label "Add0"
    function "Add"
    idx "40"
    cost 0.94
    allocation 503
    in_use 0
  ]
  node [
    id 2
    label "Regs1"
    function "Regs"
    idx "41"
    cost 2
    allocation 401
    in_use 0
    size 1
    var ""
  ]
  node [
    id 3
    label "Mult0"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 301
    in_use 0
  ]
  node [
    id 4
    label "Regs2"
    function "Regs"
    idx "41"
    cost 2
    allocation 601
    in_use 0
    size 1
    var ""
  ]
  node [
    id 5
    label "Regs3"
    function "Regs"
    idx "39"
    cost 2
    allocation 900
    in_use 0
    size 1
    var ""
  ]
  node [
    id 6
    label "Add1"
    function "Add"
    idx "40"
    cost 0.94
    allocation 304
    in_use 0
  ]
  node [
    id 7
    label "Regs4"
    function "Regs"
    idx "37"
    cost 2
    allocation 300
    in_use 0
    size 1
    var ""
  ]
  node [
    id 8
    label "Mult1"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 401
    in_use 0
  ]
  node [
    id 9
    label "Regs5"
    function "Regs"
    idx "36"
    cost 2
    allocation 300
    in_use 0
    size 1
    var ""
  ]
  node [
    id 10
    label "Regs6"
    function "Regs"
    idx "39"
    cost 2
    allocation 300
    in_use 0
    size 1
    var ""
  ]
  node [
    id 11
    label "Add2"
    function "Add"
    idx "40"
    cost 0.94
    allocation 302
    in_use 0
  ]
  node [
    id 12
    label "Regs7"
    function "Regs"
    idx "37"
    cost 2
    allocation 700
    in_use 0
    size 1
    var ""
  ]
  node [
    id 13
    label "Mult2"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 302
    in_use 0
  ]
  node [
    id 14
    label "Regs8"
    function "Regs"
    idx "36"
    cost 2
    allocation 400
    in_use 0
    size 1
    var ""
  ]
  node [
    id 15
    label "Buf0"
    function "Buf"
    size 1
    memory_module "<memory.Cache object at 0x116f672d0>"
    in_use 0
    allocation 0
  ]
  node [
    id 16
    label "Mem0"
    function "MainMem"
    size 4096
    memory_module "<memory.Memory object at 0x116f65250>"
    in_use 0
    allocation 0
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 15
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 2
    target 1
  ]
  edge [
    source 2
    target 3
  ]
  edge [
    source 2
    target 15
  ]
  edge [
    source 3
    target 1
  ]
  edge [
    source 4
    target 3
  ]
  edge [
    source 4
    target 15
  ]
  edge [
    source 5
    target 6
  ]
  edge [
    source 5
    target 15
  ]
  edge [
    source 6
    target 2
  ]
  edge [
    source 7
    target 8
  ]
  edge [
    source 7
    target 15
  ]
  edge [
    source 8
    target 6
  ]
  edge [
    source 9
    target 8
  ]
  edge [
    source 9
    target 15
  ]
  edge [
    source 10
    target 11
  ]
  edge [
    source 10
    target 15
  ]
  edge [
    source 11
    target 4
  ]
  edge [
    source 12
    target 13
  ]
  edge [
    source 12
    target 15
  ]
  edge [
    source 13
    target 11
  ]
  edge [
    source 14
    target 13
  ]
  edge [
    source 14
    target 15
  ]
  edge [
    source 15
    target 16
  ]
  edge [
    source 15
    target 0
  ]
  edge [
    source 15
    target 2
  ]
  edge [
    source 15
    target 4
  ]
  edge [
    source 15
    target 5
  ]
  edge [
    source 15
    target 7
  ]
  edge [
    source 15
    target 9
  ]
  edge [
    source 15
    target 10
  ]
  edge [
    source 15
    target 12
  ]
  edge [
    source 15
    target 14
  ]
  edge [
    source 16
    target 15
  ]
]
