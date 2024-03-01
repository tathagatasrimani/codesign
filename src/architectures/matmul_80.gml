graph [
  directed 1
  node [
    id 0
    label "Regs0"
    function "Regs"
    idx "44"
    cost 2
    allocation 76
    in_use 0
    size 1
  ]
  node [
    id 1
    label "Mult0"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 26
    in_use 0
  ]
  node [
    id 2
    label "Regs1"
    function "Regs"
    idx "43"
    cost 2
    allocation 77
    in_use 0
    size 1
  ]
  node [
    id 3
    label "Regs2"
    function "Regs"
    idx "41"
    cost 2
    allocation 26
    in_use 0
    size 1
  ]
  node [
    id 4
    label "Add0"
    function "Add"
    idx "45"
    cost 0.94
    allocation 28
    in_use 0
  ]
  node [
    id 5
    label "Regs3"
    function "Regs"
    idx "41"
    cost 2
    allocation 26
    in_use 0
    size 1
  ]
  node [
    id 6
    label "Mult1"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 26
    in_use 0
  ]
  node [
    id 7
    label "Regs4"
    function "Regs"
    idx "41"
    cost 2
    allocation 26
    in_use 0
    size 1
  ]
  node [
    id 8
    label "Regs5"
    function "Regs"
    idx "39"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 9
    label "Add1"
    function "Add"
    idx "40"
    cost 0.94
    allocation 52
    in_use 0
  ]
  node [
    id 10
    label "Regs6"
    function "Regs"
    idx "37"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 11
    label "Mult2"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 26
    in_use 0
  ]
  node [
    id 12
    label "Regs7"
    function "Regs"
    idx "36"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 13
    label "Regs8"
    function "Regs"
    idx "39"
    cost 2
    allocation 50
    in_use 0
    size 1
  ]
  node [
    id 14
    label "Add2"
    function "Add"
    idx "40"
    cost 0.94
    allocation 27
    in_use 0
  ]
  node [
    id 15
    label "Regs9"
    function "Regs"
    idx "37"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 16
    label "Mult3"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 26
    in_use 0
  ]
  node [
    id 17
    label "Regs10"
    function "Regs"
    idx "36"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 18
    label "Regs11"
    function "Regs"
    idx "39"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 19
    label "Add3"
    function "Add"
    idx "40"
    cost 0.94
    allocation 27
    in_use 0
  ]
  node [
    id 20
    label "Regs12"
    function "Regs"
    idx "37"
    cost 2
    allocation 25
    in_use 0
    size 1
  ]
  node [
    id 21
    label "Mult4"
    function "Mult"
    idx "38"
    cost 0.98
    allocation 26
    in_use 0
  ]
  node [
    id 22
    label "Regs13"
    function "Regs"
    idx "36"
    cost 2
    allocation 75
    in_use 0
    size 1
  ]
  node [
    id 23
    label "Regs14"
    function "Regs"
    idx "39"
    cost 2
    allocation 50
    in_use 0
    size 1
  ]
  node [
    id 24
    label "Add4"
    function "Add"
    idx "40"
    cost 0.94
    allocation 27
    in_use 0
  ]
  node [
    id 25
    label "Buf0"
    function "Buf"
    size 1
    memory_module "<memory.Cache object at 0x13e91bfd0>"
    in_use 0
    allocation 5100
  ]
  node [
    id 26
    label "Mem0"
    function "MainMem"
    size 2048
    memory_module "<memory.Memory object at 0x13e918f10>"
    in_use 0
    allocation 5100
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
    target 25
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
    target 25
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 3
    target 25
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
    target 25
  ]
  edge [
    source 6
    target 9
  ]
  edge [
    source 7
    target 6
  ]
  edge [
    source 7
    target 25
  ]
  edge [
    source 8
    target 9
  ]
  edge [
    source 8
    target 25
  ]
  edge [
    source 9
    target 2
  ]
  edge [
    source 10
    target 11
  ]
  edge [
    source 10
    target 25
  ]
  edge [
    source 11
    target 14
  ]
  edge [
    source 12
    target 11
  ]
  edge [
    source 12
    target 25
  ]
  edge [
    source 13
    target 14
  ]
  edge [
    source 13
    target 25
  ]
  edge [
    source 14
    target 3
  ]
  edge [
    source 15
    target 16
  ]
  edge [
    source 15
    target 25
  ]
  edge [
    source 16
    target 19
  ]
  edge [
    source 17
    target 16
  ]
  edge [
    source 17
    target 25
  ]
  edge [
    source 18
    target 19
  ]
  edge [
    source 18
    target 25
  ]
  edge [
    source 19
    target 5
  ]
  edge [
    source 20
    target 21
  ]
  edge [
    source 20
    target 25
  ]
  edge [
    source 21
    target 24
  ]
  edge [
    source 22
    target 21
  ]
  edge [
    source 22
    target 25
  ]
  edge [
    source 23
    target 24
  ]
  edge [
    source 23
    target 25
  ]
  edge [
    source 24
    target 7
  ]
  edge [
    source 25
    target 26
  ]
  edge [
    source 25
    target 0
  ]
  edge [
    source 25
    target 2
  ]
  edge [
    source 25
    target 3
  ]
  edge [
    source 25
    target 5
  ]
  edge [
    source 25
    target 7
  ]
  edge [
    source 25
    target 8
  ]
  edge [
    source 25
    target 10
  ]
  edge [
    source 25
    target 12
  ]
  edge [
    source 25
    target 13
  ]
  edge [
    source 25
    target 15
  ]
  edge [
    source 25
    target 17
  ]
  edge [
    source 25
    target 18
  ]
  edge [
    source 25
    target 20
  ]
  edge [
    source 25
    target 22
  ]
  edge [
    source 25
    target 23
  ]
  edge [
    source 26
    target 25
  ]
]
