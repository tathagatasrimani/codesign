graph [
  directed 1
  node [
    id 0
    label "Add0"
    function "Add"
    type "pe"
    in_use 0
    idx 0
    allocation "[]"
  ]
  node [
    id 1
    label "Mult0"
    function "Mult"
    type "pe"
    in_use 0
    idx 0
    allocation "[]"
  ]
  node [
    id 2
    label "Regs0"
    function "Regs"
    size 1
    type "memory"
    in_use 0
    idx 0
    var ""
    allocation "[]"
  ]
  node [
    id 3
    label "MainMem0"
    function "MainMem"
    size 1024
    type "memory"
    in_use 0
    idx 0
    memory_module "<memory.Memory object at 0x127690250>"
    allocation "[]"
  ]
  node [
    id 4
    label "Buf0"
    function "Buf"
    size 1
    type "memory"
    in_use 0
    idx 0
    memory_module "<memory.Cache object at 0x1276909d0>"
    allocation "[]"
  ]
  node [
    id 5
    label "Regs1"
    function "Regs"
    size 1
    type "memory"
    in_use 0
    idx 1
    var ""
    allocation "[]"
  ]
  node [
    id 6
    label "MainMem1"
    function "MainMem"
    size 1024
    type "memory"
    in_use 0
    idx 1
    memory_module "<memory.Memory object at 0x1275fbf50>"
    allocation "[]"
  ]
  node [
    id 7
    label "Buf1"
    function "Buf"
    size 1
    type "memory"
    in_use 0
    idx 1
    memory_module "<memory.Cache object at 0x1276a34d0>"
    allocation "[]"
  ]
  node [
    id 8
    label "Add1"
    function "Add"
    idx 1
    in_use 0
    type "pe"
  ]
  node [
    id 9
    label "Regs2"
    function "Regs"
    idx 2
    in_use 0
    type "memory"
    size 1
  ]
  node [
    id 10
    label "Regs3"
    function "Regs"
    idx 3
    in_use 0
    type "memory"
    size 1
  ]
  node [
    id 11
    label "Mult1"
    function "Mult"
    idx 1
    in_use 0
    type "pe"
  ]
  node [
    id 12
    label "Regs4"
    function "Regs"
    idx 4
    in_use 0
    type "memory"
    size 1
  ]
  node [
    id 13
    label "Add2"
    function "Add"
    idx 2
    in_use 0
    type "pe"
  ]
  node [
    id 14
    label "Regs5"
    function "Regs"
    idx 5
    in_use 0
    type "memory"
    size 1
  ]
  node [
    id 15
    label "Regs6"
    function "Regs"
    idx 6
    in_use 0
    type "memory"
    size 1
  ]
  node [
    id 16
    label "Add3"
    function "Add"
    idx 3
    in_use 0
    type "pe"
  ]
  node [
    id 17
    label "Mult2"
    function "Mult"
    idx 2
    in_use 0
    type "pe"
  ]
  node [
    id 18
    label "Mult3"
    function "Mult"
    idx 3
    in_use 0
    type "pe"
  ]
  node [
    id 19
    label "Regs7"
    function "Regs"
    idx 7
    in_use 0
    type "memory"
    size 1
  ]
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 2
  ]
  edge [
    source 0
    target 5
  ]
  edge [
    source 0
    target 8
  ]
  edge [
    source 0
    target 9
  ]
  edge [
    source 0
    target 10
  ]
  edge [
    source 0
    target 11
  ]
  edge [
    source 0
    target 12
  ]
  edge [
    source 0
    target 13
  ]
  edge [
    source 0
    target 14
  ]
  edge [
    source 0
    target 15
  ]
  edge [
    source 0
    target 16
  ]
  edge [
    source 0
    target 17
  ]
  edge [
    source 0
    target 18
  ]
  edge [
    source 0
    target 19
  ]
  edge [
    source 1
    target 0
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 1
    target 5
  ]
  edge [
    source 1
    target 8
  ]
  edge [
    source 1
    target 9
  ]
  edge [
    source 1
    target 10
  ]
  edge [
    source 1
    target 11
  ]
  edge [
    source 1
    target 12
  ]
  edge [
    source 1
    target 13
  ]
  edge [
    source 1
    target 14
  ]
  edge [
    source 1
    target 15
  ]
  edge [
    source 1
    target 16
  ]
  edge [
    source 1
    target 17
  ]
  edge [
    source 1
    target 18
  ]
  edge [
    source 1
    target 19
  ]
  edge [
    source 2
    target 4
  ]
  edge [
    source 2
    target 0
  ]
  edge [
    source 2
    target 1
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 4
    target 3
  ]
  edge [
    source 4
    target 2
  ]
  edge [
    source 4
    target 9
  ]
  edge [
    source 4
    target 10
  ]
  edge [
    source 4
    target 12
  ]
  edge [
    source 4
    target 14
  ]
  edge [
    source 4
    target 15
  ]
  edge [
    source 4
    target 19
  ]
  edge [
    source 5
    target 7
  ]
  edge [
    source 5
    target 0
  ]
  edge [
    source 5
    target 1
  ]
  edge [
    source 6
    target 7
  ]
  edge [
    source 7
    target 6
  ]
  edge [
    source 7
    target 5
  ]
  edge [
    source 7
    target 9
  ]
  edge [
    source 7
    target 10
  ]
  edge [
    source 7
    target 12
  ]
  edge [
    source 7
    target 14
  ]
  edge [
    source 7
    target 15
  ]
  edge [
    source 7
    target 19
  ]
  edge [
    source 8
    target 0
  ]
  edge [
    source 8
    target 1
  ]
  edge [
    source 8
    target 9
  ]
  edge [
    source 8
    target 10
  ]
  edge [
    source 8
    target 11
  ]
  edge [
    source 8
    target 12
  ]
  edge [
    source 8
    target 13
  ]
  edge [
    source 8
    target 14
  ]
  edge [
    source 8
    target 15
  ]
  edge [
    source 8
    target 16
  ]
  edge [
    source 8
    target 17
  ]
  edge [
    source 8
    target 18
  ]
  edge [
    source 8
    target 19
  ]
  edge [
    source 9
    target 0
  ]
  edge [
    source 9
    target 1
  ]
  edge [
    source 9
    target 8
  ]
  edge [
    source 9
    target 4
  ]
  edge [
    source 9
    target 7
  ]
  edge [
    source 10
    target 0
  ]
  edge [
    source 10
    target 1
  ]
  edge [
    source 10
    target 8
  ]
  edge [
    source 10
    target 4
  ]
  edge [
    source 10
    target 7
  ]
  edge [
    source 11
    target 0
  ]
  edge [
    source 11
    target 1
  ]
  edge [
    source 11
    target 8
  ]
  edge [
    source 11
    target 12
  ]
  edge [
    source 11
    target 13
  ]
  edge [
    source 11
    target 14
  ]
  edge [
    source 11
    target 15
  ]
  edge [
    source 11
    target 16
  ]
  edge [
    source 11
    target 17
  ]
  edge [
    source 11
    target 18
  ]
  edge [
    source 11
    target 19
  ]
  edge [
    source 12
    target 0
  ]
  edge [
    source 12
    target 1
  ]
  edge [
    source 12
    target 8
  ]
  edge [
    source 12
    target 11
  ]
  edge [
    source 12
    target 4
  ]
  edge [
    source 12
    target 7
  ]
  edge [
    source 13
    target 0
  ]
  edge [
    source 13
    target 1
  ]
  edge [
    source 13
    target 8
  ]
  edge [
    source 13
    target 11
  ]
  edge [
    source 13
    target 14
  ]
  edge [
    source 13
    target 15
  ]
  edge [
    source 13
    target 16
  ]
  edge [
    source 13
    target 17
  ]
  edge [
    source 13
    target 18
  ]
  edge [
    source 13
    target 19
  ]
  edge [
    source 14
    target 0
  ]
  edge [
    source 14
    target 1
  ]
  edge [
    source 14
    target 8
  ]
  edge [
    source 14
    target 11
  ]
  edge [
    source 14
    target 13
  ]
  edge [
    source 14
    target 4
  ]
  edge [
    source 14
    target 7
  ]
  edge [
    source 15
    target 0
  ]
  edge [
    source 15
    target 1
  ]
  edge [
    source 15
    target 8
  ]
  edge [
    source 15
    target 11
  ]
  edge [
    source 15
    target 13
  ]
  edge [
    source 15
    target 4
  ]
  edge [
    source 15
    target 7
  ]
  edge [
    source 16
    target 0
  ]
  edge [
    source 16
    target 1
  ]
  edge [
    source 16
    target 8
  ]
  edge [
    source 16
    target 11
  ]
  edge [
    source 16
    target 13
  ]
  edge [
    source 16
    target 17
  ]
  edge [
    source 16
    target 18
  ]
  edge [
    source 16
    target 19
  ]
  edge [
    source 17
    target 0
  ]
  edge [
    source 17
    target 1
  ]
  edge [
    source 17
    target 8
  ]
  edge [
    source 17
    target 11
  ]
  edge [
    source 17
    target 13
  ]
  edge [
    source 17
    target 16
  ]
  edge [
    source 17
    target 18
  ]
  edge [
    source 17
    target 19
  ]
  edge [
    source 18
    target 0
  ]
  edge [
    source 18
    target 1
  ]
  edge [
    source 18
    target 8
  ]
  edge [
    source 18
    target 11
  ]
  edge [
    source 18
    target 13
  ]
  edge [
    source 18
    target 16
  ]
  edge [
    source 18
    target 17
  ]
  edge [
    source 18
    target 19
  ]
  edge [
    source 19
    target 0
  ]
  edge [
    source 19
    target 1
  ]
  edge [
    source 19
    target 8
  ]
  edge [
    source 19
    target 11
  ]
  edge [
    source 19
    target 13
  ]
  edge [
    source 19
    target 16
  ]
  edge [
    source 19
    target 17
  ]
  edge [
    source 19
    target 18
  ]
  edge [
    source 19
    target 4
  ]
  edge [
    source 19
    target 7
  ]
]
