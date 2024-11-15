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
    memory_module "<src.memory.Memory object at 0x2fedf8750>"
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
    memory_module "<src.memory.Cache object at 0x2fa3f6490>"
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
    target 5
  ]
  edge [
    source 5
    target 4
  ]
  edge [
    source 5
    target 0
  ]
  edge [
    source 5
    target 1
  ]
]
