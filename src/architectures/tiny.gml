graph [
  directed 1
  node [
    id 0
    label "Regs0"
    type "memory"
    function "Regs"
    in_use 0
    idx 0
  ]
  node [
    id 1
    label "Regs1"
    type "memory"
    function "Regs"
    in_use 0
    idx 1
  ]
  node [
    id 2
    label "Add0"
    type "pe"
    function "Add"
    in_use 0
    idx 0
  ]
  node [
    id 3
    label "Mult0"
    type "pe"
    function "Mult"
    in_use 0
    idx 0
  ]
  node [
    id 4
    label "Eq0"
    type "pe"
    function "Eq"
    in_use 0
    idx 0
  ]
  node [
    id 5
    label "Buf0"
    type "mem"
    function "Buf"
    in_use 0
    idx 1
    size 2048
  ]
  node [
    id 6
    label "Mem0"
    type "mem"
    function "MainMem"
    in_use 0
    idx 0
    size 131072
  ]
  edge [
    source 0
    target 2
    cost 0.94
  ]
  edge [
    source 0
    target 3
    cost 0.98
  ]
  edge [
    source 1
    target 2
    cost 0.94
  ]
  edge [
    source 1
    target 3
    cost 0.98
  ]
  edge [
    source 2
    target 0
    cost 2
  ]
  edge [
    source 2
    target 1
    cost 2
  ]
  edge [
    source 2
    target 3
    cost 0.98
  ]
  edge [
    source 3
    target 2
    cost 0.94
  ]
  edge [
    source 3
    target 0
    cost 2
  ]
  edge [
    source 3
    target 1
    cost 2
  ]
  edge [
    source 0
    target 4
    cost 2
  ]
  edge [
    source 1
    target 4
    cost 2
  ]
  edge [
    source 4
    target 0
    cost 2
  ]
  edge [
    source 4
    target 1
    cost 2
  ]
  edge [
    source 6
    target 5
    cost 2
  ]
  edge [
    source 5
    target 6
    cost 2
  ]
  edge [
    source 5
    target 0
    cost 2
  ]
  edge [
    source 5
    target 1
    cost 2
  ]
  edge [
    source 0
    target 5
    cost 2
  ]
  edge [
    source 1
    target 5
    cost 2
  ]
]
