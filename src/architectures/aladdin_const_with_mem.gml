graph [
  directed 1
  node [
    id 0
    label "Add0"
    type "pe"
    function "Add"
    in_use 0
    idx 0
  ]
  node [
    id 1
    label "Regs0"
    type "memory"
    function "Regs"
    in_use 0
    size 1
    idx 0
  ]
  node [
    id 2
    label "Regs1"
    type "memory"
    function "Regs"
    in_use 0
    size 1
    idx 1
  ]
  node [
    id 3
    label "Regs2"
    type "memory"
    function "Regs"
    in_use 0
    size 1
    idx 2
  ]
  node [
    id 4
    label "Mult0"
    type "pe"
    function "Mult"
    in_use 0
    idx 0
  ]
  node [
    id 5
    label "Eq0"
    type "pe"
    function "Eq"
    in_use 0
    idx 0
  ]
  node [
    id 6
    label "Buf0"
    type "mem"
    function "Buf"
    in_use 0
    idx 0
    size 22
  ]
  node [
    id 7
    label "Buf1"
    type "mem"
    function "Buf"
    in_use 0
    idx 1
    size 22
  ]
  node [
    id 8
    label "Mem0"
    type "mem"
    function "MainMem"
    in_use 0
    idx 0
    size 1024
  ]
  node [
    id 9
    label "Regs3"
    type "memory"
    function "Regs"
    in_use 0
    size 1
    idx 3
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 2
    target 1
  ]

  edge [
    source 3
    target 9
  ]
  edge [
    source 9
    target 3
  ]
  edge [
    source 7
    target 9
  ]
  edge [
    source 9
    target 7
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
    target 3
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
    target 0
  ]
  edge [
    source 1
    target 4
  ]
  edge [
    source 1
    target 5
  ]
  edge [
    source 1
    target 7
  ]
  edge [
    source 2
    target 0
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
    source 2
    target 7
  ]
  edge [
    source 3
    target 0
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
    source 3
    target 7
  ]
  edge [
    source 4
    target 0
  ]
  edge [
    source 4
    target 1
  ]
  edge [
    source 4
    target 2
  ]
  edge [
    source 4
    target 3
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 5
    target 0
  ]
  edge [
    source 5
    target 4
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
    source 5
    target 3
  ]
  edge [
    source 7
    target 1
  ]
  edge [
    source 7
    target 8
  ]
  edge [
    source 7
    target 2
  ]
  edge [
    source 7
    target 3
  ]
  edge [
    source 8
    target 7
  ]
  edge [
    source 9
    target 0
  ]
  edge [
    source 9
    target 4
  ]
  edge [
    source 9
    target 5
  ]
  edge [
    source 0
    target 9
  ]
  edge [
    source 4
    target 9
  ]
  edge [
    source 5
    target 9
  ]
]
