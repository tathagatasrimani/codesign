graph [
  directed 1
  node [
    id 0
    label "Regs0"
    type "memory"
    function "Regs"
    in_use 0
    idx 0
    size 1
  ]
  node [
    id 1
    label "Regs1"
    type "memory"
    function "Regs"
    in_use 0
    idx 1
    size 1
  ]
  node [
    id 2
    label "Regs2"
    type "memory"
    function "Regs"
    in_use 0
    idx 2
    size 1
  ]
  node [
    id 3
    label "Regs3"
    type "memory"
    function "Regs"
    in_use 0
    idx 3
    size 1
  ]
  node [
    id 4
    label "Regs4"
    type "memory"
    function "Regs"
    in_use 0
    idx 4
    size 1
  ]
  node [
    id 5
    label "And0"
    type "pe"
    function "And"
    in_use 0
    idx 0
  ]
  node [
    id 6
    label "And1"
    type "pe"
    function "And"
    in_use 0
    idx 0
  ]
  node [
    id 7
    label "And2"
    type "pe"
    function "And"
    in_use 0
    idx 0
  ]
  node [
    id 8
    label "BitXor0"
    type "pe"
    function "BitXor"
    in_use 0
    idx 0
  ]
  node [
    id 9
    label "BitXor1"
    type "pe"
    function "BitXor"
    in_use 0
    idx 1
  ]
  node [
    id 10
    label "Buf0"
    type "mem"
    function "Buf"
    in_use 0
    size 22
    idx 0
  ]
  node [
    id 11
    label "Mem0"
    type "mem"
    function "MainMem"
    in_use 0
    idx 0
    size 256
  ]
  edge [
    source 0
    target 5
    cost 0.06
  ]
  edge [
    source 0
    target 6
    cost 0.94
  ]
  edge [
    source 0
    target 7
    cost 0.98
  ]
  edge [
    source 0
    target 8
    cost 0.06
  ]
  edge [
    source 0
    target 9
    cost 0.06
  ]
  edge [
    source 0
    target 10
  ]
  edge [
    source 1
    target 5
    cost 0.06
  ]
  edge [
    source 1
    target 6
    cost 0.94
  ]
  edge [
    source 1
    target 7
    cost 0.98
  ]
  edge [
    source 1
    target 8
    cost 0.06
  ]
  edge [
    source 1
    target 9
    cost 0.06
  ]
  edge [
    source 1
    target 10
  ]
  edge [
    source 2
    target 5
    cost 0.06
  ]
  edge [
    source 2
    target 6
    cost 0.94
  ]
  edge [
    source 2
    target 7
    cost 0.98
  ]
  edge [
    source 2
    target 8
    cost 0.06
  ]
  edge [
    source 2
    target 9
    cost 0.06
  ]
  edge [
    source 2
    target 10
  ]
  edge [
    source 3
    target 5
    cost 0.06
  ]
  edge [
    source 3
    target 6
    cost 0.94
  ]
  edge [
    source 3
    target 7
    cost 0.98
  ]
  edge [
    source 3
    target 8
    cost 0.06
  ]
  edge [
    source 3
    target 9
    cost 0.06
  ]
  edge [
    source 3
    target 10
  ]
  edge [
    source 4
    target 5
    cost 0.06
  ]
  edge [
    source 4
    target 6
    cost 0.94
  ]
  edge [
    source 4
    target 7
    cost 0.98
  ]
  edge [
    source 4
    target 8
    cost 0.06
  ]
  edge [
    source 4
    target 9
    cost 0.06
  ]
  edge [
    source 4
    target 10
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
    source 5
    target 2
    cost 2
  ]
  edge [
    source 5
    target 3
    cost 2
  ]
  edge [
    source 5
    target 4
    cost 2
  ]
  edge [
    source 5
    target 6
    cost 0.94
  ]
  edge [
    source 5
    target 7
    cost 0.98
  ]
  edge [
    source 5
    target 8
    cost 0.06
  ]
  edge [
    source 5
    target 9
    cost 0.06
  ]
  edge [
    source 6
    target 5
    cost 0.06
  ]
  edge [
    source 6
    target 0
    cost 2
  ]
  edge [
    source 6
    target 1
    cost 2
  ]
  edge [
    source 6
    target 2
    cost 2
  ]
  edge [
    source 6
    target 3
    cost 2
  ]
  edge [
    source 6
    target 4
    cost 2
  ]
  edge [
    source 6
    target 7
    cost 0.98
  ]
  edge [
    source 6
    target 8
    cost 0.06
  ]
  edge [
    source 6
    target 9
    cost 0.06
  ]
  edge [
    source 7
    target 5
    cost 0.06
  ]
  edge [
    source 7
    target 6
    cost 0.94
  ]
  edge [
    source 7
    target 0
    cost 2
  ]
  edge [
    source 7
    target 1
    cost 2
  ]
  edge [
    source 7
    target 2
    cost 2
  ]
  edge [
    source 7
    target 3
    cost 2
  ]
  edge [
    source 7
    target 4
    cost 2
  ]
  edge [
    source 7
    target 8
    cost 0.06
  ]
  edge [
    source 7
    target 9
    cost 0.06
  ]
  edge [
    source 8
    target 5
    cost 0.06
  ]
  edge [
    source 8
    target 6
    cost 0.94
  ]
  edge [
    source 8
    target 7
    cost 0.98
  ]
  edge [
    source 8
    target 0
    cost 2
  ]
  edge [
    source 8
    target 1
    cost 2
  ]
  edge [
    source 8
    target 2
    cost 2
  ]
  edge [
    source 8
    target 3
    cost 2
  ]
  edge [
    source 8
    target 4
    cost 2
  ]
  edge [
    source 8
    target 9
    cost 0.06
  ]
  edge [
    source 9
    target 5
    cost 0.06
  ]
  edge [
    source 9
    target 6
    cost 0.94
  ]
  edge [
    source 9
    target 7
    cost 0.98
  ]
  edge [
    source 9
    target 8
    cost 0.06
  ]
  edge [
    source 9
    target 0
    cost 2
  ]
  edge [
    source 9
    target 1
    cost 2
  ]
  edge [
    source 9
    target 2
    cost 2
  ]
  edge [
    source 9
    target 3
    cost 2
  ]
  edge [
    source 9
    target 4
    cost 2
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
    target 2
  ]
  edge [
    source 10
    target 3
  ]
  edge [
    source 10
    target 4
  ]
  edge [
    source 10
    target 11
  ]
  edge [
    source 11
    target 10
  ]
]
