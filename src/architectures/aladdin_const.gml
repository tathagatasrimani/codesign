graph [
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
    idx 0
  ]
  node [
    id 2
    label "Regs1"
    type "memory"
    function "Regs"
    in_use 0
    idx 1
  ]
  node [
    id 3
    label "Regs2"
    type "memory"
    function "Regs"
    in_use 0
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
    source 1
    target 4
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
]
