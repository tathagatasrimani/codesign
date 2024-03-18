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
    label "Add0"
    type "pe"
    function "Add"
    in_use 0
    idx 0
  ]
  node [
    id 2
    label "Mult0"
    type "pe"
    function "Mult"
    in_use 0
    idx 0
  ]
  edge [
    source 0
    target 1
    cost 0.94
  ]
  edge [
    source 0
    target 2
    cost 0.98
  ]
  edge [
    source 1
    target 0
    cost 2
  ]
  edge [
    source 1
    target 2
    cost 0.98
  ]
  edge [
    source 2
    target 1
    cost 0.94
  ]
  edge [
    source 2
    target 0
    cost 2
  ]
]
