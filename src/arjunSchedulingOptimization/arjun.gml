graph [
  directed 1
  node [
    id 0
    label "b_1[0][0];37"
    function "Regs"
    idx "37"
    cost 2
  ]
  node [
    id 1
    label "a_1[0][0];36"
    function "Regs"
    idx "36"
    cost 2
  ]
  node [
    id 2
    label "c_1[0][0];39"
    function "Regs"
    idx "39"
    cost 2
  ]
  node [
    id 3
    label "*;38"
    function "Mult"
    idx "38"
    cost 0.98
  ]
  node [
    id 4
    label "+;40"
    function "Add"
    idx "40"
    cost 0.94
  ]
  node [
    id 5
    label "c_1[0][0];41"
    function "Regs"
    idx "44"
    cost 2
  ]
  node [
    id 6
    label "d_1[0];43"
    function "Regs"
    idx "43"
    cost 2
  ]
  node [
    id 7
    label "+;45"
    function "Add"
    idx "45"
    cost 0.94
  ]
  node [
    id 8
    label "c_1[0][0];46"
    function "Regs"
    idx "46"
    cost 2
  ]
  node [
    id 9
    label "end"
    function "end"
  ]
  edge [
    source 0
    target 3
  ]
  edge [
    source 1
    target 3
  ]
  edge [
    source 2
    target 4
  ]
  edge [
    source 3
    target 4
  ]
  edge [
    source 4
    target 5
  ]
  edge [
    source 5
    target 7
  ]
  edge [
    source 6
    target 7
  ]
  edge [
    source 7
    target 8
  ]
  edge [
    source 8
    target 9
    weight 2
  ]
]
