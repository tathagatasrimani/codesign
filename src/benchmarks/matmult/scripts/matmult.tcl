set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY { 
    {matmult}
}

go compile

source scripts/set_libraries.tcl

go libraries

directive set -CLOCKS $clocks 

go assembly

go architect

options set Architectural/DesignGoal latency

go allocate

go extract

project report -filename memories.rpt -memories true
project report -filename bom.rpt -bom true