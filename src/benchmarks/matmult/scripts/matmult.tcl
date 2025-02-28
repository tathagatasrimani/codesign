set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY { 
    {MatMult}
}

go compile

source scripts/set_libraries.tcl

go libraries

directive set -CLOCKS $clocks 

go assembly

go architect

go allocate

go extract

solution timing -filename conv_timing.rpt -count 1000000
project report -filename memories.rpt -memories true