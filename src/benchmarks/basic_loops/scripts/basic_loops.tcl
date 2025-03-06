set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY { 
    {BasicLoops}
}

go compile

source scripts/set_libraries.tcl

go libraries

directive set -CLOCKS $clocks 

go assembly

go architect

go allocate

go extract

project report -filename memories.rpt -memories true