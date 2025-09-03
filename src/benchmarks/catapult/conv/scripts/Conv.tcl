set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY { 
    {Conv}
}

go compile

source scripts/set_libraries.tcl


solution library add "\[Block\] InputDoubleBuffer<4096,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"
solution library add "\[Block\] WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"
solution library add "\[Block\] SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"

go libraries
directive set -CLOCKS $clocks 

directive set /Conv/SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}> -MAP_TO_MODULE "\[Block\] SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"
directive set /Conv/InputDoubleBuffer<4096,${ARRAY_DIMENSION},${ARRAY_DIMENSION}> -MAP_TO_MODULE "\[Block\] InputDoubleBuffer<4096,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"
directive set /Conv/WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}> -MAP_TO_MODULE "\[Block\] WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>.v1"

directive set /Conv -FIFO_DEPTH 3
directive set /Conv/systolicArray -FIFO_DEPTH 3


directive set /Conv/outputSerializer/buffer:rsc -INTERLEAVE ${ARRAY_DIMENSION}
# directive set /Conv/outputSerializer/buffer:rsc -BLOCK_SIZE 256


go assembly

directive set /Conv/SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}> -FIFO_DEPTH 3

go architect

go allocate

go extract

solution timing -filename conv_timing.rpt -count 1000000
project report -memories