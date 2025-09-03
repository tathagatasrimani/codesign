set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY "
    {WeightDoubleBuffer<8192, ${ARRAY_DIMENSION}, ${ARRAY_DIMENSION}>} 
"

go compile

source scripts/set_libraries.tcl

go libraries
directive set -CLOCKS $clocks

go assembly

# -------------------------------
# Set the correct word widths and the stage replication
# Your code starts here
# -------------------------------
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/WeightDoubleBufferReader<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/din -WORD_WIDTH [expr ${ARRAY_DIMENSION} * 8]
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/WeightDoubleBufferWriter<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/dout -WORD_WIDTH [expr ${ARRAY_DIMENSION} * 8]
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/mem:cns -STAGE_REPLICATION 2
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/mem -WORD_WIDTH [expr ${ARRAY_DIMENSION} * 8]
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/WeightDoubleBufferWriter<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/tmp.data.value -WORD_WIDTH [expr ${ARRAY_DIMENSION} * 8]
directive set /WeightDoubleBuffer<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/WeightDoubleBufferReader<8192,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/tmp.data.value -WORD_WIDTH [expr ${ARRAY_DIMENSION} * 8]
# -------------------------------
# Your code ends here
# -------------------------------

go architect

go allocate
go extract
