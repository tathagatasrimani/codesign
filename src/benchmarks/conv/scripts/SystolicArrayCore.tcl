set blockname [file rootname [file tail [info script] ]]

source scripts/common.tcl

directive set -DESIGN_HIERARCHY "
    {SystolicArrayCore<IDTYPE, WDTYPE, ODTYPE, ${ARRAY_DIMENSION}, ${ARRAY_DIMENSION}>}
"

go compile

source scripts/set_libraries.tcl

solution library add {[CCORE] ProcessingElement<IDTYPE,ODTYPE>.v1}

go libraries
directive set -CLOCKS $clocks
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/ProcessingElement<IDTYPE,ODTYPE> -MAP_TO_MODULE {[CCORE] ProcessingElement<IDTYPE,ODTYPE>.v1}

go assembly

directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run -DESIGN_GOAL Latency
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run -CLOCK_OVERHEAD 0.000000

# -------------------------------
# Make sure that the accumulation buffer has the appropriate interleaving and block size
# Your code starts here
# -------------------------------
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/accumulation_buffer:rsc -INTERLEAVE ${ARRAY_DIMENSION}
# directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/accumulation_buffer:rsc -BLOCK_SIZE 256

# -------------------------------
# Your code ends here
# -------------------------------


# -------------------------------
# Map the input register, partial sum register, and weight register to registers and not memories, additionally set register threshold to 4096 (prevents unecessary registers in design) 
# Your code starts here
# -------------------------------
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}> -REGISTER_THRESHOLD 4096
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/input_reg:rsc -MAP_TO_MODULE {[Register]}
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/psum_reg:rsc -MAP_TO_MODULE {[Register]}
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/weight_reg:rsc -MAP_TO_MODULE {[Register]}
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/input_reg2:rsc -MAP_TO_MODULE {[Register]}
directive set /SystolicArrayCore<IDTYPE,WDTYPE,ODTYPE,${ARRAY_DIMENSION},${ARRAY_DIMENSION}>/run/psum_reg2:rsc -MAP_TO_MODULE {[Register]}
# -------------------------------
# Your code ends here
# -------------------------------

go architect

# -------------------------------
# If you try to schedule with an initiation interval of 1, you might run into a dependency error involving the accumulation buffer
# To solve this, you need to use 'ignore_memory_precedences', Iterate on the design to ignore memory precedences that show up (make sure there is no real dependency)
# Your code starts here
# -------------------------------
ignore_memory_precedences -from INNER_LOOP:if#3:for:write_mem(accumulation_buffer:rsc(0)(0).@) -to INNER_LOOP:if#2:else:for:read_mem(accumulation_buffer:rsc(0)(0).@)
for {set i 1} {$i < ${ARRAY_DIMENSION}} {incr i} {
    ignore_memory_precedences -from INNER_LOOP:if#3:for:write_mem(accumulation_buffer:rsc(0)($i).@)#$i -to INNER_LOOP:if#2:else:for:read_mem(accumulation_buffer:rsc(0)($i).@)#$i
}

# -------------------------------
# Your code ends here
# -------------------------------

go allocate
go extract
