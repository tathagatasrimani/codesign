# Check if test already exists
if {[file isdirectory test]} { 
    project load test
} else {
    project new -name test
    project save
}

options set Input/TargetPlatform x86_64
options set Input/SearchPath ./src
options set Output/OutputVHDL false
options set Architectural/DefaultRegisterThreshold 4096

flow package require /SCVerify
flow package option set /SCVerify/USE_CCS_BLOCK true
flow package option set /SCVerify/USE_NCSIM true
flow package option set /SCVerify/USE_VCS false
flow package option set /SCVerify/USE_MSIM false

flow package require /NCSim

solution options set Flows/NCSim/NC_ROOT /cad/cadence/INCISIVE15.20.022/

set clk_period 5.0
set clocks "clk \"-CLOCK_PERIOD $clk_period -CLOCK_EDGE rising -CLOCK_HIGH_TIME [expr $clk_period/2] -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND async -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high\" "

go new
solution file add ./src/Conv.cpp
solution file add ./src/ConvTb.cpp -exclude true

go analyze

directive set -DESIGN_HIERARCHY { 
    {Conv}
}

go compile
