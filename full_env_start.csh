#!/bin/tcsh

## run set_env_start.sh in bash to make sure everything is installed if needed
bash full_env_start.sh

## Now set up the csh environment variables

setenv PATH `pwd`/miniconda3/bin:$PATH
source miniconda3/etc/profile.d/conda.csh
conda activate codesign # activate the codesign environment

## verilator
set INSTALL_DIR=`pwd`/tools/verilator
setenv PATH ${INSTALL_DIR}/bin:${PATH}


################## PARSE UNIVERSITY ARGUMENT ##################
# get hostname
set host = `hostname`

if ("$host" =~ *stanford*) then
    setenv UNIVERSITY "stanford"

else if ("$host" =~ *cmu*) then
    setenv UNIVERSITY "cmu"

else
    echo "Hostname is '$host' — does not contain 'stanford' or 'cmu'."
    echo -n "Please pick your university (stanford/cmu): "
    set choice = $<
    switch ($choice)
        case "stanford":
        case "Stanford":
        case "STANFORD":
            setenv UNIVERSITY "stanford"
            breaksw

        case "cmu":
        case "CMU":
        case "Cmu":
            setenv UNIVERSITY "cmu"
            breaksw

        default:
            echo "Invalid choice. Exiting."
            exit 1
    endsw
endif

echo "UNIVERSITY set to: $UNIVERSITY"

######################################################################

# set home directory to codesign home directory
setenv HOME `pwd`


## source the appropriate cad setup scripts based on university
if ("$UNIVERSITY" == "stanford") then
    source stanford_cad_tool_setup.csh
else if ("$UNIVERSITY" == "cmu") then
    source cmu_cad_tool_setup.csh
endif




