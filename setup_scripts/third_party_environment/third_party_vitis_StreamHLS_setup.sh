## This script sets up the environment for Vitis HLS to be used with StreamHLS preprocessing tools.
##
## Third-party users: edit the lines below to match your institution's Vitis HLS installation.
##
## Option A: Source the Vitis HLS settings script directly.
##   Provide the full path to your Vitis HLS 2023.2 (or compatible) settings64.sh.
##   Example (direct path):
##     source /path/to/Xilinx/Vitis_HLS/2023.2/settings64.sh
##
## Option B: Use your institution's module system.
##   Example (environment modules):
##     source /etc/profile.d/modules.sh
##     module load vitis/2024.2
##
## Uncomment and adapt ONE of the options below, then delete the error line.

echo "ERROR: third_party_vitis_StreamHLS_setup.sh has not been configured." >&2
echo "Please edit codesign/setup_scripts/third_party_environment/third_party_vitis_StreamHLS_setup.sh" >&2
echo "to source your institution's Vitis HLS installation for StreamHLS." >&2
exit 1

## --- Option A: direct path ---
# source /path/to/Xilinx/Vitis_HLS/2023.2/settings64.sh

## --- Option B: environment modules ---
# source /etc/profile.d/modules.sh
# module load vitis/2024.2
