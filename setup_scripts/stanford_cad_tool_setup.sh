## Catapult
export MGLS_LICENSE_FILE=1717@cadlic0.stanford.edu
export PATH="${PATH}:/cad/mentor/2024.2/Mgc_home/bin/"

## Initialize module system for bash
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

## vitis:
module load vitis/2023.2