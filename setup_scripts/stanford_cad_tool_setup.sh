## Catapult
export MGLS_LICENSE_FILE=1717@cadlic0.stanford.edu
export PATH="${PATH}:/cad/mentor/2024.2/Mgc_home/bin/"

## Initialize module system for bash
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if [[ ! -e /usr/lib64/libtinfo.so.5 && ! -e /lib64/libtinfo.so.5 ]]; then
  echo "[INFO] libtinfo.so.5 missing — installing ncurses-compat-libs"
  sudo dnf install -y ncurses-compat-libs
fi

## vitis:
module load vitis/2022.1