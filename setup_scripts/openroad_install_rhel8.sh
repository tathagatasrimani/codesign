#!/bin/bash

export HOME="$(pwd)"
export PATH="$HOME/.local/bin:$(echo "$PATH")"
export CMAKE_PREFIX_PATH="$HOME/.local"

## make sure interfering packages are not installed
sudo rm -f /usr/local/lib64/libsoplex.a /usr/local/lib/libsoplex.a

cd openroad_interface/OpenROAD

sudo dnf install gcc-toolset-13
source /opt/rh/gcc-toolset-13/enable
which gcc
gcc --version

sudo dnf install qt5-qtcharts qt5-qtcharts-devel
ls /usr/lib64/libQt5Charts.so*

set +e
sudo ./etc/DependencyInstaller.sh -base 
status=$?
set -e

if [ $status -ne 0 ]; then
    echo "DependencyInstaller failed. Attempting manual pandoc install..."

    arch=amd64
    pandocVersion=3.1.11.1
    eval wget https://github.com/jgm/pandoc/releases/download/${pandocVersion}/pandoc-${pandocVersion}-linux-${arch}.tar.gz
    sudo tar xvzf pandoc-${pandocVersion}-linux-${arch}.tar.gz --strip-components 1 -C /usr/local/
    rm -rf pandoc-${pandocVersion}-linux-${arch}.tar.gz

fi

./etc/DependencyInstaller.sh -common -local


echo "\n\n\nOpenROAD dependencies installed successfully.\n\n\n"
echo "Installing OpenROAD..."

./etc/Build.sh

mkdir results

cd ..