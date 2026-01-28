#!/bin/bash

export INSTALL_DIR=$(pwd)/tools/verilator

## check if tools/verilator already exists
if [ ! -d "$INSTALL_DIR" ]; then

    mkdir -p $INSTALL_DIR

    echo "Cloning Verilator source..."
    git clone https://github.com/verilator/verilator.git
    cd verilator
    git checkout stable

    echo "Building Verilator..."
    autoconf
    ./configure --prefix=$INSTALL_DIR
    make -j$(nproc)
    make install
    cd ..
    rm -rf verilator

fi

export PATH=$INSTALL_DIR/bin:$PATH