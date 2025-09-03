#!/bin/bash
# This script converts the output rtl from catapult to the graph netlist format for all C-Cores. 

# $1 is the input verilog file
# $2 is the output graph netlist file

yosys -p "read_verilog $1; write_json netlist.json"

python3 verilog_parse.py netlist.json $2