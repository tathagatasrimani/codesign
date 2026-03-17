
import os
from collections import defaultdict
import copy
import json


import networkx as nx
import re
import xml.etree.ElementTree as ET

# misc testing file for netlist/gml parsing specifics
def test():
    netlist_file = "/scratch/gyanepss/codesign/misc_test/verbose_netlist.gml"
    netlist = nx.read_gml(netlist_file)
    netlist_op_dest_to_node = {}
    for n, d in netlist.nodes(data=True):
        ## extract name and bind->opset fields
        name = d.get('name')
        bind = d.get('bind', {})
        opset = bind.get('opset')

        # also get bitwidth for module mapping
        # note multiple pins for i/o 
        pins = d.get('pins', {})
        # bitwidth = pins.get('bw')
        print(f"node {n} has name {name}, opset {opset}, and pins {pins}")

        ## remove the slash and everything after it from the opset
        if not opset:
            #print(f"opset is None for {n},{d}")
            continue
        if '/' in opset:
            opsets = opset.split()
            for opset in opsets:
                op = opset.split('/')[0]
                if op in netlist_op_dest_to_node:
                    print(f"op {op} already exists in netlist_op_dest_to_node, skipping")
                    continue
                netlist_op_dest_to_node[op] = name
                print(f"mapping opset {op} to name {name}")
        else:
            op = opset.strip()
            if op in netlist_op_dest_to_node:
                print(f"op {op} already exists in netlist_op_dest_to_node, skipping")
                continue
            print(f"mapping opset {op} to name {name}")
            netlist_op_dest_to_node[op] = name
    print(f"logging netlist_op_dest_to_node")
    for op, node in netlist_op_dest_to_node.items():
        print(f"op: {op}, node: {node}")
    return netlist_op_dest_to_node

test()