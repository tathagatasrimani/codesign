import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)

def create_cdfg_to_netlist_mapping_vitis(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        

        #print(f"Processing directory: {subdir_path}")

        # Find _netlist.gml and _verbose_cdfg.gml files
        netlist_file = None
        verbose_cdfg_file = None
        for fname in os.listdir(subdir_path):
            if fname.endswith('verbose_netlist.gml'):
                netlist_file = os.path.join(subdir_path, fname)
            elif fname.endswith('verbose_cdfg.gml'):
                verbose_cdfg_file = os.path.join(subdir_path, fname)

        if not netlist_file or not verbose_cdfg_file:
            debug_print("ERROR: verbose CDFG or netlist not found!!")
            exit(1)
            continue

        ## read in the netlist and verbose CDFG files
        netlist = nx.read_gml(netlist_file)
        verbose_cdfg = nx.read_gml(verbose_cdfg_file)

        ## create a mapping from the CDFG nodes to the netlist nodes
        ## we will embed this mapping in the verbose CDFG nodes

        # first create a mapping from operation destination name to node in the netlist
        netlist_op_dest_to_node = {}
        for n, d in netlist.nodes(data=True):
            ## extract name and bind->opset fields
            name = d.get('name')
            bind = d.get('bind', {})
            opset = bind.get('opset')
            ## remove the slash and everything after it from the opset
            if opset and '/' in opset:
                opset = opset.split('/')[0]
            if name and opset:
                netlist_op_dest_to_node[opset] = name


        # now go through the verbose CDFG nodes and find the corresponding netlist node
        for n, d in verbose_cdfg.nodes(data=True):
            ## get the fsm_node->destination field
            fsm_node = d.get('fsm_node', {})
            dest = fsm_node.get('destination')
            ## remove the leading % from the destination
            if dest and dest.startswith('%'):
                dest = dest[1:]
            if dest and dest in netlist_op_dest_to_node:
                # we have a match, embed the netlist node name in the verbose CDFG node
                d['netlist_node'] = netlist_op_dest_to_node[dest]
            else:
                d['netlist_node'] = "NONE"

            

        # write out the updated verbose CDFG with the embedded mapping to overwrite the original file
        nx.write_gml(verbose_cdfg, verbose_cdfg_file)


def main(root_dir):
    create_cdfg_to_netlist_mapping_vitis(root_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_cdfg.py <root_directory>")
        sys.exit(1)
    main(sys.argv[1])