import os
import sys
import json
import networkx as nx


DEBUG = False

def debug_print(message):
    if DEBUG:
        print(message)


all_modules_visited = set()

def parse_complist(complist_file_path):
    '''
        Converts a complist text file into a dictionary for easy access.
        Top-level keys are comp ids, values are dicts of all other data for each entry.
        Also parses the fcode field from <opcode=... fcode="..."/> and opset from <opset=...>.
    '''
    import re

    comps = {}
    with open(complist_file_path, 'r') as f:
        lines = f.readlines()

    comp_block = []
    inside_comp = False

    for line in lines:
        line = line.strip()
        if line.startswith('<comp '):
            inside_comp = True
            comp_block = [line]
        elif line.startswith('</comp>'):
            comp_block.append(line)
            # Parse the comp block
            comp_data = {}
            # Get comp id, class, name
            comp_header = comp_block[0]
            comp_id = re.search(r'id="(\d+)"', comp_header).group(1)
            comp_class = re.search(r'class="(\d+)"', comp_header)
            comp_name = re.search(r'name="([^"]+)"', comp_header)
            comp_data['class'] = comp_class.group(1) if comp_class else None
            comp_data['name'] = comp_name.group(1) if comp_name else None

            # Parse pin_list
            pins = []
            for pin_line in comp_block:
                if pin_line.startswith('<pin '):
                    pin = {}
                    for attr in ['id', 'dir', 'index', 'bw', 'slack']:
                        m = re.search(rf'{attr}="([^"]+)"', pin_line)
                        if m:
                            pin[attr] = m.group(1)
                    pins.append(pin)
            comp_data['pins'] = pins

            # Parse bind section
            bind_data = {}
            for bind_line in comp_block:
                if '<StgValue>' in bind_line:
                    ssdm_match = re.search(r'<ssdm name="([^"]+)"', bind_line)
                    if ssdm_match:
                        bind_data['ssdm_name'] = ssdm_match.group(1)
                    memport_match = re.search(r'<MemPortTyVec>([^<]+)</MemPortTyVec>', bind_line)
                    if memport_match:
                        bind_data['memport'] = memport_match.group(1).strip()
                # Parse opcode and fcode
                opcode_fcode_match = re.search(r'<opcode=[^>]*fcode="([^"]+)"', bind_line)
                if opcode_fcode_match:
                    bind_data['fcode'] = opcode_fcode_match.group(1)
                # Parse opset
                opset_match = re.search(r'<opset="([^"]+)"', bind_line)
                if opset_match:
                    bind_data['opset'] = opset_match.group(1)
            comp_data['bind'] = bind_data

            comps[int(comp_id)] = comp_data
            inside_comp = False
            comp_block = []
        elif inside_comp:
            comp_block.append(line)

    return comps


def parse_netlist(netlist_file_path):
    '''
        Parses the netlist file and converts it into a dictionary format.
        Top-level keys are net ids, values are dicts with src and sink info.
    '''
    import re

    netlist = {}
    with open(netlist_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Match lines like: <net id="51"><net_src comp="12" pin="0"/><net_sink comp="48" pin=0"/></net>
        net_match = re.match(
            r'<net id="(\d+)"><net_src comp="(\d+)" pin="(\d+)"/><net_sink comp="(\d+)" pin="?(\d+)"?/></net>',
            line
        )
        if net_match:
            net_id = int(net_match.group(1))
            src_comp = int(net_match.group(2))
            src_pin = int(net_match.group(3))
            sink_comp = int(net_match.group(4))
            sink_pin = int(net_match.group(5))
            netlist[net_id] = {
                'src': {'comp': src_comp, 'pin': src_pin},
                'sink': {'comp': sink_comp, 'pin': sink_pin}
            }
    return netlist

def create_networkX_netlist(json_netlist, json_complist, module_name):
    '''
        Takes in the json netlist and json complist and creates one unified NetworkX graph.
        The nodes are the components, and the edges are the nets connecting them.
        Each node will have its attributes from the complist.
        Each edge will have its net id and pin info.
    '''
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes for each component, with attributes from complist
    for comp_id, comp_data in json_complist.items():
        G.add_node(str(comp_id) + "_" + module_name, **comp_data)

    # Add edges for each net, connecting src comp to sink comp
    for net_id, net_data in json_netlist.items():
        src = str(net_data['src']['comp']) + "_" + module_name
        src_pin = net_data['src']['pin']
        sink = str(net_data['sink']['comp']) + "_" + module_name
        sink_pin = net_data['sink']['pin']
        G.add_edge(src, sink, net_id=net_id, src_pin=src_pin, sink_pin=sink_pin, weight=0)

    return G
    

def create_vitis_netlist(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        debug_print(f"Processing directory: {subdir_path}")

        # Find _netlist.rpt and _complist.rpt files and open them
        netlist_file = None
        complist_file = None
        for fname in os.listdir(subdir_path):
            if fname.endswith('_netlist.rpt'):
                netlist_file = os.path.join(subdir_path, fname)
            elif fname.endswith('_complist.rpt'):
                complist_file = os.path.join(subdir_path, fname)

        if not netlist_file or not complist_file:
            debug_print("ERROR: Complist or netlist not found!!")
            continue

        ## parse through the complist file and convert it to a dictionary
        complist = parse_complist(complist_file)

        ## parse through the netlist and convert it to a dictionary
        netlist = parse_netlist(netlist_file)

        debug_print("Now printing the netlist!")
        debug_print(netlist)

        # Get the prefix from the netlist file name (before '_netlist.rpt')
        netlist_prefix = os.path.basename(netlist_file).replace('_netlist.rpt', '')

        # Write the two structures out to files with the same prefix
        with open(f"{subdir_path}/{netlist_prefix}_complist.json", "w") as f:
            json.dump(complist, f, indent=2)

        with open(f"{subdir_path}/{netlist_prefix}_netlist.json", "w") as f:
            json.dump(netlist, f, indent=2)


        ## create the networkX graph:
        final_netlist = create_networkX_netlist(netlist, complist, subdir)

        # Write out the networkX graph to a gml file
        nx.write_gml(final_netlist, f"{subdir_path}/{netlist_prefix}_netlist.gml")


def main():
    create_vitis_netlist(os.path.join(os.path.dirname(__file__), "../tmp_for_test/benchmark/jacobi/solution1/.autopilot/db"))


if __name__ == "__main__":
    main()