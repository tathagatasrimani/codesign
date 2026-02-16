import os
import sys
import json
import networkx as nx
import logging
from src import sim_util

logger = logging.getLogger(__name__)

## Enables additional logging
DEBUG = False

def debug_print(message):
    if DEBUG:
        logger.info(message)


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
        debug_print(f"Component {comp_id} compdata: {comp_data}")
        if "bind" in comp_data and "fcode" in comp_data["bind"] and comp_data["bind"]["fcode"] == "call":
            debug_print(f"Component {comp_id} is a call, fcode: {comp_data['bind']['fcode']}")
            for func_name in sim_util.get_module_map().keys():
                if comp_data["name"].find(func_name) != -1:
                    comp_data["bind"]["fcode"] = func_name
                    break
        G.add_node(str(comp_id) + "_" + module_name, **comp_data, module=module_name)

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

        #if subdir in sim_util.get_module_map().keys():
        #    debug_print(f"Skipping subdirectory {subdir} because it is a blackbox.")
        #    continue

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
            logger.error("ERROR: Complist or netlist not found!!")
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

        debug_print(f"Writing netlist to {subdir_path}/{netlist_prefix}_netlist.gml")

        # Write out the networkX graph to a gml file
        nx.write_gml(final_netlist, f"{subdir_path}/{netlist_prefix}_netlist.gml")


def extract_module_dependencies(root_dir):
    """
    Extract submodule dependency info from FSM data and write .verbose_modules.json
    files for each module. These files are consumed by MergeNetlistsVitis to know
    which submodules to recursively merge.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        subdir_name = subdir.split('/')[-1]

        # Blackboxed modules get an empty dependency list
        if subdir_name in sim_util.get_module_map().keys():
            module_path = os.path.join(subdir_path, f"{subdir}.verbose_modules.json")
            with open(module_path, 'w') as mf:
                json.dump([], mf, indent=4)
            debug_print(f"Skipping {subdir} (blackbox), wrote empty modules file.")
            continue

        # Find the FSM file
        fsm_file = None
        for fname in os.listdir(subdir_path):
            if fname.endswith('_fsm.json'):
                fsm_file = os.path.join(subdir_path, fname)
                break

        if not fsm_file:
            continue

        with open(fsm_file, 'r') as f:
            fsm_data = json.load(f)

        # Scan FSM data for 'call' operators to find instantiated submodules
        instantiated_modules = []
        for state_ops in fsm_data.values():
            for fsm_node in state_ops:
                if fsm_node['operator'] == 'call':
                    if 'function' not in fsm_node:
                        module_name = fsm_node['sources'][0]['source']
                        module_name = module_name.split('@')[1].split('<')[0] + "_" + module_name.split('<')[1].split('>')[0] + "_s"
                    else:
                        module_name = fsm_node['function']
                    if module_name not in instantiated_modules:
                        instantiated_modules.append(module_name)

        module_file = fsm_file.replace('_fsm.json', '_modules.json')
        with open(module_file, 'w') as mf:
            json.dump(instantiated_modules, mf, indent=4)
        debug_print(f"Extracted {len(instantiated_modules)} module dependencies for {subdir}.")


def create_physical_design_netlist(parse_results_dir, top_module_name, allowed_functions):
    """
    Create a modified netlist for physical design where memory/FIFO infrastructure
    nodes (port declarations, alloca, addr_gep, addr_reg) are unified into single
    nodes carrying metadata from memory_mapping.json.

    Access nodes (load/store/read/write) remain as separate functional units,
    connected to the unified memory nodes via redirected edges.

    Reads: {top_module}_full_netlist_unfiltered.gml, memory_mapping.json
    Writes: {top_module}_physical_netlist.gml
    """
    netlist_path = os.path.join(parse_results_dir, f"{top_module_name}_full_netlist_unfiltered.gml")
    mapping_path = os.path.join(parse_results_dir, "memory_mapping.json")

    if not os.path.exists(netlist_path) or not os.path.exists(mapping_path):
        logger.warning("Skipping physical design netlist: missing netlist or memory_mapping.json")
        return

    G = nx.read_gml(netlist_path)

    with open(mapping_path, 'r') as f:
        mem_mapping = json.load(f)

    flattened = mem_mapping.get("flattened", {})
    if not flattened:
        logger.warning("No flattened memory info found in memory_mapping.json")
        return

    # Build mapping: flattened memory name -> set of all related port names
    # (child port names may differ from parent, e.g. v785 -> parent_fifo v78)
    mem_port_names = {mem_name: {mem_name} for mem_name in flattened}

    for module_name, module_data in mem_mapping.items():
        if module_name == "flattened":
            continue
        for port_type in ('memory_ports', 'fifo_ports'):
            ports = module_data.get(port_type, {})
            if not ports:
                continue
            for port_name, port_info in ports.items():
                parent = port_info.get('parent_ram') or port_info.get('parent_fifo')
                child = port_info.get('child_port', port_name)
                if parent and parent in mem_port_names:
                    mem_port_names[parent].add(port_name)
                    mem_port_names[parent].add(child)
                elif child in mem_port_names:
                    mem_port_names[child].add(port_name)

    # For each flattened memory, find infrastructure nodes and unify them
    for mem_name, mem_info in flattened.items():
        all_port_names = mem_port_names.get(mem_name, {mem_name})
        related_nodes = set()

        for node, data in list(G.nodes(data=True)):
            node_name = data.get('name', '')
            bind = data.get('bind', {})
            ssdm_name = bind.get('ssdm_name', '') if isinstance(bind, dict) else ''

            # Class 1000 port/memory declaration nodes
            if node_name in all_port_names or ssdm_name in all_port_names:
                related_nodes.add(node)
                continue

            # Alloca, addr_gep, addr_reg nodes identified by name prefix
            for port_name in all_port_names:
                if (node_name.startswith(f"{port_name}_alloca") or
                    node_name.startswith(f"{port_name}_addr_gep") or
                    node_name.startswith(f"{port_name}_addr_reg")):
                    related_nodes.add(node)
                    break

        if not related_nodes:
            continue

        # Create unified memory/fifo node
        unified_id = f"mem_{mem_name}"
        mem_type = mem_info.get('type', 'unknown')
        func = 'fifo' if mem_type == 'fifo' else 'memory'

        G.add_node(unified_id,
                   name=mem_name,
                   function=func,
                   memory_type=mem_type,
                   width=mem_info.get('width', 0),
                   depth=mem_info.get('depth', 0),
                   total_size=mem_info.get('total_size', 0))

        # Redirect edges from/to related nodes to the unified node
        for node in related_nodes:
            for pred in list(G.predecessors(node)):
                if pred not in related_nodes and not G.has_edge(pred, unified_id):
                    edge_data = G.get_edge_data(pred, node)
                    G.add_edge(pred, unified_id, **(edge_data if edge_data else {'weight': 0}))
            for succ in list(G.successors(node)):
                if succ not in related_nodes and not G.has_edge(unified_id, succ):
                    edge_data = G.get_edge_data(node, succ)
                    G.add_edge(unified_id, succ, **(edge_data if edge_data else {'weight': 0}))

        G.remove_nodes_from(related_nodes)

    # Merge register nodes with their associated load/store nodes.
    # Each register (class 1005, _reg_ in name) forms a cluster with:
    #   - Load nodes that read from it (name: {var_name}_load_*)
    #   - Phi nodes (name: {var_name}_phi_*)
    #   - Directly connected store/load neighbors (fcode "store" or "load")
    # Also builds register_mapping: module -> var_name -> {width} for schedule annotation.
    register_mapping = {}
    already_merged = set()
    for node in list(G.nodes()):
        if node in already_merged:
            continue
        data = G.nodes[node]
        node_name = data.get('name', '')
        if '_reg_' not in node_name or str(data.get('class', '')) != '1005':
            continue

        var_name = node_name.split('_reg_')[0]
        node_module = data.get('module', '')
        related_nodes = {node}

        # Find load and phi nodes by variable name prefix within the same module
        for n, d in list(G.nodes(data=True)):
            if n in already_merged:
                continue
            n_name = d.get('name', '')
            if d.get('module', '') != node_module:
                continue
            if (n_name.startswith(f"{var_name}_load") or
                n_name.startswith(f"{var_name}_phi")):
                related_nodes.add(n)

        # Find directly connected store/load nodes in the same module
        for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
            if neighbor in already_merged or neighbor in related_nodes:
                continue
            n_data = G.nodes.get(neighbor)
            if n_data is None or n_data.get('module', '') != node_module:
                continue
            bind = n_data.get('bind', {})
            fcode = bind.get('fcode', '') if isinstance(bind, dict) else ''
            if fcode in ('store', 'load'):
                related_nodes.add(neighbor)

        # Extract bitwidth from register output pin
        bw = 0
        pins = data.get('pins', [])
        if isinstance(pins, list):
            for pin in pins:
                if isinstance(pin, dict) and pin.get('dir') == '1':
                    try:
                        bw = int(pin.get('bw', 0))
                    except (ValueError, TypeError):
                        pass
                    break

        # Record in register_mapping: maps each merged node's name to its register
        if node_module not in register_mapping:
            register_mapping[node_module] = {}
        register_mapping[node_module][var_name] = {"width": bw}
        # Also map the opset-derived operation names from merged nodes
        for rn in related_nodes:
            rn_data = G.nodes[rn]
            rn_bind = rn_data.get('bind', {})
            if isinstance(rn_bind, dict):
                opset = rn_bind.get('opset', '')
                if opset:
                    for op_entry in opset.split():
                        op_name = op_entry.split('/')[0].strip()
                        if op_name:
                            register_mapping[node_module][op_name] = {"width": bw, "register": var_name}

        unified_id = f"reg_{var_name}_{node_module}"
        G.add_node(unified_id,
                   name=var_name,
                   function='Register16',
                   module=node_module,
                   width=bw)

        for rn in related_nodes:
            for pred in list(G.predecessors(rn)):
                if pred not in related_nodes and not G.has_edge(pred, unified_id):
                    edge_data = G.get_edge_data(pred, rn)
                    G.add_edge(pred, unified_id, **(edge_data if edge_data else {'weight': 0}))
            for succ in list(G.successors(rn)):
                if succ not in related_nodes and not G.has_edge(unified_id, succ):
                    edge_data = G.get_edge_data(rn, succ)
                    G.add_edge(unified_id, succ, **(edge_data if edge_data else {'weight': 0}))

        G.remove_nodes_from(related_nodes)
        already_merged.update(related_nodes)

    # Save register mapping for use by schedule parser
    reg_mapping_path = os.path.join(parse_results_dir, "register_mapping.json")
    with open(reg_mapping_path, 'w') as f:
        json.dump(register_mapping, f, indent=2)

    output_path = os.path.join(parse_results_dir, f"{top_module_name}_physical_netlist.gml")
    nx.write_gml(G, output_path)
    physical_netlist_filtered = sim_util.filter_graph_by_function(G, allowed_functions)
    nx.write_gml(physical_netlist_filtered, os.path.join(parse_results_dir, f"{top_module_name}_physical_netlist_filtered.gml"))
    logger.info(f"Physical design netlist saved to {output_path}")


def main():
    create_vitis_netlist(os.path.join(os.path.dirname(__file__), "../tmp_for_test/benchmark/jacobi/solution1/.autopilot/db"))


if __name__ == "__main__":
    main()