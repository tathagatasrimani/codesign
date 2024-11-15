import os
import logging
import glob
import subprocess
import networkx as nx
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

def get_netlist_counts(netlist):
    counts = {}
    for node_id, node_data in netlist.nodes(data=True):
        func = node_data.get('function')
        if func:
            counts[func] = counts.get(func, 0) + 1
    return counts

def adjust_component_count(netlist, component, target_count):
    nodes_of_component = [
        (node_id, node_data)
        for node_id, node_data in netlist.nodes(data=True)
        if node_data.get('function') == component
    ]
    current_count = len(nodes_of_component)

    if current_count == target_count and component in ['Buf', 'Mem']:
        # Duplicate one of the nodes
        node_id_to_duplicate, node_data_to_duplicate = nodes_of_component[0]

        # Extract numeric IDs and find the maximum
        def extract_id(node_id):
            match = re.search(r'(\d+)$', node_id)
            return int(match.group(1)) if match else -1

        max_node_id = max(
            extract_id(node_id)
            for node_id in netlist.nodes()
            if node_id.startswith(component)
        )
        new_node_idx = max_node_id + 1
        new_node_id = f"{component}{new_node_idx}"

        new_node_data = node_data_to_duplicate.copy()
        new_node_data['id'] = new_node_id
        new_node_data['label'] = f"{component}{new_node_idx}"
        netlist.add_node(new_node_id, **new_node_data)
        
        # Duplicate edges for the new node
        for source, target, edge_data in netlist.edges(node_id_to_duplicate, data=True):
            if source == node_id_to_duplicate:
                netlist.add_edge(new_node_id, target, **edge_data)
            elif target == node_id_to_duplicate:
                netlist.add_edge(source, new_node_id, **edge_data)

    elif current_count > target_count:
        # Remove extra nodes
        nodes_to_remove = [node_id for node_id, _ in nodes_of_component[target_count:]]
        netlist.remove_nodes_from(nodes_to_remove)

    elif current_count < target_count:
        # Add missing nodes
        if nodes_of_component:
            component_type = nodes_of_component[0][1].get('type', 'pe')
        else:
            component_type = 'pe'  # Default type
        max_idx = max(
            [int(node_data.get('idx', -1)) for _, node_data in nodes_of_component] + [-1]
        )
        for j in range(target_count - current_count):
            new_idx = max_idx + j + 1
            new_node_id = f"{component}{new_idx}"
            new_node_data = {
                'id': new_node_id,
                'label': f"{component}{new_idx}",
                'function': component,
                'type': component_type,
                'in_use': 0,
                'idx': new_idx
            }
            netlist.add_node(new_node_id, **new_node_data)

    return netlist

def read_edp_from_output(output):
    for line in output.split('\n'):
        if 'MERP EDP:' in line:
            edp_value_str = line.split('EDP:')[1].split('E-18 Js')[0].strip()
            return float(edp_value_str)
    raise ValueError("EDP value not found in simulate.sh output")

def run_simulation(netlist_file):
    cmd = [
        "sh",
        os.path.join(current_dir, "src", "simulate.sh"),
        "-n", "matmult",
        "-c", netlist_file,
        "-p", "none"
    ]
    print(f'Running simulation: {cmd}')
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
    return read_edp_from_output(result.stdout)

def process_component(args):
    component, netlist_n_counts, netlist_n1_counts, netlist_n1_file, original_edp, netlist_pair_str, csv_lock = args
    target_count = netlist_n_counts.get(component, 0)
    current_count = netlist_n1_counts.get(component, 0)

    # Initialize edp_diff
    edp_diff = -1  # Default to -1 in case of error

    try:
        # Skip processing if counts are equal and component is not 'Buf' or 'Mem'
        if current_count == target_count and component not in ['Buf', 'Mem']:
            edp_diff = 0
        else:
            netlist_n1 = nx.read_gml(netlist_n1_file)
            modified_netlist = netlist_n1.copy()
            modified_netlist = adjust_component_count(modified_netlist, component, target_count)

            if component not in netlist_n_counts:
                # Remove component entirely
                nodes_to_remove = [
                    node_id for node_id, node_data in modified_netlist.nodes(data=True)
                    if node_data.get('function') == component
                ]
                modified_netlist.remove_nodes_from(nodes_to_remove)

            # Save and simulate modified netlist
            modified_netlist_file = f"modified_netlist_{component}.gml"
            nx.write_gml(modified_netlist, modified_netlist_file)
            try:
                modified_edp = run_simulation(modified_netlist_file)
                edp_diff = modified_edp - original_edp
                print(f"Component: {component}, EDP difference: {edp_diff}")
            except Exception as e:
                print(f"Error running simulation for component {component}: {e}")
                edp_diff = -1
            finally:
                os.remove(modified_netlist_file)
    except Exception as e:
        print(f"Error processing component {component}: {e}")
        edp_diff = -1

    # Write result to CSV immediately
    with csv_lock:
        with open('sensitivity_results.csv', 'a') as f:
            f.write(f"{netlist_pair_str},{component},{edp_diff}\n")

    return {'component': component, 'edp_diff': edp_diff}

def process_netlists(netlist_folder, start_idx, end_idx):
    netlist_files = sorted(glob.glob(os.path.join(netlist_folder, 'netlist_*.gml')))

    # Extract netlist indices and filter based on start and end indices
    netlist_files_dict = {}
    for f in netlist_files:
        match = re.search(r'netlist_(\d+)\.gml', os.path.basename(f))
        if match:
            idx = int(match.group(1))
            if start_idx <= idx <= end_idx:
                netlist_files_dict[idx] = f

    selected_indices = sorted(netlist_files_dict.keys())

    if not selected_indices:
        print("No netlist files found in the specified range.")
        return

    with open('sensitivity_results.csv', 'w') as f:
        f.write('netlist_pair,component,edp_diff\n')

    from threading import Lock
    csv_lock = Lock()

    for i in range(len(selected_indices) - 1):
        idx_n = selected_indices[i]
        idx_n1 = selected_indices[i + 1]

        netlist_n_file = netlist_files_dict[idx_n]
        netlist_n1_file = netlist_files_dict[idx_n1]

        netlist_n = nx.read_gml(netlist_n_file)
        netlist_n1 = nx.read_gml(netlist_n1_file)

        netlist_n_counts = get_netlist_counts(netlist_n)
        netlist_n1_counts = get_netlist_counts(netlist_n1)

        unique_components = set(netlist_n_counts.keys()).union(netlist_n1_counts.keys())

        # Run simulation with unmodified netlist_n1
        try:
            original_edp = run_simulation(netlist_n1_file)
        except Exception as e:
            print(f"Error running simulation for netlist {idx_n1}: {e}")
            original_edp = -1

        netlist_pair_str = f"{idx_n}-{idx_n1}"

        # Prepare arguments for parallel execution
        args_list = [
            (
                component,
                netlist_n_counts,
                netlist_n1_counts,
                netlist_n1_file,
                original_edp,
                netlist_pair_str,
                csv_lock
            )
            for component in unique_components
        ]

        # Use ThreadPoolExecutor for parallelism
        with ThreadPoolExecutor() as executor:
            future_to_component = {
                executor.submit(process_component, args): args[0] for args in args_list
            }
            for future in as_completed(future_to_component):
                component = future_to_component[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Unhandled error processing component {component}: {e}")
                    # Even though process_component should handle exceptions,
                    # we catch any unexpected exceptions here to prevent the thread pool from crashing.

if __name__ == "__main__":
    # Update the log variable as per your requirement
    log = "2024-09-27_00-51-26-great"
    netlist_folder = os.path.join(current_dir, "logs", log)

    # Specify the inclusive start and end indices
    start_idx = 0  # Replace with your desired start index
    end_idx = 6   # Replace with your desired end index

    process_netlists(netlist_folder, start_idx, end_idx)
