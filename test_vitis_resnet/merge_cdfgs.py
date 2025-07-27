import os
import sys
import json
import networkx as nx


DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)




def main(root_dir, top_level_module_name):
    """
    Main function to create CDFG for all files in the given directory.
    """
    full_cdfg = nx.DiGraph()

    ## get the names of all the submodules (the subdirectories)
    submodules = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    debug_print(f"Submodules found: {submodules}")

    ## start with the top-level module
    if top_level_module_name not in submodules:
        print(f"Error: Top-level module {top_level_module_name} does not exist in {root_dir}.")
        return
    
    full_cdfg = parse_module(root_dir, top_level_module_name)
    if full_cdfg is None:
        print(f"Error: Failed to parse the top-level module {top_level_module_name}.")
        return
    
    ## save the full CDFG to a file
    output_file_path = os.path.join(root_dir, f"{top_level_module_name}_full_cdfg.gml")
    nx.write_gml(full_cdfg, output_file_path)
    debug_print(f"Full CDFG saved to {output_file_path}")


def parse_module(root_dir, current_module):

    debug_print(f"Parsing module: {current_module}")

    ## open the _cdfg.gml file for the current module. Read it in as a NetworkX graph.
    cdfg_file_path = os.path.join(root_dir, current_module, f"{current_module}.verbose_cdfg.gml")

    if not os.path.exists(cdfg_file_path):
        print(f"Error: CDFG file {cdfg_file_path} does not exist.")
        exit(1)
        return
    else:
        full_cdfg = nx.read_gml(cdfg_file_path)

    ## read in the _modules.json file for the current module
    modules_file_path = os.path.join(root_dir, current_module, f"{current_module}.verbose_modules.json")
    if not os.path.exists(modules_file_path):
        print(f"Error: Modules file {modules_file_path} does not exist.")
        exit(1)
        return

    with open(modules_file_path, 'r') as mf:
        module_dependences = json.load(mf)

    if module_dependences:
        debug_print(f"Instantiated modules for {current_module}: {module_dependences}")

    ## get the CDFGs for each of the instantiated modules recursively
    submodule_cdfgs = {}
    for module_name in module_dependences:
        submodule_cdfgs[module_name] = parse_module(root_dir, module_name)

    ## merge the CDFGs of the submodules into the full CDFG
    ## TODO: merge properly.
    for submodule_name, submodule_cdfg in submodule_cdfgs.items():
        if submodule_cdfg is not None:
            full_cdfg = nx.compose(full_cdfg, submodule_cdfg)
            #full_cdfg = merge_cdfgs(full_cdfg, submodule_cdfg, submodule_name)
            debug_print(f"Merged CDFG for submodule {submodule_name} into full CDFG.")

    return full_cdfg

def merge_cdfgs(full_cdfg, submodule_cdfg, submodule_name):
    """
    Merge the submodule CDFG into the full CDFG.
    """
    debug_print(f"Merging CDFG for submodule {submodule_name} into full CDFG.")

    
    # Find all nodes in the full CDFG that are call functions to the submodule

    # See if there are duplicate nodes in the full CDFG that match the submodule name

    # if there are, check if they are part of the same submodule call. (i.e. multi-cycle ops)
    # if they are, remove all of them as a group, recording the inputs of the first node and the outputs of the last node.
    # then, insert the submodule CDFG in place of the removed nodes, connecting the inputs and outputs to the first and last nodes respectively.
    # make sure the start and stop nodes at the top level of the submodule CDFG no longer exist in the full CDFG, as they are now part of the full CDFG.
    # intermediate start and stop nodes in the submodule CDFG should be preserved, as they are part of the submodule's internal flow.

    # if they are not, error out for now. 

    # add stall nodes to the full CDFG to keep the timing correct between the start and stop nodes of the original CDFG (on other branches).



    return full_cdfg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_cdfgs.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])