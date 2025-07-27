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
            debug_print(f"Merged CDFG for submodule {submodule_name} into full CDFG.")

    return full_cdfg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_cdfgs.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])