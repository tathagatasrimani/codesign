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
    current_module = top_level_module_name


def parse_module(root_dir, current_module):

    ## open the _cdfg.gml file for the current module. Read it in as a NetworkX graph.
    cdfg_file_path = os.path.join(root_dir, current_module, f"{current_module}_cdfg.gml")
    debug_print(f"Opening CDFG file for module {current_module}: {cdfg_file_path}")

    if not os.path.exists(cdfg_file_path):
        print(f"Error: CDFG file {cdfg_file_path} does not exist.")
        return
    else:
        full_cdfg = nx.read_gml(cdfg_file_path)
    debug_print(f"Successfully opened CDFG for module {current_module}")

    ## 


    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_cdfg.py <root_directory> <top_level_module_name>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])