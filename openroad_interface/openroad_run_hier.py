from enum import verify
import logging
import re
import os
import copy
import shutil
import json
from math import sqrt

import logging

logger = logging.getLogger(__name__)

import networkx as nx

from . import openroad_run
from . import macro_maker

from src import sim_util

## This is the area between the die area and the core area.
DIE_CORE_BUFFER_SIZE = 50

DEBUG_PRINT = False
def debug_print(msg):
    if DEBUG_PRINT:
        logger.info(msg)

class OpenRoadRunHier:
    def __init__(self, cfg, codesign_root_dir, tmp_dir, run_openroad, circuit_model):
        """
        Initialize the OpenRoadRunHier with configuration and root directory.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        self.run_openroad = run_openroad
        self.hier_pd_base_dir = os.path.join(self.codesign_root_dir, f"{self.tmp_dir}/pd/hier/")
        self.circuit_model = circuit_model


    def run_hierarchical_openroad(
        self,
        full_graph: nx.DiGraph,
        test_file: str, 
        arg_parasitics: str,
        area_constraint: int,
        L_eff: float,
        parse_results_dir: str,
        top_module_name: str
    ):
        """
        Runs the OpenROAD flow.
        params:
            arg_parasitics: detailed, estimation, or none. determines which parasitic calculation is executed.

            parse_results_dir: the directory created during parsing of hls results. This path is relative to codesign_root_dir specified in the constructor of this class.
            top_module_name: the name of the top module in the hierarchical design. For example, for Resnet it is "forward"

        """
        self.L_eff = L_eff
        logger.info(f"Starting hierarchical place and route with parasitics: {arg_parasitics}")
        # Build mapping from original-graph edge (src, dst) -> list of net ids along the path in updated_graph
        self.edge_to_nets: dict[tuple[str, str], list[str]] = {}
        if "none" not in arg_parasitics:
            logger.info("Running setup for hierarchical place and route.")
            self.initialize_directories(os.path.join(self.codesign_root_dir, self.tmp_dir, parse_results_dir))

            self.test_file = test_file
            self.top_level_area_constraint = area_constraint
            self.L_eff = L_eff
            self.arg_parasitics = arg_parasitics

            self.edge_to_nets = self.run_pd_single_module(top_module_name)

        else: 
            raise NotImplementedError("Hierarchical 'none place and route' is not implemented.")
            
        logger.info("Hierarchical Place and route finished.")
        return self.edge_to_nets

    


    def run_pd_single_module(self, module_name, top_level=False):
        """
        Run place and route for a single module in the hierarchical design.

        """

        all_edge_to_nets = {}
        #wire_lengths_file = os.path.join(self.hier_pd_base_dir, module_name, f"{module_name}_wire_lengths.json")

        ## Base case: This module has already been placed and routed.
        # see if the pd_complete.note file exists in the module's directory
        """pd_complete_file = os.path.join(self.hier_pd_base_dir, module_name, "pd_complete.note")
        if os.path.exists(pd_complete_file):
            logger.info(f"Module {module_name} has already been placed and routed. Skipping.")

            all_edge_to_nets = sim_util.read_wirelengths(wire_lengths_file)
            return all_edge_to_nets"""

        ## first do recursive step. Make sure we have placed and routed all submodules & generated their macros. 
        # open up the file that ends in _verbose_modules.json file to get the list of submodules
        verbose_modules_file = os.path.join(self.hier_pd_base_dir, module_name, f"{module_name}.verbose_modules.json")

        if not os.path.exists(verbose_modules_file):
            verbose_modules_file = os.path.join(self.hier_pd_base_dir, module_name + "s", f"{module_name}s.verbose_modules.json")
            if not os.path.exists(verbose_modules_file):
                debug_print(f"Error: Submodule netlist file {verbose_modules_file} does not exist.")
                exit(1)
            else:
                debug_print(f"Warning: Submodule {submodule_name} not found, adding s worked though:  {submodule_name}s")
                verbose_modules_file += "s"

        logger.info(f"Opening {verbose_modules_file}")

        with open(verbose_modules_file, 'r') as f:
            verbose_modules = json.load(f)

        for submodule_name in verbose_modules:
            curr_edge_to_nets = self.run_pd_single_module(submodule_name)
            all_edge_to_nets.update(curr_edge_to_nets)

        # Read in the netlist for this module
        netlist_filtered_file = os.path.join(self.hier_pd_base_dir, module_name, f"{module_name}_netlist_hier_filtered.gml")

        if not os.path.exists(netlist_filtered_file):
            netlist_filtered_file = os.path.join(self.hier_pd_base_dir, module_name + "s", f"{module_name}s_netlist_hier_filtered.gml")
            if not os.path.exists(netlist_filtered_file):
                debug_print(f"Error: Netlist file {netlist_filtered_file} does not exist.")
                exit(1)
            else:
                debug_print(f"Warning: Netlist for {module_name} not found, adding s worked though:  {module_name}s")
                netlist_filtered_file += "s"

        logger.info(f"Opening netlist file {netlist_filtered_file}")

        # networkx.read_gml expects a path or a binary file-like object.
        # Passing the filename is simplest and avoids text-mode decoding issues.
        module_graph = nx.read_gml(netlist_filtered_file)

        # If the GML only contains the graph header (no nodes), skip this module
        # to avoid passing an empty graph like:
        #   graph [
        #     directed 1
        #   ]
        if module_graph is None or module_graph.number_of_nodes() == 0:
            logger.info(
                "Netlist %s appears empty (no nodes). Skipping place & route for module %s.",
                netlist_filtered_file,
                module_name,
            )

            pd_complete_file = os.path.join(self.hier_pd_base_dir, module_name, "pd_complete.note")
            with open(pd_complete_file, 'w') as f:
                f.write("Place and route not needed as there are no nodes in the graph.\n")

            #sim_util.write_wirelengths(all_edge_to_nets, wire_lengths_file)

            return all_edge_to_nets
        
        ## This is a list of file paths to lef files from submodules to include in this module's P&R.
        lef_files_to_include = {}
        lef_files_to_skip = {}
        
        for submodule_name in verbose_modules:
            
            ## first find the module folder for this submodule
            submodule_folder = os.path.join(self.hier_pd_base_dir, submodule_name)

            # it is possible that we need to append an "s" to the submodule name to find the folder.
            if not os.path.exists(submodule_folder):
                submodule_folder_s = os.path.join(self.hier_pd_base_dir, submodule_name + "s")
                if not os.path.exists(submodule_folder_s):
                    debug_print(f"Error: Submodule folder for {submodule_name} does not exist at {submodule_folder}.")
                    exit(1)
                else:
                    debug_print(f"Warning: Submodule {submodule_name} not found, adding s worked though:  {submodule_name}s")
                    submodule_folder = submodule_folder_s
                    submodule_name += "s"
            
            macro_name = f"HIERMODULE_{submodule_name}"
            macro_lef_file = os.path.join(self.hier_pd_base_dir, submodule_name, "pd", f"{macro_name}_macro.lef")
            
            ## ensure the file actually exists. 
            if os.path.exists(macro_lef_file):
                # the lef file exists, so add it to the list of lef files to include.
                lef_files_to_include[submodule_name] = macro_lef_file
            else:
                ## it is possible that the submodule did not need place and route, so its macro lef file was never created.
                # check to see if the pd_complete.note file exists. If it does not, then this is an error.
                pd_complete_file = os.path.join(self.hier_pd_base_dir, submodule_name, "pd_complete.note")
                if os.path.exists(pd_complete_file):
                    debug_print(f"Macro LEF file for submodule {submodule_name} not found at {macro_lef_file}, but pd_complete.note file exists. Assuming no P&R was needed for this submodule.")
                    # skip adding this lef file.
                    lef_files_to_skip[submodule_name] = macro_lef_file
                else:
                    error_message = f"Macro LEF file for submodule {submodule_name} not found at {macro_lef_file}."
                    error_message += f" Additionally, the pd_complete.note file was not found at {pd_complete_file}, indicating that place and route for this submodule was not completed."
                    raise FileNotFoundError(error_message)
                
        debug_print(f"For module {module_name}, including the following macro LEF files from submodules: {lef_files_to_include}")
        debug_print(f"For module {module_name}, skipping the following macro LEF files from submodules: {lef_files_to_skip}")

        module_graph_pruned = copy.deepcopy(module_graph)

        if len(lef_files_to_skip) > 0:
            ## make sure to remove any pruned nodes from the graph before passing to OpenROAD.
            for node in list(module_graph_pruned.nodes):
                ## see if this is a call node and if the module name is in the lef_files_to_skip list. 
                if module_graph_pruned.nodes[node].get("function") == "Call" and module_graph_pruned.nodes[node].get("call_submodule_instance_name") in lef_files_to_skip:
                    debug_print(f"Removing pruned node {node} from module graph for module {module_name}.")
                    module_graph_pruned.remove_node(node)

            # netlist_filtered_pruned_file = os.path.join(self.hier_pd_base_dir, module_name, f"{module_name}_netlist_hier_filtered_pruned.gml")
            debug_print(f"Writing pruned module graph to {netlist_filtered_file}.")
            # write out the modified graph to a new netlist_filtered_pruned_file.
            nx.write_gml(module_graph_pruned, netlist_filtered_file)

            module_graph = module_graph_pruned

        ###### then run place and route for this module, using the macros of the submodules. ######
        flat_open_road_run = openroad_run.OpenRoadRun(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=self.run_openroad, circuit_model=self.circuit_model, subdirectory=f"hier/{module_name}/pd", custom_lef_files_to_include=lef_files_to_include, top_level=top_level)
        
        edge_to_nets, _, final_area = flat_open_road_run.run(
            module_graph, self.test_file, self.arg_parasitics, self.top_level_area_constraint, self.L_eff
        )

        all_edge_to_nets.update(edge_to_nets)

        logger.info(f"Completed place and route for module {module_name}")

        ## Create the macro for this module to be used by the parent module.
        ## call macro_maker to create the macro lef file

        macro_name = f"HIERMODULE_{module_name}"
        
        macro_maker_instance = macro_maker.MacroMaker(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=self.run_openroad, subdirectory=f"hier/{module_name}/pd",
                                                      output_lef_file=f"{macro_name}_macro.lef", area_list={f"{macro_name}": final_area}, pin_list={f"{macro_name}": {"input": 16, "output": 16}}, add_ending_text=False)
        
        macro_maker_instance.create_all_macros()

        ## write a pd_complete.note file to indicate that this module has been placed and routed.
        pd_complete_file = os.path.join(self.hier_pd_base_dir, module_name, "pd_complete.note")
        with open(pd_complete_file, 'w') as f:
            f.write("Place and route completed successfully.\n")


        debug_print(f"All wirelengths = {all_edge_to_nets}")
        ## write out the final wire lengths for this module to a json file.
        #sim_util.write_wirelengths(all_edge_to_nets, wire_lengths_file)

        return all_edge_to_nets




    def initialize_directories(self, parse_results_dir):
        """
        Create necessary directories for hierarchical place and route.

        params:
            parse_results_dir: the directory created during parsing of hls results. This path is relative to codesign_root_dir specified in the constructor of this class.
        """
        if not self.run_openroad: # directories already exist from previous run
            return
        if os.path.exists(self.hier_pd_base_dir):
            shutil.rmtree(self.hier_pd_base_dir)
        os.makedirs(self.hier_pd_base_dir)

        logger.info(f"Created hierarchical PD base directory at {self.hier_pd_base_dir}")

        ## go through each subdirectory of parse_results_dir and create corresponding directories in hier_pd_base_dir
        for subdir in os.listdir(parse_results_dir):
            subdir_path = os.path.join(parse_results_dir, subdir)
            if os.path.isdir(subdir_path):
                ## only copy specific files
                dest_subdir_path = os.path.join(self.hier_pd_base_dir, subdir)
                os.makedirs(dest_subdir_path)

                ## copy the files that include the substring "_cross_module_edges.json" or "_netlist_hier_filtered.gml" or "verbose_modules.json"
                for filename in os.listdir(subdir_path):
                    if "_cross_module_edges.json" in filename or "_netlist_hier_filtered.gml" in filename or "verbose_modules.json" in filename:
                        shutil.copy2(os.path.join(subdir_path, filename), dest_subdir_path)
                        logger.info(f"Copied {filename} to {dest_subdir_path}")
        logger.info("Finished initializing hierarchical PD directories.")