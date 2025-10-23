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

## This is the area between the die area and the core area.
DIE_CORE_BUFFER_SIZE = 50

DEBUG_PRINT = True
def debug_print(msg):
    if DEBUG_PRINT:
        logger.info(msg)

class OpenRoadRunHier:
    def __init__(self, cfg, codesign_root_dir):
        """
        Initialize the OpenRoadRunHier with configuration and root directory.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.hier_pd_base_dir = os.path.join(self.codesign_root_dir, "src/tmp/pd/hier")

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
        dict = {edge: {} for edge in full_graph.edges()}
        if "none" not in arg_parasitics:
            logger.info("Running setup for place and route.")
            self.initialize_directories(os.path.join(self.codesign_root_dir, parse_results_dir))

            self.test_file = test_file
            self.top_level_area_constraint = area_constraint
            self.L_eff = L_eff
            self.arg_parasitics = arg_parasitics

            self.run_pd_single_module(top_module_name)

            exit(1)

            # full_graph, net_out_dict, node_output, lef_data, node_to_num = self.setup(full_graph, test_file, area_constraint, L_eff)
            # logger.info("Setup complete. Running extraction.")
            # dict, full_graph = self.extraction(full_graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num)
            # logger.info("Extraction complete.")
        else: 
            logger.info("No parasitics selected. Running none_place_n_route.")
            full_graph = self.none_place_n_route(full_graph)
        logger.info("Place and route finished.")
        return dict, full_graph

    


    def run_pd_single_module(self, module_name):
        """
        Run place and route for a single module in the hierarchical design.

        """

        ## Base case: This module has already been placed and routed.
        # see if the pd_complete.note file exists in the module's directory
        pd_complete_file = os.path.join(self.hier_pd_base_dir, module_name, "pd_complete.note")
        if os.path.exists(pd_complete_file):
            logger.info(f"Module {module_name} has already been placed and routed. Skipping.")
            return

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
            self.run_pd_single_module(submodule_name)

        # Read in the netlist for this module
        netlist_filtered_file = os.path.join(self.hier_pd_base_dir, module_name, f"{module_name}_netlist_filtered.gml")

        if not os.path.exists(netlist_filtered_file):
            netlist_filtered_file = os.path.join(self.hier_pd_base_dir, module_name + "s", f"{module_name}s_netlist_filtered.gml")
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

            return
        
        ## then run place and route for this module, using the macros of the submodules.
        flat_open_road_run = openroad_run.OpenRoadRun(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, subdirectory=f"hier/{module_name}/pd/")
        
        wire_length_by_edge, _ = flat_open_road_run.run(
            module_graph, self.test_file, self.arg_parasitics, self.top_level_area_constraint, self.L_eff
        )

        logger.info(f"Completed place and route for module {module_name}")

        ## write a pd_complete.note file to indicate that this module has been placed and routed.
        pd_complete_file = os.path.join(self.hier_pd_base_dir, module_name, "pd_complete.note")
        with open(pd_complete_file, 'w') as f:
            f.write("Place and route completed successfully.\n")




    def initialize_directories(self, parse_results_dir):
        """
        Create necessary directories for hierarchical place and route.

        params:
            parse_results_dir: the directory created during parsing of hls results. This path is relative to codesign_root_dir specified in the constructor of this class.
        """
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

                ## copy the files that include the substring "_cross_module_edges.json" or "_netlist_filtered.gml" or "verbose_modules.json"
                for filename in os.listdir(subdir_path):
                    if "_cross_module_edges.json" in filename or "_netlist_filtered.gml" in filename or "verbose_modules.json" in filename:
                        shutil.copy2(os.path.join(subdir_path, filename), dest_subdir_path)
                        logger.info(f"Copied {filename} to {dest_subdir_path}")
        logger.info("Finished initializing hierarchical PD directories.")