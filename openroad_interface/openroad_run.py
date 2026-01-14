from enum import verify
import logging
import re
import os
import copy
import shutil
from math import sqrt

import logging

logger = logging.getLogger(__name__)

import networkx as nx

from openroad_interface import def_generator
from . import estimation as est
from . import detailed as det
from . import scale_lef_files as scale_lef
from openroad_interface.lib_cell_generator import LibCellGenerator
from . import macro_maker as make_macros

## This is the area between the die area and the core area.
DIE_CORE_BUFFER_SIZE = 50


DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

MAX_TRACTABLE_AREA_DBU = 6e13

class OpenRoadRun:
    def __init__(self, cfg, codesign_root_dir, tmp_dir, run_openroad, circuit_model, subdirectory=None, custom_lef_files_to_include=None):
        """
        Initialize the OpenRoadRun with configuration and root directory.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        :param tmp_dir: temporary directory for OpenROAD run
        :param run_openroad: flag to run OpenROAD or use previous results
        :param circuit_model: circuit model configuration
        :param subdirectory: subdirectory for hierarchical runs
        :param custom_lef_files_to_include: custom LEF files to include
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        self.run_openroad = run_openroad
        self.directory = os.path.join(self.codesign_root_dir, f"{self.tmp_dir}/pd")
        self.subdirectory = subdirectory
        self.custom_lef_files_to_include = custom_lef_files_to_include

        ## results will be placed here. This is necessary for running the flow hierarchically. 
        if subdirectory is not None:
            self.directory = os.path.join(self.directory, subdirectory)

        self.circuit_model = circuit_model

        self.component_to_function = {
            "Mult16": "Mult16",
            "Add16": "Add16",
            "Sub16": "Sub16",
            "BUF_X4": "Not16",
            "BUF_X2": "Not16",
            "BUF_X1": "Not16",
            "BUF_X8": "Not16",
            "BUF_X16": "Not16",
            "BUF_X32": "Not16",
            "MUX2_X1": "Not16",
            "Exp16": "Exp16",
            "LShift16": "LShift16",
            "RShift16": "RShift16",
            "FloorDiv16": "FloorDiv16",
            "BitAnd16": "BitAnd16",
            "BitOr16": "BitOr16",
            "BitXor16": "BitXor16",
            "Eq16": "Eq16",
            "Not16": "Not16",
            "NotEq16": "NotEq16",
        }


    def run(
        self,
        graph: nx.DiGraph,
        test_file: str, 
        arg_parasitics: str,
        area_constraint: int,
        L_eff: float
    ):
        """
        Runs the OpenROAD flow.
        params:
            arg_parasitics: detailed, estimation, or none. Determines which parasitic calculation is executed.

        """
        self.L_eff = L_eff
        self.alpha = scale_lef.L_EFF_FREEPDK45 / self.L_eff
        self.original_graph = copy.deepcopy(graph)
        logger.info(f"Starting place and route with parasitics: {arg_parasitics}")
        d = {edge: {} for edge in graph.edges()}
        if "none" not in arg_parasitics:
            logger.info("Running setup for place and route.")

            all_call_functions = True
            for node in graph.nodes():
                if graph.nodes[node].get("function", "") != "Call":
                    all_call_functions = False
                    break

            graph, net_out_dict, node_output, lef_data, node_to_num, final_area, dbu_area_estimate = self.setup(graph, test_file, area_constraint, L_eff)

            
            

            # If all nodes in the graph have the function type "Call", skip place and route.        
            if all_call_functions:
                logger.info("All nodes in the graph have function type 'Call'. Skipping place and route.")
                return d, graph, final_area

            # if the total DBU^2 area of all macros is greater than a limit, skip place and route.
            if dbu_area_estimate > MAX_TRACTABLE_AREA_DBU:
                logger.info(f"Total DBU area {dbu_area_estimate} exceeds area constraint {area_constraint}. Skipping place and route.")
                return d, graph, final_area

            logger.info("Setup complete. Running extraction.")
            d, graph = self.extraction(graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num)
            logger.info("Extraction complete.")
        else: 
            logger.info("No parasitics selected. Running none_place_n_route.")
            graph = self.none_place_n_route(graph)
        logger.info("Place and route finished.")
        return d, graph, final_area

    def setup(
        self,
        graph: nx.DiGraph,
        test_file: str,
        area_constraint: int,
        L_eff: float
    ):
        """
        Sets up the OpenROAD environment. This method creates the working directory, copies tcl files, and generates the def file
        param:
            graph: hardware netlist graph
            test_file: tcl file
            
            area_constraint: area constraint for the placement. We will ensure that the final area constraint set to OpenROAD
                achieves at least 60% utilization based on the estimated area from the def generator.
            L_eff: effective channel length used to scale the LEF files.
            
            NOTE about outputs from setup_set_area_constraint:
                This function is very much legacy and not pretty to look at. Apologies in advance.
                There are some quantities which correspond to a "higher level" graph, where edges are from one functional unit to another.
                On the other hand, things like graph have edges for each of the 16 ports on a functional unit, as quantities are 16 bit right now.

                higher level graph: node_output
                lower level graph: graph, net_out_dict, node_to_num

        """

        old_graph = copy.deepcopy(graph)
        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, max_dim_macro, macro_dict, dbu_area_estimate = self.setup_set_area_constraint(graph, test_file, area_constraint, L_eff)

        area_constraint_old = area_constraint
        logger.info(f"Max dimension macro: {max_dim_macro}, corresponding area constraint value: {max_dim_macro**2}")
        logger.info(f"Estimated area: {area_estimate}")
        area_constraint = int(max(area_estimate, max_dim_macro**2)/0.6)
        logger.info(f"Info: Final estimated area {area_estimate} compared to area constraint {area_constraint_old}. Area constraint will be scaled from {area_constraint_old} to {area_constraint}.")
        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, max_dim_macro, macro_dict, dbu_area_estimate = self.setup_set_area_constraint(old_graph, test_file, area_constraint, L_eff)

        lib_cell_generator = LibCellGenerator()
        lib_cell_generator.generate_and_write_cells(macro_dict, self.circuit_model, self.directory + "/tcl/codesign_files/codesign_typ.lib")

        self.update_clock_period(self.directory + "/tcl/codesign_files/codesign.sdc")

        final_area = area_estimate

        return graph, net_out_dict, node_output, lef_data, node_to_num, final_area, dbu_area_estimate

    def update_clock_period(self, sdc_file: str):
        """
        Updates the clock period in the SDC file.
        param:
            sdc_file: path to the SDC file
        """
        with open(sdc_file, "r") as file:
            sdc_data = file.readlines()
        assert sdc_data[0].startswith("create_clock")
        new_clock_period = self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]
        sdc_data[0] = f"create_clock [get_ports clk] -name core_clock -period {new_clock_period}\n"
        with open(sdc_file, "w") as file:
            file.writelines(sdc_data)
        logger.info(f"Updated clock period in SDC file to {new_clock_period}")

    def setup_set_area_constraint(
        self,
        graph: nx.DiGraph,
        test_file: str,
        area_constraint: int,
        L_eff: float
    ):
        """
        This is a helper method that runs the setup for a single area constraint and provides an area estimate. 
        param:
            graph: hardware netlist graph
            test_file: tcl file
            
            area_constraint: area constraint for the placement. We will ensure that the final area constraint set to OpenROAD
                achieves at least 60% utilization based on the estimated area from the def generator.
            L_eff: effective channel length used to scale the LEF files.
        """

        logger.info("Setting up environment for place and route.")
        if self.run_openroad:
            if os.path.exists(self.directory):
                logger.info(f"Removing existing directory: {self.directory}")
                shutil.rmtree(self.directory)
            os.makedirs(self.directory)
            logger.info(f"Created directory: {self.directory}")
            shutil.copytree(os.path.dirname(os.path.abspath(__file__)) + "/tcl", self.directory + "/tcl")
            logger.info(f"Copied tcl files to {self.directory}/tcl")
            os.makedirs(self.directory + "/results")
            logger.info(f"Created results directory: {self.directory}/results")
        else:
            logger.info("Skipping setup, using previous openroad results.")

        macro_maker = make_macros.MacroMaker(self.cfg, self.codesign_root_dir, self.tmp_dir, self.run_openroad, self.subdirectory, output_lef_file=self.directory + "/tcl/codesign_files/codesign_stdcell.lef", custom_lef_files_to_include=self.custom_lef_files_to_include)

        macro_maker.create_all_macros()

        self.update_area_constraint(area_constraint)

        self.do_scale_lef = scale_lef.ScaleLefFiles(self.cfg, self.codesign_root_dir, self.tmp_dir, self.subdirectory)
        self.do_scale_lef.scale_lef_files(L_eff)

        logger.info(f"Generating DEF file for {self.codesign_root_dir}/{self.tmp_dir}/{self.subdirectory}")
        df = def_generator.DefGenerator(self.cfg, self.codesign_root_dir, self.tmp_dir, self.do_scale_lef.NEW_database_units_per_micron, self.subdirectory)

        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, macro_dict, self.node_to_component_num = df.run_def_generator(
            test_file, graph
        )

        dbu_area_estimate = area_estimate * (self.do_scale_lef.NEW_database_units_per_micron ** 2)

        logger.info(f"DEF generation complete. Area estimate: {area_estimate}")
        logger.info(f"Max dimension macro: {df.max_dim_macro}")
        logger.info(f"DBU area (area estimate in um2 * (DBU per micron)^2): {dbu_area_estimate}")

        self.scale_rc_values()

        return graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, df.max_dim_macro, macro_dict, dbu_area_estimate

    def scale_rc_values(self):
        """
        Scales the RC values in the tcl file based on the input L_eff.
        param:
            L_eff: effective channel length used to scale the LEF files.
        """
        with open(self.directory + "/tcl/codesign_files/codesign.rc", "r") as file:
            rc_data = file.readlines()
        for i, line in enumerate(rc_data):
            if line.startswith("set_layer_rc"):
                metal_layer = line.split()[2]
                rsq = self.circuit_model.tech_model.wire_parasitics["R"][metal_layer].xreplace(self.circuit_model.tech_model.base_params.tech_values) * 1e-9 # convert to kohm/um
                csq = self.circuit_model.tech_model.wire_parasitics["C"][metal_layer].xreplace(self.circuit_model.tech_model.base_params.tech_values) * 1e+15 * 1e-6 # convert to fF/um
                # need to scale up RC for mature nodes, because unscaled OpenROAD wirelengths will be too short
                # for advanced nodes the unscaled wirelengths will be too long
                resistance = rsq / self.alpha
                capacitance = csq / self.alpha
                rc_data[i] = f"set_layer_rc -layer {metal_layer} -resistance {resistance} -capacitance {capacitance}\n"
        with open(self.directory + "/tcl/codesign_files/codesign.rc", "w") as file:
            file.writelines(rc_data)

    def update_area_constraint(self, area_constraint: int):
        """
        Updates the area constraint in the tcl file based on the input area constraint.
        param:
            area_constraint: area constraint for the placement
        """
        ## edit the tcl file to have the correct area constraint
        with open(self.directory + "/tcl/codesign_top.tcl", "r") as file:
            tcl_data = file.readlines()

        ## compute the new area constraint
        new_core_sidelength = int(sqrt(area_constraint))

        #new_core_sidelength_x = new_core_sidelength * 2
        #new_core_sidelength_y = int(area_constraint / new_core_sidelength_x)

        ## find a line that contains "set die_area" and replace it with the new area constraint
        for i, line in enumerate(tcl_data):
            if "set die_area" in line:
                tcl_data[i] = f"set die_area {{0 0 {new_core_sidelength + DIE_CORE_BUFFER_SIZE*2} {new_core_sidelength + DIE_CORE_BUFFER_SIZE*2}}}\n"
                #tcl_data[i] = f"set die_area {{0 0 {new_core_sidelength_x + DIE_CORE_BUFFER_SIZE*2} {new_core_sidelength_y + DIE_CORE_BUFFER_SIZE*2}}}\n"
                logger.info(f"Updated die_area to {new_core_sidelength + DIE_CORE_BUFFER_SIZE*2}x{new_core_sidelength + DIE_CORE_BUFFER_SIZE*2}")
                #logger.info(f"Updated die_area to {new_core_sidelength_x + DIE_CORE_BUFFER_SIZE*2}x{new_core_sidelength_y + DIE_CORE_BUFFER_SIZE*2}")
            if "set core_area" in line:
                tcl_data[i] = f"set core_area {{{DIE_CORE_BUFFER_SIZE} {DIE_CORE_BUFFER_SIZE} {new_core_sidelength + DIE_CORE_BUFFER_SIZE} {new_core_sidelength + DIE_CORE_BUFFER_SIZE}}}\n"
                #tcl_data[i] = f"set core_area {{{DIE_CORE_BUFFER_SIZE} {DIE_CORE_BUFFER_SIZE} {new_core_sidelength_x + DIE_CORE_BUFFER_SIZE} {new_core_sidelength_y + DIE_CORE_BUFFER_SIZE}}}\n"
                logger.info(f"Updated core_area to {new_core_sidelength}x{new_core_sidelength}")
                #logger.info(f"Updated core_area to {new_core_sidelength_x + DIE_CORE_BUFFER_SIZE*2}x{new_core_sidelength_y + DIE_CORE_BUFFER_SIZE*2}")

        ## write the new tcl file
        with open(self.directory + "/tcl/codesign_top.tcl", "w") as file:
            file.writelines(tcl_data)
        
        logger.info(f"Wrote updated tcl file with the area constraints: {new_core_sidelength}x{new_core_sidelength}")


    def run_openroad_executable(self):
        """
        Runs the OpenROAD executable. Run this after setup.
        """
        import subprocess
        import shutil
        import os
        
        logger.info("Starting OpenROAD run.")
        old_dir = os.getcwd()
        os.chdir(self.directory + "/tcl")
        logger.info(f"Changed directory to {self.directory + '/tcl'}")
        print("running openroad. If openroad fails (check log), type exit below and hit return.")
        logger.info("Running OpenROAD command.")
        
        # Safely handle missing/malformed cfg entries for preinstalled_openroad_path.
        args_dict = self.cfg.get("args") if isinstance(self.cfg, dict) else None
        preinstalled = None
        if isinstance(args_dict, dict):
            preinstalled = args_dict.get("preinstalled_openroad_path")

        # Check if xvfb-run is available
        xvfb_available = shutil.which("xvfb-run") is not None
        
        if preinstalled:
            openroad_cmd = preinstalled
        else:
            openroad_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenROAD", "build", "src", "openroad")
            openroad_cmd = openroad_bin
        
        # Set up environment for Qt/OpenGL software rendering
        env = os.environ.copy()
        
        # Qt platform configuration - use offscreen platform for headless rendering
        # This avoids X11/XCB issues entirely and works natively without Xvfb
        env['QT_QPA_PLATFORM'] = 'offscreen'
        
        # Ensure OpenGL software rendering is available for offscreen platform
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'
        env['GALLIUM_DRIVER'] = 'llvmpipe'
        env['MESA_LOADER_DRIVER_OVERRIDE'] = 'llvmpipe'
        env['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        
        # Ensure Qt can find image format plugins (PNG support)
        # Try common system locations for Qt5 plugins
        possible_plugin_paths = [
            "/usr/lib64/qt5/plugins",
            "/usr/lib/qt5/plugins", 
            "/usr/lib/x86_64-linux-gnu/qt5/plugins",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "OpenROAD", "build", "src", "plugins")
        ]
        for qt_plugin_path in possible_plugin_paths:
            if os.path.exists(qt_plugin_path):
                env['QT_PLUGIN_PATH'] = qt_plugin_path
                logger.info(f"Set QT_PLUGIN_PATH to {qt_plugin_path}")
                break
        
        # Ensure imageformats directory is accessible for PNG support
        # Qt5 should have built-in PNG support, but plugins help
        imageformat_path = "/usr/lib64/qt5/plugins/imageformats"
        if os.path.exists(imageformat_path):
            # Set QT_PLUGIN_PATH to include the parent plugins directory
            # Qt will automatically look in plugins/imageformats subdirectory
            if 'QT_PLUGIN_PATH' not in env:
                env['QT_PLUGIN_PATH'] = "/usr/lib64/qt5/plugins"
                logger.info("Set QT_PLUGIN_PATH to include imageformats")
        
        # Also try setting QT_QPA_PLATFORM_PLUGIN_PATH if needed
        # This helps Qt find platform-specific plugins
        if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in env:
            platform_plugin_path = "/usr/lib64/qt5/plugins/platforms"
            if os.path.exists(platform_plugin_path):
                env['QT_QPA_PLATFORM_PLUGIN_PATH'] = platform_plugin_path
        
        # Build the command - with offscreen platform, we don't need Xvfb
        # Offscreen platform works natively without X server
        cmd = f"{openroad_cmd} codesign_top.tcl"
        logger.info("Using Qt offscreen platform for headless image rendering (no Xvfb needed)")

        # Redirect output to log file
        log_file = f"{self.directory}/codesign_pd.log"
        
        logger.info("Executing OpenROAD command: %s", cmd)
        logger.info("Environment: LIBGL_ALWAYS_SOFTWARE=%s, QT_QPA_PLATFORM=%s", 
                    env.get('LIBGL_ALWAYS_SOFTWARE'), env.get('QT_QPA_PLATFORM'))
        
        # Use subprocess to properly handle environment and output redirection
        with open(log_file, 'w') as log:
            result = subprocess.run(
                cmd,
                shell=True,
                env=env,
                cwd=os.getcwd(),
                stdout=log,
                stderr=subprocess.STDOUT
            )
        
        print("done")
        logger.info("OpenROAD run completed.")
        os.chdir(old_dir)
        logger.info(f"Returned to original directory {old_dir}")


    def mux_listing(self, graph, node_output, wire_length_by_edge):
        """
        goes through the graph and finds nodes that are not Muxs. If it encounters one, it will go through
        the graph to find the path of Muxs until the another non-Mux node is found. All rcl are put into a
        list and added as an edge attribute for the non-mux node to non-mux node connection

        param:
            graph: graph with the net attributes already attached
            node_output: dict of nodes and their respective outputs
        """
        #print(f"wire_length_by_edge before modification: {wire_length_by_edge}")
        logger.info("Starting mux listing.")
        edges_to_remove = set()
        for node in node_output:
            #print(f"considering node {node}")
            if "Mux" not in node:
                #print(f"outputs of {node}: {node_output[node]}")
                for output in node_output[node]:
                    path = []
                    if "Mux" in output:
                        while "Mux" in output:
                            # wire delay doesn't need to take all 16 paths into account, so just use 0. For energy, multiply by 16.
                            path.append(output + "_0")
                            output = node_output[output][0]
                        graph.add_edge(node, output)
                        node_name = graph.nodes[node]["name"]
                        output_name = graph.nodes[output]["name"]
                        #logger.info(f"Src: {node_name}, Dst: {output_name}")
                        #print(f"path from {node} to {output}: {path}")
                        if len(path) != 0 and (node_name, output_name) not in wire_length_by_edge:
                            #print(f"adding wire length by edge")
                            path_dsts = [graph.nodes[p]["name"] for p in path]
                            path_dst = path_dsts[0]
                            wire_length_by_edge[(node_name, output_name)] = wire_length_by_edge[(node_name, path_dst)]
                            edges_to_remove.add((node_name, path_dsts[0]))
                            for i in range(1, len(path)):
                                wire_length_by_edge[(node_name, output_name)]["total_wl"] += wire_length_by_edge[(path_dsts[i-1], path_dsts[i])]["total_wl"]
                                wire_length_by_edge[(node_name, output_name)]["metal1"] += wire_length_by_edge[(path_dsts[i-1], path_dsts[i])]["metal1"]
                                wire_length_by_edge[(node_name, output_name)]["metal2"] += wire_length_by_edge[(path_dsts[i-1], path_dsts[i])]["metal2"]
                                wire_length_by_edge[(node_name, output_name)]["metal3"] += wire_length_by_edge[(path_dsts[i-1], path_dsts[i])]["metal3"]
                                edges_to_remove.add((path_dsts[i-1], path_dsts[i]))
                            wire_length_by_edge[(node_name, output_name)]["total_wl"] += wire_length_by_edge[(path_dsts[-1], output_name)]["total_wl"]
                            wire_length_by_edge[(node_name, output_name)]["metal1"] += wire_length_by_edge[(path_dsts[-1], output_name)]["metal1"]
                            wire_length_by_edge[(node_name, output_name)]["metal2"] += wire_length_by_edge[(path_dsts[-1], output_name)]["metal2"]
                            wire_length_by_edge[(node_name, output_name)]["metal3"] += wire_length_by_edge[(path_dsts[-1], output_name)]["metal3"]
                            edges_to_remove.add((path_dsts[-1], output_name))
                            #print(f"wire length by edge after modification: {wire_length_by_edge[(node, output)]}")
        for edge in edges_to_remove:
            #print(f"removing edge {edge}")
            wire_length_by_edge.pop(edge)
        return wire_length_by_edge


    def mux_removal(self, graph: nx.DiGraph):
        """
        Removes the mux nodes from the graph. Does not do the connecting
        param:
            graph: graph with the new edge connections, after mux listing
        """
        logger.info("Removing mux nodes from graph.")
        reference = copy.deepcopy(graph.nodes())
        for node in reference:
            if "Mux" in node:
                graph.remove_node(node)
                logger.info(f"Removed mux node: {node}")


    def coord_scraping(
        self,
        graph: nx.DiGraph,
        node_to_num: dict,
        final_def_directory: str = None,
    ):
        """
        going through the .def file and getting macro placements and nets
        param:
            graph: digraph to add coordinate attribute to nodes
            node_to_num: dict that gives component id equivalent for node
            final_def_directory: final def directory, defaults to def directory in openroad
        return:
            graph: digraph with the new coordinate attributes
            component_nets: dict that list components for the respective net id
        """
        logger.info("Scraping coordinates and nets from DEF file.")
        pattern = r"_\w+_\s+\w+\s+\+\s+PLACED\s+\(\s*\d+\s+\d+\s*\)\s+\w+\s*;"
        net_pattern = r"-\s(_\d+_)\s((?:\(\s_\d+_\s\w+\s\)\s*)+).*"
        component_pattern = r"(_\w+_)"
        if final_def_directory is None:
            final_def_directory = self.directory + "/results/final_generated-tcl.def"
        final_def_data = open(final_def_directory)
        final_def_lines = final_def_data.readlines()
        macro_coords = {}
        component_nets = {}
        for line in final_def_lines:
            if re.search(pattern, line) is not None:
                coord = re.findall(r"\((.*?)\)", line)[0].split()
                match = re.search(component_pattern, line)
                macro_coords[match.group(0)] = {"x": float(coord[0]), "y": float(coord[1])}
                logger.info(f"Found macro {match.group(0)} at ({coord[0]}, {coord[1]})")
            if re.search(net_pattern, line) is not None:
                pins = re.findall(r"\(\s(.*?)\s\w+\s\)", line)
                match = re.search(component_pattern, line)
                component_nets[match.group(0)] = pins
                logger.info(f"Found net {match.group(0)} with pins {pins}")

        for node in node_to_num:
            coord = macro_coords[node_to_num[node]]
            graph.nodes[node]["x"] = coord["x"]
            graph.nodes[node]["y"] = coord["y"]
            logger.info(f"Assigned coordinates to node {node}: {coord}")
        logger.info("Coordinate scraping complete.")
        return graph, component_nets


        
    

    def extraction(self, graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num): 
        # 3. extract parasitics
        logger.info(f"Starting extraction with parasitics option: {arg_parasitics}")
        d = {}
        if arg_parasitics == "detailed":
            logger.info("Running detailed place and route.")
            d, graph = self.detailed_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Detailed extraction complete.")
        elif arg_parasitics == "estimation":
            logger.info("Running estimated place and route.")
            d, graph = self.estimated_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Estimated extraction complete.")

        return d, graph

    # buffers may have been inserted onto edges of the netlist so add them to a new graph
    def parse_new_netlist_graph(self):
        """
        Parses the new netlist graph and adds attributes to the graph
        """
        net_id_to_src_dsts = {}
        new_graph = nx.DiGraph()
        logger.info("Parsing new netlist graph.")
        with open(self.directory + "/results/final_generated-tcl.def", "r") as file:
            def_data = file.readlines()
        for i in range(len(def_data)):
            if def_data[i].startswith("COMPONENTS"):
                break
        for j in range(i+1, len(def_data)): # PARSE COMPONENTS
            if def_data[j].startswith("END COMPONENTS"):
                break
            component_id = def_data[j].split()[1]
            component_name = def_data[j].split()[2]
            if component_name not in self.component_to_function and "HIERMODULE" not in component_name:
                log_info(f"Component {component_name} not found in component_to_function. Skipping.")
                continue
            if "HIERMODULE" in component_name:
                component_function = "Call"
            else:
                component_function = self.component_to_function[component_name]
            new_graph.add_node(component_id, function=component_function)
            log_info(f"Added node {component_id} with function {component_function}")
        for k in range(j+1, len(def_data)):
            if def_data[k].startswith("NETS"):
                break
        
        # PARSE NETS - handle multi-line nets
        l = k + 1
        while l < len(def_data):
            if def_data[l].startswith("END NETS"):
                break
            
            # Check if this line starts a new net
            if def_data[l].strip().startswith("-"):
                # Collect all lines until the net ends with ";"
                net_lines = [def_data[l]]
                while not net_lines[-1].rstrip().endswith(";"):
                    l += 1
                    assert not (l >= len(def_data) or def_data[l].startswith("END NETS")), f"End of netlist reached before net {net_name} was fully parsed, def_data: {def_data[l]}"
                    net_lines.append(def_data[l])
                
                # Concatenate all lines for this net
                full_net_line = " ".join(net_lines)

                log_info(f"Full net line: {full_net_line}")
                # Parse the net: LINE FORMAT: - <net_name> ( <src_node> <src_pin> ) ( <dst_node_0> <dst_pin_0> ) ( <dst_node_1> <dst_pin_1> ) ...
                line_items = full_net_line.split()
                net_name = line_items[1]
                src_node = line_items[3]
                
                if src_node not in new_graph.nodes():
                    log_info(f"Source node {src_node} not found in graph. Skipping net {net_name}.")
                    l += 1
                    continue
                
                # Extract destination nodes (every 4th item after the 3rd, skipping src_pin)
                dst_nodes = []
                for idx in range(7, len(line_items), 4):
                    if idx >= len(line_items):
                        break
                    dst_node = line_items[idx]
                    if dst_node not in new_graph.nodes():
                        log_info(f"Destination node {dst_node} not found in graph. Skipping.")
                        continue
                    dst_nodes.append(dst_node)
                    new_graph.add_edge(src_node, dst_node, net=net_name)
                    log_info(f"Added edge from {src_node} to {dst_node} for net {net_name}")
                net_id_to_src_dsts[net_name] = (src_node, dst_nodes)
            l += 1
        self.export_graph(new_graph, "new_netlist_graph", self.directory)
        return new_graph, net_id_to_src_dsts

    def estimated_place_n_route(
        self,
        graph: nx.DiGraph,
        net_out_dict: dict,
        node_output: dict,
        lef_data: dict,
        node_to_num: dict,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph

        params:
            graph: networkx graph
            net_out_dict: dict that lists nodes and thier respective edges (all nodes have one output)
            node_output: dict that lists nodes and their respective output nodes
            lef_data: dict with layer information (units, res, cap, width)
            node_to_num: dict that gives component id equivalent for node
        returns:
            dict: contains list of resistance, capacitance, length, and net data
            graph: newly modified digraph with rcl attributes
        """

        # run openroad
        logger.info("Starting estimated place and route.")
        if self.run_openroad:
            self.run_openroad_executable()
        else:
            logger.info("Skipping openroad run.")

        ## if the graph has no edges, then return empty dict
        if len(graph.edges()) == 0:
            logger.info("Graph has no edges. Skipping estimated place and route.")
            return {}, graph
        
        self.updated_graph, net_id_to_src_dsts = self.parse_new_netlist_graph()

        nets = est.parse_route_guide_with_layer_breakdown(
            self.directory + "/results/codesign_codesign-tcl.route_guide",
            updated_graph=self.updated_graph,
            net_id_to_src_dsts=net_id_to_src_dsts,
        )
        for net in nets.values():
            for segment in net.segments:
                #logger.info(f"segment length for net {net.net_id} in layer {segment.layer} was {segment.length}")
                segment.length /= self.alpha * 1e6 # convert to meters
                #logger.info(f"segment length for net {net.net_id} in layer {segment.layer} is {segment.length}")

        # Build mapping from original-graph edge (src, dst) -> list of net ids along the path in updated_graph
        self.edge_to_nets: dict[tuple[str, str], list[str]] = {}

        for src, dst in self.original_graph.edges():
            src_component_name = self.original_graph.nodes[src]["name"]
            dst_component_name = self.original_graph.nodes[dst]["name"]
            src_component_num = self.node_to_component_num[src]
            dst_component_num = self.node_to_component_num[dst]
            # Only process if both endpoints exist in the updated graph
            assert src_component_num in self.updated_graph and dst_component_num  in self.updated_graph, f"Source or destination node not found in updated graph: {src}:{src_component_num}, {dst}:{dst_component_num}"

            log_info(f"Finding path from {src}:{src_component_num} to {dst}:{dst_component_num}")
            # Find a path through repeaters from src to dst in updated_graph
            # There should be a unique simple path; use shortest_simple_paths or single_source shortest path
            try:
                path_nodes = nx.shortest_path(self.updated_graph, source=src_component_num, target=dst_component_num)
            except Exception as e:
                log_warning(f"Error finding path from {src}:{src_component_num} to {dst}:{dst_component_num}: {e}")
                path_nodes = []


            # Collect net ids on each hop of the path
            nets_on_path = []
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                if (u, v) in self.updated_graph.edges():
                    nets_on_path.append(copy.deepcopy(nets[self.updated_graph.edges[u, v]["net"]]))
                else:
                    log_warning(f"Edge not found in updated graph: {u}:{v}")
            
            # add self edge if it exists, won't be captured by nx shortest path
            if src == dst:
                if (src_component_num, src_component_num) in self.updated_graph.edges():
                    nets_on_path.append(copy.deepcopy(nets[self.updated_graph.edges[src_component_num, src_component_num]["net"]]))
                else:
                    log_warning(f"Self edge not found in updated graph: {src}:{src_component_num}, {dst}:{dst_component_num}")

            self.edge_to_nets[(src_component_name, dst_component_name)] = nets_on_path
        
        log_info(f"edge_to_nets: {self.edge_to_nets}")

        # Expose for downstream consumers
        self.export_graph(graph, "estimated", self.directory)

        return self.edge_to_nets, graph


    def detailed_place_n_route(
        self,
        graph: nx.DiGraph,
        net_out_dict: dict,
        node_output: dict,
        lef_data: dict,
        node_to_num: dict,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph

        params:
            graph: networkx graph
            net_out_dict:  dict that lists nodes and their respective net (all components utilize one output, therefore this is a same assumption to use)
            node_output: dict that lists nodes and their respective output nodes
            lef_data: dict with layer information (units, res, cap, width)
            node_to_num: dict that gives component id equivalent for node
        returns:
            dict: contains list of resistance, capacitance, length, and net data
            graph: newly modified digraph with rcl attributes
        """

        # run openroad
        logger.info("Starting detailed place and route.")
        if self.run_openroad:
            self.run_openroad_executable()
        else:
            logger.info("Skipping openroad run, as resource constraints have been reached in a previous iteration.")

        # run parasitic_calc and length_calculations
        graph, _ = self.coord_scraping(graph, node_to_num)
        net_cap, net_res = self.det.parasitic_calc(self.directory + "/results/generated-tcl.spef")

        length_dict = det.length_calculations(lef_data["units"], self.directory + "/results/final_generated-tcl.def")

        # add edge attributions
        net_graph_data = []
        res_graph_data = []
        cap_graph_data = []
        len_graph_data = []
        
        for output_net in net_out_dict:
            for net in net_out_dict[output_net]:
                for node in node_output[output_net]:
                    graph[output_net][node]["net"] = net
                    graph[output_net][node]["net_length"] = length_dict[net]
                    graph[output_net][node]["net_res"] = float(net_res[net])
                    graph[output_net][node]["net_cap"] = float(net_cap[net])
                net_graph_data.append(net)
                len_graph_data.append(float(length_dict[net]))  # length
                res_graph_data.append(float(net_res[net]) if net in net_res else 0)  # ohms
                cap_graph_data.append(float(net_cap[net]) if net in net_cap else 0)  # picofarads
            
            

        self.export_graph(graph, "detailed", self.directory)

        self.mux_listing(graph, node_output)
        self.mux_removal(graph)

        self.export_graph(graph, "detailed_nomux", self.directory)

        return {
            "res": res_graph_data,
            "cap": cap_graph_data,
            "length": len_graph_data,
            "net": net_graph_data,
        }, graph


    def none_place_n_route(
        self,
        graph: nx.DiGraph,
    ) -> dict:
        """
        runs openroad, calculates rcl, and then adds attributes to the graph
        params:
            graph: networkx graph
        returns:
            graph: newly modified digraph with rcl attributes
        """

        # edge attribution
        logger.info("Running none_place_n_route: setting default edge attributes.")
        for u, v in graph.edges():
            graph[u][v]["net"] = 0
            graph[u][v]["net_length"] = 0
            graph[u][v]["net_res"] = 0
            graph[u][v]["net_cap"] = 0
            logger.info(f"Set default attributes for edge ({u}, {v})")

        logger.info("none_place_n_route finished.")
        return graph
    

    @staticmethod
    def export_graph(graph, est_or_det: str, directory: str):
        logger.info(f"Exporting graph to GML for {est_or_det}.")
        if not os.path.exists(f"{directory}/results/"):
            os.makedirs(f"{directory}/results/")
            logger.info("Created results directory.")
        nx.write_gml(
            graph, f"{directory}/results/{est_or_det}.gml"
        )
        logger.info(f"Graph exported to {directory}/results/{est_or_det}.gml")

