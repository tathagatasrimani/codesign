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

## This is the area between the die area and the core area.
DIE_CORE_BUFFER_SIZE = 50

class OpenRoadRun:
    def __init__(self, cfg, codesign_root_dir, tmp_dir, run_openroad, circuit_model):
        """
        Initialize the OpenRoadRun with configuration and root directory.

        :param cfg: top level codesign config file
        :param codesign_root_dir: root directory of codesign (where src and test are)
        """
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        self.directory = os.path.join(self.codesign_root_dir, f"{self.tmp_dir}/pd")
        self.run_openroad = run_openroad
        self.circuit_model = circuit_model

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
            arg_parasitics: detailed, estimation, or none. determines which parasitic calculation is executed.

        """
        self.L_eff = L_eff
        logger.info(f"Starting place and route with parasitics: {arg_parasitics}")
        dict = {edge: {} for edge in graph.edges()}
        if "none" not in arg_parasitics:
            logger.info("Running setup for place and route.")
            graph, net_out_dict, node_output, lef_data, node_to_num = self.setup(graph, test_file, area_constraint, L_eff)
            logger.info("Setup complete. Running extraction.")
            dict, graph = self.extraction(graph, arg_parasitics, net_out_dict, node_output, lef_data, node_to_num)
            logger.info("Extraction complete.")
        else: 
            logger.info("No parasitics selected. Running none_place_n_route.")
            graph = self.none_place_n_route(graph)
        logger.info("Place and route finished.")
        return dict, graph

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
        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, max_dim_macro, macro_dict = self.setup_set_area_constraint(graph, test_file, area_constraint, L_eff)

        area_constraint_old = area_constraint
        logger.info(f"Max dimension macro: {max_dim_macro}, corresponding area constraint value: {max_dim_macro**2}")
        logger.info(f"Estimated area: {area_estimate}")
        area_constraint = int(max(area_estimate, max_dim_macro**2)/0.6)
        logger.info(f"Info: Final estimated area {area_estimate} compared to area constraint {area_constraint_old}. Area constraint will be scaled from {area_constraint_old} to {area_constraint}.")
        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, max_dim_macro, macro_dict = self.setup_set_area_constraint(old_graph, test_file, area_constraint, L_eff)

        lib_cell_generator = LibCellGenerator()
        lib_cell_generator.generate_and_write_cells(macro_dict, self.circuit_model, self.directory + "/tcl/codesign_files/codesign_typ.lib")

        self.update_clock_period(self.directory + "/tcl/codesign_files/codesign.sdc")

        return graph, net_out_dict, node_output, lef_data, node_to_num

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

        self.update_area_constraint(area_constraint)

        self.do_scale_lef = scale_lef.ScaleLefFiles(self.cfg, self.codesign_root_dir, self.tmp_dir)
        self.do_scale_lef.scale_lef_files(L_eff)

        df = def_generator.DefGenerator(self.cfg, self.codesign_root_dir, self.tmp_dir, self.do_scale_lef.NEW_database_units_per_micron)

        graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, macro_dict = df.run_def_generator(
            test_file, graph
        )
        logger.info(f"DEF generation complete. Area estimate: {area_estimate}")
        logger.info(f"Max dimension macro: {df.max_dim_macro}")

        return graph, net_out_dict, node_output, lef_data, node_to_num, area_estimate, df.max_dim_macro, macro_dict

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
        logger.info("Starting OpenROAD run.")
        old_dir = os.getcwd()
        os.chdir(self.directory + "/tcl")
        logger.info(f"Changed directory to {self.directory + '/tcl'}")
        print("running openroad. If openroad fails (check log), type exit below and hit return.")
        logger.info("Running OpenROAD command.")
        os.system(os.path.dirname(os.path.abspath(__file__)) + "/OpenROAD/build/src/openroad codesign_top.tcl > " + self.directory + "/codesign_pd.log")#> /dev/null 2>&1
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
        dict = {}
        if arg_parasitics == "detailed":
            logger.info("Running detailed place and route.")
            dict, graph = self.detailed_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Detailed extraction complete.")
        elif arg_parasitics == "estimation":
            logger.info("Running estimated place and route.")
            dict, graph = self.estimated_place_n_route(
                graph, net_out_dict, node_output, lef_data, node_to_num
            )
            logger.info("Estimated extraction complete.")

        return dict, graph

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

        wire_length_df = est.parse_route_guide_with_layer_breakdown(self.directory + "/results/codesign_codesign-tcl.route_guide")
        wire_length_by_edge = {}
        for node in net_out_dict:
            outputs = node_output[node]
            net_outs = net_out_dict[node]
            if node.find("Mux") != -1:
                node_names = [node+"_"+str(x) for x in range(16)]
            else:
                node_names = [node] * 16
            #logger.info(f"Node: {node}, Outputs: {outputs}")
            #logger.info(f"Net outs: {net_outs}")
            for output in outputs:
                # if output is a mux, then there will be 16 instances of the mux, each with 1 output
                # if output is not a mux, then it will have 16 ports
                if output.find("Mux") != -1:
                    output_names = [output+"_"+str(x) for x in range(16)]
                else:
                    output_names = [output] * 16
                assert len(net_outs) == 16
                for idx in range(16):
                    output_name = output_names[idx]
                    node_name = node_names[idx]
                    net_name = net_outs[idx]
                    #logger.info(f"Node: {node}, Output: {output_name}, Net: {net_name}")
                    src = graph.nodes[node_name]["name"]
                    dst = graph.nodes[output_name]["name"]
                    #logger.info(f"Src: {src}, Dst: {dst}")
                    if net_name in wire_length_df.index:
                        if (src, dst) not in wire_length_by_edge:
                            wire_length_by_edge[(src, dst)] = copy.deepcopy(wire_length_df.loc[net_name])
                        else:
                            wire_length_by_edge[(src, dst)] += copy.deepcopy(wire_length_df.loc[net_name])
                    #else:
                    #    logger.warning(f"Net {net_name} not found in wire length dataframe. Skipping.")
        self.export_graph(graph, "estimated_with_mux")

        wire_length_by_edge = self.mux_listing(graph, node_output, wire_length_by_edge)
        self.mux_removal(graph)

        self.export_graph(graph, "estimated_nomux")

        # scale wire lengths to meters
        for edge in wire_length_by_edge:
            #logger.info(f"edge is {edge}")
            #logger.info(f"original wire length by edge: {wire_length_by_edge[edge]}")
            alpha_scale_factor = scale_lef.L_EFF_FREEPDK45 / self.L_eff
            wire_length_by_edge[edge]["total_wl"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal1"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal2"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal3"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal4"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal5"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal6"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal7"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal8"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal9"] /= alpha_scale_factor * 1e6 # convert to meters
            wire_length_by_edge[edge]["metal10"] /= alpha_scale_factor * 1e6 # convert to meters
            #logger.info(f"scaled wire length by edge: {wire_length_by_edge[edge]}")

        return wire_length_by_edge, graph


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
            
            

        self.export_graph(graph, "detailed")

        self.mux_listing(graph, node_output)
        self.mux_removal(graph)

        self.export_graph(graph, "detailed_nomux")

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
    def export_graph(graph, est_or_det: str):
        logger.info(f"Exporting graph to GML for {est_or_det}.")
        if not os.path.exists("openroad_interface/results/"):
            os.makedirs("openroad_interface/results/")
            logger.info("Created results directory.")
        nx.write_gml(
            graph, "openroad_interface/results/" + est_or_det + ".gml"
        )
        logger.info(f"Graph exported to openroad_interface/results/{est_or_det}.gml")

   