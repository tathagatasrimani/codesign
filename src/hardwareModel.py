import logging
import random
import yaml
import time

logger = logging.getLogger(__name__)

import networkx as nx
import sympy as sp
from . import cacti_util
from . import parameters
from . import schedule
from openroad_interface import place_n_route

HW_CONFIG_FILE = "src/params/hw_cfgs.ini"

def symbolic_convex_max(a, b, evaluate=True):
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + sp.Abs(a - b, evaluate=evaluate))

def symbolic_convex_min(a, b, evaluate=True):
    """
    Min(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b - sp.Abs(a - b, evaluate=evaluate))

class HardwareModel:
    """
    Represents a hardware model with configurable technology and hardware parameters. Provides methods
    to set up the hardware, manage netlists, and extract technology-specific timing and power data for
    optimization and simulation purposes.
    """
    def __init__(self, args):
        # HARDCODED UNTIL WE COME BACK TO MEMORY MODELING
        self.cacti_tech_node = min(
            cacti_util.valid_tech_nodes,
            key=lambda x: abs(x - 7 * 1e-3),
        )
        print(f"cacti tech node: {self.cacti_tech_node}")

        self.cacti_dat_file = (
            f"src/cacti/tech_params/{int(self.cacti_tech_node*1e3):2d}nm.dat"
        )
        print(f"self.cacti_dat_file: {self.cacti_dat_file}")
        self.params = parameters.Parameters(args.tech_node, self.cacti_dat_file)
        self.netlist = nx.DiGraph()
        self.scheduled_dfg = nx.DiGraph()
        self.parasitic_graph = nx.DiGraph()
        self.symbolic_mem = {}
        self.symbolic_buf = {}
        self.memories = []
        self.obj_fn = args.obj
        self.obj = 0
        self.obj_sub_exprs = {}
        self.symbolic_obj = 0
        self.symbolic_obj_sub_exprs = {}
        self.longest_paths = []
        self.area_constraint = args.area
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.dfg_to_netlist_edge_map = {}

    def reset_state(self):
        self.symbolic_buf = {}
        self.symbolic_mem = {}
        self.netlist = nx.DiGraph()
        self.memories = []
        self.obj = 0
        self.symbolic_obj = 0
        self.scheduled_dfg = nx.DiGraph()
        self.parasitic_graph = nx.DiGraph()
        self.longest_paths = []
        self.obj_sub_exprs = {}
        self.symbolic_obj_sub_exprs = {}
        self.execution_time = 0
        self.total_passive_energy = 0
        self.total_active_energy = 0
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.dfg_to_netlist_edge_map = {}

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.params.circuit_values["latency"],
            "dynamic_energy": self.params.circuit_values["dynamic_energy"],
            "passive_power": self.params.circuit_values["passive_power"],
            "area": self.params.circuit_values["area"], # TODO: make sure we have this
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))

    def map_netlist_to_scheduled_dfg(self, benchmark_name):

        ## create a set of all netlist nodes
        unmapped_dfg_nodes = set(self.scheduled_dfg.nodes)
        
        ### First, attempt direct name matching between scheduled DFG nodes and netlist nodes.
        if benchmark_name == "matmult":
            for node in self.scheduled_dfg:
                if self.scheduled_dfg.nodes[node]["function"] not in self.params.circuit_values["latency"]:
                    continue

                ## get the catapult name from the scheduled DFG node
                catapult_name = self.scheduled_dfg.nodes[node]["catapult_name"]

                ##E.g. catapult name: for#1-1:for-2:for-6:add_inst.run()

                ## replace all the # and - with _ to match the netlist node names
                catapult_name = catapult_name.replace("#", "_").replace("-", "_")

                ##E.g. catapult_name: for_1_1:for_2:for_6:add_inst.run()

                if "ccs_ram_sync_" in self.scheduled_dfg.nodes[node]["module"]:
                    ## This is a memory node, so we must map it to the DFG differently.
                    module_name = self.scheduled_dfg.nodes[node]["module"]

                    ## ccs_ram_sync_1R1W_wport(9,16,7,100,100,16,5)

                    ## remove the last character of module name and replace all the , and ( with _
                    module_name = module_name[:-1].replace(",", "_").replace("(", "_")


                    ### catapult_name "for#1:for:write_mem(c_chan:rsc.@)#98"
                    ## extract only the part inside the parentheses
                    catapult_name = catapult_name.split("(")[-1].split(")")[0]

                    ## c_chan:rsc.@
                    ## replace : with _ and .@ with 'i'
                    instance_name = catapult_name.replace(":", "_").replace(".@", "i")

                    ## see if there is a node in the netlist where the module name in the DFG matches part of the module type in the netlist
                    for netlist_node in self.netlist.nodes:
                        if (module_name in self.netlist.nodes[netlist_node]["module_type"] and
                            instance_name in self.netlist.nodes[netlist_node]["hierarchy_path"]):
                            ## if so, map the scheduled DFG node to the netlist node
                            #self.dfg_to_netlist_map[node] = netlist_node
                            self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                            unmapped_dfg_nodes.remove(node)
                            break

                else:
                    ## This is NOT a memory node
                    
                    ## remove the last part of the catapult name and rejoin with _
                    netlist_name = "_".join(catapult_name.split(":")[0:3])

                    ##E.g. netlist_name: for_1_1_for_2_for_6

                    ## get the function type from the scheduled DFG node
                    function_type = self.scheduled_dfg.nodes[node]["function"]

                    ## see if there is a node in the netlist with the same function type and whose hierarchy path contains the netlist_name
                    for netlist_node in self.netlist.nodes:
                        if (self.netlist.nodes[netlist_node]["function"] == function_type and
                            netlist_name in self.netlist.nodes[netlist_node]["hierarchy_path"]):
                            ## if so, map the scheduled DFG node to the netlist node
                            #self.dfg_to_netlist_map[node] = netlist_node
                            self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                            unmapped_dfg_nodes.remove(node)
                            break

        elif benchmark_name == "basic_aes":
            ### similar as matmult, but the catapult names are different.
            ## for example catapult_name "mul_inst.run()#1" in the DFG should map to label "basic_aes**basic_aes_run_inst**mul_inst_run_1_rg" in the netlist. 
            for node in self.scheduled_dfg:
                if self.scheduled_dfg.nodes[node]["function"] not in self.params.circuit_values["latency"]:
                    logger.warning(f"skipping node {node} with function {self.scheduled_dfg.nodes[node]['function']} as it is not in the circuit values")
                    continue

                ## get the catapult name from the scheduled DFG node
                catapult_name = self.scheduled_dfg.nodes[node]["catapult_name"]

                ##E.g. catapult name: mul_inst.run()#1

                ## replace all the # and - and . and () with _ to match the netlist node names
                netlist_name = catapult_name.replace("#", "_").replace("-", "_").replace(".", "_").replace("()", "_")

                ##E.g. netlist_name: mul_inst_run_1

                logger.info(f"netlist_name: {netlist_name}")

                ## get the function type from the scheduled DFG node
                function_type = self.scheduled_dfg.nodes[node]["function"]

                ## see if there is a node in the netlist with the same function type and whose hierarchy path contains the netlist_name
                found_match = False
                for netlist_node in self.netlist.nodes:
                    if (self.netlist.nodes[netlist_node]["function"] == function_type and
                        netlist_name in self.netlist.nodes[netlist_node]["hierarchy_path"]):
                        ## if so, map the scheduled DFG node to the netlist node
                        #self.dfg_to_netlist_map[node] = netlist_node
                        self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                        unmapped_dfg_nodes.remove(node)
                        found_match = True
                        break

                if not found_match:
                    logger.warning(f"no direct name match found for node {node} with catapult name {catapult_name} and function type {function_type}")


        direct_name_match_count = len(self.scheduled_dfg.nodes) - len(unmapped_dfg_nodes)
        logger.info(f"number of mapped dfg nodes using direct name match: {direct_name_match_count}")
        logger.info(f"number of unmapped dfg nodes after direct name match: {len(unmapped_dfg_nodes)}")
        
        ### Next, attempt to map the remaining unmapped DFG nodes using resource sharing information.
        ## read in the res_sharing.tcl file
        res_sharing_file = f"src/tmp/benchmark/build/{benchmark_name}.v1/res_sharing.tcl"
        res_sharing_lines = []
        with open(res_sharing_file, "r") as f:
            res_sharing_lines = f.readlines()

        ## parse the res_sharing.tcl file to get the resource sharing information
        ## E.g. of one line: directive set /MatMult/MatMult.struct/MatMult:run/MatMult:run:conc/for#1-5:for-6:for-3:mul_inst.run() RESOURCE_NAME for#1-1:for-2:for-8:mul_inst.run():rg
        res_sharing_map = {}
        for line in res_sharing_lines:
            logger.info(f"res_sharing.tcl line: {line.strip()}")
            parts = line.strip().split("RESOURCE_NAME")
            key_dirty_text = parts[0]
            value_dirty_text = parts[1]

            logger.info(f"res_sharing.tcl key: {key_dirty_text.strip()}")
            logger.info(f"res_sharing.tcl value: {value_dirty_text.strip()}")

            key_2 = "_".join(key_dirty_text.split("/")[-1].strip().split(":")[0:3])
            logger.info(f"res_sharing.tcl key_2: {key_2}")
            key = key_2.replace("#", "_").replace("-", "_").strip()  # replace all the # and - with _ to match the netlist node names

            value_2 = "_".join(value_dirty_text.split(":")[0:3])
            logger.info(f"res_sharing.tcl value_2: {value_2}")
            value = value_2.replace("#", "_").replace("-", "_").strip()  # replace all the # and - with _ to match the netlist node names
            if key not in res_sharing_map:
                res_sharing_map[key] = value
                logger.info(f"res_sharing_map: {key} -> {value}")
            else:
                logger.warning(f"duplicate key {key} in res_sharing.tcl file.")

        
        num_res_sharing_mapped = 0

        resource_sharing_mapped_nodes = set()

        ## map the unmapped dfg nodes to the netlist nodes using the resource sharing information
        for node in unmapped_dfg_nodes:
            if self.scheduled_dfg.nodes[node]["function"] not in self.params.circuit_values["latency"]:
                continue
            
            ## get the catapult name from the scheduled DFG node
            catapult_name = self.scheduled_dfg.nodes[node]["catapult_name"]

            ##E.g. catapult name: for#1-1:for-2:for-6:add_inst.run()

            ## replace all the # and - with _ to match the netlist node names
            catapult_name = catapult_name.replace("#", "_").replace("-", "_")

            ##E.g. catapult_name: for_1_1:for_2:for_6:add_inst.run()

            ## remove the last part of the catapult name and rejoin with _
            netlist_name = "_".join(catapult_name.split(":")[0:3])

            ##E.g. netlist_name: for_1_1_for_2_for_6

            ## get the function type from the scheduled DFG node
            function_type = self.scheduled_dfg.nodes[node]["function"]

            ## see if there is a node in the netlist with the same function type and whose hierarchy path contains the netlist_name
            if netlist_name in res_sharing_map:
                for netlist_node in self.netlist.nodes:
                    if (self.netlist.nodes[netlist_node]["function"] == function_type and
                        res_sharing_map[netlist_name] in self.netlist.nodes[netlist_node]["hierarchy_path"]):
                        ## if so, map the scheduled DFG node to the netlist node
                        #self.dfg_to_netlist_map[node] = netlist_node
                        self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                        resource_sharing_mapped_nodes.add(node)
                        num_res_sharing_mapped += 1
                        break

        # ## remove the resource sharing mapped nodes from the unmapped dfg nodes
        unmapped_dfg_nodes = unmapped_dfg_nodes - resource_sharing_mapped_nodes
        
        logger.info(f"number of mapped dfg nodes using resource sharing:{num_res_sharing_mapped}")
        logger.info(f"number of unmapped dfg nodes after resource sharing: {len(unmapped_dfg_nodes)}")

        ## for remaining nodes, attempt to do matching based on the the inputs and outputs of the dfg nodes.
        
        mapped_nodes_input_output_match = self.dfg_to_netlist_i_o_match(unmapped_dfg_nodes)
        logger.info(f"succesfully mapped {len(mapped_nodes_input_output_match)} nodes based on input/output matching")
        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_nodes_input_output_match
        logger.info(f"number of unmapped dfg nodes after input/output matching: {len(unmapped_dfg_nodes)}")

        ### log the unmapped dfg nodes
        if len(unmapped_dfg_nodes) > 0:
            logger.warning(f"unmapped dfg nodes after all mapping attempts: {unmapped_dfg_nodes}")


        ## print all netlist nodes that don't have a mapping in the scheduled DFG

        # create a set of all netlist nodes that are not mapped to any scheduled DFG node
        netlist_nodes_mapped = set()
        for node in self.scheduled_dfg.nodes:
            if "netlist_node" in self.scheduled_dfg.nodes[node]:
                netlist_nodes_mapped.add(self.scheduled_dfg.nodes[node]["netlist_node"])

        unmapped_netlist_nodes = set(self.netlist.nodes) - netlist_nodes_mapped
        logger.info(f"unmapped netlist nodes: {unmapped_netlist_nodes}")


        ### Finally, if there are still unmapped DFG nodes, try to match them to the netlist nodes based on the inputs and outputs of the DFG nodes and netlist nodes. 
        ### This round, if more than one resource matches, just take a random one. 
        mapped_nodes_input_output_match_approx = self.dfg_to_netlist_i_o_match(unmapped_dfg_nodes, approx_match=True)
        logger.info(f"succesfully mapped {len(mapped_nodes_input_output_match_approx)} nodes based on input/output matching")
        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_nodes_input_output_match_approx
        logger.info(f"number of unmapped dfg nodes after input/output matching: {len(unmapped_dfg_nodes)}")

        ### log the unmapped dfg nodes
        if len(unmapped_dfg_nodes) > 0:
            logger.warning(f"unmapped dfg nodes after all mapping attempts, including approximate mapping: {unmapped_dfg_nodes}")

        ### Finally, just map the rest of the nodes randomly to the netlist nodes (of the same function type).
        mapped_randomly = set()
        for node in unmapped_dfg_nodes:
            function_type = self.scheduled_dfg.nodes[node]["function"]
            netlist_nodes_of_same_type = [n for n in self.netlist.nodes if self.netlist.nodes[n]["function"] == function_type]
            if netlist_nodes_of_same_type:
                random_netlist_node = random.choice(netlist_nodes_of_same_type)
                self.scheduled_dfg.nodes[node]["netlist_node"] = random_netlist_node
                logger.info(f"mapping dfg node {node} to netlist node {random_netlist_node} (randomly)")
                mapped_randomly.add(node)

        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_randomly
        
        logger.info(f"number of unmapped dfg nodes after random mapping: {len(unmapped_dfg_nodes)}")
        if len(unmapped_dfg_nodes) > 0:
            logger.warning(f"unmapped dfg nodes after all mapping attempts, including random mapping: {unmapped_dfg_nodes}")

        with open("src/tmp/benchmark/scheduled-dfg-after-mapping.gml", "wb") as f:
            nx.write_gml(self.scheduled_dfg, f)


    def dfg_to_netlist_i_o_match(self, unmapped_dfg_nodes, approx_match=False):
        """
        Matches the unmapped DFG nodes to the netlist nodes based on the inputs and outputs of the DFG nodes and netlist nodes.
        """
        mapped_nodes_input_output_match = set()
        for node in unmapped_dfg_nodes:
            if self.scheduled_dfg.nodes[node]["function"] not in self.params.circuit_values["latency"]:
                continue

            ## get the function type from the scheduled DFG node
            function_type = self.scheduled_dfg.nodes[node]["function"]

            ## get all of the inputs and outputs of the dfg node
            dfg_inputs_unfiltered = set(self.scheduled_dfg.predecessors(node))
            dfg_outputs_unfiltered = set(self.scheduled_dfg.successors(node))

            dfg_inputs = set()
            dfg_outputs = set()

            ## filter out the edges that are resource dependencies
            for dfg_input in dfg_inputs_unfiltered:
                if self.scheduled_dfg.get_edge_data(dfg_input, node).get("resource_edge", False) == 0:
                    dfg_inputs.add(dfg_input)

            for dfg_output in dfg_outputs_unfiltered:
                if self.scheduled_dfg.get_edge_data(node, dfg_output).get("resource_edge", False) == 0:
                    dfg_outputs.add(dfg_output)

            logger.info(f"dfg node {node} inputs: {dfg_inputs}, outputs: {dfg_outputs}")

            ## see if any of the input or output nodes in the dfg from the main node have mappings to the netlist
            dfg_inputs_mapped = []
            for dfg_input in dfg_inputs:
                if "netlist_node" in self.scheduled_dfg.nodes[dfg_input]:
                    # logger.info(f"dfg input {dfg_input} is mapped to netlist node {self.scheduled_dfg.nodes[dfg_input]['netlist_node']}")
                    
                    ## store an ordered pair of the node in the dfg and the netlist node it is mapped to
                    dfg_inputs_mapped.append((dfg_input, self.scheduled_dfg.nodes[dfg_input]["netlist_node"])) 
            
            dfg_outputs_mapped = []
            for dfg_output in dfg_outputs:
                if "netlist_node" in self.scheduled_dfg.nodes[dfg_output]:
                    # logger.info(f"dfg output {dfg_output} is mapped to netlist node {self.scheduled_dfg.nodes[dfg_output]['netlist_node']}")
                    
                    ## store an ordered pair of the node in the dfg and the netlist node it is mapped to
                    dfg_outputs_mapped.append((dfg_output, self.scheduled_dfg.nodes[dfg_output]["netlist_node"]))

            logger.info(f"dfg inputs mapped: {dfg_inputs_mapped}")
            logger.info(f"dfg outputs mapped: {dfg_outputs_mapped}")

            if len(dfg_inputs_mapped) == 0 and len(dfg_outputs_mapped) == 0:
                logger.warning(f"no inputs or outputs mapped for dfg node {node}, skipping")
                continue


            ## now, see if there is a node in the netlist with the same input and output nodes as the dfg node
            potential_netlist_nodes = [] # list of potential netlist nodes that match the dfg node inputs and outputs
            
            for netlist_node in self.netlist.nodes:
                if (self.netlist.nodes[netlist_node]["function"] != function_type):
                    continue
                netlist_inputs = self.netlist.predecessors(netlist_node)
                netlist_outputs = self.netlist.successors(netlist_node)

                input_node_match = True
                for dfg_node, netlist_node in dfg_inputs_mapped:
                    if netlist_node not in netlist_inputs:
                        # logger.info(f"dfg input {dfg_node} not found in netlist inputs for node {netlist_node}")
                        input_node_match = False
                        break

                if not input_node_match:
                    continue

                output_node_match = True
                for dfg_node, netlist_node in dfg_outputs_mapped:
                    if netlist_node not in netlist_outputs:
                        # logger.info(f"dfg output {dfg_node} not found in netlist outputs for node {netlist_node}")
                        output_node_match = False
                        break

                if input_node_match and output_node_match:
                    logger.info(f"found potential netlist node {netlist_node} for dfg node {node}")
                    potential_netlist_nodes.append(netlist_node)

            if len(potential_netlist_nodes) == 0:
                logger.warning(f"no potential netlist nodes found for dfg node {node}, skipping")
            elif len(potential_netlist_nodes) > 1 and not approx_match:
                logger.warning(f"multiple potential netlist nodes found for dfg node {node}: {potential_netlist_nodes}, skipping")
            elif len(potential_netlist_nodes) > 1 and approx_match:
                logger.warning(f"multiple potential netlist nodes found for dfg node {node}: {potential_netlist_nodes}, using random one")
                random_netlist_node = random.choice(potential_netlist_nodes)
                logger.info(f"mapping dfg node {node} to netlist node {random_netlist_node}")
                self.scheduled_dfg.nodes[node]["netlist_node"] = random_netlist_node
                mapped_nodes_input_output_match.add(node)
            else:
                netlist_node = potential_netlist_nodes[0]
                logger.info(f"mapping dfg node {node} to netlist node {netlist_node}")
                self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                mapped_nodes_input_output_match.add(node)
            
        return mapped_nodes_input_output_match

    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics, benchmark_name):
        self.map_netlist_to_scheduled_dfg(benchmark_name)
        
        start_time = time.time()
        self.params.wire_length_by_edge, _ = place_n_route.place_n_route(
            self.netlist, arg_testfile, arg_parasitics
        )
        logger.info(f"wire lengths: {self.params.wire_length_by_edge}")
        
        logger.info(f"time to generate wire parasitics: {time.time()-start_time}")
        self.add_wire_delays_to_schedule()

        with open("src/tmp/benchmark/scheduled-dfg-after-openroad.gml", "wb") as f:
            nx.write_gml(self.scheduled_dfg, f)

    def add_wire_delays_to_schedule(self):
        # update scheduled dfg with wire delays
        for edge in self.scheduled_dfg.edges:
            if edge in self.dfg_to_netlist_edge_map:
                # wire delay = R * C * length^2
                self.scheduled_dfg.edges[edge]["cost"] = self.params.wire_delay(self.dfg_to_netlist_edge_map[edge])
                logger.info(f"(wire delay) {edge}: {self.scheduled_dfg.edges[edge]['cost']} ns")
                self.scheduled_dfg.edges[edge]["weight"] += self.scheduled_dfg.edges[edge]["cost"]
            else:
                self.scheduled_dfg.edges[edge]["cost"] = 0

    def update_schedule_with_latency(self):
        """
        Updates the schedule with the latency of each operation.

        Parameters:
            schedule (nx.Digraph): A list of operations in the schedule.
            latency (dict): A dictionary of operation names to their latencies.

        Returns:
            None;
            The schedule is updated in place.
        """
        for node in self.scheduled_dfg.nodes:
            if node in self.params.circuit_values["latency"]:
                self.scheduled_dfg.nodes[node]["cost"] = self.params.circuit_values["latency"][self.scheduled_dfg.nodes.data()[node]["function"]]
        for edge in self.scheduled_dfg.edges:
            func = self.scheduled_dfg.nodes.data()[edge[0]]["function"]
            self.scheduled_dfg.edges[edge]["weight"] = self.params.circuit_values["latency"][func]
        self.add_wire_delays_to_schedule()
        self.longest_paths = schedule.get_longest_paths(self.scheduled_dfg)

    def save_symbolic_memories(self):
        MemL_expr = 0
        MemReadEact_expr = 0
        MemWriteEact_expr = 0
        MemPpass_expr = 0
        OffChipIOPact_expr = 0
        BufL_expr = 0
        BufReadEact_expr = 0
        BufWriteEact_expr = 0
        BufPpass_expr = 0

        self.params.symbolic_rsc_exprs = {}
        
        for mem in self.memories:
            if self.memories[mem]["type"] == "Mem":
                MemL_expr = self.params.symbolic_mem[mem].access_time * 1e9 # convert from s to ns
                MemReadEact_expr = (self.params.symbolic_mem[mem].power.readOp.dynamic + self.params.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemWriteEact_expr = (self.params.symbolic_mem[mem].power.writeOp.dynamic + self.params.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemPpass_expr = self.params.symbolic_mem[mem].power.readOp.leakage # TODO: investigate units of this expr
                OffChipIOPact_expr = self.params.symbolic_mem[mem].io_dynamic_power * 1e-3 # convert from mW to W

            else:
                BufL_expr = self.params.symbolic_buf[mem].access_time * 1e9 # convert from s to ns
                BufReadEact_expr = (self.params.symbolic_buf[mem].power.readOp.dynamic + self.params.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufWriteEact_expr = (self.params.symbolic_buf[mem].power.writeOp.dynamic + self.params.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufPpass_expr = self.params.symbolic_buf[mem].power.readOp.leakage # TODO: investigate units of this expr

            # only need to do in first iteration
            if mem not in self.params.MemReadL:
                self.params.MemReadL[mem] = sp.symbols(f"MemReadL_{mem}")
                self.params.MemWriteL[mem] = sp.symbols(f"MemWriteL_{mem}")
                self.params.MemReadEact[mem] = sp.symbols(f"MemReadEact_{mem}")
                self.params.MemWriteEact[mem] = sp.symbols(f"MemWriteEact_{mem}")
                self.params.MemPpass[mem] = sp.symbols(f"MemPpass_{mem}")
                self.params.OffChipIOPact[mem] = sp.symbols(f"OffChipIOPact_{mem}")
                self.params.BufL[mem] = sp.symbols(f"BufL_{mem}")
                self.params.BufReadEact[mem] = sp.symbols(f"BufReadEact_{mem}")
                self.params.BufWriteEact[mem] = sp.symbols(f"BufWriteEact_{mem}")
                self.params.BufPpass[mem] = sp.symbols(f"BufPpass_{mem}")

                # update symbol table
                self.params.symbol_table[f"MemReadL_{mem}"] = self.params.MemReadL[mem]
                self.params.symbol_table[f"MemWriteL_{mem}"] = self.params.MemWriteL[mem]
                self.params.symbol_table[f"MemReadEact_{mem}"] = self.params.MemReadEact[mem]
                self.params.symbol_table[f"MemWriteEact_{mem}"] = self.params.MemWriteEact[mem]
                self.params.symbol_table[f"MemPpass_{mem}"] = self.params.MemPpass[mem]
                self.params.symbol_table[f"OffChipIOPact_{mem}"] = self.params.OffChipIOPact[mem]
                self.params.symbol_table[f"BufL_{mem}"] = self.params.BufL[mem]
                self.params.symbol_table[f"BufReadEact_{mem}"] = self.params.BufReadEact[mem]
                self.params.symbol_table[f"BufWriteEact_{mem}"] = self.params.BufWriteEact[mem]
                self.params.symbol_table[f"BufPpass_{mem}"] = self.params.BufPpass[mem]
        
            # TODO: support multiple memories in self.params
            cacti_subs_new = {
                self.params.MemReadL[mem]: MemL_expr,
                self.params.MemWriteL[mem]: MemL_expr,
                self.params.MemReadEact[mem]: MemReadEact_expr,
                self.params.MemWriteEact[mem]: MemWriteEact_expr,
                self.params.MemPpass[mem]: MemPpass_expr,
                self.params.OffChipIOPact[mem]: OffChipIOPact_expr,

                self.params.BufL[mem]: BufL_expr,
                self.params.BufReadEact[mem]: BufReadEact_expr,
                self.params.BufWriteEact[mem]: BufWriteEact_expr,
                self.params.BufPpass[mem]: BufPpass_expr,
            }
            self.params.symbolic_rsc_exprs.update(cacti_subs_new)

    def calculate_execution_time(self, symbolic):
        if symbolic:
            # take symbolic max over the critical paths
            execution_time = 0
            for path in self.longest_paths:
                logger.info(f"adding path to execution time calculation: {path}")
                path_execution_time = 0
                for i in range(len(path[1])):
                    node = path[1][i]
                    data = self.scheduled_dfg.nodes[node]
                    if node == "end" or data["function"] == "nop": continue
                    if data["function"] == "Buf" or data["function"] == "MainMem":
                        rsc_name = data["library"][data["library"].find("__")+1:]
                        logger.info(f"(execution time) rsc name: {rsc_name}, data: {data['function']}")
                        path_execution_time += self.params.symbolic_latency_wc[data["function"]]()[rsc_name]
                    else:
                        path_execution_time += self.params.symbolic_latency_wc[data["function"]]()
                    if i > 0 and (path[1][i-1], node) in self.dfg_to_netlist_edge_map:
                        path_execution_time += self.params.wire_delay(self.dfg_to_netlist_edge_map[(path[1][i-1], node)], symbolic)
                execution_time = symbolic_convex_max(execution_time, path_execution_time).simplify() if execution_time != 0 else path_execution_time

            logger.info(f"symbolic execution time: {execution_time}")
        else:
            execution_time = self.scheduled_dfg.nodes["end"]["start_time"]
        return execution_time
    
    def calculate_passive_energy(self, total_execution_time, symbolic):
        passive_power = 0
        for node in self.netlist:
            data = self.netlist.nodes[node]
            logger.info(f"calculating passive power for node {node}, data: {data}")
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                if symbolic:
                    passive_power += self.params.symbolic_power_passive[data["function"]]()[rsc_name]
                else:
                    passive_power += self.params.memories[rsc_name]["Standby leakage per bank(mW)"] * 1e6 # convert from mW to nW
            else:
                if symbolic:
                    passive_power += self.params.symbolic_power_passive[data["function"]]()
                else:
                    passive_power += self.params.circuit_values["passive_power"][data["function"]]
                logger.info(f"(passive power) {data['function']}: {self.params.circuit_values['passive_power'][data['function']]}")
        total_passive_energy = passive_power * total_execution_time*1e-9
        return total_passive_energy
        
    def calculate_active_energy(self, symbolic):
        total_active_energy = 0
        for node in self.scheduled_dfg:
            data = self.scheduled_dfg.nodes[node]
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                if symbolic:
                    total_active_energy += self.params.symbolic_energy_active[data["function"]]()[rsc_name]
                else:
                    if data["module"].find("wport") != -1:
                        total_active_energy += self.params.memories[rsc_name]["Dynamic write energy (nJ)"]
                    else:
                        total_active_energy += self.params.memories[rsc_name]["Dynamic read energy (nJ)"]
            else:
                if symbolic:
                    total_active_energy += self.params.symbolic_energy_active[data["function"]]()
                else:
                    total_active_energy += self.params.circuit_values["dynamic_energy"][data["function"]]
                logger.info(f"(active energy) {data['function']}: {total_active_energy}")
        for edge in self.scheduled_dfg.edges:
            if edge in self.dfg_to_netlist_edge_map:
                wire_energy = self.params.wire_energy(self.dfg_to_netlist_edge_map[edge], symbolic)
                logger.info(f"(wire energy) {edge}: {wire_energy} nJ")
                total_active_energy += wire_energy
        return total_active_energy
    
    def calculate_objective(self, symbolic=False):
        self.execution_time = self.calculate_execution_time(symbolic)
        self.total_passive_energy = self.calculate_passive_energy(self.execution_time, symbolic)
        self.total_active_energy = self.calculate_active_energy(symbolic)
        self.symbolic_obj_sub_exprs = {
            "execution_time": self.execution_time,
            "total_passive_energy": self.total_passive_energy,
            "total_active_energy": self.total_active_energy,
            "passive power": self.total_passive_energy/self.execution_time,
            "subthreshold leakage current": self.params.I_off,
            "gate tunneling current": self.params.I_tunnel,
            "effective threshold voltage": self.params.V_th_eff,
        }
        self.obj_sub_exprs = {
            "execution_time": self.execution_time,
            "total_passive_energy": self.total_passive_energy,
            "total_active_energy": self.total_active_energy,
            "passive power": self.total_passive_energy/self.execution_time,
        }
        if self.obj_fn == "edp":
            if symbolic:
                self.symbolic_obj = (self.total_passive_energy + self.total_active_energy) * self.execution_time
            else:
                self.obj = (self.total_passive_energy + self.total_active_energy) * self.execution_time
        elif self.obj_fn == "ed2":
            if symbolic:
                self.symbolic_obj = (self.total_passive_energy + self.total_active_energy) * (self.execution_time)**2
            else:   
                self.obj = (self.total_passive_energy + self.total_active_energy) * (self.execution_time)**2
        elif self.obj_fn == "delay":
            if symbolic:
                self.symbolic_obj = self.execution_time
            else:
                self.obj = self.execution_time
        elif self.obj_fn == "energy":
            print(f"setting energy objective to {self.total_active_energy + self.total_passive_energy}")
            if symbolic:
                self.symbolic_obj = self.total_active_energy + self.total_passive_energy
            else:
                self.obj = self.total_active_energy + self.total_passive_energy
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")