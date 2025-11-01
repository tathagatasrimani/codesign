import logging
import random
import yaml
import time
import numpy as np
import math
import copy

logger = logging.getLogger(__name__)

import networkx as nx
import sympy as sp
from src import cacti_util
from src.hardware_model.base_parameters import base_parameters
from src.hardware_model.circuit_models import circuit_model

from src.forward_pass import schedule
from src import sim_util

from src.hardware_model.tech_models import bulk_model
from src.hardware_model.tech_models import bulk_bsim4_model
from src.hardware_model.tech_models import vs_model
from src.hardware_model.tech_models import mvs_si_model
from src.hardware_model.tech_models import mvs_2_model
from src.hardware_model.tech_models import vscnfet_model
from openroad_interface import openroad_run
from openroad_interface import openroad_run_hier

import cvxpy as cp

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

def symbolic_convex_max(a, b, evaluate=True):
    """
    Max(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b + sp.Abs(a - b, evaluate=evaluate))

def symbolic_min(a, b, evaluate=True):
    """
    Min(a, b) in a format which ipopt accepts.
    """
    return 0.5 * (a + b - sp.Abs(a - b, evaluate=evaluate))

class BlockVector:
    def __init__(self):

        self.op_types = set([
            "logic",
            "memory",
            "interconnect",

            "logic_rsc",
            "memory_rsc",
            "interconnect_rsc",
        ])
        self.bound_factor = {op_type: 0 for op_type in self.op_types}
        self.normalized_bound_factor = {op_type: 0 for op_type in self.op_types}
        self.ahmdal_limit = {op_type: 0 for op_type in self.op_types}
        self.sensitivity = {op_type: 0 for op_type in self.op_types}
        self.path_mixing_factor = 0

        self.delay = 0
    
    def sensitivity_softmax(self):
        max_sensitivity = max(self.sensitivity.values())
        for op_type in self.op_types:
            self.sensitivity[op_type] = np.exp(self.sensitivity[op_type] - max_sensitivity)
        sum_sensitivity = sum(self.sensitivity.values())
        for op_type in self.op_types:
            self.sensitivity[op_type] = self.sensitivity[op_type]/sum_sensitivity

    def normalize_bound_factor(self):
        if self.delay == 0:
            return
        for op_type in self.op_types:
            self.normalized_bound_factor[op_type] = self.bound_factor[op_type]/self.delay
        self.path_mixing_factor = sum(self.normalized_bound_factor.values())
        for op_type in self.op_types:
            if self.normalized_bound_factor[op_type] == 0:
                self.ahmdal_limit[op_type] = math.inf
            else:
                self.ahmdal_limit[op_type] = 1/self.normalized_bound_factor[op_type]
    
    def __str__(self):
        return f"\n=============BlockVector=============\nDelay: {self.delay}\nSensitivity: {self.sensitivity}\nAhmdal Limit: {self.ahmdal_limit}\nBound Factor: {self.bound_factor}\nNormalized Bound Factor: {self.normalized_bound_factor}\nPath Mixing Factor: {self.path_mixing_factor}\n=============End BlockVector============="

def get_op_type_from_function(function, resource_edge=False):
    if function in ["Buf", "MainMem"]:
        return "memory" if not resource_edge else "memory_rsc"
    elif function in ["Add16", "Sub16", "Mult16", "FloorDiv16", "Mod16", "LShift16", "RShift16", "BitOr16", "BitXor16", "BitAnd16", "Eq16", "NotEq16", "Lt16", "LtE16", "Gt16", "GtE16", "USub16", "UAdd16", "IsNot16", "Not16", "Invert16", "Regs16"]:
        return "logic" if not resource_edge else "logic_rsc"
    elif function == "Wire":
        return "interconnect" if not resource_edge else "interconnect_rsc"
    else:
        return "N/A"


class HardwareModel:
    """
    Represents a hardware model with configurable technology and hardware parameters. Provides methods
    to set up the hardware, manage netlists, and extract technology-specific timing and power data for
    optimization and simulation purposes.
    """
    def __init__(self, cfg, codesign_root_dir, tmp_dir):

        args = cfg["args"]

        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
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
        with open("src/yaml/model_cfg.yaml", "r") as f:
            model_cfgs = yaml.safe_load(f)

        # model cfg is an extension of its base cfg, can create a tree of configs which need to be merged
        self.model_cfg = sim_util.recursive_cfg_merge(model_cfgs, args["model_cfg"])
        print(f"self.model_cfg: {self.model_cfg}")

        self.base_params = base_parameters.BaseParameters(args["tech_node"], self.cacti_dat_file)

        self.reset_tech_model()

        self.netlist = nx.DiGraph()
        # for catapult
        self.scheduled_dfg = nx.DiGraph()
        # for vitis
        self.scheduled_dfgs = {}
        self.loop_1x_graphs = {}
        self.loop_2x_graphs = {}
        self.top_block_name = args["benchmark"] if not args["pytorch"] else "forward"

        self.parasitic_graph = nx.DiGraph()
        self.symbolic_mem = {}
        self.symbolic_buf = {}
        self.memories = []
        self.obj_fn = args["obj"]
        self.obj = 0
        self.obj_sub_exprs = {}
        self.area_constraint = args["area"]
        self.hls_tool = args["hls_tool"]
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.constraints = []

    def reset_state(self):
        self.symbolic_buf = {}
        self.symbolic_mem = {}
        self.netlist = nx.DiGraph()
        self.memories = []
        self.obj = 0
        self.scheduled_dfg = nx.DiGraph()
        self.scheduled_dfgs = {}
        self.loop_1x_graphs = {}
        self.loop_2x_graphs = {}
        self.parasitic_graph = nx.DiGraph()
        #self.obj_sub_exprs = {}
        self.execution_time = 0
        self.total_passive_energy = 0
        self.total_active_energy = 0
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.constraints = []

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.circuit_model.circuit_values["latency"],
            "dynamic_energy": self.circuit_model.circuit_values["dynamic_energy"],
            "passive_power": self.circuit_model.circuit_values["passive_power"],
            "area": self.circuit_model.circuit_values["area"], # TODO: make sure we have this
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))

    def reset_tech_model(self):
        if self.model_cfg["model_type"] == "bulk":
            self.tech_model = bulk_model.BulkModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "bulk_bsim4":
            self.tech_model = bulk_bsim4_model.BulkBSIM4Model(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "vs":
            if self.model_cfg["vs_model_type"] == "base":
                self.tech_model = vs_model.VSModel(self.model_cfg, self.base_params)
            elif self.model_cfg["vs_model_type"] == "mvs_si":
                self.tech_model = mvs_si_model.MVSSiModel(self.model_cfg, self.base_params)
            elif self.model_cfg["vs_model_type"] == "mvs2":
                self.tech_model = mvs_2_model.MVS2Model(self.model_cfg, self.base_params)
            elif self.model_cfg["vs_model_type"] == "vscnfet":
                self.tech_model = vscnfet_model.VSCNFetModel(self.model_cfg, self.base_params)
            else:
                raise ValueError(f"Invalid vs model type: {self.model_cfg['vs_model_type']}")
        else:
            raise ValueError(f"Invalid model type: {self.model_cfg['model_type']}")
        self.tech_model.create_constraints(self.model_cfg["scaling_mode"])

        # by convention, we should always access bulk model and base params through circuit model
        self.circuit_model = circuit_model.CircuitModel(self.tech_model)

    def catapult_map_netlist_to_scheduled_dfg(self, benchmark_name):

        ## create a set of all netlist nodes
        unmapped_dfg_nodes = set(self.scheduled_dfg.nodes)
        
        ### First, attempt direct name matching between scheduled DFG nodes and netlist nodes.
        if benchmark_name == "matmult":
            for node in self.scheduled_dfg:
                if self.scheduled_dfg.nodes[node]["function"] not in self.circuit_model.circuit_values["latency"]:
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
                if self.scheduled_dfg.nodes[node]["function"] not in self.circuit_model.circuit_values["latency"]:
                    log_warning(f"skipping node {node} with function {self.scheduled_dfg.nodes[node]['function']} as it is not in the circuit values")
                    continue

                ## get the catapult name from the scheduled DFG node
                catapult_name = self.scheduled_dfg.nodes[node]["catapult_name"]

                ##E.g. catapult name: mul_inst.run()#1

                ## replace all the # and - and . and () with _ to match the netlist node names
                netlist_name = catapult_name.replace("#", "_").replace("-", "_").replace(".", "_").replace("()", "_")

                ##E.g. netlist_name: mul_inst_run_1

                log_info(f"netlist_name: {netlist_name}")

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
                    log_warning(f"no direct name match found for node {node} with catapult name {catapult_name} and function type {function_type}")


        direct_name_match_count = len(self.scheduled_dfg.nodes) - len(unmapped_dfg_nodes)
        log_info(f"number of mapped dfg nodes using direct name match: {direct_name_match_count}")
        log_info(f"number of unmapped dfg nodes after direct name match: {len(unmapped_dfg_nodes)}")
        
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
            log_info(f"res_sharing.tcl line: {line.strip()}")
            parts = line.strip().split("RESOURCE_NAME")
            key_dirty_text = parts[0]
            value_dirty_text = parts[1]

            log_info(f"res_sharing.tcl key: {key_dirty_text.strip()}")
            log_info(f"res_sharing.tcl value: {value_dirty_text.strip()}")

            key_2 = "_".join(key_dirty_text.split("/")[-1].strip().split(":")[0:3])
            log_info(f"res_sharing.tcl key_2: {key_2}")
            key = key_2.replace("#", "_").replace("-", "_").strip()  # replace all the # and - with _ to match the netlist node names

            value_2 = "_".join(value_dirty_text.split(":")[0:3])
            log_info(f"res_sharing.tcl value_2: {value_2}")
            value = value_2.replace("#", "_").replace("-", "_").strip()  # replace all the # and - with _ to match the netlist node names
            if key not in res_sharing_map:
                res_sharing_map[key] = value
                log_info(f"res_sharing_map: {key} -> {value}")
            else:
                log_warning(f"duplicate key {key} in res_sharing.tcl file.")

        
        num_res_sharing_mapped = 0

        resource_sharing_mapped_nodes = set()

        ## map the unmapped dfg nodes to the netlist nodes using the resource sharing information
        for node in unmapped_dfg_nodes:
            if self.scheduled_dfg.nodes[node]["function"] not in self.circuit_model.circuit_values["latency"]:
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
        
        log_info(f"number of mapped dfg nodes using resource sharing:{num_res_sharing_mapped}")
        log_info(f"number of unmapped dfg nodes after resource sharing: {len(unmapped_dfg_nodes)}")

        ## for remaining nodes, attempt to do matching based on the the inputs and outputs of the dfg nodes.
        
        mapped_nodes_input_output_match = self.dfg_to_netlist_i_o_match(unmapped_dfg_nodes)
        log_info(f"succesfully mapped {len(mapped_nodes_input_output_match)} nodes based on input/output matching")
        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_nodes_input_output_match
        log_info(f"number of unmapped dfg nodes after input/output matching: {len(unmapped_dfg_nodes)}")

        ### log the unmapped dfg nodes
        if len(unmapped_dfg_nodes) > 0:
            log_warning(f"unmapped dfg nodes after all mapping attempts: {unmapped_dfg_nodes}")


        ## print all netlist nodes that don't have a mapping in the scheduled DFG

        # create a set of all netlist nodes that are not mapped to any scheduled DFG node
        netlist_nodes_mapped = set()
        for node in self.scheduled_dfg.nodes:
            if "netlist_node" in self.scheduled_dfg.nodes[node]:
                netlist_nodes_mapped.add(self.scheduled_dfg.nodes[node]["netlist_node"])

        unmapped_netlist_nodes = set(self.netlist.nodes) - netlist_nodes_mapped
        log_info(f"unmapped netlist nodes: {unmapped_netlist_nodes}")


        ### Finally, if there are still unmapped DFG nodes, try to match them to the netlist nodes based on the inputs and outputs of the DFG nodes and netlist nodes. 
        ### This round, if more than one resource matches, just take a random one. 
        mapped_nodes_input_output_match_approx = self.dfg_to_netlist_i_o_match(unmapped_dfg_nodes, approx_match=True)
        log_info(f"succesfully mapped {len(mapped_nodes_input_output_match_approx)} nodes based on input/output matching")
        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_nodes_input_output_match_approx
        log_info(f"number of unmapped dfg nodes after input/output matching: {len(unmapped_dfg_nodes)}")

        ### log the unmapped dfg nodes
        if len(unmapped_dfg_nodes) > 0:
            log_warning(f"unmapped dfg nodes after all mapping attempts, including approximate mapping: {unmapped_dfg_nodes}")

        ### Finally, just map the rest of the nodes randomly to the netlist nodes (of the same function type).
        mapped_randomly = set()
        for node in unmapped_dfg_nodes:
            function_type = self.scheduled_dfg.nodes[node]["function"]
            netlist_nodes_of_same_type = [n for n in self.netlist.nodes if self.netlist.nodes[n]["function"] == function_type]
            if netlist_nodes_of_same_type:
                random_netlist_node = random.choice(netlist_nodes_of_same_type)
                self.scheduled_dfg.nodes[node]["netlist_node"] = random_netlist_node
                log_info(f"mapping dfg node {node} to netlist node {random_netlist_node} (randomly)")
                mapped_randomly.add(node)

        unmapped_dfg_nodes = unmapped_dfg_nodes - mapped_randomly
        
        log_info(f"number of unmapped dfg nodes after random mapping: {len(unmapped_dfg_nodes)}")
        if len(unmapped_dfg_nodes) > 0:
            log_warning(f"unmapped dfg nodes after all mapping attempts, including random mapping: {unmapped_dfg_nodes}")

        with open("src/tmp/benchmark/scheduled-dfg-after-mapping.gml", "wb") as f:
            nx.write_gml(self.scheduled_dfg, f)


    def dfg_to_netlist_i_o_match(self, unmapped_dfg_nodes, approx_match=False):
        """
        Matches the unmapped DFG nodes to the netlist nodes based on the inputs and outputs of the DFG nodes and netlist nodes.
        """
        mapped_nodes_input_output_match = set()
        for node in unmapped_dfg_nodes:
            if self.scheduled_dfg.nodes[node]["function"] not in self.circuit_model.circuit_values["latency"]:
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

            log_info(f"dfg node {node} inputs: {dfg_inputs}, outputs: {dfg_outputs}")

            ## see if any of the input or output nodes in the dfg from the main node have mappings to the netlist
            dfg_inputs_mapped = []
            for dfg_input in dfg_inputs:
                if "netlist_node" in self.scheduled_dfg.nodes[dfg_input]:
                    # log_info(f"dfg input {dfg_input} is mapped to netlist node {self.scheduled_dfg.nodes[dfg_input]['netlist_node']}")
                    
                    ## store an ordered pair of the node in the dfg and the netlist node it is mapped to
                    dfg_inputs_mapped.append((dfg_input, self.scheduled_dfg.nodes[dfg_input]["netlist_node"])) 
            
            dfg_outputs_mapped = []
            for dfg_output in dfg_outputs:
                if "netlist_node" in self.scheduled_dfg.nodes[dfg_output]:
                    # log_info(f"dfg output {dfg_output} is mapped to netlist node {self.scheduled_dfg.nodes[dfg_output]['netlist_node']}")
                    
                    ## store an ordered pair of the node in the dfg and the netlist node it is mapped to
                    dfg_outputs_mapped.append((dfg_output, self.scheduled_dfg.nodes[dfg_output]["netlist_node"]))

            log_info(f"dfg inputs mapped: {dfg_inputs_mapped}")
            log_info(f"dfg outputs mapped: {dfg_outputs_mapped}")

            if len(dfg_inputs_mapped) == 0 and len(dfg_outputs_mapped) == 0:
                log_warning(f"no inputs or outputs mapped for dfg node {node}, skipping")
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
                        # log_info(f"dfg input {dfg_node} not found in netlist inputs for node {netlist_node}")
                        input_node_match = False
                        break

                if not input_node_match:
                    continue

                output_node_match = True
                for dfg_node, netlist_node in dfg_outputs_mapped:
                    if netlist_node not in netlist_outputs:
                        # log_info(f"dfg output {dfg_node} not found in netlist outputs for node {netlist_node}")
                        output_node_match = False
                        break

                if input_node_match and output_node_match:
                    log_info(f"found potential netlist node {netlist_node} for dfg node {node}")
                    potential_netlist_nodes.append(netlist_node)

            if len(potential_netlist_nodes) == 0:
                log_warning(f"no potential netlist nodes found for dfg node {node}, skipping")
            elif len(potential_netlist_nodes) > 1 and not approx_match:
                log_warning(f"multiple potential netlist nodes found for dfg node {node}: {potential_netlist_nodes}, skipping")
            elif len(potential_netlist_nodes) > 1 and approx_match:
                log_warning(f"multiple potential netlist nodes found for dfg node {node}: {potential_netlist_nodes}, using random one")
                random_netlist_node = random.choice(potential_netlist_nodes)
                log_info(f"mapping dfg node {node} to netlist node {random_netlist_node}")
                self.scheduled_dfg.nodes[node]["netlist_node"] = random_netlist_node
                mapped_nodes_input_output_match.add(node)
            else:
                netlist_node = potential_netlist_nodes[0]
                log_info(f"mapping dfg node {node} to netlist node {netlist_node}")
                self.scheduled_dfg.nodes[node]["netlist_node"] = netlist_node
                mapped_nodes_input_output_match.add(node)
            
        return mapped_nodes_input_output_match

    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics, benchmark_name, run_openroad, area_constraint=None):
        if self.hls_tool == "catapult":
            self.catapult_map_netlist_to_scheduled_dfg(benchmark_name)
        
        start_time = time.time()

        netlist_copy = copy.deepcopy(self.netlist)

        logger.info(f"num nodes in netlist before openroad: {len(netlist_copy.nodes)}")

        L_eff = self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.L]
        logger.info(f"current L_eff for get_wire_parascitics: {L_eff}")

        ## hierarchical openroad run
        hier_open_road_run = openroad_run_hier.OpenRoadRunHier(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=run_openroad)

        hls_parse_results_dir = f"benchmark/parse_results"

        hier_open_road_run.run_hierarchical_openroad(
            netlist_copy,
            arg_testfile,
            arg_parasitics,
            area_constraint,
            L_eff,
            hls_parse_results_dir,
            "forward"
        )

        exit(1)

        ## flat openroad run
        # open_road_run = openroad_run.OpenRoadRun(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=run_openroad)

        # self.circuit_model.wire_length_by_edge, _, _ = open_road_run.run(
        #     netlist_copy, arg_testfile, arg_parasitics, area_constraint, L_eff
        # )

        log_info(f"wire lengths: {self.circuit_model.wire_length_by_edge}")
        
        logger.info(f"time to generate wire parasitics: {time.time()-start_time}")

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

        self.circuit_model.symbolic_rsc_exprs = {}
        
        for mem in self.memories:
            if self.memories[mem]["type"] == "Mem":
                MemL_expr = self.circuit_model.symbolic_mem[mem].access_time * 1e9 # convert from s to ns
                MemReadEact_expr = (self.circuit_model.symbolic_mem[mem].power.readOp.dynamic + self.circuit_model.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemWriteEact_expr = (self.circuit_model.symbolic_mem[mem].power.writeOp.dynamic + self.circuit_model.symbolic_mem[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                MemPpass_expr = self.circuit_model.symbolic_mem[mem].power.readOp.leakage # TODO: investigate units of this expr
                OffChipIOPact_expr = self.circuit_model.symbolic_mem[mem].io_dynamic_power * 1e-3 # convert from mW to W

            else:
                BufL_expr = self.circuit_model.symbolic_buf[mem].access_time * 1e9 # convert from s to ns
                BufReadEact_expr = (self.circuit_model.symbolic_buf[mem].power.readOp.dynamic + self.circuit_model.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufWriteEact_expr = (self.circuit_model.symbolic_buf[mem].power.writeOp.dynamic + self.circuit_model.symbolic_buf[mem].power.searchOp.dynamic) * 1e9 # convert from J to nJ
                BufPpass_expr = self.circuit_model.symbolic_buf[mem].power.readOp.leakage # TODO: investigate units of this expr

            # only need to do in first iteration
            if mem not in self.circuit_model.tech_model.base_params.MemReadL:
                self.circuit_model.tech_model.base_params.MemReadL[mem] = sp.symbols(f"MemReadL_{mem}")
                self.circuit_model.tech_model.base_params.MemWriteL[mem] = sp.symbols(f"MemWriteL_{mem}")
                self.circuit_model.tech_model.base_params.MemReadEact[mem] = sp.symbols(f"MemReadEact_{mem}")
                self.circuit_model.tech_model.base_params.MemWriteEact[mem] = sp.symbols(f"MemWriteEact_{mem}")
                self.circuit_model.tech_model.base_params.MemPpass[mem] = sp.symbols(f"MemPpass_{mem}")
                self.circuit_model.tech_model.base_params.OffChipIOPact[mem] = sp.symbols(f"OffChipIOPact_{mem}")
                self.circuit_model.tech_model.base_params.BufL[mem] = sp.symbols(f"BufL_{mem}")
                self.circuit_model.tech_model.base_params.BufReadEact[mem] = sp.symbols(f"BufReadEact_{mem}")
                self.circuit_model.tech_model.base_params.BufWriteEact[mem] = sp.symbols(f"BufWriteEact_{mem}")
                self.circuit_model.tech_model.base_params.BufPpass[mem] = sp.symbols(f"BufPpass_{mem}")

                # update symbol table
                self.circuit_model.tech_model.base_params.symbol_table[f"MemReadL_{mem}"] = self.circuit_model.tech_model.base_params.MemReadL[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"MemWriteL_{mem}"] = self.circuit_model.tech_model.base_params.MemWriteL[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"MemReadEact_{mem}"] = self.circuit_model.tech_model.base_params.MemReadEact[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"MemWriteEact_{mem}"] = self.circuit_model.tech_model.base_params.MemWriteEact[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"MemPpass_{mem}"] = self.circuit_model.tech_model.base_params.MemPpass[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"OffChipIOPact_{mem}"] = self.circuit_model.tech_model.base_params.OffChipIOPact[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"BufL_{mem}"] = self.circuit_model.tech_model.base_params.BufL[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"BufReadEact_{mem}"] = self.circuit_model.tech_model.base_params.BufReadEact[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"BufWriteEact_{mem}"] = self.circuit_model.tech_model.base_params.BufWriteEact[mem]
                self.circuit_model.tech_model.base_params.symbol_table[f"BufPpass_{mem}"] = self.circuit_model.tech_model.base_params.BufPpass[mem]
        
            # TODO: support multiple memories in self.params
            cacti_subs_new = {
                self.circuit_model.tech_model.base_params.MemReadL[mem]: MemL_expr,
                self.circuit_model.tech_model.base_params.MemWriteL[mem]: MemL_expr,
                self.circuit_model.tech_model.base_params.MemReadEact[mem]: MemReadEact_expr,
                self.circuit_model.tech_model.base_params.MemWriteEact[mem]: MemWriteEact_expr,
                self.circuit_model.tech_model.base_params.MemPpass[mem]: MemPpass_expr,
                self.circuit_model.tech_model.base_params.OffChipIOPact[mem]: OffChipIOPact_expr,

                self.circuit_model.tech_model.base_params.BufL[mem]: BufL_expr,
                self.circuit_model.tech_model.base_params.BufReadEact[mem]: BufReadEact_expr,
                self.circuit_model.tech_model.base_params.BufWriteEact[mem]: BufWriteEact_expr,
                self.circuit_model.tech_model.base_params.BufPpass[mem]: BufPpass_expr,
            }
            self.circuit_model.symbolic_rsc_exprs.update(cacti_subs_new)

    def calculate_active_energy_vitis(self):
        total_active_energy = 0
        for basic_block_name in self.scheduled_dfgs:
            total_active_energy += self.calculate_active_energy_basic_block(basic_block_name, self.scheduled_dfgs[basic_block_name])
        return total_active_energy

    def get_rsc_edge(self, edge, dfg):
        if "rsc" in dfg.nodes[edge[0]] and "rsc" in dfg.nodes[edge[1]]:
            return (dfg.nodes[edge[0]]["rsc"], dfg.nodes[edge[1]]["rsc"])
        else:
            return edge

    def calculate_active_energy_basic_block(self, basic_block_name, dfg, is_loop=False):
        total_active_energy_basic_block = 0
        loop_count = 1
        loop_energy = 0
        for node, data in dfg.nodes(data=True):
            if data["function"] == "II": 
                loop_count = int(data["count"])
                if is_loop:
                    loop_energy = self.calculate_active_energy_basic_block(basic_block_name, self.loop_1x_graphs[basic_block_name], is_loop=True)
            elif data["function"] == "Wire":
                src = data["src_node"]
                dst = data["dst_node"]
                rsc_edge = self.get_rsc_edge((src, dst), dfg)
                if rsc_edge in self.circuit_model.wire_length_by_edge:
                    total_active_energy_basic_block += self.circuit_model.wire_energy(rsc_edge)
                    log_info(f"edge {rsc_edge} is in circuit_model.wire_length_by_edge")
                    log_info(f"wire energy for {node}: {self.circuit_model.wire_energy(rsc_edge)}")
                else:
                    log_info(f"edge {rsc_edge} is not in circuit_model.wire_length_by_edge")
            else:
                total_active_energy_basic_block += self.circuit_model.symbolic_energy_active[data["function"]]()
                log_info(f"active energy for {node}: {self.circuit_model.symbolic_energy_active[data['function']]()}")
        log_info(f"total active energy for {basic_block_name}: {total_active_energy_basic_block}")
        log_info(f"loop count for {basic_block_name}: {loop_count}")
        if is_loop:
            log_info(f"loop energy for {basic_block_name}: {loop_energy}")
            return total_active_energy_basic_block * (loop_count-1)
        else:
            return total_active_energy_basic_block + (loop_count-1) * loop_energy

    
    def calculate_passive_power_vitis(self, total_execution_time):
        total_passive_power = 0
        for node, data in self.netlist.nodes(data=True):
            total_passive_power += self.circuit_model.symbolic_power_passive[data["function"]]()
            log_info(f"passive power for {node}: {self.circuit_model.symbolic_power_passive[data['function']]()}")
        self.total_passive_power = total_passive_power
        return total_passive_power * total_execution_time

    def update_execution_time_vitis(self):
        start_time = time.time()
        self.circuit_model.update_uarch_parameters()
        self.circuit_model.create_constraints_cvx(self.scale_cvx)
        

        # if our performance not that sensitive to frequency, just hold frequency constant
        if self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_ahmdal_limit] > 10:
            logger.info(f"holding frequency constant because logic_ahmdal_limit > 10")
            clk_period_constraints = [self.circuit_model.clk_period_cvx == self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]]
        else:
            logger.info(f"allowing frequency to vary because logic_ahmdal_limit <= 10")
            clk_period_constraints = self.circuit_model.constraints_cvx
        
        for constr in clk_period_constraints:
            log_info(f"clock period constraint final: {constr}")
        logger.info(f"time to create constraints cvx: {time.time()-start_time}")
        start_time = time.time()
        #prob = cp.Problem(cp.Minimize(self.graph_delays_cvx[self.top_block_name]), self.constr_cvx+clk_period_constraints)
        prob = cp.Problem(cp.Minimize(self.circuit_model.clk_period_cvx), clk_period_constraints)
        prob.solve()
        #self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.node_arrivals_end] = self.graph_delays_cvx[self.top_block_name].value / self.scale_cvx
        logger.info(f"time to update execution time with cvxpy: {time.time()-start_time}")
        self.calculate_block_vectors(self.top_block_name)
        #return self.block_vectors[self.top_block_name]["top"].delay
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.node_arrivals_end] = self.block_vectors[self.top_block_name]["top"].delay
        return self.circuit_model.tech_model.base_params.node_arrivals_end

    def update_state_after_cvxpy_solve(self):
        """for block_name in self.graph_delays:
            for node in self.node_arrivals[block_name]["full"]:
                if self.node_arrivals_cvx[block_name]["full"][node].value is not None:
                    self.circuit_model.tech_model.base_params.tech_values[self.node_arrivals[block_name]["full"][node]] = self.node_arrivals_cvx[block_name]["full"][node].value / self.scale_cvx
                if node in self.node_arrivals[block_name]["loop_1x"] and self.node_arrivals_cvx[block_name]["loop_1x"][node].value is not None:
                    self.circuit_model.tech_model.base_params.tech_values[self.node_arrivals[block_name]["loop_1x"][node]] = self.node_arrivals_cvx[block_name]["loop_1x"][node].value / self.scale_cvx
                if node in self.node_arrivals[block_name]["loop_2x"] and self.node_arrivals_cvx[block_name]["loop_2x"][node].value is not None:
                    self.circuit_model.tech_model.base_params.tech_values[self.node_arrivals[block_name]["loop_2x"][node]] = self.node_arrivals_cvx[block_name]["loop_2x"][node].value / self.scale_cvx
            for node in self.node_arrivals_cvx[block_name]["loop_1x"]:
                if self.node_arrivals_cvx[block_name]["loop_1x"][node].value is not None:
                    self.circuit_model.tech_model.base_params.tech_values[self.node_arrivals[block_name]["loop_1x"][node]] = self.node_arrivals_cvx[block_name]["loop_1x"][node].value / self.scale_cvx
            for node in self.graph_delays:
                self.circuit_model.tech_model.base_params.tech_values[self.graph_delays[node]] = self.graph_delays_cvx[node].value / self.scale_cvx
        for block_name in self.graph_delays:
            log_info(f"graph delays for {block_name}: {sim_util.xreplace_safe(self.graph_delays[block_name], self.circuit_model.tech_model.base_params.tech_values)}")
        for block_name in self.node_arrivals:
            for graph_type in self.node_arrivals[block_name]:
                for node in self.node_arrivals[block_name][graph_type]:
                    if self.node_arrivals_cvx[block_name][graph_type][node].value is not None:
                        log_info(f"node arrivals for {block_name} {graph_type} {node}: {self.node_arrivals_cvx[block_name][graph_type][node].value / self.scale_cvx}")
                    else:
                        log_info(f"node arrivals for {block_name} {graph_type} {node}: None")"""
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.node_arrivals_end] = self.graph_delays_cvx[self.top_block_name].value / self.scale_cvx
        log_info(f"graph delay cvx for top block: {self.graph_delays_cvx[self.top_block_name].value / self.scale_cvx}")

    def calculate_block_vectors(self, top_block_name):
        self.circuit_model.update_uarch_parameters()
        logger.info("calculating block vectors")
        self.block_vectors = {}
        for basic_block_name in self.scheduled_dfgs:
            self.block_vectors[basic_block_name] = {}
        self.block_vectors[top_block_name]["top"] = self.calculate_block_vector_basic_block(top_block_name, "full", self.scheduled_dfgs[top_block_name])
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["logic"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_resource_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["logic_rsc"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["logic"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_resource_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["logic_rsc"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.interconnect_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["interconnect"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.interconnect_resource_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["interconnect_rsc"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.interconnect_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["interconnect"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.interconnect_resource_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["interconnect_rsc"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.memory_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["memory"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.memory_resource_sensitivity] = self.block_vectors[top_block_name]["top"].sensitivity["memory_rsc"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.memory_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["memory"]
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.memory_resource_ahmdal_limit] = self.block_vectors[top_block_name]["top"].ahmdal_limit["memory_rsc"]

    def make_graph_one_op_type(self, basic_block_name, graph_type, op_type, eps, dfg):
        G_new = dfg.copy()
        for edge in G_new.edges:
            fn = G_new.nodes[edge[0]]["function"]
            base_delay = self.block_vectors[basic_block_name][graph_type][edge].delay if op_type == "all" else self.block_vectors[basic_block_name][graph_type][edge].bound_factor[op_type]
            # eps is zero unless we are calculating sensitivity
            percent_add = eps * self.block_vectors[basic_block_name][graph_type][edge].sensitivity[op_type] if op_type != "all" else eps
            G_new.edges[edge]["weight"] = (1+percent_add) * base_delay
        return G_new


    def calculate_top_vector(self, basic_block_name, graph_type, dfg):
        eps = 1e-2
        vector_top = BlockVector()
        iso_op_type_graphs = {}
        # calculate bound factor and delay
        for op_type in vector_top.op_types:
            iso_op_type_graphs[op_type] = self.make_graph_one_op_type(basic_block_name, graph_type, op_type, 0, dfg)
            crit_path = nx.dag_longest_path(iso_op_type_graphs[op_type])
            log_info(f"crit path for {op_type}: {crit_path}")
            for i in range(len(crit_path)-1):
                vector_top.bound_factor[op_type] += self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].bound_factor[op_type]
            log_info(f"crit path delay for {op_type}: {vector_top.bound_factor[op_type]}")
        all_op_graph = self.make_graph_one_op_type(basic_block_name, graph_type, "all", 0, dfg)
        crit_path = nx.dag_longest_path(all_op_graph)
        for i in range(len(crit_path)-1):
            log_info(f"crit path delay for all: {self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].delay}")
            vector_top.delay += self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].delay
    
        iso_op_type_graphs_with_eps = {}
        # calculate sensitivity, should range between 0 and 1
        for op_type in vector_top.op_types:
            # using full graph for this part
            iso_op_type_graphs_with_eps[op_type] = self.make_graph_one_op_type(basic_block_name, graph_type, "all", eps, dfg)
            crit_path = nx.dag_longest_path(iso_op_type_graphs_with_eps[op_type])
            crit_path_eps = 0
            for i in range(len(crit_path)-1):
                log_info(f"crit path bound factor for {op_type}: {self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].bound_factor[op_type]}")
                log_info(f"crit path sensitivity for {op_type}: {self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].sensitivity[op_type]}")
                crit_path_eps += self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].delay * (1+eps*self.block_vectors[basic_block_name][graph_type][(crit_path[i], crit_path[i+1])].sensitivity[op_type])
            log_info(f"crit path eps for {op_type}: {crit_path_eps}")
            log_info(f"crit path delay for {op_type}: {vector_top.delay}")
            vector_top.sensitivity[op_type] = 0 if vector_top.delay == 0 else (crit_path_eps - vector_top.delay) / (eps * vector_top.delay)
            assert vector_top.sensitivity[op_type] >= 0-eps and vector_top.sensitivity[op_type] <= 1+eps, f"sensitivity for {op_type} is {vector_top.sensitivity[op_type]}"
        vector_top.normalize_bound_factor()
        log_info(f"top vector for {basic_block_name} {graph_type}: {str(vector_top)}")
        if vector_top.delay != 0:
            assert 1-eps <= sum(vector_top.normalized_bound_factor.values()) <= 6+eps, f"sum of normalized bound factors for {basic_block_name} {graph_type} is {sum(vector_top.normalized_bound_factor.values())}"

        return vector_top

    def calculate_block_vector_basic_block(self, basic_block_name, graph_type, dfg):
        self.block_vectors[basic_block_name][graph_type] = {}
        for node in dfg.nodes:
            for pred in dfg.predecessors(node):
                if (pred, node) in self.block_vectors[basic_block_name][graph_type]:
                    continue # nothing to be done
                # calculate vector for II delay based on the resource constrained 1x loop iteration graph
                if dfg.nodes[pred]["function"] == "II":
                    loop_1x_vector = self.calculate_block_vector_basic_block(basic_block_name, "loop_1x", self.loop_1x_graphs[basic_block_name])
                    loop_1x_vector.delay *= int(dfg.nodes[pred]["count"])-1
                    for op_type in loop_1x_vector.op_types:
                        loop_1x_vector.bound_factor[op_type] *= int(dfg.nodes[pred]["count"])-1
                    loop_1x_vector.normalize_bound_factor()
                    self.block_vectors[basic_block_name][graph_type][(pred, node)] = loop_1x_vector
                # calculate vector for sub-function call
                elif dfg.nodes[pred]["function"] == "Call":
                    sub_block_name = dfg.nodes[pred]["call_function"]
                    if sub_block_name not in self.block_vectors or "top" not in self.block_vectors[sub_block_name]:
                        self.block_vectors[sub_block_name]["top"] = self.calculate_block_vector_basic_block(sub_block_name, graph_type, self.scheduled_dfgs[sub_block_name])
                    self.block_vectors[basic_block_name][graph_type][(pred, node)] = self.block_vectors[sub_block_name]["top"]
                # calculate vector for a basic operation
                else:
                    self.block_vectors[basic_block_name][graph_type][(pred, node)] = self.calculate_block_vector_edge(pred, node, basic_block_name, graph_type, dfg)

        return self.calculate_top_vector(basic_block_name, graph_type, dfg)

    def calculate_block_vector_edge(self, src, dst, basic_block_name, graph_type, dfg):
        fn = dfg.nodes[src]["function"]
        vector = BlockVector()
        if dfg.edges[src, dst]["resource_edge"]:
            # TODO add interconnect resource dependency case
            vector.delay = self.circuit_model.clk_period_cvx.value
            if fn in ["Buf", "MainMem"]:
                vector.bound_factor["memory_rsc"] = vector.delay
                vector.sensitivity["memory_rsc"] = 1
            else: # logic
                vector.bound_factor["logic_rsc"] = vector.delay
                vector.sensitivity["logic_rsc"] = 1
        else:
            if fn == "Wire":
                src_for_wire = dfg.nodes[src]["src_node"]
                dst_for_wire = dfg.nodes[src]["dst_node"]
                rsc_edge = self.get_rsc_edge((src_for_wire, dst_for_wire), dfg)
                if rsc_edge in self.circuit_model.wire_length_by_edge:
                    vector.delay = self.circuit_model.wire_delay_uarch_cvx(rsc_edge).value
                    log_info(f"added wire delay {vector.delay} for {rsc_edge}, which has length {self.circuit_model.wire_length(rsc_edge)}")
                else:
                    vector.delay = 0
                    log_info(f"edge {rsc_edge} not in wire_length_by_edge")
                vector.bound_factor["interconnect"] = vector.delay
                vector.sensitivity["interconnect"] = 1
            elif fn in ["Buf", "MainMem"]:
                # TODO actually fetch memory latency once implemented, this is a placeholder
                vector.delay = sim_util.xreplace_safe(self.circuit_model.symbolic_latency_wc[fn](), self.circuit_model.tech_model.base_params.tech_values)
                vector.bound_factor["memory"] = vector.delay
                vector.sensitivity["memory"] = 1
            else: # logic
                vector.delay = sim_util.xreplace_safe(self.circuit_model.symbolic_latency_wc[fn](), self.circuit_model.tech_model.base_params.tech_values)
                vector.bound_factor["logic"] = vector.delay
                vector.sensitivity["logic"] = 1
        vector.normalize_bound_factor()
        log_info(f"block vector for {src, dst}: {str(vector)}")
        return vector


    def calculate_execution_time_vitis(self, top_block_name, clk_period_opt=False, form_dfg=True):
        if not form_dfg:
            return self.update_execution_time_vitis()
        self.circuit_model.update_uarch_parameters()
        #self.node_arrivals = {}
        self.node_arrivals_cvx = {}
        self.graph_delays = {}
        self.graph_delays_cvx = {}
        self.constr_cvx = []

        self.scale_cvx = 1e-6

        log_info(f"scheduled dfgs: {self.scheduled_dfgs.keys()}")
        start_time = time.time()

        for basic_block_name in self.scheduled_dfgs:
            #self.node_arrivals[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}}
            self.node_arrivals_cvx[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}}

        self.graph_delays_cvx[top_block_name] = self.calculate_execution_time_vitis_recursive(top_block_name, self.scheduled_dfgs[top_block_name])

        if not clk_period_opt:
            clk_period_constr = [self.circuit_model.clk_period_cvx== self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]]
        else:
            self.circuit_model.create_constraints_cvx(self.scale_cvx)
            if self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.logic_ahmdal_limit] > 10:
                logger.info(f"holding frequency constant because logic_ahmdal_limit > 10")
                clk_period_constr = [self.circuit_model.clk_period_cvx == self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]]
            else:
                logger.info(f"allowing frequency to vary because logic_ahmdal_limit <= 10")
                clk_period_constr = self.circuit_model.constraints_cvx
        for constr in self.constr_cvx:
            log_info(f"constraint final: {constr}")
        for constr in clk_period_constr:
            log_info(f"clock period constraint final: {constr}")
        for node in self.node_arrivals_cvx[top_block_name]["full"]:
            log_info(f"node arrivals cvx var for {top_block_name} full {node}: {self.node_arrivals_cvx[top_block_name]['full'][node]}")
        logger.info(f"time to create cvxpy problem: {time.time()-start_time}")
        start_time = time.time()
        #prob = cp.Problem(cp.Minimize(self.graph_delays_cvx[top_block_name]), self.constr_cvx+clk_period_constr)
        prob = cp.Problem(cp.Minimize(self.circuit_model.clk_period_cvx), clk_period_constr)
        prob.solve()
        logger.info(f"time to solve cvxpy problem: {time.time()-start_time}")
        #self.update_state_after_cvxpy_solve()
        self.calculate_block_vectors(top_block_name)
        self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.node_arrivals_end] = self.block_vectors[top_block_name]["top"].delay
        return self.circuit_model.tech_model.base_params.node_arrivals_end


    def calculate_execution_time_vitis_recursive(self, basic_block_name, dfg, graph_end_node="graph_end", graph_type="full", resource_delays_only=False):
        log_info(f"calculating execution time for {basic_block_name} with graph end node {graph_end_node}")
        for node in dfg.nodes:
            #self.node_arrivals[basic_block_name][graph_type][node] = sp.symbols(f"node_arrivals_{basic_block_name}_{graph_type}_{node}")
            self.node_arrivals_cvx[basic_block_name][graph_type][node] = cp.Variable(pos=True) 
        for node in dfg.nodes:   
            for pred in dfg.predecessors(node):
                pred_delay_cvx = 0.0
                if dfg.edges[pred, node]["resource_edge"]:
                    if dfg.nodes[pred]["function"] == "II":
                        delay_1x_cvx = self.calculate_execution_time_vitis_recursive(basic_block_name, self.loop_1x_graphs[basic_block_name], graph_end_node="loop_end_1x", graph_type="loop_1x", resource_delays_only=True)
                        #delay_2x, delay_2x_cvx = self.calculate_execution_time_vitis_recursive(basic_block_name, self.loop_2x_graphs[basic_block_name], graph_end_node="loop_end_2x", graph_type="loop_2x")
                        # TODO add dependence of II on loop-carried dependency
                        #pred_delay = delay_1x * (dfg.nodes[pred]["count"]-1)
                        #print(dfg.nodes[pred]["count"])
                        pred_delay_cvx = delay_1x_cvx * (int(dfg.nodes[pred]["count"])-1)
                    else:
                        #pred_delay = self.circuit_model.tech_model.base_params.clk_period # convert to ns
                        pred_delay_cvx = self.circuit_model.clk_period_cvx * self.scale_cvx
                elif dfg.nodes[pred]["function"] == "Call": # if function call, recursively calculate its delay 
                    if dfg.nodes[pred]["call_function"] not in self.graph_delays:
                        self.graph_delays_cvx[dfg.nodes[pred]["call_function"]] = self.calculate_execution_time_vitis_recursive(dfg.nodes[pred]["call_function"], self.scheduled_dfgs[dfg.nodes[pred]["call_function"]])
                    #pred_delay = self.graph_delays[dfg.nodes[pred]["call_function"]]
                    pred_delay_cvx = self.graph_delays_cvx[dfg.nodes[pred]["call_function"]]
                elif not resource_delays_only:
                    if dfg.nodes[pred]["function"] == "Wire":
                        src = dfg.nodes[pred]["src_node"]
                        dst = dfg.nodes[pred]["dst_node"]
                        rsc_edge = self.get_rsc_edge((src, dst), dfg)
                        if rsc_edge in self.circuit_model.wire_length_by_edge:
                            pred_delay_cvx = self.circuit_model.wire_delay_uarch_cvx(rsc_edge) * self.scale_cvx
                            log_info(f"added wire delay {self.circuit_model.wire_delay_uarch_cvx(rsc_edge)} for edge {rsc_edge}")
                        else:
                            log_info(f"no wire delay for edge {rsc_edge}")
                    else:
                        #pred_delay = self.circuit_model.symbolic_latency_wc[dfg.nodes[pred]["function"]]()
                        pred_delay_cvx = self.circuit_model.uarch_lat_cvx[dfg.nodes[pred]["function"]] * self.scale_cvx
                log_info(f"pred_delay_cvx: {pred_delay_cvx}")
                if isinstance(pred_delay_cvx, cp.Expression):
                    log_info(f"pred_delay_cvx value: {pred_delay_cvx.value}")
                #log_info(f"pred_delay: {pred_delay}")
                assert pred_delay_cvx is not None and not isinstance(pred_delay_cvx, sp.Expr), f"pred_delay_cvx is {pred_delay_cvx}, type: {type(pred_delay_cvx)}"
                #self.constraints.append(self.node_arrivals[basic_block_name][graph_type][node] >= self.node_arrivals[basic_block_name][graph_type][pred] + pred_delay)

                #log_info(f"constraint: {self.node_arrivals[basic_block_name][graph_type][node] >= self.node_arrivals[basic_block_name][graph_type][pred] + pred_delay}")
                self.constr_cvx.append(self.node_arrivals_cvx[basic_block_name][graph_type][node] >= self.node_arrivals_cvx[basic_block_name][graph_type][pred] + pred_delay_cvx)
                log_info(f"constraint cvx: {self.node_arrivals_cvx[basic_block_name][graph_type][node] >= self.node_arrivals_cvx[basic_block_name][graph_type][pred] + pred_delay_cvx}")
        return self.node_arrivals_cvx[basic_block_name][graph_type][graph_end_node]

    def calculate_execution_time(self, symbolic):
        if symbolic:
            # reset the constraints
            self.circuit_model.tech_model.create_constraints(self.model_cfg["scaling_mode"])
            #self.circuit_model.set_uarch_parameters()
            #self.circuit_model.set_uarch_constraints()
            # take symbolic max over the critical paths
            execution_time = 0
            node_arrivals = {}
            node_arrivals_cvx = {}
            constr_cvx = []
            for node in self.scheduled_dfg.nodes:
                if node == "end":
                    node_arrivals[node] = self.circuit_model.tech_model.base_params.node_arrivals_end
                    node_arrivals_cvx[node] = cp.Variable()
                elif len(list(self.scheduled_dfg.predecessors(node))) == 0:
                    node_arrivals[node] = 0
                    node_arrivals_cvx[node] = 0
                    continue
                else:
                    node_arrivals[node] = sp.symbols(f"node_arrivals_{node}")
                    node_arrivals_cvx[node] = cp.Variable()
                for pred in self.scheduled_dfg.predecessors(node):
                    assert self.scheduled_dfg.nodes[pred]["function"] != "nop"
                    if self.scheduled_dfg.edges[pred, node]["resource_edge"]:
                        pred_delay = self.circuit_model.tech_model.base_params.clk_period # convert to ns
                    elif self.scheduled_dfg.nodes[pred]["function"] in ["Buf", "MainMem"]:
                        rsc_name = self.scheduled_dfg.nodes[pred]["library"][self.scheduled_dfg.nodes[pred]["library"].find("__")+1:]
                        pred_delay = self.circuit_model.symbolic_latency_wc[self.scheduled_dfg.nodes[pred]["function"]]()[rsc_name]
                    else:
                        pred_delay = self.circuit_model.symbolic_latency_wc[self.scheduled_dfg.nodes[pred]["function"]]()
                    rsc_edge = self.get_rsc_edge((pred, node), self.scheduled_dfg)
                    if rsc_edge in self.circuit_model.wire_length_by_edge:
                        pred_delay += self.circuit_model.wire_delay(rsc_edge, symbolic)
                    self.constraints.append(node_arrivals[node] >= node_arrivals[pred] + pred_delay)
                    constr_cvx.append(node_arrivals_cvx[node] >= node_arrivals_cvx[pred] + sim_util.xreplace_safe(pred_delay, self.circuit_model.tech_model.base_params.tech_values))
            obj = node_arrivals_cvx["end"]
            prob = cp.Problem(cp.Minimize(obj), constr_cvx)
            prob.solve()
            for node in node_arrivals:
                if type(node_arrivals_cvx[node]) != int:
                    self.circuit_model.tech_model.base_params.tech_values[node_arrivals[node]] = node_arrivals_cvx[node].value
            print(f"cvxpy symbolic execution time: {prob.value}")
            self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.node_arrivals_end] = self.scheduled_dfg.nodes["end"]["start_time"]
            execution_time = self.circuit_model.tech_model.base_params.node_arrivals_end
            print(f"at the end of symbolic execution time calc, there are {len(self.circuit_model.tech_model.constraints)} constraints")
        else:
            execution_time = self.scheduled_dfg.nodes["end"]["start_time"]
        return execution_time
    
    def calculate_passive_energy(self, total_execution_time, symbolic):
        passive_power = 0
        for node in self.netlist:
            data = self.netlist.nodes[node]
            log_info(f"calculating passive power for node {node}, data: {data}")
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                if symbolic:
                    passive_power += self.circuit_model.symbolic_power_passive[data["function"]]()[rsc_name]
                else:
                    passive_power += self.circuit_model.memories[rsc_name]["Standby leakage per bank(mW)"] * 1e6 # convert from mW to nW
            else:
                if symbolic:
                    passive_power += self.circuit_model.symbolic_power_passive[data["function"]]()
                else:
                    passive_power += self.circuit_model.circuit_values["passive_power"][data["function"]]
                log_info(f"(passive power) {data['function']}: {self.circuit_model.circuit_values['passive_power'][data['function']]}")
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
                    total_active_energy += self.circuit_model.symbolic_energy_active[data["function"]]()[rsc_name]
                else:
                    if data["module"].find("wport") != -1:
                        total_active_energy += self.circuit_model.memories[rsc_name]["Dynamic write energy (nJ)"]
                    else:
                        total_active_energy += self.circuit_model.memories[rsc_name]["Dynamic read energy (nJ)"]
            else:
                if symbolic:
                    total_active_energy += self.circuit_model.symbolic_energy_active[data["function"]]()
                else:
                    total_active_energy += self.circuit_model.circuit_values["dynamic_energy"][data["function"]]
                log_info(f"(active energy) {data['function']}: {total_active_energy}")
        for edge in self.scheduled_dfg.edges:
            rsc_edge = self.get_rsc_edge(edge, self.scheduled_dfg)
            if rsc_edge in self.circuit_model.wire_length_by_edge:
                wire_energy = self.circuit_model.wire_energy(rsc_edge, symbolic)
                log_info(f"(wire energy) {edge}: {wire_energy} nJ")
                total_active_energy += wire_energy
        return total_active_energy

    def save_obj_vals(self, execution_time, execution_time_override=False, execution_time_override_val=0):
        if self.model_cfg["model_type"] == "bulk_bsim4":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "subthreshold leakage current": self.circuit_model.tech_model.I_sub,
                "gate tunneling current": self.circuit_model.tech_model.I_tunnel,
                "GIDL current": self.circuit_model.tech_model.I_GIDL,
                "long channel threshold voltage": self.circuit_model.tech_model.base_params.V_th,
                "effective threshold voltage": self.circuit_model.tech_model.V_th_eff,
                "supply voltage": self.circuit_model.tech_model.base_params.V_dd,
                "wire RC": self.circuit_model.tech_model.m1_Rsq * self.circuit_model.tech_model.m1_Csq,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                "f": self.circuit_model.tech_model.base_params.f,
            }
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "bulk":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "subthreshold leakage current": self.circuit_model.tech_model.I_off,
                "gate tunneling current": self.circuit_model.tech_model.I_tunnel,
                "FN term": self.circuit_model.tech_model.FN_term,
                "WKB term": self.circuit_model.tech_model.WKB_term,
                "GIDL current": self.circuit_model.tech_model.I_GIDL,
                "effective threshold voltage": self.circuit_model.tech_model.V_th_eff,
                "supply voltage": self.circuit_model.tech_model.base_params.V_dd,
                "wire RC": self.circuit_model.tech_model.m1_Rsq * self.circuit_model.tech_model.m1_Csq,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                "f": self.circuit_model.tech_model.base_params.f,
            }
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "vs":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "gate length": self.circuit_model.tech_model.param_db["L"],
                "gate width": self.circuit_model.tech_model.param_db["W"],
                "subthreshold leakage current": self.circuit_model.tech_model.param_db["I_sub"],
                "long channel threshold voltage": self.circuit_model.tech_model.param_db["V_th"],
                "effective threshold voltage": self.circuit_model.tech_model.param_db["V_th_eff"],
                "supply voltage": self.circuit_model.tech_model.param_db["V_dd"],
                "wire RC": self.circuit_model.tech_model.param_db["wire RC"],
                "on current per um": self.circuit_model.tech_model.param_db["I_on_per_um"],
                "off current per um": self.circuit_model.tech_model.param_db["I_off_per_um"],
                "gate tunneling current per um": self.circuit_model.tech_model.param_db["I_tunnel_per_um"],
                "subthreshold leakage current per um": self.circuit_model.tech_model.param_db["I_sub_per_um"],
                "DIBL factor": self.circuit_model.tech_model.param_db["DIBL factor"],
                "SS": self.circuit_model.tech_model.param_db["SS"],
                "t_ox": self.circuit_model.tech_model.param_db["t_ox"],
                "eot": self.circuit_model.tech_model.param_db["eot"],
                "scale length": self.circuit_model.tech_model.param_db["scale_length"],
                "C_load": self.circuit_model.tech_model.param_db["C_load"],
                "C_wire": self.circuit_model.tech_model.param_db["C_wire"],
                "R_wire": self.circuit_model.tech_model.param_db["R_wire"],
                "R_device": self.circuit_model.tech_model.param_db["V_dd"]/self.circuit_model.tech_model.param_db["I_on"],
                "F_f": self.circuit_model.tech_model.param_db["F_f"],
                "F_s": self.circuit_model.tech_model.param_db["F_s"],
                "vx0": self.circuit_model.tech_model.param_db["vx0"],
                "v": self.circuit_model.tech_model.param_db["v"],
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                #"f": self.circuit_model.tech_model.base_params.f,
                "parasitic capacitance": self.circuit_model.tech_model.param_db["parasitic capacitance"],
                "k_gate": self.circuit_model.tech_model.param_db["k_gate"],
                "delay": self.circuit_model.tech_model.delay,
                "multiplier delay": self.circuit_model.symbolic_latency_wc["Mult16"](),
                #"scaled power": self.total_passive_power * self.circuit_model.tech_model.capped_power_scale_total + self.total_active_energy/(execution_time * self.circuit_model.tech_model.capped_delay_scale_total),
                "logic_sensitivity": self.circuit_model.tech_model.base_params.logic_sensitivity,
                "logic_resource_sensitivity": self.circuit_model.tech_model.base_params.logic_resource_sensitivity,
                "logic_ahmdal_limit": self.circuit_model.tech_model.base_params.logic_ahmdal_limit,
                "logic_resource_ahmdal_limit": self.circuit_model.tech_model.base_params.logic_resource_ahmdal_limit,
                "interconnect sensitivity": self.circuit_model.tech_model.base_params.interconnect_sensitivity,
                "interconnect resource sensitivity": self.circuit_model.tech_model.base_params.interconnect_resource_sensitivity,
                "interconnect ahmdal limit": self.circuit_model.tech_model.base_params.interconnect_ahmdal_limit,
                "interconnect resource ahmdal limit": self.circuit_model.tech_model.base_params.interconnect_resource_ahmdal_limit,
                "memory sensitivity": self.circuit_model.tech_model.base_params.memory_sensitivity,
                "memory resource sensitivity": self.circuit_model.tech_model.base_params.memory_resource_sensitivity,
                "memory ahmdal limit": self.circuit_model.tech_model.base_params.memory_ahmdal_limit,
                "memory resource ahmdal limit": self.circuit_model.tech_model.base_params.memory_resource_ahmdal_limit,
                "m1_Rsq": self.circuit_model.tech_model.m1_Rsq,
                "m2_Rsq": self.circuit_model.tech_model.m2_Rsq,
                "m3_Rsq": self.circuit_model.tech_model.m3_Rsq,
                "m1_Csq": self.circuit_model.tech_model.m1_Csq,
                "m2_Csq": self.circuit_model.tech_model.m2_Csq,
                "m3_Csq": self.circuit_model.tech_model.m3_Csq,
                "m1_rho": self.circuit_model.tech_model.base_params.m1_rho,
                "m2_rho": self.circuit_model.tech_model.base_params.m2_rho,
                "m3_rho": self.circuit_model.tech_model.base_params.m3_rho,
                "m1_k": self.circuit_model.tech_model.base_params.m1_k,
                "m2_k": self.circuit_model.tech_model.base_params.m2_k,
                "m3_k": self.circuit_model.tech_model.base_params.m3_k,
            }
            if self.circuit_model.tech_model.model_cfg["vs_model_type"] == "base":
                self.obj_sub_exprs["t_1"] = self.circuit_model.tech_model.param_db["t_1"]
            elif self.circuit_model.tech_model.model_cfg["vs_model_type"] == "mvs_si":
                self.obj_sub_exprs["R_s"] = self.circuit_model.tech_model.param_db["R_s"]
                self.obj_sub_exprs["R_d"] = self.circuit_model.tech_model.param_db["R_d"]
                self.obj_sub_exprs["L_ov"] = self.circuit_model.tech_model.param_db["L_ov"]
            elif self.circuit_model.tech_model.model_cfg["vs_model_type"] == "vscnfet":
                self.obj_sub_exprs["Vth_rolloff"] = self.circuit_model.tech_model.param_db["Vth_rolloff"]
                self.obj_sub_exprs["d"] = self.circuit_model.tech_model.param_db["d"]
                self.obj_sub_exprs["L_c"] = self.circuit_model.tech_model.param_db["L_c"]
                self.obj_sub_exprs["H_c"] = self.circuit_model.tech_model.param_db["H_c"]
                self.obj_sub_exprs["H_g"] = self.circuit_model.tech_model.param_db["H_g"]
                self.obj_sub_exprs["k_cnt"] = self.circuit_model.tech_model.param_db["k_cnt"]

        else: 
            raise ValueError(f"Objective function {self.obj_fn} not supported")
        self.obj_sub_plot_names = {
            "execution_time": "Execution Time over generations (ns)",
            "passive power": "Passive Power over generations (W)",
            "active power": "Active Power over generations (W)",
            "gate length": "Gate Length over generations (m)",
            "gate width": "Gate Width over generations (m)",
            "subthreshold leakage current": "Subthreshold Leakage Current over generations (nA)",
            "long channel threshold voltage": "Long Channel Threshold Voltage (V)",
            "effective threshold voltage": "Effective Threshold Voltage over generations (V)",
            "supply voltage": "Supply Voltage over generations (V)",
            "wire RC": "Wire RC over generations (s)",
            "on current per um": "On Current per um over generations (A/um)",
            "off current per um": "Off Current per um over generations (A/um)",
            "gate tunneling current per um": "Gate Tunneling Current per um over generations (A/um)",
            "subthreshold leakage current per um": "Subthreshold Leakage Current per um over generations (A/um)",
            "DIBL factor": "DIBL Factor over generations (V/V)",
            "SS": "Subthreshold Slope over generations (V/V)",
            "Vth_rolloff": "Vth Rolloff over generations (V)",
            "t_ox": "Gate Oxide Thickness over generations (m)",
            "eot": "Electrical Oxide Thickness over generations (m)",
            "scale length": "Scale Length over generations (m)",
            "C_load": "Load Capacitance over generations (F)",
            "C_wire": "Wire Capacitance over generations (F)",
            "R_wire": "Wire Resistance over generations (Ohm)",
            "R_device": "Device Resistance over generations (Ohm)",
            "F_f": "F_f over generations",
            "F_s": "F_s over generations",
            "vx0": "virtual source injection velocity over generations (m/s)",
            "v": "effective injection velocity over generations (m/s)",
            "t_1": "T1 over generations (s)",
            "clk_period": "Clock Period over generations (ns)",
            "f": "Frequency over generations (Hz)",
            "parasitic capacitance": "Parasitic Capacitance over generations (F)",
            "L_ov": "L_ov over generations (m)",
            "R_s": "R_s over generations (Ohm)",
            "R_d": "R_d over generations (Ohm)",
            "Vth_rolloff": "Vth Rolloff over generations (V)",
            "d": "CNT diameter over generations (m)",
            "L_c": "CNT contact length over generations (m)",
            "H_c": "CNT contact height over generations (m)",
            "H_g": "CNT gate height over generations (m)",
            "k_cnt": "CNT Dielectric Constant over generations (F/m)",
            "k_gate": "Gate Dielectric Constant over generations (F/m)",
            "delay": "Transistor Delay over generations (s)",
            "multiplier delay": "Multiplier Delay over generations (s)",
            "scaled power": "Scaled Power over generations (W)",
            "logic_sensitivity": "Logic Sensitivity over generations",
            "logic_resource_sensitivity": "Logic Resource Sensitivity over generations",
            "logic_ahmdal_limit": "Logic Ahmdal Limit over generations",
            "logic_resource_ahmdal_limit": "Logic Resource Ahmdal Limit over generations",
            "interconnect sensitivity": "Interconnect Sensitivity over generations",
            "interconnect resource sensitivity": "Interconnect Resource Sensitivity over generations",
            "interconnect ahmdal limit": "Interconnect Ahmdal Limit over generations",
            "interconnect resource ahmdal limit": "Interconnect Resource Ahmdal Limit over generations",
            "memory sensitivity": "Memory Sensitivity over generations",
            "memory resource sensitivity": "Memory Resource Sensitivity over generations",
            "memory ahmdal limit": "Memory Ahmdal Limit over generations",
            "memory resource ahmdal limit": "Memory Resource Ahmdal Limit over generations",
            "m1_Rsq": "Metal 1 Resistance per Square over generations (Ohm/m)",
            "m2_Rsq": "Metal 2 Resistance per Square over generations (Ohm/m)",
            "m3_Rsq": "Metal 3 Resistance per Square over generations (Ohm/m)",
            "m1_Csq": "Metal 1 Capacitance per Square over generations (F/m)",
            "m2_Csq": "Metal 2 Capacitance per Square over generations (F/m)",
            "m3_Csq": "Metal 3 Capacitance per Square over generations (F/m)",
            "m1_rho": "Metal 1 Resistivity over generations (Ohm-m)",
            "m2_rho": "Metal 2 Resistivity over generations (Ohm-m)",
            "m3_rho": "Metal 3 Resistivity over generations (Ohm-m)",
            "m1_k": "Metal 1 Permittivity over generations (F/m)",
            "m2_k": "Metal 2 Permittivity over generations (F/m)",
            "m3_k": "Metal 3 Permittivity over generations (F/m)",
        }
        if execution_time_override:
            execution_time = execution_time_override_val
        if self.obj_fn == "edp":
            self.obj = (self.total_passive_energy + self.total_active_energy) * execution_time
            self.obj_scaled = (self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale + self.total_active_energy) * execution_time * self.circuit_model.tech_model.capped_delay_scale
        elif self.obj_fn == "ed2":
            self.obj = (self.total_passive_energy + self.total_active_energy) * (execution_time)**2
            self.obj_scaled = (self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale + self.total_active_energy) * (execution_time * self.circuit_model.tech_model.capped_delay_scale)**2
        elif self.obj_fn == "delay":
            self.obj = execution_time
            self.obj_scaled = execution_time * self.circuit_model.tech_model.capped_delay_scale
        elif self.obj_fn == "energy":
            self.obj = self.total_active_energy + self.total_passive_energy
            self.obj_scaled = (self.total_active_energy + self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale)
        elif self.obj_fn == "eplusd":
            self.obj = ((self.total_active_energy + self.total_passive_energy) * sim_util.xreplace_safe(execution_time, self.circuit_model.tech_model.base_params.tech_values) 
                        + execution_time * sim_util.xreplace_safe(self.total_active_energy + self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values))
            self.obj_scaled = ((self.total_active_energy + self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale) * sim_util.xreplace_safe(execution_time, self.circuit_model.tech_model.base_params.tech_values) 
                        + execution_time * self.circuit_model.tech_model.capped_delay_scale * sim_util.xreplace_safe(self.total_active_energy + self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values))
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")
    
    def calculate_objective(self, clk_period_opt=False, form_dfg=True):
        start_time = time.time()
        self.constraints = []
        if self.hls_tool == "vitis":
            self.execution_time = self.calculate_execution_time_vitis(self.top_block_name, clk_period_opt, form_dfg)
            self.total_passive_energy = self.calculate_passive_power_vitis(self.execution_time)
            self.total_active_energy = self.calculate_active_energy_vitis()
        else: # catapult
            # always use symbolic calculation. If you want concrete value later then sub tech values in.
            self.execution_time = self.calculate_execution_time(symbolic=True)
            self.total_passive_energy = self.calculate_passive_energy(self.execution_time, symbolic=True)
            self.total_active_energy = self.calculate_active_energy(symbolic=True)
        self.save_obj_vals(self.execution_time)
        logger.info(f"time to calculate objective: {time.time()-start_time}")

    def display_objective(self, message):
        obj = float(self.obj.xreplace(self.circuit_model.tech_model.base_params.tech_values))
        sub_exprs = {}
        for key in self.obj_sub_exprs:
            if not isinstance(self.obj_sub_exprs[key], float):
                sub_exprs[key] = float(self.obj_sub_exprs[key].xreplace(self.circuit_model.tech_model.base_params.tech_values))
            else:   
                sub_exprs[key] = self.obj_sub_exprs[key]
        print(f"{message}\n {self.obj_fn}: {obj}, sub expressions: {sub_exprs}")