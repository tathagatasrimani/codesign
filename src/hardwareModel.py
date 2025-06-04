import logging
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

    def map_netlist_to_scheduled_dfg(self):
        # TODO: include resource sharing information
        for node in self.scheduled_dfg:
            if self.scheduled_dfg.nodes[node]["function"] not in self.params.circuit_values["latency"]:
                self.dfg_to_netlist_map[node] = None
            else:
                for elem in self.netlist:
                    if self.inst_name_map[node] in elem:
                        self.dfg_to_netlist_map[node] = elem
                        break
                if node not in self.dfg_to_netlist_map:
                    logger.warning(f"node {node}, {self.inst_name_map[node]} not found in netlist")

        for edge in self.scheduled_dfg.edges:
            if edge[0] not in self.dfg_to_netlist_map or edge[1] not in self.dfg_to_netlist_map: continue
            edge_mapped = (self.dfg_to_netlist_map[edge[0]], self.dfg_to_netlist_map[edge[1]])
            if edge_mapped not in self.netlist.edges:
                logger.info(f"edge {edge} not found in netlist")
            else:
                self.dfg_to_netlist_edge_map[edge] = edge_mapped
                logger.info(f"edge {edge} mapped to {edge_mapped}")

    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics):
        start_time = time.time()
        self.params.wire_length_by_edge, _ = place_n_route.place_n_route(
            self.netlist, arg_testfile, arg_parasitics
        )
        self.map_netlist_to_scheduled_dfg()
        logger.info(f"time to generate wire parasitics: {time.time()-start_time}")
        self.add_wire_delays_to_schedule()

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
        execution_time = self.calculate_execution_time(symbolic)
        total_passive_energy = self.calculate_passive_energy(execution_time, symbolic)
        total_active_energy = self.calculate_active_energy(symbolic)
        self.symbolic_obj_sub_exprs = {
            "execution_time": execution_time,
            "total_passive_energy": total_passive_energy,
            "total_active_energy": total_active_energy,
            "passive power": total_passive_energy/execution_time,
            "subthreshold leakage current": self.params.I_off,
            "gate tunneling current": self.params.I_tunnel,
            "effective threshold voltage": self.params.V_th_eff,
        }
        self.obj_sub_exprs = {
            "execution_time": execution_time,
            "total_passive_energy": total_passive_energy,
            "total_active_energy": total_active_energy,
            "passive power": total_passive_energy/execution_time,
        }
        if self.obj_fn == "edp":
            if symbolic:
                self.symbolic_obj = (total_passive_energy + total_active_energy) * execution_time
            else:
                self.obj = (total_passive_energy + total_active_energy) * execution_time
        elif self.obj_fn == "ed2":
            if symbolic:
                self.symbolic_obj = (total_passive_energy + total_active_energy) * (execution_time)**2
            else:   
                self.obj = (total_passive_energy + total_active_energy) * (execution_time)**2
        elif self.obj_fn == "delay":
            if symbolic:
                self.symbolic_obj = execution_time
            else:
                self.obj = execution_time
        elif self.obj_fn == "energy":
            if symbolic:
                self.symbolic_obj = total_active_energy + total_passive_energy
            else:
                self.obj = total_active_energy + total_passive_energy/ execution_time
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")