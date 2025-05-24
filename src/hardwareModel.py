import logging
import yaml
import time

logger = logging.getLogger(__name__)

import networkx as nx
import sympy as sp
from . import cacti_util
from . import parameters
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

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.params.circuit_values["latency"],
            "dynamic_energy": self.params.circuit_values["dynamic_energy"],
            "passive_power": self.params.circuit_values["passive_power"],
            "area": self.params.circuit_values["area"], # TODO: make sure we have this
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))
    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics):
        start_time = time.time()
        self.params.wire_length_by_edge, _ = place_n_route.place_n_route(
            self.netlist, arg_testfile, arg_parasitics
        )
        logger.info(f"time to generate wire parasitics: {time.time()-start_time}")
        # update scheduled dfg with wire delays
        for edge in self.scheduled_dfg.edges:
            if edge in self.netlist.edges:
                # wire delay = R * C * length^2
                self.scheduled_dfg.edges[edge]["cost"] = self.params.wire_delay(edge)
                logger.info(f"(wire delay) {edge}: {self.scheduled_dfg.edges[edge]['cost']} ns")
                self.scheduled_dfg.edges[edge]["weight"] += self.scheduled_dfg.edges[edge]["cost"]
            else:
                self.scheduled_dfg.edges[edge]["cost"] = 0

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
                    if i > 0:
                        path_execution_time += self.params.wire_delay((path[1][i-1], node), symbolic)
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
            if edge in self.netlist.edges:
                wire_energy = self.params.wire_energy(edge, symbolic)
                logger.info(f"(wire energy) {edge}: {wire_energy} nJ")
                total_active_energy += wire_energy
        return total_active_energy
    
    def calculate_objective(self, symbolic=False):
        if self.obj_fn == "edp":
            execution_time = self.calculate_execution_time(symbolic)
            total_passive_energy = self.calculate_passive_energy(execution_time, symbolic)
            total_active_energy = self.calculate_active_energy(symbolic)
            if symbolic:
                self.symbolic_obj = total_passive_energy + total_active_energy * execution_time
                self.symbolic_obj_sub_exprs = {
                    "execution_time": execution_time,
                    "total_passive_energy": total_passive_energy,
                    "total_active_energy": total_active_energy,
                }
            else:
                self.obj = (total_passive_energy + total_active_energy) * execution_time
                self.obj_sub_exprs = {
                    "execution_time": execution_time,
                    "total_passive_energy": total_passive_energy,
                    "total_active_energy": total_active_energy,
                }
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")