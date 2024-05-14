import json
import re
from collections import deque
import ast
import configparser as cp
import yaml

import graphviz as gv
from sympy import *
import networkx as nx

from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils
from memory import Memory, Cache
from config_dicts import op2sym_map
from rcgen import generate_optimization_params
from cacti_util import gen_vals


HW_CONFIG_FILE = "hw_cfgs.ini"

benchmark = "simple"
expr_to_node = {}
func_ref = {}

## WRAP ALL OF THESE METHODS INTO A 'NETLIST' CLASS
## Shoudl have an NX graph object as the main instance variable.


def get_nodes_by_filter(netlist, key, value):
    """
    returns dict of nodes:data that satisfy the filter
    """
    return {k: v for k, v in dict(netlist.nodes.data()).items() if v[key] == value}


def get_in_use_nodes(netlist):
    return get_nodes_by_filter(netlist, "in_use", True)


def get_nodes_with_func(netlist, func):
    """
    should i refactor 'function' to 'operation'?
    """
    return get_nodes_by_filter(netlist, "function", func)


def num_nodes_with_func(netlist, func):
    return len(get_nodes_with_func(netlist, func))


def un_allocate_all_in_use_elements(netlist):
    for k, v in dict(netlist.nodes.data()).items():
        v["in_use"] = False
        v["allocation"] = []


def get_memory_node(netlist):
    """
    returns dict
    """
    return get_nodes_with_func(netlist, "MainMem")


class HardwareModel:
    def __init__(
        self,
        cfg=None,
        path_to_graphml=None,
        id=None,
        bandwidth=None,
        mem_layers=None,
        pitch=None,
        transistor_size=None,
        cache_size=None,
        mem_size=None,
        bus_width=None
    ):
        """
        Simulates the effect of 2 different constructors. Either supply cfg (config), or supply the rest of the arguments.
        In this form for backward compatability. I want to deprecate the manual construction soon.
        cfg here refers to config not a control flow graph, the name collision is unfortunate.
        """
        if cfg is None:
            self.set_hw_config_vars(
                id, bandwidth, mem_layers, pitch, transistor_size, cache_size
            )
        else:
            config = cp.ConfigParser()
            config.read(HW_CONFIG_FILE)
            path_to_graphml = f"architectures/{cfg}.gml"
            self.set_hw_config_vars(
                config.getint(cfg, "id"),
                config.getint(cfg, "bandwidth"),
                config.getint(cfg, "nummemlayers"),
                config.getint(cfg, "interconnectpitch"),
                config.getint(cfg, "transistorsize"),
                config.getint(cfg, "cachesize"),
                config.getint(cfg, "frequency"),
                config.getfloat(cfg, "V_dd")
            )
        self.hw_allocated = {}

        if path_to_graphml is not None:
            self.netlist = nx.read_gml(path_to_graphml)
            # print(f"netlist: {self.netlist.nodes.data()}")
        else:
            self.netlist = nx.Graph()

        self.init_misc_vars()
        self.set_technology_parameters()

    def set_hw_config_vars(
        self,
        id,
        bandwidth,
        mem_layers,
        pitch,
        transistor_size,
        cache_size,
        frequency,
        V_dd
    ):
        self.id = id
        self.max_bw = bandwidth  # this doesn't really get used. deprecate?
        self.bw_avail = bandwidth  # deprecate?
        self.mem_layers = mem_layers
        self.pitch = pitch
        self.transistor_size = transistor_size
        self.cache_size = cache_size
        self.frequency = frequency * 1.0
        self.V_dd = V_dd

    def init_memory(self, mem_needed, nvm_mem_needed):
        """
        Add a Memory Module to the netlist for each MainMem node.
        Add a Cache Module to the netlist for each Buf node.
        Params:
        mem_needed: int
        nvm_mem_needed: int - not yet implemented
        """
        for node, data in dict(
            filter(lambda x: x[1]["function"] == "MainMem", self.netlist.nodes.data())
        ).items():  # should only have 1
            data["memory_module"] = Memory(mem_needed)
            data["size"] = mem_needed

        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Buf", self.netlist.nodes.data())
        ).items():
            edges = self.netlist.edges(node)
            for edge in edges:
                if self.netlist.nodes[edge[1]]["function"] == "MainMem":
                    # for now there is only one neighbor that is MainMem.
                    data["memory_module"] = Cache(
                        data["size"],
                        self.netlist.nodes[edge[1]]["memory_module"],
                        var_size=1,
                    )

        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Regs", self.netlist.nodes.data())
        ).items():
            data["var"] = ""  # reg keeps track of which variable it is allocated

        self.mem_size = mem_needed
        self.bus_width = 256 # ask if want as input

    def set_technology_parameters(self):
        """
        I Want to Deprecate everything that takes into account 3D with indexing by pitch size
        and number of mem layers.
        """
        tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

        self.area = tech_params["area"][self.transistor_size]
        self.latency = tech_params["latency"][self.transistor_size]

        self.dynamic_power = tech_params["dynamic_power"][self.transistor_size]
        self.leakage_power = tech_params["leakage_power"][self.transistor_size]

        # ADDED 
        self.dynamic_energy = tech_params["dynamic_energy"][self.transistor_size]
        # END ADDED
        # print(f"t_size: {self.transistor_size}, cache: {self.cache_size}, mem_layers: {self.mem_layers}, pitch: {self.pitch}")
        # print(f"tech_params[mem_area][t_size][cache_size][mem_layers]: {tech_params['mem_area'][self.transistor_size][self.cache_size][self.mem_layers]}")

        # this reg stuff should have its own numbers. Those mem numbers are for SRAM cache
        # self.area["Regs"] = tech_params['mem_area'][self.transistor_size][self.cache_size][self.mem_layers][self.pitch]
        # self.latency["Regs"] = tech_params['mem_latency'][self.cache_size][self.mem_layers][self.pitch]
        # self.dynamic_power["Regs"] = tech_params['mem_dynamic_power'][self.cache_size][self.mem_layers][self.pitch]
        # self.leakage_power["Regs"] = 1e-6*tech_params['mem_leakage_power'][self.cache_size][self.mem_layers][self.pitch]

        self.mem_area = tech_params["mem_area"][self.transistor_size][self.cache_size][
            self.mem_layers
        ][self.pitch]
        # units of mW
        self.mem_leakage_power = tech_params["mem_leakage_power"][self.cache_size][
            self.mem_layers
        ][self.pitch]
        # how does mem latency get incorporated?
        ## DO THIS!!!!

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.latency,
            "dynamic_power": self.dynamic_power,
            "leakage_power": self.leakage_power,
            "area": self.area,
            "f": self.frequency,
            "V_dd": self.V_dd,
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))

    def update_technology_parameters(
        self, rc_params_file="rcs_current.yaml", coeff_file="coefficients.yaml"
    ):
        """
        For full codesign loop, need to update the technology parameters after a run of the inverse pass.
        Local States:
            latency - dictionary of latencies in cycles
            dynamic_power - dictionary of active power in nW
            leakage_power - dictionary of passive power in nW
            V_dd - voltage in V
            f - frequency in Hz
        Inputs:
            C - dictionary of capacitances in F
            R - dictionary of resistances in Ohms
            rcs[other]:
                f: frequency in Hz
                V_dd: voltage in V
                MemReadL: memory read latency in s
                MemWriteL: memory write latency in s
                MemReadPact: memory read active power in W
                MemWritePact: memory write active power in W
                MemPpass: memory passive power in W
        """
        print(f"Updating Technology Parameters...")
        rcs = yaml.load(open(rc_params_file, "r"), Loader=yaml.Loader)
        C = rcs["Ceff"]
        R = rcs["Reff"]
        self.frequency = rcs["other"]["f"]
        self.V_dd = rcs["other"]["V_dd"]

        self.latency["MainMem"] = (
            rcs["other"]["MemReadL"] + rcs["other"]["MemWriteL"]
        ) * self.frequency / 2
        self.dynamic_power["MainMem"] = (
            rcs["other"]["MemReadPact"] + rcs["other"]["MemWritePact"]
        ) / 2 * 1e9
        self.leakage_power["MainMem"] = rcs["other"]["MemPpass"] * 1e9

        beta = yaml.load(open(coeff_file, "r"), Loader=yaml.Loader)["beta"]

        for key in C:
            self.dynamic_power[key] = (
                0.5 * C[key] * self.V_dd * self.V_dd * self.frequency * 1e9
            )  # convert to nW
            self.latency[key] = R[key] * C[key] * self.frequency  # convert to cycles
            self.leakage_power[key] = (
                beta[key] * self.V_dd**2 / (R["Not"] * self.R_off_on_ratio) * 1e9
            )  # convert to nW

    def get_optimization_params_from_tech_params(self):
        """
        Generate R,C, etc from the latency, power tech parameters.
        """
        rcs = generate_optimization_params(
            self.latency,
            self.dynamic_power,
            self.leakage_power,
            self.V_dd,
            self.frequency,
        )
        self.R_off_on_ratio = rcs["other"]["Roff_on_ratio"]
        return rcs

    def update_cache_size(self, cache_size):
        pass

    def init_misc_vars(self):
        self.compute_operation_totals = {}

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable] = False
        self.cycles = 0

        # what is this doing?
        for op in op2sym_map:
            self.compute_operation_totals[op] = 0

        self.hw_allocated = {}
        self.hw_allocated["Regs"] = 0

        for key in op2sym_map.keys():
            self.hw_allocated[key] = 0

    def set_loop_counts(self, loop_counts):
        self.loop_counts = loop_counts

    def set_var_sizes(self, var_sizes):
        self.var_sizes = var_sizes

    def print_stats(self):
        s = """
	   cycles={cycles}
	   allocated={allocated}
	   utilized={utilized}
	   """.format(
            cycles=self.cycles, allocated=str(self.hw_allocated)
        )

    def get_total_area(self):
        bw_scaling = 0.1  # check this
        total_area = 0
        for node, data in self.netlist.nodes.data():
            scaling = 1
            if data["function"] in ["Regs", "Buf", "MainMem"]:
                scaling = data["size"]
            total_area += self.area[data["function"]] * scaling
        bw = 0
        for node in filter(
            lambda x: x[1]["function"] == "Buf", self.netlist.nodes.data()
        ):
            # print(f"node: {node[0]}")
            in_edges = self.netlist.in_edges(node[0])
            filtered_edges = list(filter(lambda x: "MainMem" not in x[0], in_edges))
            bw += len(filtered_edges)
        total_area += (bw - 1) * bw_scaling * self.area["MainMem"]

        return total_area * 1e-6  # convert from nm^2 to um^2

    def get_mem_compute_bw(self):
        """
        get edges between buf0 and Regs
        """
        edges = list(filter(lambda x: "Reg" in x[0], self.netlist.in_edges("Buf0")))
        n = len(edges)
        return n
    
    def gen_cacti_results(self):

        # 1. add bit width, mem size, cache size DONE
        # 2. add IO DONE
        # 3. check simulate.py for energy instead of power ASK about this

        # 1 details
        # sizes mem set on hwmodel memsize var, add variable for cache size in hardWareModel that we set later - init small
        # add instance for all -> cache size, bit width

        # 2 details
        # add I/O 
        # latency and power to each main mem read or write
        # add new dict entry, off-chip IO [latency and power], add that whenever mem interaction DONE

        # 3 details
        # careful cacti gen energy 
        # need to change power to energy -> just for buf and main mem DONE 
        # add dynamic energy, change simulate.py  
        # store read and write energy separate, buf read & write -> dynamic energy, key for read and write DONE

        buf_vals = gen_vals("base_cache", 131072, 64,
                                      "cache", 512)
        mem_vals = gen_vals("mem_cache", 131072, 64,
                                      "main memory", 512)

        self.latency["Buf"] = float(buf_vals[0][0])
        self.latency["MainMem"] = float(mem_vals[0][0])

        self.dynamic_energy["Buf"]["Read"] = float(buf_vals[0][2])
        self.dynamic_energy["Buf"]["Write"] = float(buf_vals[0][3])
        
        self.dynamic_energy["MainMem"]["Read"] = float(mem_vals[0][2])
        self.dynamic_energy["MainMem"]["Write"] = float(mem_vals[0][3])

        self.leakage_power["Buf"] = float(buf_vals[0][4])
        self.leakage_power["MainMem"] = float(mem_vals[0][4])

        self.latency["OffChipIO"] = float(mem_vals[1][5]) if mem_vals[1][5] != "N/A" else 0.0
        self.dynamic_power["OffChipIO"] = float(mem_vals[1][2]) if mem_vals[1][2] != "N/A" else 0.0

        return
