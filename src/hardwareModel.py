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
            )
        self.hw_allocated = {}

        if path_to_graphml is not None:
            self.netlist = nx.read_gml(path_to_graphml)
            print(f"netlist: {self.netlist.nodes.data()}")
        else:
            self.netlist = nx.Graph()

        self.init_misc_vars()

        self.dynamic_allocation = False
        if cfg is not None:
            self.allocate_hw_from_config(config[cfg])

        self.set_technology_parameters()

    def set_hw_config_vars(
        self, id, bandwidth, mem_layers, pitch, transistor_size, cache_size
    ):
        self.id = id
        self.max_bw = bandwidth # this doesn't really get used. deprecate?
        self.bw_avail = bandwidth # deprecate?
        self.mem_layers = mem_layers
        self.pitch = pitch
        self.transistor_size = transistor_size
        self.cache_size = cache_size

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
        ).items(): # should only have 1
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
                        data["size"], self.netlist.nodes[edge[1]]["memory_module"], var_size=1
                    )

        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Regs", self.netlist.nodes.data())
        ).items():
            data["var"] = '' # reg keeps track of which variable it is allocated

    ## Deprecated
    def allocate_hw_from_config(self, config):
        """
        allocate hardware from a config file
        """
        self.hw_allocated = dict(config)
        # remove the other config variables from the hardware allocated dict
        # hardware allocated should be refactored to PEs allocated or something like that.
        self.hw_allocated.pop("id")
        self.hw_allocated.pop("bandwidth")
        self.hw_allocated.pop("nummemlayers")
        self.hw_allocated.pop("interconnectpitch")
        self.hw_allocated.pop("transistorsize")
        self.hw_allocated.pop("cachesize")

        # convert lower case to Camel Case
        self.hw_allocated["Add"] = self.hw_allocated.pop("add")
        self.hw_allocated["Regs"] = self.hw_allocated.pop("regs")
        self.hw_allocated["Mult"] = self.hw_allocated.pop("mult")
        self.hw_allocated["Sub"] = self.hw_allocated.pop("sub")
        self.hw_allocated["FloorDiv"] = self.hw_allocated.pop("floordiv")
        self.hw_allocated["Gt"] = self.hw_allocated.pop("gt")
        self.hw_allocated["And"] = self.hw_allocated.pop("and")
        self.hw_allocated["Or"] = self.hw_allocated.pop("or")
        self.hw_allocated["Mod"] = self.hw_allocated.pop("mod")
        self.hw_allocated["LShift"] = self.hw_allocated.pop("lshift")
        self.hw_allocated["RShift"] = self.hw_allocated.pop("rshift")
        self.hw_allocated["BitOr"] = self.hw_allocated.pop("bitor")
        self.hw_allocated["BitXor"] = self.hw_allocated.pop("bitxor")
        self.hw_allocated["BitAnd"] = self.hw_allocated.pop("bitand")
        self.hw_allocated["Eq"] = self.hw_allocated.pop("eq")
        self.hw_allocated["NotEq"] = self.hw_allocated.pop("noteq")
        self.hw_allocated["Lt"] = self.hw_allocated.pop("lt")
        self.hw_allocated["LtE"] = self.hw_allocated.pop("lte")
        self.hw_allocated["GtE"] = self.hw_allocated.pop("gte")
        self.hw_allocated["IsNot"] = self.hw_allocated.pop("isnot")
        self.hw_allocated["USub"] = self.hw_allocated.pop("usub")
        self.hw_allocated["UAdd"] = self.hw_allocated.pop("uadd")
        self.hw_allocated["Not"] = self.hw_allocated.pop("not")
        self.hw_allocated["Invert"] = self.hw_allocated.pop("invert")

        for k, v in self.hw_allocated.items():
            self.hw_allocated[k] = int(v)

        tmp = True
        for key, value in self.hw_allocated.items():
            tmp &= value == -1
        self.dynamic_allocation = tmp
        # print(f"dynamic_allocation flag: {self.dynamic_allocation}")

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
        total_area = 0
        for node, data in self.netlist.nodes.data():
            scaling = 1
            if data["function"] in ["Regs", "Buf", "MainMem"]:
                scaling = data["size"]
            total_area += self.area[data["function"]] * scaling
        return total_area * 1e-6 # convert from nm^2 to um^2

    def get_mem_compute_bw(self):
        """
        get edges between buf0 and Regs
        """
        edges = list(filter(lambda x: "Reg" in x[0], self.netlist.in_edges("Buf0")))
        n = len(edges)
        return n