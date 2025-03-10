import json
import re
from collections import deque
import ast
import configparser as cp
import yaml
import os
import shutil
import logging

logger = logging.getLogger(__name__)

import graphviz as gv
import sympy as sp
from sympy import *
import networkx as nx
import shutil
from staticfg.builder import CFGBuilder
import numpy as np

from .ast_utils import ASTUtils
from .memory import Memory, Cache
from .config_dicts import op2sym_map
from . import rcgen
from . import cacti_util
from . import sim_util
from . import hw_symbols
from .global_constants import SYSTEM_BUS_SIZE
from openroad_interface.var import directory
from openroad_interface import place_n_route


HW_CONFIG_FILE = "src/params/hw_cfgs.ini"

benchmark = "simple"
expr_to_node = {}
func_ref = {}

## WRAP ALL OF THESE METHODS INTO A 'NETLIST' CLASS
## Shoudl have an NX graph object as the main instance variable.


def get_nodes_by_filter(netlist, key, value) -> dict:
    """
    returns dict of nodes:data that satisfy the filter
    """
    return {k: v for k, v in dict(netlist.nodes.data()).items() if v[key] == value}


def get_in_use_nodes(netlist) -> dict:
    return get_nodes_by_filter(netlist, "in_use", True)


def get_nodes_with_func(netlist, func) -> dict:
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


def get_unique_funcs(netlist):
    return set(map(lambda v: v["function"], dict(netlist.nodes.data()).values()))


def get_func_count(netlist):
    return {
        func: num_nodes_with_func(netlist, func) for func in get_unique_funcs(netlist)
    }


class HardwareModel:
    # set transistor override flag to tell hw to expect values for transistor_size and optionally for cacti_transistor_size
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
        V_dd=None,
        bus_width=None,
        transistor_override=False,
        cacti_transistor_size=None
    ):
        """
        Simulates the effect of 2 different constructors. Either supply cfg (config), or supply the rest of the arguments.
        In this form for backward compatability. I want to deprecate the manual construction soon.
        cfg here refers to config not a control flow graph, the name collision is unfortunate.
        """
        if cfg is None:
            self.set_hw_config_vars(
                id, bandwidth, mem_layers, pitch, transistor_size, cache_size, V_dd
            )
        else:
            config = cp.ConfigParser()
            config.read(HW_CONFIG_FILE)
            self.path_to_graphml = f"src/architectures/{cfg}.gml"
            try:
                self.set_hw_config_vars(
                    config.getint(cfg, "id"),
                    config.getint(cfg, "bandwidth"),
                    config.getint(cfg, "nummemlayers"),
                    config.getint(cfg, "interconnectpitch"),
                    config.getint(cfg, "transistorsize"),
                    config.getint(cfg, "cachesize"),
                    config.getfloat(cfg, "V_dd"),
                )
            except cp.NoSectionError:
                self.set_hw_config_vars(
                    config.getint("DEFAULT", "id"),
                    config.getint("DEFAULT", "bandwidth"),
                    config.getint("DEFAULT", "nummemlayers"),
                    config.getint("DEFAULT", "interconnectpitch"),
                    config.getint("DEFAULT", "transistorsize"),
                    config.getint("DEFAULT", "cachesize"),
                    config.getfloat("DEFAULT", "V_dd"),
                )
        # allow for architecture configs to have their tech nodes swapped out
        if transistor_override:
            self.transistor_size = transistor_size
        # cacti transistor size can optionally be specified separately from original transistor size
        # if not, then set them equal
        if cacti_transistor_size:
            self.cacti_transistor_size = cacti_transistor_size
        else:
            self.cacti_transistor_size = self.transistor_size
        self.hw_allocated = {}

        if self.path_to_graphml is not None and os.path.exists(self.path_to_graphml):
            self.netlist = nx.read_gml(self.path_to_graphml)
            self.update_netlist()
        else:
            self.netlist = nx.Graph()

        self.parasitic_graph = None  # can be removed if better organization for this
        self.parasitics = "none" # by default, can be overwritten later

        self.init_misc_vars()
        self.set_technology_parameters()

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

    def init_memory(self, mem_needed, nvm_mem_needed, mallocs, buffer_size=64, gen_cacti=True, gen_symbolic=True):
        """
        Add a Memory Module to the netlist for each MainMem node.
        Add a Cache Module to the netlist for each Buf node.
        Params:
        mem_needed: int - determined by total size of malloc operations. Since we are scheduling statically,
                          we just assume that variables will be in memory when we access them. Separately,
                          we can dynamically track how much memory will actually be needed (i.e. if a compiler
                          is freeing up space when variables are dead), but that is not relevant here.
        nvm_mem_needed: int - not yet implemented
        buffer_size: int - default 64 bits equal to one var size.
        """
        mem_object = Memory(mem_needed)
        # allocate space for all variables we will need to access
        for malloc in mallocs:
            self.process_memory_operation(malloc, mem_object)
        for node, data in dict(
            filter(lambda x: x[1]["function"] == "MainMem", self.netlist.nodes.data())
        ).items():
            data["memory_module"] = mem_object
            data["size"] = mem_needed

        buffer_object = Cache(
            data["size"],
            mem_object,
            var_size=None,
        )
        # loop enables multiple unshared buffers
        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Buf", self.netlist.nodes.data())
        ).items():
            edges = self.netlist.edges(node)
            for edge in edges:
                if self.netlist.nodes[edge[1]]["function"] == "MainMem":
                    data["memory_module"] = buffer_object

        for node, data in dict(
            filter(lambda x: x[1]["function"] == "Regs", self.netlist.nodes.data())
        ).items():
            data["var"] = ""  # reg keeps track of which variable it is allocated
        self.mem_size = mem_needed
        self.nvm_mem_size = nvm_mem_needed
        self.buffer_size = buffer_size
        if gen_cacti: self.gen_cacti_results(gen_symbolic)

    def set_hw_config_vars(
        self,
        id,
        bandwidth,
        mem_layers,
        pitch,
        transistor_size,
        cache_size,
        V_dd,
    ):
        self.id = id
        self.max_bw = bandwidth  # this doesn't really get used. deprecate?
        self.bw_avail = bandwidth  # deprecate?
        self.mem_layers = mem_layers
        self.pitch = pitch
        self.transistor_size = transistor_size
        self.cache_size = cache_size
        self.V_dd = V_dd
        self.buffer_bus_width = SYSTEM_BUS_SIZE
        self.memory_bus_width = SYSTEM_BUS_SIZE

    def set_technology_parameters(self):
        """
        I Want to Deprecate everything that takes into account 3D with indexing by pitch size
        and number of mem layers.
        """
        tech_params = yaml.load(
            open("src/params/tech_params.yaml", "r"), Loader=yaml.Loader
        )
        self.area = tech_params["area"][self.transistor_size]
        self.latency = tech_params["latency"][self.transistor_size]

        self.dynamic_power = tech_params["dynamic_power"][self.transistor_size]
        self.leakage_power = tech_params["leakage_power"][self.transistor_size]
        self.dynamic_energy = tech_params["dynamic_energy"][self.transistor_size]

        self.cacti_tech_node = min(
            cacti_util.valid_tech_nodes,
            key=lambda x: abs(x - self.transistor_size * 1e-3),
        )
        # DEBUG
        print(f"cacti tech node: {self.cacti_tech_node}, specified cacti transistor size: {self.cacti_transistor_size}")

        self.cacti_dat_file = (
            f"src/cacti/tech_params/{int(self.cacti_tech_node*1e3):2d}nm.dat"
        )
        print(f"self.cacti_dat_file: {self.cacti_dat_file}")

    def set_var_sizes(self, var_sizes):
        """
        Deprecated?
        """
        self.var_sizes = var_sizes

    def duplicate_config_section(self, cfg, new_cfg):
        """
        Duplicate a section in a config file.
        """
        config = cp.ConfigParser()
        config.read(HW_CONFIG_FILE)
        config.add_section(new_cfg)
        for key, value in config.items(cfg):
            config.set(new_cfg, key, value)
        with open(HW_CONFIG_FILE, "w") as configfile:
            config.write(configfile)

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.latency,
            "dynamic_power": self.dynamic_power,
            "leakage_power": self.leakage_power,
            "dynamic_energy": self.dynamic_energy,
            "area": self.area,
            "V_dd": self.V_dd,
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))

    def update_technology_parameters(
        self,
        rc_params_file="src/params/rcs_current.yaml",
        coeff_file="src/params/coefficients.yaml",
        gen_symbolic=True
    ):
        """
        For full codesign loop, need to update the technology parameters after a run of the inverse pass.
        Local States:
            latency - dictionary of latencies in cycles
            dynamic_power - dictionary of active power in nW
            leakage_power - dictionary of passive power in nW
            V_dd - voltage in V
        Inputs:
            C - dictionary of capacitances in F
            R - dictionary of resistances in Ohms
            rcs[other]:
                V_dd: voltage in V
                MemReadL: memory read latency in s
                MemWriteL: memory write latency in s
                MemReadPact: memory read active power in W
                MemWritePact: memory write active power in W
                MemPpass: memory passive power in W
        """
        logger.info("Updating Technology Parameters")
        rcs = yaml.load(open(rc_params_file, "r"), Loader=yaml.Loader)
        C = rcs["Ceff"]  # nF
        R = rcs["Reff"]  # Ohms
        self.V_dd = rcs["other"]["V_dd"]

        opt_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

        # update dat file with new parameters and re-run cacti
        cacti_util.update_dat(rcs, self.cacti_dat_file)
        self.gen_cacti_results(gen_symbolic)

        beta = yaml.load(open(coeff_file, "r"), Loader=yaml.Loader)["beta"]

        for key in C:
            self.latency[key] = R[key] * C[key]  # ns
            self.dynamic_power[key] = 0.5 * self.V_dd * self.V_dd * 1e9 / R[key]  # nW

            self.leakage_power[key] = (
                beta[key] * self.V_dd**2 * 1e9 / (R["Not"] * self.R_off_on_ratio)
            )  # convert to nW

    def update_netlist(self):
        logger.info(
            f"Old bus_widths -> buf: {self.buffer_bus_width}, mem: {self.memory_bus_width}"
        )
        self.buffer_bus_width = (
            num_nodes_with_func(self.netlist, "Buf") * SYSTEM_BUS_SIZE
        )
        self.memory_bus_width = (
            num_nodes_with_func(self.netlist, "MainMem") * SYSTEM_BUS_SIZE
        )
        logger.info(
            f"New bus_widths -> buf: {self.buffer_bus_width}, mem: {self.memory_bus_width}"
        )

    def update_cache_size(self, cache_size):
        pass

    def print_stats(self):
        """
        Deprecated?
        """
        s = """
	   cycles={cycles}
	   allocated={allocated}
	   utilized={utilized}
	   """.format(
            cycles=self.cycles, allocated=str(self.hw_allocated)
        )

    def get_optimization_params_from_tech_params(self):
        """
        Generate R,C, etc from the latency, power tech parameters.
        """
        rcs = rcgen.generate_optimization_params(
            self.latency,
            self.dynamic_power,
            self.dynamic_energy,
            self.leakage_power,
            self.V_dd,
            self.cacti_dat_file,
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

    def get_total_area(self, include_mem=True):
        """
        Calculate on chip and off chip area.
        TODO: Get Area breakdown of cache (area efficiency) from cacti and integrate here.
        """
        self.on_chip_area = 0

        for node, data in self.netlist.nodes.data():
            if data["function"] in ["Buf", "MainMem"]:
                continue
            self.on_chip_area += self.area[data["function"]]
        logger.info(f"Area of all PEs: {self.on_chip_area}")

        self.off_chip_area = self.area["MainMem"]

        if include_mem: 
            self.on_chip_area += self.area["Buf"] + self.area[
                "Buf"
            ] * self.buf_peripheral_area_proportion * (
                num_nodes_with_func(self.netlist, "Buf") - 1
            )

            # bw = 0
            # for node in filter(
            #     lambda x: x[1]["function"] == "Buf", self.netlist.nodes.data()
            # ):
            #     # print(f"node: {node[0]}")
            #     in_edges = self.netlist.in_edges(node[0])
            #     filtered_edges = list(filter(lambda x: "MainMem" not in x[0], in_edges))
            #     bw += len(filtered_edges)
            # self.on_chip_area += (bw - 1) * bw_scaling * self.area["MainMem"]
            self.off_chip_area += self.area["OffChipIO"]

        logger.info(f"on_chip_area: {self.on_chip_area}")
        logger.info(f"off_chip_area: {self.off_chip_area}")

        return self.on_chip_area # in um^2

    def gen_cacti_results(self, gen_symbolic=True):
        """
        Generate buffer and memory latency and energy numbers from Cacti.
        """
        self.buffer_size = 2048
        self.mem_size = 131072
        buf_vals = cacti_util.gen_vals(
            "base_cache",
            cache_size=self.buffer_size,
            block_size=64,
            cache_type="cache",
            bus_width=self.buffer_bus_width,
        )
        logger.info(f"BUFFER VALS: read/write time {buf_vals['Access time (ns)']} ns, read energy {buf_vals['Dynamic read energy (nJ)']} nJ, write energy {buf_vals['Dynamic write energy (nJ)']} nJ, leakage power {buf_vals['Standby leakage per bank(mW)']}")
        logger.info(f"Buffer cacti with: {self.buffer_size} bytes, {self.buffer_bus_width} bus width")
        buf_opt = {
            "ndwl": buf_vals["Ndwl"],
            "ndbl": buf_vals["Ndbl"],
            "nspd": buf_vals["Nspd"],
            "ndcm": buf_vals["Ndcm"],
            "ndsam1": buf_vals["Ndsam_level_1"],
            "ndsam2": buf_vals["Ndsam_level_2"],
            "repeater_spacing": buf_vals["Repeater spacing"],
            "repeater_size": buf_vals["Repeater size"],
        }

        mem_vals = cacti_util.gen_vals(
            "mem_cache",
            cache_size=self.mem_size,
            block_size=64,
            cache_type="main memory",
            bus_width=self.memory_bus_width,
        )
        logger.info(f"MEMORY VALS: read/write time {mem_vals['Access time (ns)']} ns, read energy {mem_vals['Dynamic read energy (nJ)']} nJ, write energy {mem_vals['Dynamic write energy (nJ)']} nJ, leakage power {mem_vals['Standby leakage per bank(mW)']}")
        mem_opt = {
            "ndwl": mem_vals["Ndwl"],
            "ndbl": mem_vals["Ndbl"],
            "nspd": mem_vals["Nspd"],
            "ndcm": mem_vals["Ndcm"],
            "ndsam1": mem_vals["Ndsam_level_1"],
            "ndsam2": mem_vals["Ndsam_level_2"],
            "repeater_spacing": mem_vals["Repeater spacing"],
            "repeater_size": mem_vals["Repeater size"],
        }
        logger.info(
            f"Memory cacti with: {self.mem_size} bytes, {self.memory_bus_width} bus width"
        )

        self.area["Buf"] = float(buf_vals["Area (mm2)"]) * 1e6  # convert to um^2
        self.area["MainMem"] = float(mem_vals["Area (mm2)"]) * 1e6  # convert to um^2
        self.area["OffChipIO"] = (
            float(mem_vals["IO area"]) * 1e6
            if mem_vals["IO area"] not in ["N/A", "inf", "-inf", "nan", "-nan"]
            else 0.0  # convert to um^2
        )
        logger.info(f"Buf area from cacti: {self.area['Buf']} nm^2")
        logger.info(f"Mem area from cacti: {self.area['MainMem']} nm^2")
        logger.info(f"OffChipIO area from cacti: {self.area['OffChipIO']} nm^2")

        self.buf_peripheral_area_proportion = (
            100 - float(buf_vals["Data arrary area efficiency %"])
        ) / 100
        self.mem_peripheral_area_proportion = (
            100 - float(mem_vals["Data arrary area efficiency %"])
        ) / 100

        logger.info(
            f"Buf peripheral area proportion from cacti: {self.buf_peripheral_area_proportion}"
        )
        logger.info(
            f"Mem peripheral area proportion from cacti: {self.mem_peripheral_area_proportion}"
        )

        self.latency["Buf"] = float(buf_vals["Access time (ns)"])
        self.latency["MainMem"] = float(mem_vals["Access time (ns)"])

        self.dynamic_energy["Buf"]["Read"] = float(buf_vals["Dynamic read energy (nJ)"])
        self.dynamic_energy["Buf"]["Write"] = float(
            buf_vals["Dynamic write energy (nJ)"]
        )

        self.dynamic_energy["MainMem"]["Read"] = float(
            mem_vals["Dynamic read energy (nJ)"]
        )
        self.dynamic_energy["MainMem"]["Write"] = float(
            mem_vals["Dynamic write energy (nJ)"]
        )

        # convert to nW
        self.leakage_power["Buf"] = (
            float(buf_vals["Standby leakage per bank(mW)"]) * 1e6
        )
        self.leakage_power["MainMem"] = (
            float(mem_vals["Standby leakage per bank(mW)"]) * 1e6
        )

        # get the IO parameters from the memory run
        self.latency["OffChipIO"] = (
            float(mem_vals["IO latency (s)"]) * 1e9
            if mem_vals["IO latency (s)"] != "N/A"
            else 0.0
        )
        # This comes in mW
        self.dynamic_power["OffChipIO"] = (
            float(mem_vals["IO power dynamic"]) * 1e6  # convert to nW
            if mem_vals["IO power dynamic"] != "N/A"
            else 0.0
        )

        base_cache_cfg = "cfg/base_cache.cfg"
        mem_cache_cfg = "cfg/mem_cache.cfg"

        # TODO: This only needs to be triggered if we're doing inverse pass (ie symbolic simulate or codesign)
        if gen_symbolic:
            cacti_util.gen_symbolic("Buf", base_cache_cfg, buf_opt, use_piecewise=False)
            cacti_util.gen_symbolic("Mem", mem_cache_cfg, mem_opt, use_piecewise=False)

        return

    def get_wire_parasitics(self, arg_testfile, arg_parasitics):
        design_name = self.path_to_graphml.split("/")[
            len(self.path_to_graphml.split("/")) - 1
        ]
        _, graph = place_n_route.place_n_route(
            design_name, arg_testfile, arg_parasitics
        )
        self.parasitics = arg_parasitics
        self.parasitic_graph = graph

    def process_memory_operation(self, mem_op, mem_module: Memory):
        """
        Processes a memory operation, handling memory allocation or deallocation based on the operation type.

        This function interprets and acts on memory operations (like allocating or freeing memory)
        within the hardware simulation. It modifies the state of the memory module according to the
        specified operation, updating the memory allocation status as necessary.

        Parameters:
        - mem_op (list): A list representing a memory operation. The first element is the operation type
        ('malloc' or 'free'), the second element is the size of the memory block, and subsequent elements
         provide additional context or dimensions for the operation.

        Usage:
        - If `mem_op` is a 'malloc' operation, the function allocates memory of the specified size and
            potentially with specified dimensions.
        - If `mem_op` is a 'free' operation, the function deallocates the memory associated with the
             given variable name.

        This function is typically called during the simulation process to dynamically manage
        memory as the simulated program executes different operations that require memory allocation
        and deallocation.
        """
        var_name = mem_op[2]
        size = int(mem_op[1])
        status = mem_op[0]
        if status == "malloc":
            dims = []
            num_elem = 1
            if len(mem_op) > 3:
                dims = sim_util.get_dims(mem_op[3:])
                # print(f"dims: {dims}")
                num_elem = np.prod(dims)
            mem_module.malloc(var_name, size, dims=dims, elem_size=size // num_elem)
        elif status == "free":
            mem_module.free(var_name)
