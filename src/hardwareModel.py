import logging
import configparser as cp
import yaml
import time

logger = logging.getLogger(__name__)

import networkx as nx

from . import rcgen
from . import cacti_util
from openroad_interface import place_n_route

HW_CONFIG_FILE = "src/params/hw_cfgs.ini"

class HardwareModel:
    """
    Represents a hardware model with configurable technology and hardware parameters. Provides methods
    to set up the hardware, manage netlists, and extract technology-specific timing and power data for
    optimization and simulation purposes.
    """
    def __init__(self, args, cfg="default"):
        config = cp.ConfigParser()
        config.read(HW_CONFIG_FILE)
        try:
            self.set_hw_config_vars(
                config.getint(cfg, "id"),
                config.getint(cfg, "transistorsize"),
                config.getfloat(cfg, "V_dd"),
                config.getint(cfg, "frequency")
            )
        except cp.NoSectionError:
            self.set_hw_config_vars(
                config.getint("DEFAULT", "id"),
                config.getint("DEFAULT", "transistorsize"),
                config.getfloat("DEFAULT", "V_dd"),
                config.getint("DEFAULT", "frequency")
            )
        self.area_constraint = args.area
        if hasattr(args, "logic_node"):
            self.transistor_size = args.logic_node
        if hasattr(args, "mem_node"):
            self.cacti_tech_node = args.mem_node
        self.set_technology_parameters()
        self.netlist = nx.DiGraph()
        self.symbolic_mem = {}
        self.symbolic_buf = {}
        self.memories = []

    def reset_state(self):
        self.symbolic_buf = {}
        self.symbolic_mem = {}
        self.netlist = nx.DiGraph()
        self.memories = []

    def set_hw_config_vars(
        self,
        id,
        transistor_size,
        V_dd,
        f
    ):
        self.id = id
        self.transistor_size = transistor_size
        self.V_dd = V_dd
        self.f = f
        self.cacti_tech_node = min(
            cacti_util.valid_tech_nodes,
            key=lambda x: abs(x - self.transistor_size * 1e-3),
        )

    def set_technology_parameters(self):
        """
        Load and set technology-specific parameters (area, latency, power, energy) from the YAML file
        for the given transistor size. Also determines the closest CACTI technology node and associated
        data file.

        Returns:
            None
        """
        tech_params = yaml.load(
            open("src/params/tech_params.yaml", "r"), Loader=yaml.Loader
        )
        self.area = tech_params["area"][self.transistor_size]
        self.latency = tech_params["latency"][self.transistor_size]

        self.dynamic_power = tech_params["dynamic_power"][self.transistor_size]
        self.leakage_power = tech_params["leakage_power"][self.transistor_size]
        self.dynamic_energy = tech_params["dynamic_energy"][self.transistor_size]

        # DEBUG
        print(f"cacti tech node: {self.cacti_tech_node}")

        self.cacti_dat_file = (
            f"src/cacti/tech_params/{int(self.cacti_tech_node*1e3):2d}nm.dat"
        )
        print(f"self.cacti_dat_file: {self.cacti_dat_file}")

    def get_optimization_params_from_tech_params(self):
        """
        Generate optimization parameters (R, C, etc.) from the loaded latency and power technology
        parameters.

        Returns:
            rcs: Dictionary containing optimization parameters for the hardware model.
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

        # update dat file with new parameters
        cacti_util.update_dat(rcs, self.cacti_dat_file)

        beta = yaml.load(open(coeff_file, "r"), Loader=yaml.Loader)["beta"]

        for key in C:
            self.latency[key] = R[key] * C[key]  # ns

            # power calculations may need to change with reintroduction of clock frequency
            self.dynamic_power[key] = 0.5 * self.V_dd * self.V_dd * 1e9 / R[key]  # nW

            self.leakage_power[key] = (
                beta[key] * self.V_dd**2 * 1e9 / (R["Not"] * self.R_off_on_ratio)
            )  # convert to nW
    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics):
        start_time = time.time()
        _, graph = place_n_route.place_n_route(
            self.netlist, arg_testfile, arg_parasitics
        )
        logger.info(f"time to generate wire parasitics: {time.time()-start_time}")
        self.parasitics = arg_parasitics
        self.parasitic_graph = graph