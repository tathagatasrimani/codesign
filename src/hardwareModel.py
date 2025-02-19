import logging
import configparser as cp
import yaml

logger = logging.getLogger(__name__)

from . import rcgen
from . import cacti_util

HW_CONFIG_FILE = "src/params/hw_cfgs.ini"

class HardwareModel:
    def __init__(self, cfg="default"):
        config = cp.ConfigParser()
        config.read(HW_CONFIG_FILE)
        try:
            self.set_hw_config_vars(
                config.getint(cfg, "id"),
                config.getint(cfg, "transistorsize"),
                config.getfloat(cfg, "V_dd"),
            )
        except cp.NoSectionError:
            self.set_hw_config_vars(
                config.getint("DEFAULT", "id"),
                config.getint("DEFAULT", "transistorsize"),
                config.getfloat("DEFAULT", "V_dd"),
            )
        self.set_technology_parameters()

    def set_hw_config_vars(
        self,
        id,
        transistor_size,
        V_dd,
    ):
        self.id = id
        self.transistor_size = transistor_size
        self.V_dd = V_dd

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
        print(f"cacti tech node: {self.cacti_tech_node}")

        self.cacti_dat_file = (
            f"src/cacti/tech_params/{int(self.cacti_tech_node*1e3):2d}nm.dat"
        )
        print(f"self.cacti_dat_file: {self.cacti_dat_file}")

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