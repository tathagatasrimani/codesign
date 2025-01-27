# The point of this test is to compare how the scheduling works for the first and second runs of the forward pass,
# particularly with the first run of the architecture search

import networkx as nx
import os
import sys
import yaml
import logging

logger = logging.getLogger("codesign")
os.chdir("..")
sys.path.append(os.getcwd())
from src import hardwareModel
from src import architecture_search
from src import sim_util

benchmark = "add_and_mult.py"
config = "aladdin_const_with_mem"

openroad_testfile = "openroad_interface/tcl/test_nangate45_bigger.tcl"

save_dir = "notebooks/test_files"
os.remove(f"{save_dir}/schedule_test.log")
logging.basicConfig(filename=f"{save_dir}/schedule_test.log", level=logging.INFO)

log_dir = sim_util.get_latest_log_dir()

tech_params_init = yaml.load(open(log_dir+"/initial_tech_params.yaml", "r"), Loader=yaml.Loader)
tech_params_next = yaml.load(open(log_dir+"/tech_params_0.yaml", "r"), Loader=yaml.Loader)

logger.info(
    f"Setting up architecture search; benchmark: {benchmark}, config: {config}"
)
(
    sim,
    hw,
    computation_dfg,
) = architecture_search.setup_arch_search(benchmark, config, gen_cacti=False, gen_symbolic=False)

hw.latency = tech_params_init["latency"]

hw.get_wire_parasitics(openroad_testfile, "none")

scheduled_dfg = sim.schedule(computation_dfg, hw, "sdc")