import argparse
import logging
import os
import sys

logger = logging.getLogger("inverse walkthrough example")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
os.chdir(project_root)

# Now `current_directory` should reflect the new working directory
current_directory = os.getcwd()
print(current_directory)

# Add the parent directory to sys.path
sys.path.insert(0, current_directory)

# Now you can safely import modules that rely on the correct working directory
from src import coefficients
from src import symbolic_simulate
from src import hardwareModel
from src import hw_symbols
from src import optimize
from src import sim_util
from src import architecture_search

args = None
initial_tech_params = None
final_tech_params = None
hw = None
scheduled_dfg = None

def parse_output(f):
    # mostly copied from codesign.py
    tech_params = {}
    lines = f.readlines()
    mapping = {}
    i = 0
    while lines[i][0] != "x":
        i += 1
    while lines[i][0] == "x":
        mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
            hw_symbols.symbol_table[lines[i].split(" ")[-1][:-1]]
        )
        i += 1
    while i < len(lines) and lines[i].find("x") != 4:
        i += 1
    i += 2
    for _ in range(len(mapping)):
        key = lines[i].split(":")[0].lstrip().rstrip()
        value = float(lines[i].split(":")[2][1:-1])
        tech_params[mapping[key]] = value  # just know that self.tech_params contains all dat
        i += 1
    return tech_params

def run_forward_pass():

    # initialize hw model, override architecture config to have current tech node
    logger.info(
        f"Setting up architecture search; benchmark: {args.benchmark}, config: {args.architecture_config}"
    )
    (
        simulator,
        hw,
        computation_dfg,
    ) = architecture_search.setup_arch_search(args.benchmark, args.architecture_config)
    hw.init_memory(
        sim_util.find_nearest_power_2(simulator.memory_needed),
        sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
    )
    scheduled_dfg = simulator.schedule(computation_dfg, hw)
    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
    simulator.simulate(scheduled_dfg, hw)
    simulator.calculate_edp()
    logger.warning(f"edp of forward pass: {simulator.edp} E-18 Js")
    
    rcs = hw.get_optimization_params_from_tech_params()
    initial_tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

def run_inverse_passes(num_iters):
    symbolic_sim = symbolic_simulate.SymbolicSimulator()
    symbolic_sim.simulator_prep(args.benchmark, hw.latency)
    coefficients.create_and_save_coefficients([hw.transistor_size])
    hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

    symbolic_sim.simulate(scheduled_dfg, hw)
    symbolic_sim.calculate_edp(hw)

    current_tech_params = initial_tech_params.copy()

    for _ in range(num_iters):
        stdout = sys.stdout
        with open(f"{args.savedir}/ipopt_out.txt", "w") as sys.stdout:
            optimize.optimize(current_tech_params, symbolic_sim.edp, "ipopt", regularization=0.1)
        sys.stdout = stdout
        f = open(f"{args.savedir}/ipopt_out.txt", "r")
        current_tech_params = parse_output(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Inverse Walkthrough Example",
        description="Script that walks through multiple runs of the inverse pass for illustrative purposes.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "benchmark", 
        metavar="B", 
        type=str,
        default="matmult.py"
    )
    parser.add_argument(
        "-c",
        "--architecture_config",
        type=str,
        default="mm_test",
        help="Path to the architecture config file",
    )
    parser.add_argument(
        "-f",
        "--savedir",
        type=str,
        default="src/inverse_validation/inverse_walkthrough_log_dir",
        help="Path to the save logs",
    )
    parser.add_argument(
        "N",
        "--num_iters",
        type=int,
        default="3",
    )
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    else:  # clear the directory
        files = os.listdir(args.savedir)
        for file in files:
            os.remove(f"{args.savedir}/{file}")
    with open(f"{args.savedir}/log.txt", "a") as f:
        f.write("Codesign Log\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Architecture Config: {args.architecture_config}\n")

    logging.basicConfig(filename=f"{args.savedir}/log.txt", level=logging.WARNING)

    # run forward pass once to setup inverse pass
    run_forward_pass()

    # run multiple iterations of inverse pass to demonstrate how gradients of variables change
    run_inverse_passes(args.num_iters)