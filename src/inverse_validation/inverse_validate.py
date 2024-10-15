import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
    
logger = logging.getLogger("inverse validation")

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

# 40, 7, 5
logic_tech_nodes = [7, 5]
# 90, 45, 32, 22
cacti_nodes = [45, 22]
default_cacti = 22
default_logic = 7
tech_nodes = []
initial_tech_params = {}
final_tech_params = {}
rcs_m = {}
params_exclusion_list = []

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

def plot_diff(tech_node_pair):
    # use final tech params to determine params to plot because not every tech param appears in edp equation
    tech_node_0_keys = final_tech_params[tech_node_pair[0]].keys()
    params_to_plot = [tech_param for tech_param in tech_node_0_keys if tech_param not in params_exclusion_list]

    tech_node_0_vals = [final_tech_params[tech_node_pair[0]][tech_param] for tech_param in params_to_plot]
    tech_node_1_vals = [initial_tech_params[tech_node_pair[1]][tech_param] for tech_param in params_to_plot]

    # ratio of optimized params of higher tech node to initial params of lower tech node
    ratios = [tech_node_0_vals[i] / tech_node_1_vals[i] for i in range(len(tech_node_1_vals))]
    i = 0
    while (i < len(params_to_plot)):
        num_params_on_fig = min(5, len(params_to_plot)-i)
        X_axis = np.arange(num_params_on_fig)
        plt.bar(X_axis, ratios[i:i+num_params_on_fig])
        plt.xticks(X_axis, params_to_plot[i:i+num_params_on_fig])
        plt.xlabel("tech params")
        plt.ylabel("tech param ratios")
        plt.title(f"optimized {args.test_type} params for {tech_node_pair[0]} nm divided by initial params for {tech_node_pair[1]} nm")
        plt.savefig(f"src/inverse_validation/figs/{args.test_type}_{tech_node_pair[0]}_{tech_node_pair[1]}_compare_{i/5}.png")
        plt.close()
        i += 5

def run_initial():
    edps = []
    hws = {}
    dfgs = {}
    logic_tech_node, cacti_tech_node = None, None
    for tech_node in tech_nodes:
        # TODO: configure cacti tech node
        if (args.test_type == "logic"):
            cacti_tech_node = default_cacti
            logic_tech_node = tech_node
        elif (args.test_type == "cacti"):
            logic_tech_node = default_logic
            cacti_tech_node = tech_node
        # initialize hw model, override architecture config to have current tech node
        logger.info(
            f"Setting up architecture search; benchmark: {args.benchmark}, config: {args.architecture_config}"
        )
        (
            simulator,
            hw,
            computation_dfg,
        ) = architecture_search.setup_arch_search(args.benchmark, args.architecture_config, True, logic_tech_node, cacti_tech_node)
        hw.init_memory(
            sim_util.find_nearest_power_2(simulator.memory_needed),
            sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
        )
        scheduled_dfg = simulator.schedule(computation_dfg, hw)
        hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
        hw.get_wire_parasitics("", "none")
        simulator.simulate(scheduled_dfg, hw)
        simulator.calculate_edp()
        logger.warning(f"edp of simulation running on {tech_node} nm {args.test_type} tech node: {simulator.edp} E-18 Js")
        
        rcs = hw.get_optimization_params_from_tech_params()
        initial_tech_params[tech_node] = sim_util.generate_init_params_from_rcs_as_symbols(rcs)
        
        edps.append(simulator.edp)
        rcs_m[tech_node] = rcs
        hws[tech_node] = hw
        dfgs[tech_node] = scheduled_dfg
    return edps, hws, dfgs

def run_pairwise(tech_node_pair, improvement, hws, dfgs):
    logger.warning(f"tech node pair {tech_node_pair}, asking inverse pass for {improvement} edp improvement")
    symbolic_sim = symbolic_simulate.SymbolicSimulator()
    symbolic_sim.simulator_prep(args.benchmark, hws[tech_node_pair[0]].latency)
    coefficients.create_and_save_coefficients([hws[tech_node_pair[0]].transistor_size])
    hardwareModel.un_allocate_all_in_use_elements(hws[tech_node_pair[0]].netlist)

    symbolic_sim.simulate(dfgs[tech_node_pair[0]], hws[tech_node_pair[0]])
    cacti_subs = symbolic_sim.calculate_edp(hws[tech_node_pair[0]])

    for cacti_var in cacti_subs:
        initial_tech_params[tech_node_pair[0]][cacti_var] = cacti_subs[cacti_var].xreplace(initial_tech_params[tech_node_pair[0]]).evalf()

    # we only want to optimize the variables specified by the test type, so keep the others constant by substituting concrete values
    if (args.test_type == "cacti"):
        logic_params = sim_util.generate_logic_init_params_from_rcs_as_symbols(rcs_m[tech_node_pair[0]])
        symbolic_sim.edp = symbolic_sim.edp.xreplace(logic_params)
        #logger.warning(f"symbolic edp after subbing out logic params: {symbolic_sim.edp}")
    elif (args.test_type == "logic"):
        cacti_params = {}
        for cacti_var in cacti_subs:
            cacti_params[cacti_var] = initial_tech_params[tech_node_pair[0]][cacti_var]
        symbolic_sim.edp = symbolic_sim.edp.xreplace(cacti_params)
        #logger.warning(f"symbolic edp after subbing out cacti params: {symbolic_sim.edp}")

    improvement = (symbolic_sim.edp.xreplace(initial_tech_params[tech_node_pair[0]]) / symbolic_sim.edp.xreplace(initial_tech_params[tech_node_pair[1]])).simplify()
    logger.warning(f"jk actually asking for {improvement} edp improvement")

    stdout = sys.stdout
    with open(f"{args.savedir}/ipopt_out_{tech_node_pair[0]}.txt", "w") as sys.stdout:
        optimize.optimize(initial_tech_params[tech_node_pair[0]], symbolic_sim.edp, "ipopt", cacti_subs, improvement, regularization=0.1)
    sys.stdout = stdout
    f = open(f"{args.savedir}/ipopt_out_{tech_node_pair[0]}.txt", "r")
    final_tech_params[tech_node_pair[0]] = parse_output(f)
    logger.warning(f"final value of edp is {symbolic_sim.edp.xreplace(final_tech_params[tech_node_pair[0]]).simplify()} vs inital {symbolic_sim.edp.xreplace(initial_tech_params[tech_node_pair[0]]).simplify()}")
    logger.warning(f"initial value of next tech node edp is {symbolic_sim.edp.xreplace(initial_tech_params[tech_node_pair[1]]).simplify()}")
    logger.warning(f"final tech params for {tech_node_pair[0]} nm: {final_tech_params[tech_node_pair[0]]}")
    logger.warning(f"initial tech params for {tech_node_pair[1]} nm: {initial_tech_params[tech_node_pair[1]]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Inverse Validation",
        description="Validation script for inverse pass.",
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
        default="src/inverse_validation/inverse_val_log_dir",
        help="Path to the save logs",
    )
    parser.add_argument(
        "-t",
        "--test_type",
        type=str,
        default="logic",
        help="Determines whether we sweep logic or cacti tech nodes",
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
    if (args.test_type == "logic"):
        tech_nodes = logic_tech_nodes
    elif (args.test_type == "cacti"):
        tech_nodes = cacti_nodes
    else:
        raise Exception("invalid test type: must be logic or cacti")

    edps, hws, dfgs = run_initial()

    for i in range(0, len(tech_nodes)-1):
        improvement = edps[i]/edps[i+1]
        run_pairwise(tech_nodes[i:i+2], improvement, hws, dfgs)
        plot_diff(tech_nodes[i:i+2])