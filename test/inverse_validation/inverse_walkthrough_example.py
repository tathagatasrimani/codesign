import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import itertools

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

class InverseWalkthrough:
    def __init__(self):
        self.args = None
        self.initial_tech_params = None
        self.hw = None
        self.scheduled_dfg = None
        self.dE_mult, self.dVth, self.dC_ox, self.dC_mult, self.dR_mult = [], [], [], [], []

    def plot_3d(self, edp, symbol_list, final_tech_params):
        tech_params_to_sub_out = {}
        for param in final_tech_params:
            if not (param == symbol_list[0] or param == symbol_list[1]):
                tech_params_to_sub_out[param] = final_tech_params[param]

        edp_with_two_symbols = edp.xreplace(tech_params_to_sub_out)

        lam = sp.lambdify(symbol_list, edp_with_two_symbols, 'numpy')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, 1, 30)
        y = np.linspace(0, 1, 30)

        points = list(itertools.product(x,y))
        x_new = list(map(lambda p: p[0], points))
        y_new = list(map(lambda p: p[1], points))
        Z = list(map(lambda tup: lam(tup[0], tup[1]), points))

        ax.scatter(x_new, y_new, Z)
        plt.xlabel(symbol_list[0])
        plt.ylabel(symbol_list[1])
        ax.set_zlabel("EDP")
        fig_save_dir = "test/inverse_validation/walkthrough_figs"
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        plt.savefig(f"{fig_save_dir}/3d.png")

    def plot_results(self):
        plt.plot(self.dE_mult, label="E_mult")
        plt.plot(self.dC_ox, label="C_ox")
        plt.plot(self.dVth, label="Vth")
        plt.plot(self.dC_mult, label="C_mult")
        plt.plot(self.dR_mult, label="R_mult")
        plt.xlabel("iteration")
        plt.ylabel("gradient value")
        plt.legend()
        plt.title("Gradients of different simulator variables across iterations")
        fig_save_dir = "test/inverse_validation/walkthrough_figs"
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        plt.savefig(f"{fig_save_dir}/grads.png")

    # plotting Cmult, Rmult, Emult, Vth, C_ox
    def record_grads(self, edp, cacti_exprs, current_tech_params):

        self.dC_mult.append(sp.diff(edp, hw_symbols.Ceff["Mult"]).xreplace(current_tech_params))
        self.dR_mult.append(sp.diff(edp, hw_symbols.Reff["Mult"]).xreplace(current_tech_params))
        print("done with R and C gradient calculation")
        self.dVth.append(sp.diff(edp, hw_symbols.Vth).xreplace(current_tech_params))
        print("done with Vth gradient calculation")
        self.dC_ox.append(sp.diff(edp, hw_symbols.C_ox).xreplace(current_tech_params))

        # E = CV^2 -> C = E/V^2
        dC_dE = 1/(current_tech_params[hw_symbols.V_dd]**2)
        self.dE_mult.append(dC_dE * self.dC_mult[-1]) # dC/dE * dEDP/dC = d/dE

    def parse_output(self, f):
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

    def run_forward_pass(self):

        # initialize hw model, override architecture config to have current tech node
        logger.info(
            f"Setting up architecture search; benchmark: {self.args.benchmark}, config: {self.args.architecture_config}"
        )
        (
            simulator,
            self.hw,
            computation_dfg,
        ) = architecture_search.setup_arch_search(self.args.benchmark, self.args.architecture_config)
        self.hw.init_memory(
            sim_util.find_nearest_power_2(simulator.memory_needed),
            sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
        )
        self.scheduled_dfg = simulator.schedule(computation_dfg, self.hw)
        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)
        self.hw.get_wire_parasitics("", "none")
        simulator.simulate(self.scheduled_dfg, self.hw)
        simulator.calculate_edp()
        logger.warning(f"edp of forward pass: {simulator.edp} E-18 Js")
        
        rcs = self.hw.get_optimization_params_from_tech_params()
        self.initial_tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

    def run_inverse_passes(self, num_iters):
        symbolic_sim = symbolic_simulate.SymbolicSimulator()
        symbolic_sim.simulator_prep(self.args.benchmark, self.hw.latency)
        coefficients.create_and_save_coefficients([self.hw.transistor_size])
        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)

        symbolic_sim.simulate(self.scheduled_dfg, self.hw)
        cacti_subs = symbolic_sim.calculate_edp(self.hw)

        current_tech_params = self.initial_tech_params.copy()

        for _ in range(num_iters):
            stdout = sys.stdout
            with open(f"{self.args.savedir}/ipopt_out.txt", "w") as sys.stdout:
                optimize.optimize(current_tech_params, symbolic_sim.edp, "ipopt", cacti_subs, regularization=0.1)
            sys.stdout = stdout
            f = open(f"{self.args.savedir}/ipopt_out.txt", "r")
            current_tech_params = self.parse_output(f)
            self.record_grads(symbolic_sim.edp, symbolic_sim.cacti_exprs, current_tech_params)
        
        self.plot_results()
        
        self.plot_3d(symbolic_sim.edp, ["Ceff_Mult", "C_g_ideal"], current_tech_params)

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
        default="test/inverse_validation/inverse_walkthrough_log_dir",
        help="Path to the save logs",
    )
    parser.add_argument(
        "-N",
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

    inv_walkthrough = InverseWalkthrough()
    inv_walkthrough.args = args

    # run forward pass once to setup inverse pass
    inv_walkthrough.run_forward_pass()

    # run multiple iterations of inverse pass to demonstrate how gradients of variables change
    inv_walkthrough.run_inverse_passes(args.num_iters)