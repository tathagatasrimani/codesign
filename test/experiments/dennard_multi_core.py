import argparse
import copy
import logging
import os
import sys
import yaml
import matplotlib.pyplot as plt

logger = logging.getLogger("dennard multi core")

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

class DennardMultiCore:
    def __init__(self, args):
        self.args = args
        self.logic_node = 7
        self.cacti_node = 22
        self.tech_params = {}
        self.hw = None
        self.params_over_iterations = []
        self.plot_list = set([
            hw_symbols.V_dd,
            hw_symbols.Reff["Add"],
            hw_symbols.Ceff["Add"],
            hw_symbols.Reff["Mult"],
            hw_symbols.Ceff["Mult"],
            hw_symbols.Reff["Regs"],
            hw_symbols.Ceff["Regs"]
        ])


    def write_back_rcs(self, rcs_path="src/params/rcs_current.yaml"):
        rcs = {"Reff": {}, "Ceff": {}, "Cacti": {}, "Cacti_IO": {}, "other": {}}
        for elem in self.tech_params:
            if (
                elem.name == "f"
                or elem.name == "V_dd"
                or elem.name.startswith("Mem")
                or elem.name.startswith("Buf")
                or elem.name.startswith("OffChipIO")
            ):
                rcs["other"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            elif elem.name in hw_symbols.cacti_tech_params:
                rcs["Cacti"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            elif elem.name in hw_symbols.cacti_io_tech_params:
                rcs["Cacti_IO"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            else:
                rcs[elem.name[: elem.name.find("_")]][
                    elem.name[elem.name.find("_") + 1 :]
                ] = self.tech_params[elem]
        with open(rcs_path, "w") as f:
            f.write(yaml.dump(rcs))

    def parse_output(self, f):
        # mostly copied from codesign.py
        tech_params_out = {}
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
            tech_params_out[mapping[key]] = value 
            i += 1
        return tech_params_out
    
    def save_dat(self):
        # Save tech node info to another file prefixed by prev_ so we can restore
        org_dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(org_dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"
        with open(org_dat_file, "r") as src_file, open(prev_dat_file, "w") as dest_file:
            for line in src_file:
                dest_file.write(line)

    def restore_dat(self):
        dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"

        with open(prev_dat_file, "r") as src_file, open(dat_file, "w") as dest_file:
            for line in src_file:
                dest_file.write(line)
        os.remove(prev_dat_file)

    def plot_params_over_iterations(self):
        fig_save_dir = "test/experiments/dennard_multi_core_figs"
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        for param in self.plot_list:
            values = []
            for i in range(len(self.params_over_iterations)):
                values.append(self.params_over_iterations[i][param])
                plt.plot(values)
                plt.xlabel("iteration")
                plt.ylabel("value")
                plt.title(f"{param.name} over iterations")
                plt.yscale("log")
                plt.savefig(f"{fig_save_dir}/{param.name}_over_iters.png")
                plt.close()

    def run_experiment(self):
        # set up framework
        print(
            f"Setting up architecture search; benchmark: {self.args.benchmark}, config: {self.args.architecture_config}"
        )
        (
            simulator,
            self.hw,
            computation_dfg,
        ) = architecture_search.setup_arch_search(self.args.benchmark, self.args.architecture_config, True, self.logic_node, self.cacti_node)
        self.hw.init_memory(
            sim_util.find_nearest_power_2(simulator.memory_needed),
            sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
        )
        scheduled_dfg = simulator.schedule(computation_dfg, self.hw)
        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)
        self.hw.get_wire_parasitics("", "none")
        simulator.simulate(scheduled_dfg, self.hw)
        simulator.calculate_edp()
        print(f"edp of simulation running on {self.logic_node} nm logic tech node, {self.cacti_node} cacti tech node: {simulator.edp} E-18 Js")
        
        rcs = self.hw.get_optimization_params_from_tech_params()
        self.tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

        # save previous dat file
        self.save_dat()

        # run initial symbolic simulation to set up inverse passes
        symbolic_sim = symbolic_simulate.SymbolicSimulator()
        symbolic_sim.simulator_prep(self.args.benchmark, self.hw.latency)
        coefficients.create_and_save_coefficients([self.hw.transistor_size])
        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)

        symbolic_sim.simulate(scheduled_dfg, self.hw)
        cacti_subs = symbolic_sim.calculate_edp(self.hw, concrete_sub=True)
        #print(f"edp: {symbolic_sim.edp}")

        for cacti_var in cacti_subs:
            self.tech_params[cacti_var] = cacti_subs[cacti_var]#.xreplace(self.tech_params).evalf()
        symbolic_sim.edp = symbolic_sim.edp.xreplace(cacti_subs)

        inverse_edp = symbolic_sim.edp.xreplace(self.tech_params).evalf()
        inverse_exec_time = symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
        active_energy = symbolic_sim.total_active_energy.xreplace(self.tech_params).evalf()
        passive_energy = symbolic_sim.total_passive_energy.xreplace(self.tech_params).evalf()

        assert len(inverse_edp.free_symbols) == 0

        print(
            f"Initial EDP: {inverse_edp} E-18 Js. Active Energy: {active_energy} nJ. Passive Energy: {passive_energy} nJ. Execution time: {inverse_exec_time} ns"
        )

        self.params_over_iterations.append(self.tech_params)

        # run technology optimization to simulate dennard scaling.
        # show that over time, as Vdd comes up to limit, benefits are more difficult to find
        for i in range(self.args.num_opt_iters):
            initial_tech_params = copy.copy(self.tech_params)
            stdout = sys.stdout
            with open(f"{self.args.savedir}/ipopt_out_{i}.txt", "w") as sys.stdout:
                optimize.optimize(self.tech_params, symbolic_sim.edp, "ipopt", cacti_subs, regularization=1e-9)
            sys.stdout = stdout
            f = open(f"{self.args.savedir}/ipopt_out_{i}.txt", "r")
            self.tech_params = self.parse_output(f)
            for param in initial_tech_params:
                if param not in self.tech_params:
                    self.tech_params[param] = initial_tech_params[param]
            logger.warning(f"tech params after iteration {i}: {self.tech_params}")
            inverse_edp = symbolic_sim.edp.xreplace(self.tech_params).evalf()
            total_active_energy = (symbolic_sim.total_active_energy).xreplace(self.tech_params).evalf()
            total_passive_energy = (symbolic_sim.total_passive_energy).xreplace(self.tech_params).evalf()
            execution_time = symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
            print(
                f"EDP after iteration {i}: {inverse_edp} E-18 Js. Active Energy: {total_active_energy} nJ. Passive Energy: {total_passive_energy} nJ. Execution time: {execution_time} ns"
            )
            regularization = 0
            for var in self.tech_params:
                regularization += (self.tech_params[var]-initial_tech_params[var])**2
            logger.warning(f"regularization in iteration {i}: {regularization}")
            self.write_back_rcs()
            self.hw.update_technology_parameters()
            self.hw.write_technology_parameters(
                f"{self.args.savedir}/tech_params_{i}.yaml"
            )
            self.params_over_iterations.append(self.tech_params)

        self.plot_params_over_iterations()
        
        # now run architecture search to demonstrate how parallelism can be added
        # to combat diminishing tech benefits at the end of dennard scaling
        sim_util.update_schedule_with_latency(computation_dfg, self.hw.latency)
        sim_util.update_schedule_with_latency(scheduled_dfg, self.hw.latency)

        self.hw.get_wire_parasitics("", "none")
        simulator.simulate(scheduled_dfg, self.hw)
        simulator.calculate_edp()
        edp = simulator.edp
        logger.warning(f"edp after new forward pass: {edp} E-18 Js. {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns")

        new_schedule, new_hw = architecture_search.run_arch_search(
            simulator,
            self.hw,
            computation_dfg,
            self.args.area,
            best_edp=edp,
        )
        print(
            f"Final EDP  : {simulator.edp} E-18 Js. Active Energy: {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns"
        )

        # restore dat file
        self.restore_dat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dennard Scaling, Multi-Core Architecture Experiment",
        description="Script demonstrating recreation of historical technology/architecture trends.",
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
        default="test/experiments/dennard_multi_core_log_dir",
        help="Path to the save logs",
    )
    parser.add_argument(
        "-N",
        "--num_opt_iters",
        type=int,
        default=3,
        help="Number of tech optimization iterations"
    )
    parser.add_argument(
        "--num_arch_search_iters",
        type=int,
        default=1,
        help="Number of Architecture Search iterations to run",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="test/experiments/dennard_multi_core_figs",
        help="path to save figs"
    )
    parser.add_argument("-a", "--area", type=float, default=100000, help="Max Area of the chip in um^2")
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

    logging.basicConfig(filename=f"{args.savedir}/log.txt", level=logging.INFO)

    experiment = DennardMultiCore(args)
    experiment.run_experiment()    