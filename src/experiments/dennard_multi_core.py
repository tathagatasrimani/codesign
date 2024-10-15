import argparse
import copy
import logging
import os
import sys
import yaml

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
        self.cacti_node = 45
        self.tech_params = {}


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

    def run_experiment(self):
        # set up framework
        logger.warning(
            f"Setting up architecture search; benchmark: {self.args.benchmark}, config: {self.args.architecture_config}"
        )
        (
            simulator,
            hw,
            computation_dfg,
        ) = architecture_search.setup_arch_search(self.args.benchmark, self.args.architecture_config, True, self.logic_node, self.cacti_node)
        hw.init_memory(
            sim_util.find_nearest_power_2(simulator.memory_needed),
            sim_util.find_nearest_power_2(simulator.nvm_memory_needed),
        )
        scheduled_dfg = simulator.schedule(computation_dfg, hw)
        hardwareModel.un_allocate_all_in_use_elements(hw.netlist)
        hw.get_wire_parasitics("", "none")
        simulator.simulate(scheduled_dfg, hw)
        simulator.calculate_edp()
        logger.warning(f"edp of simulation running on {self.logic_node} nm logic tech node, {self.cacti_node} cacti tech node: {simulator.edp} E-18 Js")
        
        rcs = hw.get_optimization_params_from_tech_params()
        self.tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

        # run initial symbolic simulation to set up inverse passes
        symbolic_sim = symbolic_simulate.SymbolicSimulator()
        symbolic_sim.simulator_prep(self.args.benchmark, hw.latency)
        coefficients.create_and_save_coefficients([hw.transistor_size])
        hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

        symbolic_sim.simulate(scheduled_dfg, hw)
        cacti_subs = symbolic_sim.calculate_edp(hw)

        for cacti_var in cacti_subs:
            self.tech_params[cacti_var] = cacti_subs[cacti_var].xreplace(self.tech_params).evalf()

        inverse_edp = symbolic_sim.edp.xreplace(self.tech_params).evalf()
        inverse_exec_time = symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
        active_energy = symbolic_sim.total_active_energy.xreplace(self.tech_params).evalf()
        passive_energy = symbolic_sim.total_passive_energy.xreplace(self.tech_params).evalf()

        assert len(inverse_edp.free_symbols) == 0

        logger.warning(
            f"Initial EDP: {inverse_edp} E-18 Js. Active Energy: {active_energy} nJ. Passive Energy: {passive_energy} nJ. Execution time: {inverse_exec_time} ns"
        )

        # run technology optimization to simulate dennard scaling.
        # show that over time, as Vdd comes up to limit, benefits are more difficult to find
        for i in range(self.args.num_opt_iters):
            initial_tech_params = copy.copy(self.tech_params)
            stdout = sys.stdout
            with open(f"{self.args.savedir}/ipopt_out_{i}.txt", "w") as sys.stdout:
                optimize.optimize(self.tech_params, symbolic_sim.edp, "ipopt", cacti_subs)
            sys.stdout = stdout
            f = open(f"{self.args.savedir}/ipopt_out_{i}.txt", "r")
            self.tech_params = self.parse_output(f)
            logger.warning(f"tech params after iteration {i}: {self.tech_params}")
            inverse_edp = symbolic_sim.edp.xreplace(self.tech_params).evalf()
            total_active_energy = (symbolic_sim.total_active_energy).xreplace(self.tech_params).evalf()
            total_passive_energy = (symbolic_sim.total_passive_energy).xreplace(self.tech_params).evalf()
            execution_time = symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
            logger.warning(
                f"EDP after iteration {i}: {inverse_edp} E-18 Js. Active Energy: {total_active_energy} nJ. Passive Energy: {total_passive_energy} nJ. Execution time: {execution_time} ns"
            )
            regularization = 0
            for var in self.tech_params:
                regularization += (self.tech_params[var]-initial_tech_params[var])**2
            logger.warning(f"regularization in iteration {i}: {regularization}")
            self.write_back_rcs()
            hw.update_technology_parameters()
            hw.write_technology_parameters(
                f"{self.args.savedir}/tech_params_{i}.yaml"
            )
        
        # now run architecture search to demonstrate how parallelism can be added
        # to combat diminishing tech benefits at the end of dennard scaling
        sim_util.update_schedule_with_latency(computation_dfg, hw.latency)
        sim_util.update_schedule_with_latency(scheduled_dfg, hw.latency)

        hw.get_wire_parasitics("", "none")
        simulator.simulate(scheduled_dfg, hw)
        simulator.calculate_edp()
        edp = simulator.edp
        logger.warning(f"edp after new forward pass: {edp} E-18 Js. {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns")

        new_schedule, new_hw = architecture_search.run_arch_search(
            simulator,
            hw,
            computation_dfg,
            self.args.area,
            best_edp=edp,
        )
        print(
            f"Final EDP  : {simulator.edp} E-18 Js. Active Energy: {simulator.active_energy} nJ. Passive Energy: {simulator.passive_energy} nJ. Execution time: {simulator.execution_time} ns"
        )

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
        default="src/experiments/dennard_multi_core_log_dir",
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

    logging.basicConfig(filename=f"{args.savedir}/log.txt", level=logging.WARNING)

    experiment = DennardMultiCore(args)
    experiment.run_experiment()    