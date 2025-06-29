import argparse
import copy
import logging
import os
import sys
import yaml
import shutil
import matplotlib.pyplot as plt

logger = logging.getLogger("dennard multi core")

"""project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
os.chdir(project_root)

# Now `current_directory` should reflect the new working directory
current_directory = os.getcwd()
print(current_directory)

# Add the parent directory to sys.path
sys.path.insert(0, current_directory)"""

# Now you can safely import modules that rely on the correct working directory
from src import codesign

class DennardMultiCore:
    def __init__(self, args):
        self.args = args
        self.codesign_module = codesign.Codesign(
            self.args
        )
        self.dummy_app = args.dummy
        self.params_over_iterations = []
        self.plot_list = set([
            self.codesign_module.hw.params.V_dd,
            self.codesign_module.hw.params.V_th,
            #self.codesign_module.hw.params.u_n,
            self.codesign_module.hw.params.L,
            self.codesign_module.hw.params.W,
            self.codesign_module.hw.params.Cox,
        ])

    def plot_params_over_iterations(self):
        fig_save_dir = "test/experiments/dennard_multi_core_figs"
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        f = open(f"{fig_save_dir}/param_data.txt", 'w')
        f.write(str(self.params_over_iterations))
        for param in self.plot_list:
            values = []
            for i in range(len(self.params_over_iterations)):
                values.append(self.params_over_iterations[i][param])
                plt.plot(values)
                plt.xlabel("iteration")
                plt.ylabel("value")
                plt.title(f"{param.name} over iterations")
                plt.yscale("log")
                plt.grid(True)
                plt.savefig(f"{fig_save_dir}/{param.name}_over_iters.png")
                plt.close()

    def run_dummy_forward_pass(self):
        #self.codesign_module.hw.params.L = 5e-7
        #self.codesign_module.hw.params.W = 1.51e-6
        #self.codesign_module.hw.params.Cox = 0.002
        self.num_inverters = 1e8
        self.utilization = 0.1
        self.codesign_module.hw.params.tech_values[self.codesign_module.hw.params.f] = 100e6
        self.cycle_time = 1e9/self.codesign_module.hw.params.f # ns
        self.codesign_module.hw.execution_time = (self.codesign_module.hw.params.delay*(1e5/self.utilization)).subs(self.codesign_module.hw.params.tech_values) #ns
        self.plot_list.add(self.codesign_module.hw.params.f)
        self.codesign_module.hw.total_passive_energy = (self.num_inverters * self.codesign_module.hw.params.P_pass_inv * self.codesign_module.hw.execution_time).subs(self.codesign_module.hw.params.tech_values)
        self.codesign_module.hw.total_active_energy = (self.num_inverters * self.codesign_module.hw.params.C_gate * self.codesign_module.hw.params.V_dd**2 * self.codesign_module.hw.params.f * self.codesign_module.hw.execution_time * self.utilization).subs(self.codesign_module.hw.params.tech_values)
        if self.args.obj == "edp":
            self.codesign_module.hw.obj = (self.codesign_module.hw.total_passive_energy + self.codesign_module.hw.total_active_energy) * self.codesign_module.hw.execution_time
        elif self.args.obj == "delay":
            self.codesign_module.hw.obj = self.codesign_module.hw.execution_time
        self.codesign_module.hw.obj_sub_exprs = {
            "execution_time": self.codesign_module.hw.execution_time,
            "total_passive_energy": self.codesign_module.hw.total_passive_energy,
            "total_active_energy": self.codesign_module.hw.total_active_energy,
            "passive power": self.codesign_module.hw.total_passive_energy/self.codesign_module.hw.execution_time,
        }
        self.codesign_module.display_objective("after forward pass")

        print(f"initial area: {(self.num_inverters * self.codesign_module.hw.params.A_gate * 2).subs(self.codesign_module.hw.params.tech_values)}")



    def run_dummy_inverse_pass(self):

        self.codesign_module.hw.execution_time = self.codesign_module.hw.params.delay*(1e5/self.utilization)
        self.codesign_module.hw.total_passive_energy = self.num_inverters * self.codesign_module.hw.params.P_pass_inv * self.codesign_module.hw.execution_time
        self.codesign_module.hw.total_active_energy = self.num_inverters * self.codesign_module.hw.params.C_gate * self.codesign_module.hw.params.V_dd**2 * self.codesign_module.hw.params.f * self.codesign_module.hw.execution_time * self.utilization
        if self.args.obj == "edp":
            self.codesign_module.hw.symbolic_obj = (self.codesign_module.hw.total_passive_energy + self.codesign_module.hw.total_active_energy) * self.codesign_module.hw.execution_time
        elif self.args.obj == "delay":
            self.codesign_module.hw.symbolic_obj = self.codesign_module.hw.execution_time
        self.codesign_module.hw.symbolic_obj_sub_exprs = {
            "execution_time": self.codesign_module.hw.execution_time,
            "passive power": self.codesign_module.hw.total_passive_energy/self.codesign_module.hw.execution_time,
            "active power": self.codesign_module.hw.total_active_energy/self.codesign_module.hw.execution_time,
            "subthreshold leakage current": self.codesign_module.hw.params.I_off,
            "gate tunneling current": self.codesign_module.hw.params.I_tunnel,
            "FN term": self.codesign_module.hw.params.FN_term,
            "WKB term": self.codesign_module.hw.params.WKB_term,
            "effective threshold voltage": self.codesign_module.hw.params.V_th_eff,
            "supply voltage": self.codesign_module.hw.params.V_dd,
        }
        self.codesign_module.display_objective("before inverse pass", symbolic=True)

        self.disabled_knobs = [self.codesign_module.hw.params.f]

        stdout = sys.stdout
        with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
            self.codesign_module.opt.optimize("ipopt", improvement=self.codesign_module.inverse_pass_improvement, disabled_knobs=self.disabled_knobs)
        sys.stdout = stdout
        f = open("src/tmp/ipopt_out.txt", "r")
        self.codesign_module.parse_output(f)

        self.codesign_module.write_back_params()

        self.codesign_module.display_objective("after inverse pass", symbolic=True)

    def run_experiment(self):
        if self.dummy_app:
            self.run_dummy_forward_pass()
        else:
            self.codesign_module.forward_pass()
        self.codesign_module.log_forward_tech_params()

        self.params_over_iterations.append(copy.copy(self.codesign_module.hw.params.tech_values))

        # run technology optimization to simulate dennard scaling.
        # show that over time, as Vdd comes up to limit, benefits are more difficult to find
        for i in range(self.args.num_opt_iters):
            initial_tech_params = copy.copy(self.codesign_module.hw.params.tech_values)
            if self.dummy_app:
                self.run_dummy_inverse_pass()
            else:
                self.codesign_module.inverse_pass()
            
            regularization = 0
            for var in self.codesign_module.hw.params.tech_values:
                regularization += (max(self.codesign_module.hw.params.tech_values[var]/initial_tech_params[var] - 1,
                                initial_tech_params[var]/self.codesign_module.hw.params.tech_values[var] - 1)**2)
            logger.info(f"regularization in iteration {i}: {regularization}")
            self.codesign_module.log_all_to_file(i)
            self.params_over_iterations.append(copy.copy(self.codesign_module.hw.params.tech_values))

            # update schedule with modified technology parameters
            if not self.dummy_app:
                self.codesign_module.hw.update_schedule_with_latency()

        self.plot_params_over_iterations()
        
        # now run forward pass to demonstrate how parallelism can be added
        # to combat diminishing tech benefits at the end of Dennard scaling
        #self.codesign_module.forward_pass()

        # restore dat file
        self.codesign_module.restore_dat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dennard Scaling, Multi-Core Architecture Experiment",
        description="Script demonstrating recreation of historical technology/architecture trends.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        help="Name of benchmark to run"
    )
    parser.add_argument(
        "-f",
        "--savedir",
        type=str,
        default="test/experiments/dennard_multi_core_logs",
        help="Path to the save new architecture file",
    )
    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        default="estimation",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="src/tmp/pd/tcl/codesign_top.tcl",
        help="what tcl file will be executed for openroad",
    )
    parser.add_argument(
        "-a",
        "--area",
        type=float,
        default=1000000,
        help="Area constraint in um2",
    )
    parser.add_argument(
        "--no_memory",
        type=bool,
        default=False,
        help="disable memory modeling",
    )
    parser.add_argument(
        "-N",
        "--num_opt_iters",
        type=int,
        default=3,
        help="Number of tech optimization iterations"
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default="test/experiments/dennard_multi_core_figs",
        help="path to save figs"
    )
    parser.add_argument(
        "--debug_no_cacti",
        type=bool,
        default=False,
        help="disable cacti in the first iteration to decrease runtime when debugging"
    )
    parser.add_argument(
        "--tech_node",
        type=str,
        default="default",
        help="tech node in nm"
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="edp",
        help="objective function to optimize"
    )
    parser.add_argument(
        "--dummy",
        type=bool,
        default=False,
        help="dummy application"
    )
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    experiment = DennardMultiCore(args)
    experiment.run_experiment()    