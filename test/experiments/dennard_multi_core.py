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
from src import trend_plot

class DennardMultiCore:
    def __init__(self, args):
        self.args = args
        self.codesign_module = codesign.Codesign(
            self.args
        )
        self.dummy_app = args.dummy
        self.params_over_iterations = []
        self.edp_over_iterations = []
        self.lag_factor_over_iterations = [1.0]

    def run_dummy_forward_pass(self):

        self.num_inverters = 1e5
        self.utilization = 0.1
        self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values[self.codesign_module.hw.circuit_model.tech_model.base_params.f] = 100e6
        self.cycle_time = 1e9/self.codesign_module.hw.circuit_model.tech_model.base_params.f # ns
        self.codesign_module.hw.execution_time = (self.codesign_module.hw.circuit_model.tech_model.delay*(1e5/self.utilization)).subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values) #ns

        self.codesign_module.hw.total_passive_energy = (self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.P_pass_inv * self.codesign_module.hw.execution_time).subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values)
        self.codesign_module.hw.total_active_energy = (self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.C_gate * self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd**2 * self.codesign_module.hw.circuit_model.tech_model.base_params.f * self.codesign_module.hw.execution_time * self.utilization).subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values)
        if self.args.obj == "edp":
            self.codesign_module.hw.obj = (self.codesign_module.hw.total_passive_energy + self.codesign_module.hw.total_active_energy) * self.codesign_module.hw.execution_time
        elif self.args.obj == "delay":
            self.codesign_module.hw.obj = self.codesign_module.hw.execution_time
        elif self.args.obj == "energy":
            self.codesign_module.hw.obj = self.codesign_module.hw.total_active_energy + self.codesign_module.hw.total_passive_energy
        self.codesign_module.hw.obj_sub_exprs = {
            "execution_time": self.codesign_module.hw.execution_time,
            "total_passive_energy": self.codesign_module.hw.total_passive_energy,
            "total_active_energy": self.codesign_module.hw.total_active_energy,
            "passive power": self.codesign_module.hw.total_passive_energy/self.codesign_module.hw.execution_time,
        }
        self.codesign_module.display_objective("after forward pass")

        print(f"initial area: {(self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.A_gate * 2).subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values)}")



    def run_dummy_inverse_pass(self):

        self.codesign_module.hw.execution_time = self.codesign_module.hw.circuit_model.tech_model.delay*(1e5/self.utilization)
        self.codesign_module.hw.total_passive_energy = self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.P_pass_inv * self.codesign_module.hw.execution_time
        self.codesign_module.hw.total_active_energy = self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.C_gate * self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd**2 * self.codesign_module.hw.circuit_model.tech_model.base_params.f * self.codesign_module.hw.execution_time * self.utilization
        if self.args.obj == "edp":
            self.codesign_module.hw.symbolic_obj = (self.codesign_module.hw.total_passive_energy + self.codesign_module.hw.total_active_energy) * self.codesign_module.hw.execution_time
        elif self.args.obj == "delay":
            self.codesign_module.hw.symbolic_obj = self.codesign_module.hw.execution_time
        elif self.args.obj == "energy":
            self.codesign_module.hw.symbolic_obj = self.codesign_module.hw.total_active_energy + self.codesign_module.hw.total_passive_energy
        self.codesign_module.hw.symbolic_obj_sub_exprs = {
            "execution_time": self.codesign_module.hw.execution_time,
            "passive power": self.codesign_module.hw.total_passive_energy/self.codesign_module.hw.execution_time,
            "active power": self.codesign_module.hw.total_active_energy/self.codesign_module.hw.execution_time,
            "subthreshold leakage current": self.codesign_module.hw.circuit_model.tech_model.I_off,
            "gate tunneling current": self.codesign_module.hw.circuit_model.tech_model.I_tunnel,
            "FN term": self.codesign_module.hw.circuit_model.tech_model.FN_term,
            "WKB term": self.codesign_module.hw.circuit_model.tech_model.WKB_term,
            "GIDL current": self.codesign_module.hw.circuit_model.tech_model.I_GIDL,
            "effective threshold voltage": self.codesign_module.hw.circuit_model.tech_model.V_th_eff,
            "supply voltage": self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd,
            "wire RC": self.codesign_module.hw.circuit_model.tech_model.m1_Rsq * self.codesign_module.hw.circuit_model.tech_model.m1_Csq,
        }
        self.codesign_module.display_objective("before inverse pass", symbolic=True)

        self.disabled_knobs = [self.codesign_module.hw.circuit_model.tech_model.base_params.f, self.codesign_module.hw.circuit_model.tech_model.base_params.u_n]

        stdout = sys.stdout
        with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
            lag_factor, error = self.codesign_module.opt.optimize("ipopt", improvement=self.codesign_module.inverse_pass_improvement, disabled_knobs=self.disabled_knobs)
            self.codesign_module.inverse_pass_lag_factor *= lag_factor
        sys.stdout = stdout
        f = open("src/tmp/ipopt_out.txt", "r")
        if not error:
            self.codesign_module.parse_output(f)

        self.codesign_module.write_back_params()
        print(f"inverse pass lag factor: {self.codesign_module.inverse_pass_lag_factor}")

        self.codesign_module.display_objective("after inverse pass", symbolic=True)

    def update_params_over_iterations(self):
        self.params_over_iterations.append(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values.copy())

    def run_experiment(self):
        if self.dummy_app:
            self.run_dummy_forward_pass()
        else:
            self.codesign_module.forward_pass()
        self.codesign_module.log_forward_tech_params()

        self.update_params_over_iterations()
        self.edp_over_iterations.append(self.codesign_module.hw.obj)

        # run technology optimization to simulate dennard scaling.
        # show that over time, as Vdd comes up to limit, benefits are more difficult to find
        for i in range(self.args.num_opt_iters):
            initial_tech_params = copy.copy(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values)
            if self.dummy_app:
                self.run_dummy_inverse_pass()
            else:
                self.codesign_module.inverse_pass()
                self.codesign_module.hw.circuit_model.update_circuit_values()
            self.edp_over_iterations.append(self.codesign_module.hw.symbolic_obj.subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values))
            self.lag_factor_over_iterations.append(self.codesign_module.inverse_pass_lag_factor)

            regularization = 0
            for var in self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values:
                regularization += (max(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values[var]/initial_tech_params[var] - 1,
                                initial_tech_params[var]/self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values[var] - 1)**2)
            logger.info(f"regularization in iteration {i}: {regularization}")
            self.codesign_module.log_all_to_file(i)
            self.update_params_over_iterations()

            # update schedule with modified technology parameters
            if not self.dummy_app:
                self.codesign_module.hw.update_schedule_with_latency()
            
            if self.codesign_module.inverse_pass_lag_factor >= 2.0 and not self.dummy_app:
                self.codesign_module.hw.reset_state()
                self.codesign_module.forward_pass()
                self.codesign_module.log_forward_tech_params()
                self.edp_over_iterations.append(self.codesign_module.hw.obj)
            else:
                self.edp_over_iterations.append(self.codesign_module.hw.symbolic_obj.subs(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values))

        trend_plotter = trend_plot.TrendPlot(self.codesign_module, self.params_over_iterations, self.edp_over_iterations, self.lag_factor_over_iterations, self.codesign_module.save_dir + "/figs")
        trend_plotter.plot_params_over_iterations()
        trend_plotter.plot_edp_over_iterations()
        trend_plotter.plot_lag_factor_over_iterations()
        
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
        "-i",
        "--inverse_pass_improvement",
        type=float,
        default=10,
        help="Improvement factor for inverse pass",
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
        default="test/experiments/dennard_multi_core_logs/figs",
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
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="default",
        help="symbolic model configuration"
    )
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    experiment = DennardMultiCore(args)
    experiment.run_experiment()    