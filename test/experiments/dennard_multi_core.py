import argparse
import copy
import logging
import os
import sys
import yaml
import shutil
import matplotlib.pyplot as plt

logger = logging.getLogger("dennard multi core")

# Now you can safely import modules that rely on the correct working directory
from src import codesign
from src import trend_plot

class DennardMultiCore:
    def __init__(self, args):
        self.args = args
        self.codesign_module = codesign.Codesign(
            self.args
        )
        # for this experiment, we don't use the scaled obj feature, instead just bake the parallelism scaling into the obj
        max_parallel_val = 1
        self.codesign_module.hw.circuit_model.tech_model.init_scale_factors(max_parallel_val)
        self.dummy_app = args.dummy
        self.params_over_iterations = []
        self.edp_over_iterations = []
        self.lag_factor_over_iterations = [1.0]

    def calculate_objective(self):
        self.codesign_module.hw.execution_time = self.codesign_module.hw.circuit_model.tech_model.delay*(self.num_switches_per_inverter/self.utilization)
        self.codesign_module.hw.total_passive_energy = self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.P_pass_inv * self.codesign_module.hw.execution_time
        self.codesign_module.hw.total_active_energy = self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.E_act_inv * self.num_switches_per_inverter

    def run_dummy_forward_pass(self):

        self.num_inverters = 1e5
        self.utilization = 0.02
        self.num_switches_per_inverter = 1e4
        self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values[self.codesign_module.hw.circuit_model.tech_model.base_params.f] = 100e6
        self.cycle_time = 1e9/self.codesign_module.hw.circuit_model.tech_model.base_params.f # ns
        self.calculate_objective()
        self.codesign_module.hw.save_obj_vals()
        self.codesign_module.display_objective("after forward pass")

        print(f"initial area: {(self.num_inverters * self.codesign_module.hw.circuit_model.tech_model.A_gate * 2).xreplace(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values).evalf()}")



    def run_dummy_inverse_pass(self, k_gate_disabled=True):

        self.calculate_objective()
        self.codesign_module.hw.save_obj_vals()

        self.codesign_module.display_objective("before inverse pass")

        self.disabled_knobs = [self.codesign_module.hw.circuit_model.tech_model.base_params.f, self.codesign_module.hw.circuit_model.tech_model.base_params.u_n]
        if k_gate_disabled:
            self.disabled_knobs.append(self.codesign_module.hw.circuit_model.tech_model.base_params.k_gate)

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

        self.codesign_module.display_objective("after inverse pass")

    def update_params_over_iterations(self):
        self.params_over_iterations.append(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values.copy())

    def run_experiment(self):
        if self.dummy_app:
            self.run_dummy_forward_pass()
        else:
            self.codesign_module.forward_pass()
        self.codesign_module.log_forward_tech_params()

        self.update_params_over_iterations()
        self.edp_over_iterations.append(self.codesign_module.hw.obj.xreplace(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values).evalf())

        # run technology optimization to simulate dennard scaling.
        # show that over time, as Vdd comes up to limit, benefits are more difficult to find
        for i in range(self.args.num_opt_iters):
            initial_tech_params = copy.copy(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values)
            if self.dummy_app:
                self.run_dummy_inverse_pass(k_gate_disabled=(i<11)) # k_gate_disabled=(i<11) as alternative
            else:
                self.codesign_module.inverse_pass()
                self.codesign_module.hw.circuit_model.update_circuit_values()
            self.edp_over_iterations.append(self.codesign_module.hw.obj.xreplace(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values).evalf())
            self.lag_factor_over_iterations.append(self.codesign_module.inverse_pass_lag_factor)

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
                self.edp_over_iterations.append(self.codesign_module.hw.obj.xreplace(self.codesign_module.hw.circuit_model.tech_model.base_params.tech_values).evalf())
            self.codesign_module.hw.reset_tech_model()
        
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
        help="symbolic model configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="codesign configuration file"
    )
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    try:
        experiment = DennardMultiCore(args)
        experiment.run_experiment()    
    finally:
        experiment.codesign_module.end_of_run_plots(experiment.edp_over_iterations, experiment.lag_factor_over_iterations, experiment.params_over_iterations)