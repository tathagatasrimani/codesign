import argparse
import os
import yaml
import sys
import datetime
import logging
logger = logging.getLogger("codesign")

from sympy import sympify
import networkx as nx

import architecture_search
import coefficients
import symbolic_simulate
import optimize
import hw_symbols
import sim_util
import hardwareModel


class Codesign:
    def __init__(self, benchmark, area, config, save_dir, opt):
        self.save_dir = os.path.join(
            save_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.opt_cfg = opt

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:  # clear the directory
            files = os.listdir(self.save_dir)
            for file in files:
                os.remove(f"{self.save_dir}/{file}")
        with open(f"{self.save_dir}/log.txt", "a") as f:
            f.write("Codesign Log\n")
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"Architecture Config: {config}\n")
            f.write(f"Area: {area}\n")
            f.write(f"Optimization: {opt}\n")

        logging.basicConfig(filename=f"{self.save_dir}/log.txt", level=logging.INFO)

        self.area_constraint = area
        self.forward_edp = 0
        self.inverse_edp = 0
        self.tech_params = None
        self.initial_tech_params = None
        self.full_tech_params = {}

        logger.info(f"Setting up architecture search; benchmark: {benchmark}, config: {config}")
        (
            self.sim,
            self.hw,
            self.computation_dfg,
        ) = architecture_search.setup_arch_search(benchmark, config)

        logger.info(f"Scheduling computation graph")
        self.scheduled_dfg = self.sim.schedule(
            self.computation_dfg,
            hw_counts=hardwareModel.get_func_count(self.hw.netlist),
        )

        self.symbolic_sim = symbolic_simulate.SymbolicSimulator()
        self.symbolic_sim.simulator_prep(benchmark, self.hw.latency)

        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)

        # starting point set by the config we load into the HW model
        coefficients.create_and_save_coefficients([self.hw.transistor_size])

        rcs = self.hw.get_optimization_params_from_tech_params()
        initial_tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

        self.set_technology_parameters(initial_tech_params)

        logger.info(f"Running initial forward pass")
        self.sim.simulate(self.scheduled_dfg, self.hw)
        self.sim.calculate_edp(self.hw)
        self.forward_edp = self.sim.edp

        print(
            f"\nInitial EDP: {self.forward_edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
        )

        with open("rcs_current.yaml", "w") as f:
            f.write(yaml.dump(rcs))

    def set_technology_parameters(self, tech_params):
        if self.initial_tech_params == None:
            self.initial_tech_params = tech_params
        self.tech_params = tech_params

    def forward_pass(self):
        print("\nRunning Forward Pass")
        logger.info("Running Forward Pass")

        sim_util.update_schedule_with_latency(self.scheduled_dfg, self.hw.latency)
        sim_util.update_schedule_with_latency(self.computation_dfg, self.hw.latency)

        self.sim.simulate(self.scheduled_dfg, self.hw)
        self.sim.calculate_edp(self.hw)
        edp = self.sim.edp
        print(
            f"Initial EDP: {edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
        )

        # hw updated in place, schedule and edp returned.
        new_edp, new_schedule = architecture_search.run_arch_search(
            self.sim,
            self.hw,
            self.computation_dfg,
            self.area_constraint,
            best_edp=edp,
        )
        self.scheduled_dfg = new_schedule
        self.forward_edp = new_edp
        print(
            f"New EDP    : {new_edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
        )

    def parse_output(self, f):
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
            self.tech_params[mapping[key]] = value
            i += 1

    def write_back_rcs(self, rcs_path="rcs_current.yaml"):
        rcs = {"Reff": {}, "Ceff": {}, "other": {}}
        for elem in self.tech_params:
            if elem.name == "f" or elem.name == "V_dd" or elem.name.startswith("Mem") or elem.name.startswith("Buf") or elem.name.startswith("OffChipIO"):
                rcs["other"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            else:
                rcs[elem.name[: elem.name.find("_")]][
                    elem.name[elem.name.find("_") + 1 :]
                ] = self.tech_params[elem]
        with open(rcs_path, "w") as f:
            f.write(yaml.dump(rcs))

    def clear_dfg_allocations(self):
        for k, computation_graph in self.cfg_node_to_hw_map.items():
            for node in computation_graph.nodes:
                if "allocation" in computation_graph.nodes[node]:
                    del computation_graph.nodes[node]["allocation"]

    def inverse_pass(self):
        print("\nRunning Inverse Pass")

        hardwareModel.un_allocate_all_in_use_elements(self.hw.netlist)

        self.symbolic_sim.simulate(self.scheduled_dfg, self.hw)
        self.symbolic_sim.calculate_edp(self.hw)
        self.symbolic_sim.save_edp_to_file()

        self.inverse_edp = self.symbolic_sim.edp.subs(self.tech_params)
        self.inverse_edp_ceil = self.symbolic_sim.edp_ceil.subs(self.tech_params)

        print(
            f"Initial EDP: {self.inverse_edp} E-18 Js. Active Energy: {(self.symbolic_sim.total_active_energy).subs(self.tech_params)} nJ. Passive Energy: {(self.symbolic_sim.total_passive_energy).subs(self.tech_params)} nJ. Execution time: {self.symbolic_sim.cycles.subs(self.tech_params)} ns"
        )

        if (self.opt_cfg == "ipopt"):
            stdout = sys.stdout
            with open("ipopt_out.txt", "w") as sys.stdout:
                optimize.optimize(self.tech_params, self.symbolic_sim.edp, self.opt_cfg)
            sys.stdout = stdout
            f = open("ipopt_out.txt", "r")
            self.parse_output(f)
        else:
            self.tech_params = optimize.optimize(
                self.tech_params, self.symbolic_sim.edp, self.opt_cfg
            )
        self.write_back_rcs()
        self.inverse_edp = self.symbolic_sim.edp.subs(self.tech_params)
        self.inverse_edp_ceil = self.symbolic_sim.edp_ceil.subs(self.tech_params)

        print(
            f"Final EDP  : {self.inverse_edp} E-18 Js. Active Energy: {(self.symbolic_sim.total_active_energy).subs(self.tech_params)} nJ. Passive Energy: {(self.symbolic_sim.total_passive_energy).subs(self.tech_params)} nJ. Execution time: {self.symbolic_sim.cycles.subs(self.tech_params)} ns"
        )

    def log_all_to_file(self, iter_number):
        with open(f"{self.save_dir}/results.txt", "a") as f:
            f.write(f"{iter_number}\n")
            f.write(f"Forward EDP: {self.forward_edp}\n")
            f.write(f"Inverse EDP: {self.inverse_edp}\n")
        nx.write_gml(
            self.hw.netlist,
            f"{self.save_dir}/netlist_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        self.write_back_rcs(f"{self.save_dir}/rcs_{iter_number}.yaml")
        # save latency, power, and tech params
        self.hw.write_technology_parameters(
            f"{self.save_dir}/tech_params_{iter_number}.yaml"
        )


def main():
    codesign_module = Codesign(args.benchmark, args.area, args.architecture_config, args.savedir, args.opt)

    i = 0
    while i < 10:

        codesign_module.inverse_pass()
        codesign_module.hw.update_technology_parameters()

        codesign_module.log_all_to_file(i)

        codesign_module.forward_pass()
        # TODO: create stopping condition
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Codesign",
        description="Runs a two-step loop to optimize architecture and technology for a given application.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument("-a", "--area", type=float, help="Max Area of the chip in um^2")
    parser.add_argument(
        "-c",
        "--architecture_config",
        type=str,
        default="aladdin_const_with_mem",
        help="Path to the architecture config file",
    )
    parser.add_argument(
        "-f",
        "--savedir",
        type=str,
        default="codesign_log_dir",
        help="Path to the save new architecture file",
    )
    parser.add_argument("-o", "--opt", type=str, default="ipopt")
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace: {args.notrace}, architecture_cfg: {args.architecture_config}, area: {args.area}, optimization: {args.opt}"
    )

    main()
