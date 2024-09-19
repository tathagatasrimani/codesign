import argparse
import os
import yaml
import sys
import datetime
import logging
import shutil

logger = logging.getLogger("codesign")

import sympy as sp
import networkx as nx

from . import architecture_search
from . import coefficients
from . import symbolic_simulate
from . import optimize
from . import hw_symbols
from . import sim_util
from . import hardwareModel


class Codesign:
    def __init__(self, benchmark, area, config, arch_search_iters, save_dir, opt):
        self.save_dir = os.path.join(
            save_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.opt_cfg = opt
        self.num_arch_search_iters = arch_search_iters

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

        # copy the benchmark and instrumented files;
        shutil.copy(benchmark, f"{self.save_dir}/benchmark.py")
        shutil.copy(f"src/instrumented_files/output.txt", f"{self.save_dir}/output.txt")

        logging.basicConfig(filename=f"{self.save_dir}/log.txt", level=logging.INFO)

        self.area_constraint = area  # in um^2
        self.forward_edp = 0
        self.inverse_edp = 0
        self.tech_params = None
        self.initial_tech_params = None
        self.full_tech_params = {}

        logger.info(
            f"Setting up architecture search; benchmark: {benchmark}, config: {config}"
        )
        (
            self.sim,
            self.hw,
            self.computation_dfg,
        ) = architecture_search.setup_arch_search(benchmark, config)

        nx.write_gml(self.computation_dfg, f"{self.save_dir}/computation_dfg.gml")

        logger.info(f"Scheduling computation graph")
        self.scheduled_dfg = self.sim.schedule(
            self.computation_dfg,
            self.hw,
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
        self.sim.calculate_edp()
        self.forward_edp = self.sim.edp

        print(
            f"\nInitial EDP: {self.forward_edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
        )

        with open("src/params/rcs_current.yaml", "w") as f:
            f.write(yaml.dump(rcs))

        # Save tech node info to another file prefixed by prev_ so we can restore
        org_dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(org_dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"
        with open(org_dat_file, "r") as src_file, open(prev_dat_file, "w") as dest_file:
            for line in src_file:
                dest_file.write(line)

    def set_technology_parameters(self, tech_params):
        if self.initial_tech_params == None:
            self.initial_tech_params = tech_params
        self.tech_params = tech_params

    def forward_pass(self):
        print("\nRunning Forward Pass")
        logger.info("Running Forward Pass")

        sim_util.update_schedule_with_latency(self.computation_dfg, self.hw.latency)
        sim_util.update_schedule_with_latency(self.scheduled_dfg, self.hw.latency)

        self.sim.simulate(self.scheduled_dfg, self.hw)
        self.sim.calculate_edp()
        edp = self.sim.edp
        print(
            f"Initial EDP: {edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
        )

        # sim updated in place, hw, schedule and edp returned.
        # TODO: test whether hardware can be updated in place.
        new_schedule, new_hw = architecture_search.run_arch_search(
            self.sim,
            self.hw,
            self.computation_dfg,
            self.area_constraint,
            self.num_arch_search_iters,
            best_edp=edp,
        )

        self.hw = new_hw
        self.scheduled_dfg = new_schedule
        self.forward_edp = self.sim.edp
        print(
            f"Final EDP  : {self.sim.edp} E-18 Js. Active Energy: {self.sim.active_energy} nJ. Passive Energy: {self.sim.passive_energy} nJ. Execution time: {self.sim.execution_time} ns"
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
            self.tech_params[mapping[key]] = (
                value  # just know that self.tech_params contains all dat
            )
            i += 1

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

        self.inverse_edp = self.symbolic_sim.edp.xreplace(self.tech_params).evalf()
        inverse_exec_time = self.symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
        active_energy = self.symbolic_sim.total_active_energy.xreplace(self.tech_params).evalf()
        passive_energy = self.symbolic_sim.total_passive_energy.xreplace(self.tech_params).evalf()

        assert len(self.inverse_edp.free_symbols) == 0

        print(
            f"Initial EDP: {self.inverse_edp} E-18 Js. Active Energy: {active_energy} nJ. Passive Energy: {passive_energy} nJ. Execution time: {inverse_exec_time} ns"
        )
        print(
            f"edp: {self.inverse_edp}, should equal cycles * (active + passive): {inverse_exec_time * (active_energy + passive_energy)}"
        )

        if self.opt_cfg == "ipopt":
            stdout = sys.stdout
            with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
                optimize.optimize(self.tech_params, self.symbolic_sim.edp, self.opt_cfg)
            sys.stdout = stdout
            f = open("src/tmp/ipopt_out.txt", "r")
            self.parse_output(f)
        else:
            self.tech_params = optimize.optimize(
                self.tech_params, self.symbolic_sim.edp, self.opt_cfg
            )
        self.write_back_rcs()

        self.inverse_edp = self.symbolic_sim.edp.xreplace(self.tech_params).evalf()
        total_active_energy = (self.symbolic_sim.total_active_energy).xreplace(
            self.tech_params
        ).evalf()
        total_passive_energy = (self.symbolic_sim.total_passive_energy).xreplace(
            self.tech_params
        ).evalf()
        execution_time = self.symbolic_sim.execution_time.xreplace(self.tech_params).evalf()

        print(
            f"Final EDP  : {self.inverse_edp} E-18 Js. Active Energy: {total_active_energy} nJ. Passive Energy: {total_passive_energy} nJ. Execution time: {execution_time} ns"
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
        nx.write_gml(
            self.scheduled_dfg,
            f"{self.save_dir}/schedule_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        self.write_back_rcs(f"{self.save_dir}/rcs_{iter_number}.yaml")
        shutil.copy(
            "src/tmp/symbolic_edp.txt",
            f"{self.save_dir}/symbolic_edp_{iter_number}.txt",
        )
        shutil.copy("src/tmp/ipopt_out.txt", f"{self.save_dir}/ipopt_{iter_number}.txt")
        shutil.copy(
            "src/tmp/solver_out.txt", f"{self.save_dir}/solver_{iter_number}.txt"
        )
        # save latency, power, and tech params
        self.hw.write_technology_parameters(
            f"{self.save_dir}/tech_params_{iter_number}.yaml"
        )

    def restore_dat(self):
        dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"

        with open(prev_dat_file, "r") as src_file, open(dat_file, "w") as dest_file:
            for line in src_file:
                dest_file.write(line)
        os.remove(prev_dat_file)

    def cleanup(self):
        self.restore_dat()

    def execute(self, num_iters):
        i = 0
        while i < num_iters:
            self.inverse_pass()
            self.hw.update_technology_parameters()

            self.log_all_to_file(i)

            self.forward_pass()
            i += 1

        # cleanup
        self.cleanup()


def main():
    codesign_module = Codesign(
        args.benchmark,
        args.area,
        args.architecture_config,
        args.num_arch_search_iters,
        args.savedir,
        args.opt,
    )
    try:
        codesign_module.execute(args.num_iters)
    except Exception as e:
        codesign_module.cleanup()
        raise e



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
        default="logs",
        help="Path to the save new architecture file",
    )
    parser.add_argument("-o", "--opt", type=str, default="ipopt")
    parser.add_argument(
        "-N",
        "--num_iters",
        type=int,
        default=10,
        help="Number of Codesign iterations to run",
    )
    parser.add_argument(
        "--num_arch_search_iters",
        type=int,
        default=1,
        help="Number of Architecture Search iterations to run",
    )
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, trace: {args.notrace}, architecture_cfg: {args.architecture_config}, area: {args.area}, optimization: {args.opt}"
    )

    main()
