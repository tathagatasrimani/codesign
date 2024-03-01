import argparse
import os
import simulate
import architecture_search
import symbolic_simulate
import optimize
import hw_symbols
import yaml
from sympy import sympify

from sim_util import generate_init_params_from_rcs



initial_tech_params = {
    hw_symbols.f: 2e9,
    hw_symbols.V_dd: 1.1,
}

class Codesign:
    def __init__(self, benchmark):
        self.tech_params = None
        self.full_tech_params = {}
        (
            self.sim,
            self.hw,
            self.cfg,
            self.cfg_node_to_hw_map,
            self.saved_elem,
            self.max_continuous,
        ) = architecture_search.setup_arch_search(benchmark)
        self.symbolic_sim = symbolic_simulate.SymbolicSimulator()
        self.symbolic_sim.simulator_prep(benchmark, self.hw.latency)
        self.symbolic_sim.id_to_node = self.sim.id_to_node
        self.symbolic_sim.update_data_path(self.sim.data_path)

    def set_technology_parameters(self, tech_params):
        self.tech_params = tech_params
        # self.create_full_tech_params()

    def forward_pass(self, area_constraint):
        print("\nRunning Forward Pass")
        new_hw, new_edp = architecture_search.run_architecture_search(
            self.sim,
            self.hw,
            self.cfg,
            self.cfg_node_to_hw_map,
            self.saved_elem,
            self.max_continuous,
            area_constraint
        )
        self.hw = new_hw
        self.symbolic_sim.id_to_node = self.sim.id_to_node

        print(f"New EDP: {new_edp} mJs")

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
            key = lines[i].lstrip()[0]
            value = float(lines[i].split(":")[2][1:-1])
            self.tech_params[mapping[key]] = value
            i += 1
        # print("mapping:", mapping)
        # print("tech params:", self.tech_params)

    def create_full_tech_params(self):
        self.full_tech_params["symbolic_latency_wc"] = (
            hw_symbols.symbolic_latency_wc.copy()
        )
        for key in self.full_tech_params["symbolic_latency_wc"]:
            self.full_tech_params["symbolic_latency_wc"][key] = self.full_tech_params[
                "symbolic_latency_wc"
            ][key].subs(self.tech_params)
            # for the rest of the parameters, just give them back their initial values
            self.full_tech_params["symbolic_latency_wc"][key] = self.full_tech_params[
                "symbolic_latency_wc"
            ][key].subs(initial_tech_params)
        self.full_tech_params["symbolic_power_active"] = (
            hw_symbols.symbolic_power_active.copy()
        )
        for key in self.full_tech_params["symbolic_power_active"]:
            self.full_tech_params["symbolic_power_active"][key] = self.full_tech_params[
                "symbolic_power_active"
            ][key].subs(self.tech_params)
            self.full_tech_params["symbolic_power_active"][key] = self.full_tech_params[
                "symbolic_power_active"
            ][key].subs(initial_tech_params)
        self.full_tech_params["symbolic_power_passive"] = (
            hw_symbols.symbolic_power_passive.copy()
        )
        for key in self.full_tech_params["symbolic_power_passive"]:
            self.full_tech_params["symbolic_power_passive"][key] = (
                self.full_tech_params["symbolic_power_passive"][key].subs(
                    self.tech_params
                )
            )
            self.full_tech_params["symbolic_power_passive"][key] = (
                self.full_tech_params["symbolic_power_passive"][key].subs(
                    initial_tech_params
                )
            )
        # print(self.full_tech_params)

    def write_back_rcs(self, rcs_path="rcs_current.yaml"):
        rcs = {"Reff": {}, "Ceff": {}, "other": {}}
        for elem in self.tech_params:
            if elem.name == "f" or elem.name == "V_dd" or elem.name.startswith("Mem"):
                rcs["other"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            else:
                rcs[elem.name[: elem.name.find("_")]][
                    elem.name[elem.name.find("_") + 1 :]
                ] = self.tech_params[elem]
        with open(rcs_path, "w") as f:
            f.write(yaml.dump(rcs))

    def inverse_pass(self):
        print("\nRunning Inverse Pass")
        self.symbolic_sim.simulate(self.cfg, self.cfg_node_to_hw_map, self.hw)
        self.symbolic_sim.calculate_edp(self.hw)
        self.symbolic_sim.save_edp_to_file()

        edp = self.symbolic_sim.edp

        print(f"initial edp: {edp.subs(self.tech_params)} units?")
        os.system("python optimize.py > ipopt_out.txt")

        f = open("ipopt_out.txt", "r")
        self.parse_output(f)
        self.write_back_rcs()
        self.create_full_tech_params()
        print(f"final edp: {edp.subs(self.tech_params)} units?")


def main():
    print("instrumenting application")
    os.system("python instrument.py " + args.benchmark)
    os.system(
        "python instrumented_files/xformed-"
        + args.benchmark.split("/")[-1]
        + " > instrumented_files/output.txt"
    )

    codesign_module = Codesign(args.benchmark)

    # starting point set by the config we load into the HW model
    rcs = codesign_module.hw.get_optimization_params_from_tech_params()
    initial_tech_params = generate_init_params_from_rcs(rcs)

    codesign_module.set_technology_parameters(initial_tech_params)
    
    # for elem in rcs["Reff"]:
    #     initial_tech_params[hw_symbols.symbol_table["Reff_" + elem]] = rcs["Reff"][elem]
    #     initial_tech_params[hw_symbols.symbol_table["Ceff_" + elem]] = rcs["Ceff"][elem]
    with open("rcs_current.yaml", "w") as f:
        f.write(yaml.dump(rcs))
    
    i = 0
    while i < 2:
        codesign_module.forward_pass(args.area)
        codesign_module.symbolic_sim.update_data_path(codesign_module.sim.data_path)

        codesign_module.inverse_pass()
        codesign_module.hw.update_technology_parameters()
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
    
    args = parser.parse_args()
    print(f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}")

    main()
