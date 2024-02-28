import argparse
import os
import simulate
import symbolic_simulate
import optimize
import hw_symbols
import yaml


initial_tech_params = {
    hw_symbols.f: 1e6,
    hw_symbols.V_dd: 1,
}

class Codesign:
    def __init__(self):
        self.tech_params = initial_tech_params
        self.full_tech_params = {}
        self.architecture = None
    
    def forward_pass(self):
        print("starting forward pass")
        self.architecture = simulate.main(args) 

    def parse_output(self, f):
        lines = f.readlines()
        mapping = {}
        i = 0
        while lines[i][0] == 'x':
            mapping[lines[i][lines[i].find('[')+1:lines[i].find(']')]] = hw_symbols.symbol_table[lines[i].split(' ')[-1][:-1]]
            i += 1
        while i < len(lines) and lines[i].find('x') != 4:
            i += 1
        i += 2
        for _ in range(len(mapping)):
            key = lines[i].lstrip()[0]
            value = float(lines[i].split(':')[2][1:-1])
            self.tech_params[mapping[key]] = value
            i += 1
        print("mapping:", mapping)
        print("tech params:", self.tech_params)

    def create_full_tech_params(self):
        self.full_tech_params["symbolic_latency_wc"] = hw_symbols.symbolic_latency_wc
        for key in self.full_tech_params["symbolic_latency_wc"]:
            self.full_tech_params["symbolic_latency_wc"][key] = self.full_tech_params["symbolic_latency_wc"][key].subs(self.tech_params)
            # for the rest of the parameters, just give them back their initial values
            self.full_tech_params["symbolic_latency_wc"][key] = self.full_tech_params["symbolic_latency_wc"][key].subs(initial_tech_params)
        self.full_tech_params["symbolic_power_active"] = hw_symbols.symbolic_power_active
        for key in self.full_tech_params["symbolic_power_active"]:
            self.full_tech_params["symbolic_power_active"][key] = self.full_tech_params["symbolic_power_active"][key].subs(self.tech_params)
            self.full_tech_params["symbolic_power_active"][key] = self.full_tech_params["symbolic_power_active"][key].subs(initial_tech_params)
        self.full_tech_params["symbolic_power_passive"] = hw_symbols.symbolic_power_passive
        for key in self.full_tech_params["symbolic_power_passive"]:
            self.full_tech_params["symbolic_power_passive"][key] = self.full_tech_params["symbolic_power_passive"][key].subs(self.tech_params)
            self.full_tech_params["symbolic_power_passive"][key] = self.full_tech_params["symbolic_power_passive"][key].subs(initial_tech_params)
        print(self.full_tech_params)

    def inverse_pass(self):
        print("starting inverse pass")
        symbolic_simulate.main(args)
        os.system('python3 optimize.py > ipopt_out.txt')
        f = open("ipopt_out.txt", 'r')
        self.parse_output(f)
        self.create_full_tech_params()


def main():
    codesign_module = Codesign()
    print("instrumenting application")
    os.system('python3 instrument.py '+args.benchmark)
    os.system('python3 instrumented_files/xformed-' + args.benchmark.split('/')[-1] + ' > instrumented_files/output.txt')
    rcs = yaml.load(open("rcs.yaml", "r"), Loader=yaml.Loader)
    rcs["other"] = {"f": 1e6, "V_dd": 1}
    for elem in rcs["Reff"]:
        initial_tech_params[hw_symbols.symbol_table["Reff_"+elem]] = rcs["Reff"][elem]
        initial_tech_params[hw_symbols.symbol_table["Ceff_"+elem]] = rcs["Ceff"][elem]
    with open("rcs_current.yaml", 'w') as f:
        f.write(yaml.dump(rcs))
    while True:
        codesign_module.forward_pass()
        codesign_module.inverse_pass()
        # TODO: create stopping condition
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Codesign",
        description="Runs a two-step loop to optimize architecture and technology for a given application.",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("benchmark", metavar="B", type=str)
    parser.add_argument("--notrace", action="store_true")
    parser.add_argument("-a", "--area", type=float, help="Max Area of the chip in um^2")
    parser.add_argument("-s", "--archsearch", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-b", "--bw", type=float, help="Compute - Memory Bandwidth in ??GB/s??"
    )

    args = parser.parse_args()
    print(f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}")

    main()