import argparse
import os
import simulate
import symbolic_simulate
import optimize


initial_tech_params = {
    "f": 1e6,
    "C_int_inv": 1e-8,
    "V_dd": 1,
    "C_input_inv": 1e-9
}

class Codesign:
    def __init__(self):
        self.tech_params = initial_tech_params
        self.architecture = None
    
    def forward_pass(self):
        print("starting forward pass")
        self.architecture = simulate.main(args) 
    
    def inverse_pass(self):
        print("starting inverse pass")
        symbolic_simulate.main(args)
        optimize.main()


def main():
    codesign_module = Codesign()
    print("instrumenting application")
    os.system('python3 instrument.py '+args.benchmark)
    os.system('python3 instrumented_files/xformed-' + args.benchmark.split('/')[-1] + ' > instrumented_files/output.txt')
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