import argparse
from instrument import instrument_and_run
import os


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
        print("instrumenting")
        instrument_and_run(args.benchmark)
        os.system('python3 instrumented_files/xformed-'+args.benchmark.split('/')[-1])
        return 
    
    def inverse_pass(self):
        return

def main():
    codesign_module = Codesign()
    while True:
        codesign_module.forward_pass()
        codesign_module.inverse_pass()
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

    args = parser.parse_args()
    print(f"args: benchmark: {args.benchmark}, trace:{args.notrace}, area:{args.area}")

    main()