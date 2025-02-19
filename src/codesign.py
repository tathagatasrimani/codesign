import argparse
import os
import yaml
import sys
import datetime
import logging
import shutil

import networkx as nx

logger = logging.getLogger("codesign")
os.chdir("..")
sys.path.append(os.getcwd())

class Codesign:
    def __init__(self, benchmark, save_dir, openroad_testfile, parasitics, no_cacti):
        self.save_dir = os.path.join(
            save_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:  # clear the directory
            files = os.listdir(self.save_dir)
            for file in files:
                os.remove(f"{self.save_dir}/{file}")
        with open(f"{self.save_dir}/codesign.log", "a") as f:
            f.write("Codesign Log\n")
            f.write(f"Benchmark: {benchmark}\n")

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
            self.forward_pass()
            self.log_forward_tech_params()
            self.inverse_pass()
            self.hw.update_technology_parameters()
            self.log_all_to_file(i)
            i += 1

        # cleanup
        self.cleanup()

        


def main(args):
    codesign_module = Codesign(
        args.benchmark,
        args.savedir,
        args.openroad_testfile,
        args.parasitics,
        args.debug_no_cacti
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
        default="logs",
        help="Path to the save new architecture file",
    )
    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        default="detailed",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="openroad_interface/tcl/test_nangate45_bigger.tcl",
        help="what tcl file will be executed for openroad",
    )
    parser.add_argument(
        "-N",
        "--num_iters",
        type=int,
        default=10,
        help="Number of Codesign iterations to run",
    )
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, parasitics: {args.parasitics}, num iterations: {args.num_iters}"
    )

    main(args)
