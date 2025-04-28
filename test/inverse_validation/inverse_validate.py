import argparse
import os
import yaml
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("inverse_validation")

from src import codesign
from src import hw_symbols

class InverseValidation:
    def __init__(self, args):
        self.logic_tech_nodes = [7, 5]
        self.cacti_tech_nodes = [45, 32]
        self.tech_nodes = self.logic_tech_nodes if args.test_type == "logic" else self.cacti_tech_nodes
        self.codesign_args = args
        self.codesign_modules = {}

    def run_initial(self):
        for tech_node in self.tech_nodes:
            if self.codesign_args.test_type == "logic":
                self.codesign_args.logic_node = tech_node
            else:
                self.codesign_args.mem_node = tech_node
            codesign_module = codesign.Codesign(
                self.codesign_args
            )
            self.codesign_modules[tech_node] = codesign_module
            logger.info(f"Running forward pass for tech node {tech_node}")
            codesign_module.forward_pass()

    def run_pairwise(self):
        for i in range(len(self.tech_nodes)-1):
            improvement = self.codesign_modules[self.tech_nodes[i]].forward_edp / self.codesign_modules[self.tech_nodes[i+1]].forward_edp
            self.codesign_modules[self.tech_nodes[i]].inverse_pass_improvement = improvement
            logger.info(f"Running inverse pass for tech node {self.tech_nodes[i]} with improvement {improvement}")
            self.codesign_modules[self.tech_nodes[i]].inverse_pass()
            self.codesign_modules[self.tech_nodes[i]].cleanup()

            # plot results
            self.plot_diff(self.tech_nodes[i], self.tech_nodes[i+1])

    def plot_diff(self, tech_node1, tech_node2):
        params_to_plot = self.codesign_modules[tech_node1].symbolic_sim.edp.free_symbols
        logger.info(f"params_to_plot: {params_to_plot}")
        # Convert to sorted list of strings for consistent ordering
        param_names = sorted([str(p) for p in params_to_plot])
        tech_params1 = self.codesign_modules[tech_node1].tech_params
        tech_params2 = self.codesign_modules[tech_node2].tech_params
        logger.info(f"tech_params1: {tech_params1}")
        logger.info(f"tech_params2: {tech_params2}")
        ratios = []
        for name in param_names:
            try:
                v1 = tech_params1[hw_symbols.symbol_table[name]]
                v2 = tech_params2[hw_symbols.symbol_table[name]]
                ratio = v1 / v2 if v2 != 0 else float('inf')
            except KeyError:
                ratio = float('nan')
            logger.info(f"Ratio for {name}: {ratio}")
            ratios.append(ratio)
        # Make figs dir if needed
        figs_dir = os.path.join(os.path.dirname(__file__), 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        # Split into chunks for readability
        chunk_size = 12
        for i in range(0, len(param_names), chunk_size):
            chunk_names = param_names[i:i+chunk_size]
            chunk_ratios = ratios[i:i+chunk_size]
            fig, ax = plt.subplots(figsize=(max(8, len(chunk_names)*0.8), 6))
            ax.bar(chunk_names, chunk_ratios)
            ax.set_xlabel('Parameter')
            ax.set_ylabel(f'Ratio {tech_node1}/{tech_node2}')
            ax.set_title(f'Parameter Ratios: {tech_node1} vs {tech_node2} (params {i+1}-{i+len(chunk_names)})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fname = f'ratios_{tech_node1}_vs_{tech_node2}_chunk{i//chunk_size+1}.png'
            fpath = os.path.join(figs_dir, fname)
            plt.savefig(fpath)
            plt.close(fig)

def main(args):
    inverse_validation = InverseValidation(args)
    inverse_validation.run_initial()
    inverse_validation.run_pairwise()

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
        default="test/inverse_validation/logs",
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
        "--test_type",
        type=str,
        choices=["logic", "cacti"],
        default="logic",
        help="determines what type of technology parameters are used",
    )
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    parser.add_argument("-c", "--checkpoint", type=bool, default=False, help="save a design checkpoint upon exit")
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, parasitics: {args.parasitics}, num iterations: {args.num_iters}, checkpointing: {args.checkpoint}, area: {args.area}, memory included: {not args.no_memory}"
    )

    main(args)