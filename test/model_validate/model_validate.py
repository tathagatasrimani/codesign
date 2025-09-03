import argparse
from src import codesign
import copy
import sympy as sp
import logging
import datetime
import os
import yaml

logger = logging.getLogger(__name__)

class ModelValidate:
    def __init__(self, args):
        self.args = args
        log_save_dir = os.path.join(os.path.dirname(__file__), "logs", f"{self.args.model_cfg}_vs_ref_{self.args.ref_model_cfg}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        logging.basicConfig(level=logging.INFO, filename=os.path.join(log_save_dir, "bsim4_validate.log"))
        logger.info(args.tech_node)
        self.ref_model_vals = {}
        self.model_vals = {}
        with open(os.path.join(os.path.dirname(__file__), "model_params.yaml"), "r") as f:
            self.model_params = yaml.safe_load(f)
        with open(os.path.join(os.path.dirname(__file__), "tech_node_sets.yaml"), "r") as f:
            self.tech_node_sets = yaml.safe_load(f)
        with open(os.path.join(os.path.dirname(__file__), "param_value_sets.yaml"), "r") as f:
            self.param_value_sets = yaml.safe_load(f)
            """for value in self.param_value_sets[self.args.param_value_set]:
                print(type(value))"""
        if self.args.just_one_node:
            self.tech_nodes = [self.args.tech_node]
        else:
            self.tech_nodes = self.tech_node_sets[self.args.tech_node_set]

    def get_model_objects(self, tech_node):
        orig_args = copy.deepcopy(self.args)
        orig_args.tech_node = tech_node
        codesign_model = codesign.Codesign(orig_args)
        new_args = copy.deepcopy(orig_args)
        new_args.model_cfg = orig_args.ref_model_cfg
        codesign_ref_model = codesign.Codesign(new_args)
        return codesign_model, codesign_ref_model

    def validate_model(self):
        for tech_node in self.tech_nodes:
            codesign_model, codesign_ref_model = self.get_model_objects(tech_node)
            self.ref_model_vals[tech_node] = {}
            self.model_vals[tech_node] = {}
            for param in self.model_params[self.args.params]:
                self.ref_model_vals[tech_node][param] = codesign_ref_model.hw.circuit_model.tech_model.param_db[param].xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
                self.model_vals[tech_node][param] = codesign_model.hw.circuit_model.tech_model.param_db[param].xreplace(codesign_model.hw.circuit_model.tech_model.base_params.tech_values).evalf()
                logger.info(f"{param} for {tech_node} ({self.args.model_cfg}): {self.model_vals[tech_node][param]}")
                logger.info(f"{param} for {tech_node} ({self.args.ref_model_cfg}): {self.ref_model_vals[tech_node][param]}")

        for param in self.model_params[self.args.params]:
            logger.info(f"{param} ({self.args.model_cfg}): {self.model_vals[param]}")
            logger.info(f"{param} ({self.args.ref_model_cfg}): {self.ref_model_vals[param]}")
    
    def validate_one_param_range(self, param):
        assert self.args.just_one_node, "Must specify a single node to validate a single parameter"
        logger.info(f"Validating {param} for {self.args.tech_node} ({self.args.model_cfg})")
        self.param_value_set = self.param_value_sets[self.args.param_value_set]
        codesign_model, codesign_ref_model = self.get_model_objects(self.args.tech_node)
        logger.info(f"==========BEGINNING ONE PARAMETER RANGE VALIDATION==========")
        tech_params_modified = {}
        for param_value in self.param_value_set:
            tech_params_modified[param_value] = copy.deepcopy(codesign_model.hw.circuit_model.tech_model.base_params.tech_values)
            tech_params_modified[param_value][codesign_model.hw.circuit_model.tech_model.param_db[param]] = param_value

        for eval_param in self.model_params[self.args.params]:
            self.model_vals[eval_param] = {}
            self.ref_model_vals[eval_param] = {}
            for param_value in self.param_value_set:
                self.model_vals[eval_param][param_value] = codesign_model.hw.circuit_model.tech_model.param_db[eval_param].xreplace(tech_params_modified[param_value])
                if not type(self.model_vals[eval_param][param_value]) == float:
                    self.model_vals[eval_param][param_value] = self.model_vals[eval_param][param_value].evalf()
                self.ref_model_vals[eval_param][param_value] = codesign_ref_model.hw.circuit_model.tech_model.param_db[eval_param].xreplace(tech_params_modified[param_value])
                if not type(self.ref_model_vals[eval_param][param_value]) == float:
                    self.ref_model_vals[eval_param][param_value] = self.ref_model_vals[eval_param][param_value].evalf()
                logger.info(f"{eval_param} for {param}={param_value} ({self.args.model_cfg}): {self.model_vals[eval_param][param_value]}")
                logger.info(f"{eval_param} for {param}={param_value} ({self.args.ref_model_cfg}): {self.ref_model_vals[eval_param][param_value]}\n")
            logger.info("================================================")
        

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
        default="matmult",
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
        default="estimation",
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        default="openroad_interface/tcl/codesign_top.tcl",
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
        default=True,
        help="disable memory modeling",
    )
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    parser.add_argument("-c", "--checkpoint", type=bool, default=False, help="save a design checkpoint upon exit")
    parser.add_argument("--logic_node", type=int, default=7, help="logic node size")
    parser.add_argument("--mem_node", type=int, default=32, help="memory node size")
    parser.add_argument("--inverse_pass_improvement", type=float, help="improvement factor for inverse pass")
    parser.add_argument("--tech_node", "-T", type=str, help="technology node to use as starting point")
    parser.add_argument("--obj", type=str, default="edp", help="objective function")
    parser.add_argument("--model_cfg", type=str, default="bsim4_limited", help="symbolic model configuration")
    parser.add_argument("--ref_model_cfg", type=str, default="bsim4_limited", help="reference model configuration")
    parser.add_argument("--params", type=str, default="bsim4_limited", help="parameters to validate")
    parser.add_argument("--tech_node_set", type=str, default="every_other", help="technology node set to use")
    parser.add_argument("--just_one_node", type=bool, default=False, help="just run for one node specified by --tech_node")
    parser.add_argument("--param_value_set", type=str, default="tox", help="parameter value set to use")

    args = parser.parse_args()
    model_validate = ModelValidate(args)
    if args.just_one_node:
        model_validate.validate_one_param_range(args.param_value_set)
    else:
        model_validate.validate_model()