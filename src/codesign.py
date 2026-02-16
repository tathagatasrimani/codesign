import argparse
import os
import yaml
import sys
import datetime
import json
import math
import logging
import shutil
import subprocess
import copy
import shlex
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("codesign")

from src import sim_util
from src.hardware_model import hardwareModel
from src.inverse_pass import optimize
from src.forward_pass import schedule_vitis
from src.forward_pass.scale_hls_port_fix import scale_hls_port_fix
from src.generate_blackbox_files import generate_blackbox_files
from src.forward_pass.vitis_create_netlist import create_vitis_netlist, extract_module_dependencies, create_physical_design_netlist
from src.forward_pass.vitis_parse_verbose_rpt import parse_verbose_rpt
from src.forward_pass.vitis_memory_mapping import build_memory_mapping, write_mapping, load_mapping
from src.forward_pass.vitis_merge_netlists import MergeNetlistsVitis
from src import trend_plot
import time
from test import checkpoint_controller

FLOW_SUCCESS_MSG = "FLOW END: Design flow completed successfully for iterations = "

class Codesign:
    def __init__(self, args):
        """
        Initializes a Codesign object, sets up directories and logging, initializes hardware and
        simulation models, and prepares technology parameters.

        Args:
            benchmark_name (str): Name of the benchmark to use.
            save_dir (str): Directory to save logs and results.
            openroad_testfile (str): Path to the OpenROAD test file.
            parasitics (Any): Parasitics configuration.
            no_cacti (bool): If True, disables CACTI memory modeling for the first iteration.
            area (float): Area constraint in um2.
            no_memory (bool): If True, disables memory modeling in general.
        Returns:
            None
        """
        self.set_config(args)

        ## save the root directory that the program started in
        self.codesign_root_dir = os.getcwd()

        ## parse the university name from the setup scripts folder
        with open(f"{self.codesign_root_dir}/setup_scripts/university_name.txt", "r") as f:
            self.university_name = f.read().strip()

        self.hls_tool = self.cfg["args"]["hls_tool"]
        self.benchmark = f"src/benchmarks/{self.hls_tool}/{self.cfg['args']['benchmark']}"
        self.benchmark_name = self.cfg["args"]["benchmark"]
        self.obj_fn = self.cfg["args"]["obj"]
        self.tmp_dir = self.get_tmp_dir()
        print(f"tmp_dir: {self.tmp_dir}")
        self.benchmark_dir = f"{self.tmp_dir}/benchmark"
        self.benchmark_setup_dir = f"{self.tmp_dir}/benchmark_setup"
        tmp_dir_suffix = self.tmp_dir.split("/")[-1]
        self.save_dir = os.path.join(
            self.cfg["args"]["savedir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + tmp_dir_suffix
        )
        self.hls_tcl_script = "tcl_script.tcl" if self.cfg["args"]["arch_opt_pipeline"] == "scalehls" else "hls.tcl"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:  # clear the directory
            files = os.listdir(self.save_dir)
            for file in files:
                os.remove(f"{self.save_dir}/{file}")
        with open(f"{self.save_dir}/codesign.log", "a") as f:
            f.write("Codesign Log\n")
            f.write(f"Benchmark: {self.benchmark}\n")
            f.write(f"Tmp dir: {self.tmp_dir}\n")

        logging.basicConfig(filename=f"{self.save_dir}/codesign.log", level=logging.INFO)
        logger.info(f"args: {self.cfg['args']}")

        self.block_vectors_save_dir = f"{self.save_dir}/block_vectors"
        os.makedirs(self.block_vectors_save_dir, exist_ok=True)

        self.checkpoint_controller = checkpoint_controller.CheckpointController(self.cfg, self.codesign_root_dir, self.tmp_dir)

        if self.cfg["args"]["checkpoint_start_step"]!= "none" and self.cfg["args"]["checkpoint_load_dir"] != "none":
            self.checkpoint_controller.load_checkpoint()

        self.forward_obj = 0
        self.inverse_obj = 0
        self.openroad_testfile = self.cfg['args']['openroad_testfile']
        self.parasitics = self.cfg["args"]["parasitics"]
        self.run_cacti = not self.cfg["args"]["debug_no_cacti"]
        self.no_memory = self.cfg["args"]["no_memory"]
        self.hw = hardwareModel.HardwareModel(self.cfg, self.codesign_root_dir, self.tmp_dir)
        self.opt = optimize.Optimizer(self.hw, self.tmp_dir, self.save_dir, max_power=self.cfg["args"]["max_power"], max_power_density=self.cfg["args"]["max_power_density"], opt_pipeline=self.cfg["args"]["opt_pipeline"])
        self.module_map = {}
        self.inverse_pass_improvement = self.cfg["args"]["inverse_pass_improvement"]
        self.inverse_pass_lag_factor = 1

        self.params_over_iterations = [copy.copy(self.hw.circuit_model.tech_model.base_params.tech_values)]
        self.sensitivities_over_iterations = []
        self.constraint_slack_over_iterations = []
        self.obj_over_iterations = []
        self.lag_factor_over_iterations = [1.0]
        self.max_unroll = 64

        self.save_dat()

        #with open(f"{self.tmp_dir}/tech_params_0.yaml", "w") as f:
        #    f.write(yaml.dump(self.hw.circuit_model.tech_model.base_params.tech_values))

        self.hw.write_technology_parameters(self.save_dir+"/initial_tech_params.yaml")

        self.iteration_count = 0

        # configure starting resource usage based on the minimum DSP and BRAM usage
        self.dsp_multiplier = 1/self.cfg["args"]["min_dsp"] * self.cfg["args"]["area"] / sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values)
        self.bram_multiplier = 1/self.cfg["args"]["min_dsp"] * self.cfg["args"]["area"] / sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values)

        self.wire_lengths_over_iterations = []
        self.wire_delays_over_iterations = []
        self.device_delays_over_iterations = []
        self.cur_dsp_usage = None # to be set later
        self.last_dsp_count_set = False
        if os.path.exists(f"{self.tmp_dir}/last_dsp_count.yaml"):
            with open(f"{self.tmp_dir}/last_dsp_count.yaml", "r") as f:
                self.cur_dsp_usage = yaml.load(f, Loader=yaml.FullLoader)["last_dsp_count"]
                self.last_dsp_count_set = True
                logger.info(f"loaded last_dsp_count from {self.tmp_dir}/last_dsp_count.yaml: {self.cur_dsp_usage}")
        self.max_rsc_reached = False

        self.config_json_path_scalehls = "ScaleHLS-HIDA/test/Transforms/Directive/config.json"
        self.config_json_path_streamhls = "Stream-HLS/test/Transforms/Directive/config.json"
        self.config_json_path = self.benchmark_setup_dir + "/config.json"

    # any arguments specified on CLI will override the default config
    def set_config(self, args):
        with open(f"src/yaml/codesign_cfg.yaml", "r") as f:
            cfgs = yaml.load(f, Loader=yaml.FullLoader)

        # open each additonal config in test/additional_configs. Add them to the cfgs
        for additional_config_file in os.listdir("test/additional_configs"):
            if additional_config_file.endswith(".yaml"):
                with open(f"test/additional_configs/{additional_config_file}", "r") as f:
                    additional_cfg = yaml.load(f, Loader=yaml.FullLoader)

                    # check that there aren't any duplicate keys
                    for key in additional_cfg:
                        if key in cfgs:
                            raise Exception(f"Duplicate key {key} found in additional config {additional_config_file}")

                    cfgs = {**cfgs, **additional_cfg}

        if args.additional_cfg_file is not None:
            with open(args.additional_cfg_file, "r") as f:
                additional_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # print(f"Loaded additional config from {args.additional_cfg_file}: {additional_cfg}")

                # check that there aren't any duplicate keys
                for key in additional_cfg:
                    if key in cfgs:
                        raise Exception(f"Duplicate key {key} found in additional config {args.additional_cfg_file}")

                cfgs = {**cfgs, **additional_cfg}
        
        # print(f"Final cfgs before applying CLI args: {cfgs}")
        
        overwrite_args_all = vars(args)
        overwrite_args = {}
        for key, value in overwrite_args_all.items():
            if value is not None and value != 'none':
                overwrite_args[key] = value
        overwrite_cfg = {"base_cfg": args.config, "args": overwrite_args}
        cfgs["overwrite_cfg"] = overwrite_cfg
        self.cfg = sim_util.recursive_cfg_merge(cfgs, "overwrite_cfg")
        assert self.cfg["args"]["arch_opt_pipeline"] in ["scalehls", "streamhls"]
        print(f"args: {self.cfg['args']}")

    def get_tmp_dir(self):
        idx = 0

        ## make base tmp directory
        if not os.path.exists("src/tmp"):
            os.makedirs("src/tmp", exist_ok=True)

        tmp_base_dir = "src/tmp"
        # use .get() so missing key won't raise; fall back to default
        tmp_dir_arg = None
        if isinstance(self.cfg.get("args"), dict):
            tmp_dir_arg = self.cfg["args"].get("tmp_dir")
        if tmp_dir_arg:
            tmp_base_dir = tmp_dir_arg

        while True:
            tmp_dir = f"{tmp_base_dir}/tmp_{self.benchmark_name}_{self.obj_fn}_{idx}"
            tmp_dir_full = os.path.join(self.codesign_root_dir, tmp_dir)
            if not os.path.exists(tmp_dir_full):
                os.makedirs(tmp_dir_full)
                return tmp_dir
            idx += 1

    def log_forward_tech_params(self):
        latency = self.hw.circuit_model.circuit_values["latency"]
        logger.info(f"latency (ns):\n {latency}")

        active_energy_logic = self.hw.circuit_model.circuit_values["dynamic_energy"]
        logger.info(f"active energy (nW):\n {active_energy_logic}")

        passive_power = self.hw.circuit_model.circuit_values["passive_power"]
        logger.info(f"passive power (nW):\n {passive_power}")

        #logger.info(f"compute operation totals in fw pass:\n {self.hw.compute_operation_totals}")

    def set_resource_constraint_scalehls(self, unlimited=False):
        """
        Sets the resource constraint and op latencies for ScaleHLS.
        """
        with open(self.config_json_path, "r") as f:
            config = json.load(f)
        if unlimited:
            config["dsp"] = 1024
            config["bram"] = 1024
        else:
            config["dsp"] = int(self.cfg["args"]["area"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
            config["bram"] = int(self.cfg["args"]["area"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.bram_multiplier))
            # setting cur_dsp_usage here instead of with parse_dsp_usage after running scaleHLS
            # because I observed that for a small amount of resources, scaleHLS won't generate the csv file that we need to parse
            self.cur_dsp_usage = config["dsp"] 

        ## Manual override for max_dsp from command line. This is primarily used for the YARCH experiments where we only want to run the forward pass with a specific DSP constraint.
        if "max_dsp" in self.cfg["args"]:
            self.max_dsp = self.cfg["args"]["max_dsp"]
            config["dsp"] = self.max_dsp
            config["bram"] = self.max_dsp # set bram to same as dsp for yarch experiments
            print(f"Using user specified max_dsp: {self.max_dsp}")
            print(f"The current dsp used in the iteration is: {self.cur_dsp_usage}")

        # I don't think "100MHz" has any meaning because scaleHLS should be agnostic to frequency
        config["100MHz"]["fadd"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Add16"] / self.clk_period)
        config["100MHz"]["fmul"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Mult16"] / self.clk_period)
        config["100MHz"]["fdiv"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["FloorDiv16"] / self.clk_period)
        config["100MHz"]["fcmp"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["GtE16"] / self.clk_period)

        config["max_iter_num"] = self.cfg["args"]["max_iter_num_scalehls"]
        with open(self.config_json_path, "w") as f:
            json.dump(config, f)

    def set_resource_constraint_streamhls(self):
        """
        Sets the resource constraint and op latencies for StreamHLS.
        """
        with open(self.config_json_path, "r") as f:
            config = json.load(f)

        config["dsp"] = self.cur_dsp_usage
        config["bram"] = self.cur_dsp_usage

        # Set operation latencies based on hardware model
        config["latency"]["fadd"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Add16"] / self.clk_period)
        config["latency"]["fsub"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Sub16"] / self.clk_period)
        config["latency"]["fmul"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Mult16"] / self.clk_period)
        config["latency"]["fdiv"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["FloorDiv16"] / self.clk_period)
        config["latency"]["fcmp"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["GtE16"] / self.clk_period)
        config["latency"]["fexp"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Exp16"] / self.clk_period)

        # Set DSP usage based on hardware model
        config["dsp_usage"]["fadd"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["Add16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
        config["dsp_usage"]["fsub"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["Sub16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
        config["dsp_usage"]["fmul"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["Mult16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
        config["dsp_usage"]["fdiv"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["FloorDiv16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
        config["dsp_usage"]["fcmp"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["GtE16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
        config["dsp_usage"]["fexp"] = math.ceil(self.hw.circuit_model.circuit_values["area"]["Exp16"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))

        with open(self.config_json_path, "w") as f:
            json.dump(config, f)

    def run_streamhls(self, save_dir, setup=False, iteration_count=0):
        """
        Runs StreamHLS synthesis tool in a different environment with modified PATH and PYTHONPATH.
        Updates the memory configuration and logs the output.
        Handles directory changes and cleans up temporary files.
        """
        start_time = time.time()
        logger.info(f"Running StreamHLS with save_dir: {save_dir}")
        ## get CWD
        cwd = os.getcwd()
        print(f"Running StreamHLS in {cwd}")

        if not setup:
            if self.last_dsp_count_set and iteration_count == 0:
                self.cur_dsp_usage = self.cur_dsp_usage
                self.last_dsp_count_set = False
            elif self.cfg["args"]["fixed_area_increase_pattern"] and iteration_count > 0:
                self.cur_dsp_usage = self.cur_dsp_usage * 4
            else: 
                self.cur_dsp_usage = int(self.cfg["args"]["area"] / (sim_util.xreplace_safe(self.hw.circuit_model.tech_model.param_db["A_gate"], self.hw.circuit_model.tech_model.base_params.tech_values) * self.dsp_multiplier))
                self.cur_dsp_usage = max(self.cur_dsp_usage, self.cfg["args"]["min_dsp"])
            tilelimit = 1
        else:
            self.cur_dsp_usage = 10000
            tilelimit = 1

        self.set_resource_constraint_streamhls()

        save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)
        config_path = os.path.join(cwd, self.config_json_path)
        streamhls_opt_level = int(self.cfg["args"].get("streamhls_opt_level", 5))
        cmd = [
            'bash', '-c',
            f'''
            cd {cwd}
            source miniconda3/etc/profile.d/conda.sh
            cd Stream-HLS
            pwd
            source setup-env.sh
            cd examples
            python run_streamhls.py -b {save_path} -d {save_path} -k {self.benchmark_name} -O {streamhls_opt_level} --dsps {self.cur_dsp_usage} --timelimit {2} --tilelimit {tilelimit} --tech-config {config_path} --bufferize 1
            '''
        ]

        log_path = f"{save_path}/streamhls_out.log"
        with open(log_path, "w") as outfile:
            p = subprocess.Popen(
                cmd,
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env={}  # clean environment
            )
            p.wait()
        with open(log_path, "r") as f:
            logger.info(f"StreamHLS output:\n{f.read()}")

        if p.returncode != 0:
            tail = ""
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                tail = "".join(lines[-50:])
            except Exception:
                tail = "(unable to read streamhls_out.log)"
            raise Exception(
                "StreamHLS failed. "
                f"Return code: {p.returncode}. "
                f"Log: {log_path}\n"
                f"Last lines:\n{tail}"
            )
        logger.info(f"time to run StreamHLS: {time.time()-start_time}")
        shutil.copy(f"{save_path}/{self.benchmark_name}/hls/src/{self.benchmark_name}.cpp", f"{save_path}/{self.benchmark_name}.cpp")
        shutil.copy(f"{save_path}/{self.benchmark_name}/hls/hls.tcl", f"{save_path}/hls.tcl")

    def run_scalehls(self, save_dir, opt_cmd,setup=False):
        """
        Runs ScaleHLS synthesis tool in a different environment with modified PATH and PYTHONPATH.
        Updates the memory configuration and logs the output.
        Handles directory changes and cleans up temporary files.

        Args:
            save_dir: Directory to save the ScaleHLS output files.
        Returns:
            None
        """
        start_time = time.time()
        self.set_resource_constraint_scalehls(unlimited=setup)

        logger.info(f"Running scaleHLS with save_dir: {save_dir}, opt_cmd: {opt_cmd}")
            
        ## get CWD
        cwd = os.getcwd()
        print(f"Running scaleHLS in {cwd}")

        if self.cfg["args"]["pytorch"]:
            logger.info("Running scaleHLS with pytorch for benchmark_name: "+self.benchmark_name)
            cmd = [
                'bash', '-c',
                f'''
                cd {cwd}
                cd ScaleHLS-HIDA/
                export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
                export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core
                source mlir_venv/bin/activate
                cd {os.path.join(os.path.dirname(__file__), "..", save_dir)}
                python3 {self.benchmark_name}.py > {self.benchmark_name}.mlir 
                {opt_cmd} | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > {os.path.join(os.path.dirname(__file__), "..", save_dir)}/{self.benchmark_name}.cpp
                deactivate
                pwd
                '''
            ]
        else:
            logger.info("Running scaleHLS without pytorch for benchmark_name: "+self.benchmark_name)
            cmd = [
                'bash', '-c',
                f'''
                cd {cwd}
                cd ScaleHLS-HIDA/
                export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
                export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core
                source mlir_venv/bin/activate
                cd {os.path.join(os.path.dirname(__file__), "..", save_dir)}
                cgeist {self.benchmark_name}.c \
                                                -function='*'\
                                                -S \
                                                -memref-fullrank \
                                                -raise-scf-to-affine \
                                                -std=c11 \
                                                -I{os.path.join(cwd, "ScaleHLS-HIDA/polygeist/tools/cgeist/Test/polybench/utilities")} \
                                                -I/usr/include \
                                                -I/usr/lib/gcc/x86_64-linux-gnu/13/include \
                                                -I/usr/local/include \
                                                -resource-dir $(clang -print-resource-dir) \
                                                > {self.benchmark_name}.mlir
                {opt_cmd} | scalehls-translate -scalehls-emit-hlscpp > {os.path.join(os.path.dirname(__file__), "..", save_dir)}/{self.benchmark_name}.cpp
                deactivate
                pwd
                '''
            ]

        with open(f"{self.tmp_dir}/scalehls_out.log", "w") as outfile:
            p = subprocess.Popen(
                cmd,
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env={}  # clean environment
            )
            p.wait()
        with open(f"{self.tmp_dir}/scalehls_out.log", "r") as f:
            logger.info(f"scaleHLS output:\n{f.read()}")

        if p.returncode != 0:
            raise Exception(f"scaleHLS failed with error: {p.stderr}")
        logger.info(f"time to run scaleHLS: {time.time()-start_time}")

    def parse_design_space_for_mlir(self, read_dir):
        df = pd.read_csv(f"{read_dir}/{self.vitis_top_function}_space.csv")
        mlir_idx_that_exist = []
        best_idx = None
        best_dsp = -1
        for i in range(len(df['dsp'].values)):
            file_exists = os.path.exists(f"{read_dir}/{self.vitis_top_function}_pareto_{i}.mlir")
            if file_exists:
                mlir_idx_that_exist.append(i)
                # Select the design point that uses the MOST DSPs (up to constraint) to maximize loop unrolling
                if df['dsp'].values[i] <= self.max_dsp and df['dsp'].values[i] > best_dsp:
                    best_dsp = df['dsp'].values[i]
                    best_idx = i
        if best_idx is not None:
            logger.info(f"Selected design point {best_idx} with {best_dsp} DSPs (constraint: {self.max_dsp})")
            return f"{read_dir}/{self.vitis_top_function}_pareto_{best_idx}.mlir", best_idx
        if len(mlir_idx_that_exist) == 0:
            raise Exception(f"No Pareto solutions found for {self.benchmark_name} with dsp usage {self.max_dsp}")
        return f"{read_dir}/{self.vitis_top_function}_pareto_{mlir_idx_that_exist[-1]}.mlir", mlir_idx_that_exist[-1]

    def parse_dsp_usage_and_latency(self, mlir_idx, save_dir):
        if self.cfg["args"]["arch_opt_pipeline"] == "scalehls":
            df = pd.read_csv(f'''{os.path.join(os.path.dirname(__file__), "..", self.benchmark_setup_dir)}/function_hier_output/{self.vitis_top_function}_space.csv''')
            logger.info(f"dsp usage: {df['dsp'].values[mlir_idx]}, latency: {df['cycle'].values[mlir_idx]}")
            return df['dsp'].values[mlir_idx], df['cycle'].values[mlir_idx]
        elif self.cfg["args"]["arch_opt_pipeline"] == "streamhls":
            log_file = sim_util.get_latest_log_dir_streamhls(save_dir)
            if log_file is None:
                log_file = sim_util.get_latest_log_dir_streamhls(os.path.join(save_dir, self.benchmark_name))
            assert log_file is not None, f"No log file found for {self.benchmark_name} in {save_dir}"
            latency, dsp = None, None
            last_nonzero_latency, last_nonzero_dsp = None, None
            combined_latency = None
            parallel_latency = None
            perm_latency = None
            crash_line = None
            with open(log_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Combined Latency:" in line:
                        combined_latency = float(line.split("Combined Latency:")[1].strip())
                        if combined_latency > 0:
                            last_nonzero_latency = combined_latency
                    if "Parallel Latency:" in line:
                        parallel_latency = float(line.split("Parallel Latency:")[1].strip())
                        if parallel_latency > 0:
                            last_nonzero_latency = parallel_latency
                    if "Permutation solver: latency:" in line:
                        perm_latency = float(line.split("Permutation solver: latency:")[1].strip())
                        if perm_latency > 0:
                            last_nonzero_latency = perm_latency
                    if "Total DSPs:" in line:
                        dsp = int(line.split("Total DSPs:")[1].strip())
                        if dsp > 0:
                            last_nonzero_dsp = dsp
                        if self.cur_dsp_usage is None:
                            self.cur_dsp_usage = dsp
                    if crash_line is None and ("Assertion" in line or "Aborted" in line or "core dumped" in line):
                        crash_line = line.strip()
            if combined_latency is not None:
                latency = combined_latency
            elif parallel_latency is not None:
                latency = parallel_latency
            elif perm_latency is not None:
                latency = perm_latency
            assert latency is not None and dsp is not None, f"No latency or dsp found for {self.benchmark_name} in {log_file}"
            if (latency == 0 or dsp == 0) and (last_nonzero_latency is not None or last_nonzero_dsp is not None):
                latency = last_nonzero_latency if last_nonzero_latency is not None else latency
                dsp = last_nonzero_dsp if last_nonzero_dsp is not None else dsp
            if dsp <= 0:
                raise ValueError(
                    f"StreamHLS reported {dsp} DSPs for {self.benchmark_name}. "
                    f"{'StreamHLS crashed: ' + crash_line + '. ' if crash_line else ''}"
                    f"Check the StreamHLS log for details: {log_file}"
                )
            return dsp, latency

    def parse_vitis_data(self, save_dir):

        ## print the cwd
        print(f"Current working directory in vitis parse data: {os.getcwd()}")

        if self.no_memory:
            allowed_functions_netlist = set(self.hw.circuit_model.circuit_values["area"].keys()).difference({"N/A", "Buf", "MainMem", "Call", "read", "write", "load", "store", "fifo", "memory"})
            allowed_functions_schedule = allowed_functions_netlist.union({"Call", "II"})
            allowed_functions_pd = allowed_functions_netlist
        else:
            allowed_functions_netlist = set(self.hw.circuit_model.circuit_values["area"].keys()).difference({"N/A", "Call"})
            allowed_functions_schedule = allowed_functions_netlist.union({"Call", "II"})
            allowed_functions_pd = allowed_functions_netlist.difference({"read", "write", "load", "store"})

        parse_results_dir = f"{save_dir}/parse_results"

        logger.info(f"parse results dir: {parse_results_dir}")

        if self.checkpoint_controller.check_checkpoint("netlist", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            ## Do preprocessing to the vitis data for the next scripts
            parse_verbose_rpt(f"{save_dir}/{self.benchmark_name}/solution1/.autopilot/db", parse_results_dir)

            ## Build cross-hierarchy memory mapping from generated Verilog
            verilog_dir = f"{save_dir}/{self.benchmark_name}/solution1/impl/verilog"
            if os.path.isdir(verilog_dir):
                logger.info("Building cross-hierarchy memory mapping")
                mem_mapping = build_memory_mapping(verilog_dir, self.vitis_top_function, parse_results_dir)
                self.mem_mapping = mem_mapping
                write_mapping(mem_mapping, parse_results_dir)

            ## Create the netlist
            logger.info("Creating Vitis netlist")
            create_vitis_netlist(parse_results_dir)

            ## Extract module dependencies from FSM data (needed for netlist merging)
            logger.info("Extracting module dependencies")
            extract_module_dependencies(parse_results_dir)

            ## Merge the netlists recursivley through the module hierarchy to produce overall netlist
            logger.info("Recursivley merging vitis netlists")
            vitis_netlist_merger = MergeNetlistsVitis(self.cfg, self.codesign_root_dir, allowed_functions_netlist)
            vitis_netlist_merger.merge_netlists_vitis(parse_results_dir, self.vitis_top_function)

            ## Create physical design netlist with unified memory/FIFO nodes
            logger.info("Creating physical design netlist")
            create_physical_design_netlist(parse_results_dir, self.vitis_top_function, allowed_functions_pd)

            logger.info("Vitis netlist parsing complete")
            logger.info(f"time to parse vitis netlist: {time.time()-start_time}")
        else:
            logger.info("Skipping Vitis netlist parsing")
            self.mem_mapping = load_mapping(parse_results_dir)

        self.checkpoint_controller.check_end_checkpoint("netlist", self.iteration_count)

        if self.checkpoint_controller.check_checkpoint("schedule", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            logger.info("Parsing Vitis schedule")
            schedule_parser = schedule_vitis.vitis_schedule_parser(save_dir, self.benchmark_name, self.vitis_top_function, self.clk_period, allowed_functions_schedule, self.mem_mapping)
            
            logger.info("Creating DFGs from Vitis CDFGs")
            schedule_parser.create_dfgs()
            for basic_block_name in schedule_parser.basic_blocks:
                if schedule_parser.basic_blocks[basic_block_name].is_dataflow_pipeline:
                    self.hw.scheduled_dfgs[basic_block_name] = schedule_parser.basic_blocks[basic_block_name].dfg.G_flattened_standard_with_wire_ops
                    self.hw.dataflow_blocks.add(basic_block_name)
                else:
                    self.hw.scheduled_dfgs[basic_block_name] = schedule_parser.basic_blocks[basic_block_name].dfg.G_standard_with_wire_ops
            for basic_block_name in schedule_parser.basic_blocks:
                for loop_name in schedule_parser.basic_blocks[basic_block_name].dfg.loop_dfgs:
                    for iter_num in schedule_parser.basic_blocks[basic_block_name].dfg.loop_dfgs[loop_name]:
                        self.hw.loop_1x_graphs[loop_name] = {True: schedule_parser.basic_blocks[basic_block_name].dfg.loop_dfgs[loop_name][iter_num][True].G_standard_with_wire_ops, False: schedule_parser.basic_blocks[basic_block_name].dfg.loop_dfgs[loop_name][iter_num][False].G_standard_with_wire_ops}

            #self.hw.loop_2x_graphs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_loop_2x_standard"] for basic_block_name in schedule_parser.basic_blocks if "G_loop_2x" in schedule_parser.basic_blocks[basic_block_name]}

            logger.info("Vitis schedule parsing complete")
            logger.info(f"time to parse vitis schedule: {time.time()-start_time}")
        else:
            for file in os.listdir(parse_results_dir):
                if os.path.isdir(os.path.join(parse_results_dir, file)):
                    if file in sim_util.get_module_map().keys(): # dont parse blackboxed functions
                        continue
                    if os.path.exists(f"{parse_results_dir}/{file}/{file}_flattened_graph_standard_with_wire_ops.gml"):
                        self.hw.scheduled_dfgs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_flattened_graph_standard_with_wire_ops.gml")
                        self.hw.dataflow_blocks.add(file)
                    else:
                        self.hw.scheduled_dfgs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_standard_with_wire_ops.gml")
                    for subfile in os.listdir(f"{parse_results_dir}/{file}"):
                        logger.info(f"subfile: {subfile}")
                        if subfile.endswith("rsc_delay_only_graph_standard_with_wire_ops_1.gml"):
                            loop_name = subfile.replace("_rsc_delay_only_graph_standard_with_wire_ops_1.gml", "")
                            logger.info(f"matched resource constrained delay only graph for loop {loop_name}")
                            self.hw.loop_1x_graphs[loop_name] = {
                                True: nx.read_gml(f"{parse_results_dir}/{file}/{subfile}"),
                                False: nx.read_gml(f"{parse_results_dir}/{file}/{subfile.replace('rsc_delay_only_', '')}")
                            }
            logger.info("Skipping Vitis schedule parsing")
        
        logger.info(f"scheduled dfgs: {self.hw.scheduled_dfgs}")
        logger.info(f"loop 1x graphs: {self.hw.loop_1x_graphs}")

        self.checkpoint_controller.check_end_checkpoint("schedule", self.iteration_count)

        print(f"Current working directory at end of vitis parse data: {os.getcwd()}")

        self.hw.netlist = nx.read_gml(f"{parse_results_dir}/{self.vitis_top_function}_physical_netlist_filtered.gml")
        self.hw.set_memory_models(self.mem_mapping)

        ## print the cwd
        print(f"Current working directory in vitis parse data: {os.getcwd()}")

        """## write the netlist to a file
        with open(f"{parse_results_dir}/netlist-from-vitis.gml", "wb") as f:
            nx.write_gml(self.hw.netlist, f)

        ## write the scheduled dfg to a file
        with open(f"{parse_results_dir}/cdfg-from-vitis.gml", "wb") as f:
            nx.write_gml(self.hw.scheduled_dfg, f)"""

    def vitis_forward_pass(self, save_dir, iteration_count, setup=False):
        """
        Runs Vitis version of forward pass, updates the memory configuration, and logs the output.
        """
        self.vitis_top_function = self.benchmark_name if not self.cfg["args"]["pytorch"] and self.cfg["args"]["arch_opt_pipeline"] != "streamhls" else "forward"
        self.scalehls_pipeline = "hida-pytorch-dse-pipeline" if self.cfg["args"]["pytorch"] else "scalehls-dse-pipeline"

        # prep before running scalehls
        if self.cfg["args"]["arch_opt_pipeline"] == "scalehls":
            self.set_resource_constraint_scalehls(unlimited=setup)
            if setup:
                opt_cmd = f'''scalehls-opt {self.benchmark_name}.mlir -{self.scalehls_pipeline}=\"top-func={self.vitis_top_function} target-spec={os.path.join(os.path.dirname(__file__), "..", self.config_json_path)}\"'''
                mlir_idx = 0
            else:
                mlir_file, mlir_idx = self.parse_design_space_for_mlir(os.path.join(os.path.dirname(__file__), "..", f"{self.tmp_dir}/benchmark_setup/function_hier_output"))
                opt_cmd = f"cat {mlir_file}"
            if (self.checkpoint_controller.check_checkpoint("arch_opt", self.iteration_count) and not setup) or (self.checkpoint_controller.check_checkpoint("setup", self.iteration_count) and setup) and not self.max_rsc_reached:
                self.run_scalehls(save_dir, opt_cmd, setup)
            else:
                logger.info("Skipping ScaleHLS")
            # set scale factors if in setup or first iteration
        elif self.cfg["args"]["arch_opt_pipeline"] == "streamhls":
            if (self.checkpoint_controller.check_checkpoint("arch_opt", self.iteration_count) and not setup) or (self.checkpoint_controller.check_checkpoint("setup", self.iteration_count) and setup):
                self.run_streamhls(save_dir, setup, iteration_count)
            mlir_idx = 0
        dsp_usage, latency = self.parse_dsp_usage_and_latency(mlir_idx, save_dir)

        sim_util.change_clk_period_in_script(f"{save_dir}/{self.hls_tcl_script}", self.clk_period, self.cfg["args"]["hls_tool"])

        if setup:
            self.max_dsp = dsp_usage + 2 # allow some margin above initial dsp usage
            self.max_latency = latency
        elif iteration_count == 0:
            self.max_speedup_factor = float(latency / self.max_latency)
            if dsp_usage <= 0:
                raise ValueError(
                    f"Invalid DSP usage ({dsp_usage}) for {self.benchmark_name}. "
                    "StreamHLS parsing likely failed or reported zero DSPs."
                )
            self.max_area_increase_factor = float(self.max_dsp / dsp_usage)
        if not setup:
            self.checkpoint_controller.check_end_checkpoint("arch_opt", self.iteration_count)
        if setup: # setup step ends here, don't need to run rest of forward pass
            return

        self.cur_dsp_usage = dsp_usage

        self.hw.circuit_model.tech_model.init_scale_factors(self.max_speedup_factor, self.max_area_increase_factor)

        os.chdir(os.path.join(os.path.dirname(__file__), "..", save_dir))
        if self.cfg["args"]["arch_opt_pipeline"] == "scalehls":
            scale_hls_port_fix(f"{self.benchmark_name}.cpp", self.benchmark_name, self.cfg["args"]["pytorch"])
        generate_blackbox_files(f"{self.benchmark_name}.cpp", os.path.join(os.getcwd(), "blackbox_files"), os.path.join(os.getcwd(), self.hls_tcl_script), self.hw.circuit_model.circuit_values["latency"], self.clk_period)
        logger.info(f"Vitis top function: {self.vitis_top_function}")


        import time
        # Build the path to the university-specific setup script
        arch_opt_pipeline = self.cfg["args"]["arch_opt_pipeline"]
        arch_opt_pipeline_capitalized = arch_opt_pipeline.capitalize()  # "scalehls" -> "Scalehls", "streamhls" -> "Streamhls"
        # Handle capitalization: "scalehls" -> "ScaleHLS", "streamhls" -> "StreamHLS"
        if arch_opt_pipeline == "scalehls":
            arch_opt_pipeline_capitalized = "ScaleHLS"
        elif arch_opt_pipeline == "streamhls":
            arch_opt_pipeline_capitalized = "StreamHLS"
        
        setup_script_path = os.path.join(
            self.codesign_root_dir,
            "setup_scripts",
            f"{self.university_name}_environment",
            f"{self.university_name}_vitis_{arch_opt_pipeline_capitalized}_setup.sh"
        )
        
        # Build the base command
        base_command = ["vitis_hls", "-f", "tcl_script_new.tcl"] if arch_opt_pipeline == "scalehls" else ["vitis_hls", "hls_new.tcl", self.benchmark_name, "syn", "-l", "syn.log"]
        
        # Convert command list to string for bash -c with proper shell escaping
        command_str = " ".join(shlex.quote(arg) for arg in base_command)
        
        # Build the full command with source
        full_command = [
            'bash', '-c',
            f'source {shlex.quote(setup_script_path)} && {command_str}'
        ]
        
        if self.checkpoint_controller.check_checkpoint("vitis", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            # Clean up any existing Vitis HLS project directory to avoid stale blackbox file references
            project_dir = f"hls_{self.benchmark_name}"
            if os.path.exists(project_dir):
                logger.info(f"Cleaning up existing Vitis HLS project directory: {project_dir}")
                shutil.rmtree(project_dir)
            logger.info(f"Sourcing setup script: {setup_script_path}")
            # Start the process and write output to vitis_hls.log
            with open("vitis_hls.log", "w") as logfile:
                p = subprocess.Popen(full_command, stdout=logfile, stderr=subprocess.STDOUT, text=True)

            completed_required_sections = False
            while True:
                time.sleep(1)
                if os.path.exists("vitis_hls.log"):
                    with open("vitis_hls.log", "r") as f:
                        for line in f:
                            if "Finished Generating all RTL models" in line:
                                completed_required_sections = True
                                break
                if completed_required_sections:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except Exception:
                        p.kill()
                    logger.info("Vitis HLS process terminated early after RTL models generated.")
                    break
                # Check if process already exited
                if p.poll() is not None:
                    break

            elapsed = time.time() - start_time
            logger.info(f"Vitis HLS command elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

            # Read the log for output
            if p.returncode not in (0, None) and not completed_required_sections:
                logger.error(f"Vitis HLS command failed: see vitis_hls.log")
                raise Exception(f"Vitis HLS command failed: see vitis_hls.log")
            if self.cfg["args"]["arch_opt_pipeline"] == "streamhls":
                save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)
                shutil.copytree(f"{save_path}/hls_{self.benchmark_name}/solution1", f"{save_path}/{self.benchmark_name}/solution1")
        else:
            logger.info("Skipping Vitis")
        self.checkpoint_controller.check_end_checkpoint("vitis", self.iteration_count)
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        # PARSE OUTPUT, set schedule and read netlist
        self.parse_vitis_data(save_dir=save_dir)
        

    def set_workload_size(self, dir_name):
        """
        Sets the workload size for the benchmark.
        """
        if os.path.exists(f"{dir_name}/{self.benchmark_name}.c"):
            with open(f"{dir_name}/{self.benchmark_name}.c", "r") as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    if "#define N" in line:
                        new_lines.append(f"#define N {self.cfg['args']['workload_size']}\n")
                    else:
                        new_lines.append(line)
            with open(f"{dir_name}/{self.benchmark_name}.c", "w") as f:
                f.writelines(new_lines)

    def forward_pass(self, iteration_count, save_dir, setup=False):
        """
        Executes the forward pass of the codesign process: prepares benchmark, updates ccore
        delays/areas, runs hls, calculates wire parasitics, and parses timing reports.

        Args:
            iteration_count: The iteration count of the forward pass
            setup: forward pass as a setup step, where we run with maximum resources and see how much benefit we can get from parallelism (only supports vitis)
        Returns:
            None
        """
        if setup:
            print("Running setup forward pass")
        else:
            print("\nRunning Forward Pass")
            logger.info("Running Forward Pass")


        ## clear out the existing tmp benchmark directory and copy the benchmark files from the desired benchmark
        if (iteration_count != 0 or self.cfg["args"]["checkpoint_start_step"] == "none") and not self.max_rsc_reached:
            logger.info("Resetting benchmark directory.")
            if os.path.exists(self.benchmark_dir):
                while os.path.exists(self.benchmark_dir):
                    shutil.rmtree(self.benchmark_dir, ignore_errors=True)
                    time.sleep(10)
            shutil.copytree(self.benchmark, self.benchmark_dir)
            shutil.copy(os.path.join(self.benchmark, "../..", "arith_ops.c"), os.path.join(self.benchmark_dir, "arith_ops.c"))
        else:
            logger.info("Skipping benchmark directory reset.")

        self.set_workload_size(self.benchmark_dir if not setup else self.benchmark_setup_dir)

        self.clk_period = self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period] # ns

        if self.hls_tool == "vitis":
            self.vitis_forward_pass(save_dir=save_dir, iteration_count=iteration_count, setup=setup)
        else:
            raise ValueError(f"Invalid hls tool: {self.hls_tool}")
        if setup: return

        # calculate wire parasitics 
        # NOTE: we don't check the checkpoint status here because it is handled in the function call (see run_openroad variable)
        self.calculate_wire_parasitics()

        ## log the scheduled_dfgs from hardware model
        for basic_block_name in self.hw.scheduled_dfgs:
            logger.info(f"Scheduled DFG for basic block {basic_block_name}:")

        ## create the obj equation 
        self.hw.calculate_objective(log_top_vectors=True)
        #self.hw.dump_top_vectors_to_file(f"{self.block_vectors_save_dir}/block_vectors_forward_pass_{iteration_count}.json")

        """if setup:
            self.max_parallel_initial_objective_value = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf()
            print(f"objective value with max parallelism: {self.max_parallel_initial_objective_value}")
        elif iteration_count == 0:
            self.hw.circuit_model.tech_model.max_speedup_factor = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() / self.max_parallel_initial_objective_value
            self.hw.circuit_model.tech_model.max_area_increase_factor = self.max_dsp / self.cur_dsp_usage
            print(f"max parallelism factor: {self.hw.circuit_model.tech_model.max_speedup_factor}")
            self.hw.circuit_model.tech_model.init_scale_factors(self.hw.circuit_model.tech_model.max_speedup_factor, self.hw.circuit_model.tech_model.max_area_increase_factor)"""

        # Ensure clock period is no less than the longest wire delay
        max_wire_delay = 0.0
        for edge in self.hw.circuit_model.edge_to_nets:
            wire_delay = sim_util.xreplace_safe(
                self.hw.circuit_model.wire_delay(edge),
                self.hw.circuit_model.tech_model.base_params.tech_values
            )
            max_wire_delay = max(max_wire_delay, wire_delay)
        
        if self.clk_period < max_wire_delay:
            old_clk_period = self.clk_period
            # Update clock period to be at least as large as the longest wire delay
            self.hw.circuit_model.tech_model.base_params.tech_values[
                self.hw.circuit_model.tech_model.base_params.clk_period
            ] = max_wire_delay
            self.clk_period = max_wire_delay
            logger.warning(
                f"Clock period ({old_clk_period} ns) was less than the longest wire delay "
                f"({max_wire_delay} ns). Updated clock period to {max_wire_delay} ns."
            )

        self.hw.display_objective("after forward pass")

        self.checkpoint_controller.check_end_checkpoint("pd", self.iteration_count)
        self.obj_over_iterations.append(sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values))

    def calculate_wire_parasitics(self):
        """
        Prepares the schedule by setting the end node's start time and getting the longest paths.
        Also updates the hardware netlist with wire parasitics and determines the longest paths.

        Args:
            None
        Returns:
            None
        """

        # If the user requested zero wirelength costs, we must not run (or try to reuse) OpenROAD.
        # In this mode, all wirelength-related quantities are assumed to be 0.
        if self.cfg["args"].get("zero_wirelength_costs", False):
            logger.info("zero_wirelength_costs enabled: skipping OpenROAD; assuming all wirelengths/delays/energies are 0.")
            # Make this explicit so other parts of the flow don't accidentally use stale OpenROAD results.
            self.hw.circuit_model.edge_to_nets = {}
            return

        # netlist_dfg = self.hw.scheduled_dfg.copy() 
        # netlist_dfg.remove_node("end")
        # self.hw.netlist = netlist_dfg

        # update netlist and scheduled dfg with wire parasitics
        run_openroad = (
            self.cfg["args"]["always_run_openroad"] or (
            self.checkpoint_controller.check_checkpoint("pd", self.iteration_count) 
            and not self.max_rsc_reached
            and not os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp")
            )
        )
        if os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp"):
            shutil.rmtree(f"{self.tmp_dir}/pd", ignore_errors=True)
            shutil.copytree(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp", f"{self.tmp_dir}/pd")
            logger.info(f"copied pd_{self.cur_dsp_usage}_dsp to pd, will use results from previous run")
        if not self.checkpoint_controller.check_checkpoint("pd", self.iteration_count):
            logger.info(f"will not run openroad because of checkpoint, will use results from previous run")
        self.hw.get_wire_parasitics(self.openroad_testfile, self.parasitics, self.benchmark_name, run_openroad, self.cfg["args"]["area"])

    def write_back_params(self, params_path="src/yaml/params_current.yaml"):
        """
        Writes the technology parameters back to a YAML file

        Args:
            params_path (str): Path to the output YAML file. Defaults to 'src/yaml/params_current.yaml'.

        Returns:
            None
        """
        with open(params_path, "w") as f:
            d = {}
            for key in self.hw.circuit_model.tech_model.base_params.tech_values:
                if isinstance(key, str):
                    d[key] = float(self.hw.circuit_model.tech_model.base_params.tech_values[key])
                elif key in self.hw.circuit_model.tech_model.base_params.names:
                    d[self.hw.circuit_model.tech_model.base_params.names[key]] = float(self.hw.circuit_model.tech_model.base_params.tech_values[key])
            f.write(yaml.dump(d))

    def inverse_pass(self):
        """
        Executes the inverse pass of the codesign process: generates symbolic CACTI values for
        memories, computes obj, runs optimizer to find better technology
        parameters, and logs results.

        Args:
            None
        Returns:
            None
        """
        print("\nRunning Inverse Pass")
        logger.info("Running Inverse Pass")

        stdout = sys.stdout
        with open(f"{self.tmp_dir}/ipopt_out.txt", "w") as sys.stdout:
            lag_factor, error = self.opt.optimize(self.cfg["args"]["solver"], iteration=self.iteration_count, improvement=self.inverse_pass_improvement)
            self.inverse_pass_lag_factor *= lag_factor
        sys.stdout = stdout

        self.write_back_params()

        self.hw.display_objective("after inverse pass")

        self.obj_over_iterations.append(sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values))
        self.lag_factor_over_iterations.append(self.inverse_pass_lag_factor)

    def log_all_to_file(self, iter_number):
        wire_lengths={}
        wire_delays={}
        for edge in self.hw.circuit_model.edge_to_nets:
            wire_lengths[edge] = self.hw.circuit_model.wire_length(edge)
            wire_delays[edge] = sim_util.xreplace_safe(self.hw.circuit_model.wire_delay(edge), self.hw.circuit_model.tech_model.base_params.tech_values)
        self.wire_lengths_over_iterations.append(wire_lengths)
        self.wire_delays_over_iterations.append(wire_delays)
        device_delay = sim_util.xreplace_safe(self.hw.circuit_model.tech_model.delay, self.hw.circuit_model.tech_model.base_params.tech_values)
        self.device_delays_over_iterations.append(device_delay)
        json.dump(list(wire_lengths.values()), open(f"{self.save_dir}/wire_lengths_{iter_number}.json", "w"))
        nx.write_gml(
            self.hw.netlist,
            f"{self.save_dir}/netlist_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        nx.write_gml(
            self.hw.scheduled_dfg,
            f"{self.save_dir}/schedule_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        self.write_back_params(f"{self.save_dir}/tech_params_{iter_number}.yaml")
        shutil.copy(f"{self.tmp_dir}/ipopt_out.txt", f"{self.save_dir}/ipopt_{iter_number}.txt")
        if not os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp") and os.path.exists(f"{self.tmp_dir}/pd"):
            shutil.copytree(f"{self.tmp_dir}/pd", f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp")
        if os.path.exists(f"{self.tmp_dir}/pd/results/design_snapshot-tcl.png"):
            shutil.copy(f"{self.tmp_dir}/pd/results/design_snapshot-tcl.png", f"{self.save_dir}/design_snapshot_{iter_number}.png")
        """for mem in self.hw.circuit_model.memories:
            shutil.copy(
                f"{self.tmp_dir}/cacti_exprs_{mem}.txt", f"{self.save_dir}/cacti_exprs_{mem}_{iter_number}.txt"
            )"""
        #TODO: copy cacti expressions to file, read yaml file from notebook, call sim util fn to get xreplace structure
        #TODO: fw pass save cacti params of interest, with logger unique starting string, then write parsing script in notebook to look at them
        # save latency, power, and tech params
        self.hw.write_technology_parameters(
            f"{self.save_dir}/circuit_values_{iter_number}.yaml"
        )
        self.params_over_iterations.append(copy.copy(self.hw.circuit_model.tech_model.base_params.tech_values))
        self.sensitivities_over_iterations.append(copy.copy(self.hw.sensitivities))
        self.constraint_slack_over_iterations.append({})
        #self.hw.dump_top_vectors_to_file(f"{self.block_vectors_save_dir}/block_vectors_inverse_pass_{iter_number}.json")
        for constraint in self.opt.constraints:
            if constraint.label in self.hw.constraints_to_plot:
                assert len(self.params_over_iterations) > 1, "params over iterations has less than 2 elements"
                self.constraint_slack_over_iterations[-1][constraint.label] = sim_util.xreplace_safe(constraint.slack, self.params_over_iterations[-2])

    def save_dat(self):
        pass

    def restore_dat(self):
        pass

    def save_last_dsp_count(self, last_dsp_count_path="src/yaml/last_dsp_count.yaml"):
        with open(last_dsp_count_path, "w") as f:
            yaml.dump({"last_dsp_count": self.cur_dsp_usage}, f)

    def cleanup(self):
        self.restore_dat()

    def end_of_run_plots(self, obj_over_iterations, lag_factor_over_iterations, params_over_iterations, wire_lengths_over_iterations, wire_delays_over_iterations, device_delays_over_iterations, sensitivities_over_iterations, constraint_slack_over_iterations, visualize_block_vectors=False):
        if not hasattr(self.hw, "obj_sub_plot_names"):
            logger.warning("Skipping plots: missing obj_sub_plot_names on hardware model.")
            return
        if not obj_over_iterations:
            logger.warning("Skipping plots: no objective data collected.")
            return
        obj = "Energy Delay Product"
        units = "nJ*ns"
        if self.obj_fn == "energy":
            obj = "Energy"
            units = "nJ"
        elif self.obj_fn == "delay":
            obj = "Delay"
            units = "ns"
        trend_plotter = trend_plot.TrendPlot(self, params_over_iterations, self.hw.circuit_model.tech_model.base_params.names, obj_over_iterations, lag_factor_over_iterations, wire_lengths_over_iterations, wire_delays_over_iterations, device_delays_over_iterations, sensitivities_over_iterations, constraint_slack_over_iterations, self.save_dir + "/figs", obj, units, self.obj_fn)
        logger.info(f"plotting wire lengths over generations")
        trend_plotter.plot_wire_lengths_over_generations()
        logger.info(f"plotting wire delays over generations")
        trend_plotter.plot_wire_delays_over_generations()
        logger.info(f"plotting params over generations")
        trend_plotter.plot_params_over_generations()
        logger.info(f"plotting obj over generations")
        trend_plotter.plot_obj_over_generations()
        logger.info(f"plotting lag factor over generations")
        trend_plotter.plot_lag_factor_over_generations()
        logger.info(f"plotting sensitivities over generations")
        trend_plotter.plot_sensitivities_over_generations()
        logger.info(f"plotting constraint slack over generations")
        trend_plotter.plot_constraint_slack_over_generations()
        #if visualize_block_vectors:
        #    logger.info(f"plotting block vectors over generations")
        #    trend_plotter.plot_block_vectors_over_generations()

    def setup(self):
        if not os.path.exists(self.benchmark_setup_dir):
            shutil.copytree(self.benchmark, self.benchmark_setup_dir)
            shutil.copy(os.path.join(self.benchmark, "../..", "arith_ops.c"), os.path.join(self.benchmark_setup_dir, "arith_ops.c"))
        if self.cfg["args"]["arch_opt_pipeline"] == "scalehls":
            assert os.path.exists(self.config_json_path_scalehls)
            shutil.copy(self.config_json_path_scalehls, self.config_json_path)
        elif self.cfg["args"]["arch_opt_pipeline"] == "streamhls":
            assert os.path.exists(self.config_json_path_streamhls)
            shutil.copy(self.config_json_path_streamhls, self.config_json_path)
        self.forward_pass(0, save_dir=self.benchmark_setup_dir, setup=True)

    def execute(self, num_iters):
        self.iteration_count = 0
        self.setup()
        self.checkpoint_controller.check_end_checkpoint("setup", self.iteration_count)
        while self.iteration_count < num_iters:
            start_time = time.time()
            self.forward_pass(self.iteration_count, self.benchmark_dir)
            self.log_forward_tech_params()
            self.inverse_pass()
            start_time_after_inverse_pass = time.time()
            self.hw.circuit_model.update_circuit_values()
            self.log_all_to_file(self.iteration_count)
            self.hw.reset_state()
            self.hw.reset_tech_model()
            logger.info(f"time to update state after inverse pass iteration {self.iteration_count}: {time.time()-start_time_after_inverse_pass}")
            logger.info(f"time to execute iteration {self.iteration_count}: {time.time()-start_time}")
            self.iteration_count += 1
            params_over_iterations_start_ind = 0 if not self.cfg["args"]["fixed_area_increase_pattern"] else 1
            self.end_of_run_plots(self.obj_over_iterations, self.lag_factor_over_iterations, self.params_over_iterations[params_over_iterations_start_ind:], self.wire_lengths_over_iterations, self.wire_delays_over_iterations, self.device_delays_over_iterations, self.sensitivities_over_iterations, self.constraint_slack_over_iterations, visualize_block_vectors=False)
            logger.info(f"current dsp usage: {self.cur_dsp_usage}, max dsp: {self.max_dsp}")
            if self.cur_dsp_usage == self.max_dsp:
                logger.info("Resource constraints have been reached, will skip forward pass steps from now on.")
                self.max_rsc_reached = True        

        # cleanup
        self.cleanup()

        print(f"{FLOW_SUCCESS_MSG}{self.iteration_count}")

        
def main(args):
    codesign_module = Codesign(
        args
    )
    try:
        codesign_module.execute(codesign_module.cfg["args"]["num_iters"])
    finally:
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        # dump latest tech params to tmp dir for replayability
        codesign_module.write_back_params(f"{codesign_module.tmp_dir}/tech_params_latest.yaml")
        codesign_module.save_last_dsp_count(f"{codesign_module.tmp_dir}/last_dsp_count.yaml")
        
        params_over_iterations_start_ind = 0 if not codesign_module.cfg["args"]["fixed_area_increase_pattern"] else 1
        codesign_module.end_of_run_plots(codesign_module.obj_over_iterations, codesign_module.lag_factor_over_iterations, codesign_module.params_over_iterations[params_over_iterations_start_ind:], codesign_module.wire_lengths_over_iterations, codesign_module.wire_delays_over_iterations, codesign_module.device_delays_over_iterations, codesign_module.sensitivities_over_iterations, codesign_module.constraint_slack_over_iterations, visualize_block_vectors=True)
        codesign_module.cleanup()

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
        help="Path to the save new architecture file",
    )
    parser.add_argument(
        "--parasitics",
        type=str,
        choices=["detailed", "estimation", "none"],
        help="determines what type of parasitic calculations are done for wires",
    )

    parser.add_argument(
        "--openroad_testfile",
        type=str,
        help="what tcl file will be executed for openroad",
    )
    parser.add_argument(
        "-N",
        "--num_iters",
        type=int,
        help="Number of Codesign iterations to run",
    )
    parser.add_argument(
        "-a",
        "--area",
        type=float,
        help="Area constraint in um2",
    )
    parser.add_argument(
        "--no_memory",
        type=bool,
        help="disable memory modeling",
    )
    parser.add_argument(
        "--additional_cfg_file",
        type=str,
        help="path to an additional configuration file",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        help="path to store the tmp dir for this run",
    )
    parser.add_argument(
        "--preinstalled_openroad_path",
        type=str,
        help="Path to a pre-installed OpenROAD installation. This is primarily useful for CI testing where OpenRoad is pre-installed on the system.",
    )
    parser.add_argument("--arch_opt_pipeline", type=str, help="architecture optimization pipeline to use")
    parser.add_argument("--streamhls_opt_level", type=str, help="StreamHLS optimization level to use")
    parser.add_argument('--debug_no_cacti', type=bool,
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    parser.add_argument("--logic_node", type=int, help="logic node size")
    parser.add_argument("--mem_node", type=int, help="memory node size")
    parser.add_argument("--inverse_pass_improvement", type=float, help="improvement factor for inverse pass")
    parser.add_argument("--tech_node", "-T", type=str, help="technology node to use as starting point")
    parser.add_argument("--obj", type=str, help="objective function")
    parser.add_argument("--model_cfg", type=str, help="symbolic model configuration")
    parser.add_argument("--hls_tool", type=str, help="hls tool to use")
    parser.add_argument("--config", type=str, default="default", help="config to use")
    parser.add_argument("--checkpoint_load_dir", type=str, help="directory to load checkpoint")
    parser.add_argument("--checkpoint_save_dir", type=str, help="directory to save checkpoint")
    parser.add_argument("--checkpoint_start_step", type=str, help="checkpoint step to resume from (the flow will start normal execution AFTER this step)")
    parser.add_argument("--always_run_openroad", type=bool, help="always run openroad, even if max rsc reached")
    parser.add_argument("--stop_at_checkpoint", type=str, help="checkpoint step to stop at (will complete this step and then stop)")
    parser.add_argument("--workload_size", type=int, help="workload size to use, such as the dimension of the matrix for gemm. Only applies to certain benchmarks")
    parser.add_argument("--opt_pipeline", type=str, help="optimization pipeline to use for inverse pass")
    parser.add_argument("--num_dsps", type=str, help="the number of DSPs to use as a constraint for ScaleHLS. Only use this if attempting to only run the FORWARD PASS.")
    parser.add_argument("--zero_wirelength_costs", type=bool, help="set all wirelength costs to zero (wire_length and wire_energy will return 0)")
    parser.add_argument("--constant_wire_length_cost", type=int, help="Instead of estimating the wirelength based on physical layout, use a constant cost for every wire in the design.")  
    parser.add_argument("--min_dsp", type=int, help="minimum DSP usage to start with")
    parser.add_argument("--max_power_density", type=float, help="maximum power density to allow")
    parser.add_argument("--max_power", type=float, help="maximum total power to allow")
    parser.add_argument("--solver", type=str, help="solver to use for inverse pass")
    parser.add_argument("--fixed_area_increase_pattern", type=bool, help="number of resources increases by some factor for each iteration")
    parser.add_argument("--leakage_restriction", type=bool, help="restrict the passive power to be less than 1/3 of the total power")
    parser.add_argument("--MUL_restriction", type=bool, help="restrict the MUL flag to be 1")
    args = parser.parse_args()

    main(args)
