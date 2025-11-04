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
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from src.forward_pass.netlist_parse import parse_yosys_json

logger = logging.getLogger("codesign")

from src import cacti_util
from src import sim_util
from src.hardware_model import hardwareModel
from src.inverse_pass import optimize
from src.forward_pass import schedule
from src import memory
from src.forward_pass import ccore_update
from src.forward_pass import schedule_vitis
from src.forward_pass.scale_hls_port_fix import scale_hls_port_fix
from src.forward_pass.vitis_create_netlist import create_vitis_netlist
from src.forward_pass.vitis_parse_verbose_rpt import parse_verbose_rpt
from src.forward_pass.vitis_merge_netlists import MergeNetlistsVitis
from src.forward_pass.vitis_create_cdfg import create_cdfg_vitis
from src import trend_plot
import time
from test import checkpoint_controller

DEBUG_YOSYS = False  # set to True to debug yosys output.

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

        self.forward_obj = 0
        self.inverse_obj = 0
        self.openroad_testfile = self.cfg['args']['openroad_testfile']
        self.parasitics = self.cfg["args"]["parasitics"]
        self.run_cacti = not self.cfg["args"]["debug_no_cacti"]
        self.no_memory = self.cfg["args"]["no_memory"]
        self.hw = hardwareModel.HardwareModel(self.cfg, self.codesign_root_dir, self.tmp_dir)
        self.opt = optimize.Optimizer(self.hw, self.tmp_dir, opt_pipeline=self.cfg["args"]["opt_pipeline"])
        self.module_map = {}
        self.inverse_pass_improvement = self.cfg["args"]["inverse_pass_improvement"]
        self.inverse_pass_lag_factor = 1

        self.params_over_iterations = [copy.copy(self.hw.circuit_model.tech_model.base_params.tech_values)]
        self.obj_over_iterations = []
        self.lag_factor_over_iterations = [1.0]
        self.max_unroll = 64

        self.save_dat()

        #with open(f"{self.tmp_dir}/tech_params_0.yaml", "w") as f:
        #    f.write(yaml.dump(self.hw.circuit_model.tech_model.base_params.tech_values))

        self.hw.write_technology_parameters(self.save_dir+"/initial_tech_params.yaml")

        self.iteration_count = 0
        
        self.checkpoint_controller = checkpoint_controller.CheckpointController(self.cfg, self.codesign_root_dir, self.tmp_dir)

        if self.cfg["args"]["checkpoint_start_step"]!= "none" and self.cfg["args"]["checkpoint_load_dir"] != "none":
            self.checkpoint_controller.load_checkpoint()

        # configure to start with 3 dsp and 3 bram
        self.dsp_multiplier = 1/3 * self.cfg["args"]["area"] / (3e-6 * 9e-6)
        self.bram_multiplier = 1/3 * self.cfg["args"]["area"] / (3e-6 * 9e-6)

        self.wire_lengths_over_iterations = []

        self.cur_dsp_usage = 0
        self.max_rsc_reached = False

        self.config_json_path_scalehls = "ScaleHLS-HIDA/test/Transforms/Directive/config.json"
        self.config_json_path = self.benchmark_setup_dir + "/config.json"

    # any arguments specified on CLI will override the default config
    def set_config(self, args):
        with open(f"src/yaml/codesign_cfg.yaml", "r") as f:
            cfgs = yaml.load(f, Loader=yaml.FullLoader)
        overwrite_args_all = vars(args)
        overwrite_args = {}
        for key, value in overwrite_args_all.items():
            if value is not None:
                overwrite_args[key] = value
        overwrite_cfg = {"base_cfg": args.config, "args": overwrite_args}
        cfgs["overwrite_cfg"] = overwrite_cfg
        self.cfg = sim_util.recursive_cfg_merge(cfgs, "overwrite_cfg")
        print(f"args: {self.cfg['args']}")

    def get_tmp_dir(self):
        idx = 0

        ## make base tmp directory
        if not os.path.exists("src/tmp"):
                os.makedirs("src/tmp")

        while True:
            tmp_dir = f"src/tmp/tmp_{self.benchmark_name}_{self.obj_fn}_{idx}"
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


    def run_catapult(self):
        """
        Runs the Catapult synthesis tool, updates the memory configuration, and logs the output.
        Handles directory changes and cleans up temporary files.

        Args:
            None
        Returns:
            None
        """
        self.module_map = {}
        # if no_memory flag set, scheduler will cut out all memory nodes
        if not self.no_memory:
            self.module_map = {
                "ccs_ram_sync_1R1W_rwport": "Rsc",
                "ccs_ram_sync_1R1W_rport": "Rsc",
                "ccs_ram_sync_1R1W_wport": "Rsc",
                "nop": "nop"
            }
        for unit in self.hw.circuit_model.circuit_values["area"].keys():
            self.module_map[unit.lower()] = unit

        os.chdir(self.benchmark_dir)
        
        # add area constraint
        sim_util.add_area_constraint_to_script(f"scripts/{self.benchmark_name}.tcl", self.hw.area_constraint)

        p = subprocess.run(["make", "clean"], capture_output=True, text=True)
        cmd = ["make", "build_design"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"first catapult run output: {p.stdout}")
        if p.returncode != 0:
            raise Exception(p.stderr)
        os.chdir("../../..")
        # if not self.no_memory:
        #     ## If we are running with memory, we need to run Cacti to get the memory parameters and then rerun Catapult. 
        #     pre_assign_counts = memory.get_pre_assign_counts(f"{self.benchmark_dir}/bom.rpt", self.module_map)
        #     logger.info(f"hw.circuit_model.circuit_values: {self.hw.circuit_model.circuit_values}")
        #     self.hw.circuit_model.set_memories(memory.customize_catapult_memories(f"{self.benchmark_dir}/memories.rpt", self.benchmark_name, self.hw, pre_assign_counts))
        #     logger.info(f"custom catapult memories: {self.hw.circuit_model.memories}")
        #     os.chdir(self.benchmark_dir)
        #     p = subprocess.run(["make", "clean"], capture_output=True, text=True)
        #     p = subprocess.run(cmd, capture_output=True, text=True)
        #     logger.info(f"custom memory catapult run output: {p.stdout}")
        #     if p.returncode != 0:
        #         raise Exception(p.stderr)
        #     os.chdir("../../..")
        # else:
        logger.info("skipping memory customization and second catapult run")
        self.hw.circuit_model.set_memories({})

        # Extract Hardware Netlist
        cmd = ["yosys", "-p", f"read_verilog {self.benchmark_dir}/build/{self.benchmark_name}.v1/rtl.v; hierarchy -top {self.benchmark_name}; proc; write_json {self.benchmark_dir}/netlist.json"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if DEBUG_YOSYS:
            logger.info(f"Yosys output: {p.stdout}")
        if p.returncode != 0:
            raise Exception(f"Yosys failed with error: {p.stderr}")

        self.hw.netlist, full_netlist = parse_yosys_json(f"{self.benchmark_dir}/netlist.json", include_memories=(not self.no_memory), top_level_module_type=self.benchmark_name)

        ## write the netlist to a file
        with open(f"{self.benchmark_dir}/netlist-from-catapult.gml", "wb") as f:
            nx.write_gml(self.hw.netlist, f)

        with open(f"{self.benchmark_dir}/full_netlist_debug.gml", "wb") as f:
            nx.write_gml(full_netlist, f)

    def set_resource_constraint_scalehls(self, unlimited=False):
        """
        Sets the resource constraint and op latencies for ScaleHLS.
        """
        with open(self.config_json_path, "r") as f:
            config = json.load(f)
        if unlimited:
            config["dsp"] = 10000
            config["bram"] = 10000
        else:
            config["dsp"] = int(self.cfg["args"]["area"] / (self.hw.circuit_model.tech_model.param_db["A_gate"].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() * self.dsp_multiplier))
            config["bram"] = int(self.cfg["args"]["area"] / (self.hw.circuit_model.tech_model.param_db["A_gate"].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() * self.bram_multiplier))
            # setting cur_dsp_usage here instead of with parse_dsp_usage after running scaleHLS
            # because I observed that for a small amount of resources, scaleHLS won't generate the csv file that we need to parse
            self.cur_dsp_usage = config["dsp"] 

        # I don't think "100MHz" has any meaning because scaleHLS should be agnostic to frequency
        config["100MHz"]["fadd"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Add16"] / self.clk_period)
        config["100MHz"]["fmul"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Mult16"] / self.clk_period)
        config["100MHz"]["fdiv"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["FloorDiv16"] / self.clk_period)
        config["100MHz"]["fcmp"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["GtE16"] / self.clk_period)

        config["max_iter_num"] = self.cfg["args"]["max_iter_num_scalehls"]
        with open(self.config_json_path, "w") as f:
            json.dump(config, f)

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
                scalehls-opt {self.benchmark_name}.mlir -hida-pytorch-pipeline=\"top-func={self.vitis_top_function} loop-tile-size=8 loop-unroll-factor=4\" | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > {os.path.join(os.path.dirname(__file__), "..", save_dir)}/{self.benchmark_name}.cpp
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
                                                -function={self.benchmark_name}\
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
        df = pd.read_csv(f"{read_dir}/{self.benchmark_name}_space.csv")
        mlir_idx_that_exist = []
        for i in range(len(df['dsp'].values)):
            file_exists = os.path.exists(f"{read_dir}/{self.benchmark_name}_pareto_{i}.mlir")
            if file_exists:
                mlir_idx_that_exist.append(i)
            if df['dsp'].values[i] <= self.cur_dsp_usage and file_exists:
                return f"{read_dir}/{self.benchmark_name}_pareto_{i}.mlir", i
        if len(mlir_idx_that_exist) == 0:
            raise Exception(f"No Pareto solutions found for {self.benchmark_name} with dsp usage {self.cur_dsp_usage}")
        return f"{read_dir}/{self.benchmark_name}_pareto_{mlir_idx_that_exist[-1]}.mlir", mlir_idx_that_exist[-1]

    def parse_dsp_usage_and_latency(self, mlir_idx):
        df = pd.read_csv(f'''{os.path.join(os.path.dirname(__file__), "..", self.benchmark_setup_dir)}/{self.benchmark_name}_space.csv''')
        logger.info(f"dsp usage: {df['dsp'].values[mlir_idx]}, latency: {df['cycle'].values[mlir_idx]}")
        return df['dsp'].values[mlir_idx], df['cycle'].values[mlir_idx]

    def parse_vitis_data(self, save_dir):

        ## print the cwd
        print(f"Current working directory in vitis parse data: {os.getcwd()}")

        if self.no_memory:
            allowed_functions_netlist = {"Add16", "Mult16"}
            allowed_functions_schedule = {"Add16", "Mult16", "Call", "II"}
        else:
            allowed_functions_netlist = {"Add16", "Mult16", "Buf", "MainMem"}
            allowed_functions_schedule = {"Add16", "Mult16", "Call", "II", "Buf", "MainMem"}

        parse_results_dir = f"{save_dir}/parse_results"

        logger.info(f"parse results dir: {parse_results_dir}")

        if self.checkpoint_controller.check_checkpoint("netlist", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            ## Do preprocessing to the vitis data for the next scripts
            parse_verbose_rpt(f"{save_dir}/{self.benchmark_name}/solution1/.autopilot/db", parse_results_dir)

            ## Create the netlist
            logger.info("Creating Vitis netlist")
            create_vitis_netlist(parse_results_dir)

            ## Create the CDFGs for each FSM
            logger.info("Creating Vitis CDFGs")
            create_cdfg_vitis(parse_results_dir)

            ## Create the mapping from CDFG nodes to netlist nodes
            #create_cdfg_to_netlist_mapping_vitis(parse_results_dir)

            ## Merge the CDFGs recursivley through the FSM module hierarchy to produce overall CDFG
            #merge_cdfgs_vitis(parse_results_dir, self.vitis_top_function)

            ## Merge the netlists recursivley through the module hierarchy to produce overall netlist
            logger.info("Recursivley merging vitis netlists")
            vitis_netlist_merger = MergeNetlistsVitis(self.cfg, self.codesign_root_dir, allowed_functions_netlist)
            vitis_netlist_merger.merge_netlists_vitis(parse_results_dir, self.vitis_top_function)
            logger.info("Vitis netlist parsing complete")
            logger.info(f"time to parse vitis netlist: {time.time()-start_time}")
        else:
            logger.info("Skipping Vitis netlist parsing")

        self.checkpoint_controller.check_end_checkpoint("netlist")

        if self.checkpoint_controller.check_checkpoint("schedule", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            logger.info("Parsing Vitis schedule")
            schedule_parser = schedule_vitis.vitis_schedule_parser(save_dir, self.benchmark_name, self.vitis_top_function, self.clk_period, allowed_functions_schedule)
            
            logger.info("Creating DFGs from Vitis CDFGs")
            schedule_parser.create_dfgs()
            self.hw.scheduled_dfgs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_standard_with_wire_ops"] for basic_block_name in schedule_parser.basic_blocks}
            self.hw.loop_1x_graphs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_loop_1x_standard_with_wire_ops"] for basic_block_name in schedule_parser.basic_blocks if "G_loop_1x" in schedule_parser.basic_blocks[basic_block_name]}
            #self.hw.loop_2x_graphs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_loop_2x_standard"] for basic_block_name in schedule_parser.basic_blocks if "G_loop_2x" in schedule_parser.basic_blocks[basic_block_name]}
            logger.info(f"scheduled dfgs: {self.hw.scheduled_dfgs}")
            logger.info(f"loop 1x graphs: {self.hw.loop_1x_graphs}")

            logger.info("Vitis schedule parsing complete")
            logger.info(f"time to parse vitis schedule: {time.time()-start_time}")
        else:
            for file in os.listdir(parse_results_dir):
                if os.path.isdir(os.path.join(parse_results_dir, file)):
                    self.hw.scheduled_dfgs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_standard_with_wire_ops.gml")
                    if os.path.exists(f"{parse_results_dir}/{file}/{file}_graph_loop_1x_standard.gml"):
                        self.hw.loop_1x_graphs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_loop_1x_standard_with_wire_ops.gml")
                        #assert os.path.exists(f"{parse_results_dir}/{file}/{file}_graph_loop_2x_standard.gml")
                        #self.hw.loop_2x_graphs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_loop_2x_standard_with_wire_ops.gml")
            logger.info("Skipping Vitis schedule parsing")

        self.checkpoint_controller.check_end_checkpoint("schedule")

        print(f"Current working directory at end of vitis parse data: {os.getcwd()}")

        self.hw.netlist = nx.read_gml(f"{parse_results_dir}/{self.vitis_top_function}_full_netlist.gml")

        logger.info("Now calculating execution time of Vitis design with top function "+self.vitis_top_function)
        start_time = time.time()
        execution_time = self.hw.calculate_execution_time_vitis(self.vitis_top_function)
        logger.info(f"time to calculate execution time: {time.time()-start_time}")
        print(f"Execution time: {execution_time}")

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
        self.vitis_top_function = self.benchmark_name if not self.cfg["args"]["pytorch"] else "forward"

        # prep before running scalehls
        self.set_resource_constraint_scalehls(unlimited=setup)
        if setup:
            opt_cmd = f'''scalehls-opt {self.benchmark_name}.mlir -scalehls-dse-pipeline=\"top-func={self.vitis_top_function} target-spec={os.path.join(os.path.dirname(__file__), "..", self.config_json_path)}\"'''
            mlir_idx = 0
        elif not self.cfg["args"]["pytorch"]: # pytorch scalehls dse not yet working
            mlir_file, mlir_idx = self.parse_design_space_for_mlir(os.path.join(os.path.dirname(__file__), "..", f"{self.tmp_dir}/benchmark_setup"))
            opt_cmd = f"cat {mlir_file}"
        else:
            opt_cmd = ""

        # run scalehls
        if (self.checkpoint_controller.check_checkpoint("scalehls", self.iteration_count) and not setup) or (self.checkpoint_controller.check_checkpoint("setup", self.iteration_count) and setup) and not self.max_rsc_reached:
            self.run_scalehls(save_dir, opt_cmd, setup)
        else:
            logger.info("Skipping ScaleHLS")

        # set scale factors if in setup or first iteration
        if self.cfg["args"]["pytorch"]:
            # TODO replace once pytorch dse working
            if setup:
                dsp_usage, latency = 10, 10
            else:
                dsp_usage, latency = 1, 1
        else:
            dsp_usage, latency = self.parse_dsp_usage_and_latency(mlir_idx)
        if setup:
            self.max_dsp = dsp_usage
            self.max_latency = latency
        elif iteration_count == 0:
            self.max_speedup_factor = float(latency / self.max_latency)
            self.max_area_increase_factor = float(self.max_dsp / dsp_usage)
        if not setup:
            self.checkpoint_controller.check_end_checkpoint("scalehls")
        if setup: # setup step ends here, don't need to run rest of forward pass
            return

        self.cur_dsp_usage = dsp_usage

        self.hw.circuit_model.tech_model.init_scale_factors(self.max_speedup_factor, self.max_area_increase_factor)

        os.chdir(os.path.join(os.path.dirname(__file__), "..", save_dir))
        scale_hls_port_fix(f"{self.benchmark_name}.cpp", self.benchmark_name, self.cfg["args"]["pytorch"])
        logger.info(f"Vitis top function: {self.vitis_top_function}")


        import time
        command = ["vitis_hls", "-f", "tcl_script.tcl"]
        if self.checkpoint_controller.check_checkpoint("vitis", self.iteration_count) and not self.max_rsc_reached:
            start_time = time.time()
            # Start the process and write output to vitis_hls.log
            with open("vitis_hls.log", "w") as logfile:
                p = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, text=True)

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
        else:
            logger.info("Skipping Vitis")
        self.checkpoint_controller.check_end_checkpoint("vitis")
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        # PARSE OUTPUT, set schedule and read netlist
        self.parse_vitis_data(save_dir=save_dir)
        


    def catapult_forward_pass(self):
        """
        Runs Catapult version of forward pass, updates the memory configuration, and logs the output.
        """
        shutil.copytree("src/forward_pass/ccores_base", f"{self.benchmark_dir}/src/ccores")

        # update delay and area of ccores
        ccore_update.update_ccores(self.hw.circuit_model.circuit_values["area"], self.hw.circuit_model.circuit_values["latency"])

        if self.benchmark_name == "matmult":
            # change the unroll factors in the matmult_basic.cpp file
            with open(f"{self.benchmark}/src/matmult_basic.cpp", "r") as f:
                lines = f.readlines()
            with open(f"{self.benchmark_dir}/src/matmult_basic.cpp", "w") as f:
                unroll_stmts = []
                unroll_lines = {}
                matrix_size = 0
                for i in range(len(lines)):
                    if "pragma hls_unroll no" in lines[i]:
                        unroll_stmts.append(lines[i])
                        unroll_lines[lines[i]] = i
                    if "#define MATRIX_SIZE" in lines[i]:
                        matrix_size = int(lines[i].strip().split()[-1])
                assert matrix_size > 0
                unroll_ordering = [2, 1, 0] # order of unroll statements to change (innermost first)
                cur_max_unroll = min(self.max_unroll, int(self.inverse_pass_lag_factor))
                for i in unroll_ordering:
                    amount_to_unroll = cur_max_unroll
                    lines[unroll_lines[unroll_stmts[i]]] = unroll_stmts[i].replace("no", str(min(amount_to_unroll, matrix_size)))
                    cur_max_unroll //= min(amount_to_unroll, matrix_size)

                f.writelines(lines)
        elif self.benchmark_name == "basic_aes":
            # change the unroll factors in the basic_aes.cpp file
            with open(f"{self.benchmark}/src/basic_aes.cpp", "r") as f:
                lines = f.readlines()
            with open(f"{self.benchmark_dir}/src/basic_aes.cpp", "w") as f:
                unroll_stmts = []
                unroll_lines = []
                block_size = 0
                for i in range(len(lines)):
                    if "pragma hls_unroll no" in lines[i]:
                        unroll_stmts.append(lines[i])
                        unroll_lines.append(i)
                    if "#define BLOCK_SIZE" in lines[i]:
                        block_size = int(lines[i].strip().split()[2])
                #print(f"unroll stmts: {unroll_stmts}")
                #print(f"unroll lines: {unroll_lines}")
                assert block_size > 0
                assert len(unroll_stmts) == 3
                max_unrolls = [block_size, 4, block_size] # maximum value of unroll for each statement
                cur_max_unroll = min(self.max_unroll, int(self.inverse_pass_lag_factor))
                for i in range(3):
                    amount_to_unroll = cur_max_unroll
                    lines[unroll_lines[i]] = unroll_stmts[i].replace("no", str(min(amount_to_unroll, max_unrolls[i])))
                    #print(f"new line: {lines[unroll_lines[i]]}")
                    #print(f"unrolling {unroll_stmts[i]} by {min(amount_to_unroll, max_unrolls[i])}")

                f.writelines(lines)

        # run catapult with custom memory configurations
        self.run_catapult()

        # parse catapult timing report and create schedule
        self.parse_catapult_timing()

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
        else:
            logger.info("Skipping benchmark directory reset.")

        self.set_workload_size(self.benchmark_dir if not setup else self.benchmark_setup_dir)

        self.clk_period = self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period] # ns

        if self.hls_tool == "catapult":
            sim_util.change_clk_period_in_script(f"{self.benchmark_dir}/scripts/common.tcl", self.clk_period, self.cfg["args"]["hls_tool"])
            self.catapult_forward_pass()
        else:
            sim_util.change_clk_period_in_script(f"{save_dir}/tcl_script.tcl", self.clk_period, self.cfg["args"]["hls_tool"])
            self.vitis_forward_pass(save_dir=save_dir, iteration_count=iteration_count, setup=setup)
        if setup: return

        if self.checkpoint_controller.check_checkpoint("pd", iteration_count) and not self.max_rsc_reached:
            # calculate wire parasitics
            self.calculate_wire_parasitics()

            ## create the obj equation 
            self.hw.calculate_objective()

            if iteration_count == 0:
                self.params_over_iterations[0].update(
                    {
                        self.hw.circuit_model.tech_model.base_params.logic_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.logic_resource_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_resource_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.logic_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_ahmdal_limit],
                        self.hw.circuit_model.tech_model.base_params.logic_resource_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.logic_resource_ahmdal_limit],
                        self.hw.circuit_model.tech_model.base_params.interconnect_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.interconnect_resource_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_resource_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.interconnect_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_ahmdal_limit],
                        self.hw.circuit_model.tech_model.base_params.interconnect_resource_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.interconnect_resource_ahmdal_limit],
                        self.hw.circuit_model.tech_model.base_params.memory_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.memory_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.memory_resource_sensitivity: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.memory_resource_sensitivity],
                        self.hw.circuit_model.tech_model.base_params.memory_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.memory_ahmdal_limit],
                        self.hw.circuit_model.tech_model.base_params.memory_resource_ahmdal_limit: self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.memory_resource_ahmdal_limit],
                    }
                )

            """if setup:
                self.max_parallel_initial_objective_value = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf()
                print(f"objective value with max parallelism: {self.max_parallel_initial_objective_value}")
            elif iteration_count == 0:
                self.hw.circuit_model.tech_model.max_speedup_factor = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() / self.max_parallel_initial_objective_value
                self.hw.circuit_model.tech_model.max_area_increase_factor = self.max_dsp / self.cur_dsp_usage
                print(f"max parallelism factor: {self.hw.circuit_model.tech_model.max_speedup_factor}")
                self.hw.circuit_model.tech_model.init_scale_factors(self.hw.circuit_model.tech_model.max_speedup_factor, self.hw.circuit_model.tech_model.max_area_increase_factor)"""

            self.hw.display_objective("after forward pass")

        self.checkpoint_controller.check_end_checkpoint("pd")
        self.obj_over_iterations.append(sim_util.xreplace_safe(self.hw.obj, self.hw.circuit_model.tech_model.base_params.tech_values))

    def parse_catapult_timing(self):
        """
        Parses the Catapult timing report, extracts and schedules the data flow graph (DFG).

        Args:
            None
        Returns:
            None
        """
        # make sure to use parasitics here
        build_dir = os.listdir(f"{self.benchmark_dir}/build")
        schedule_dir = None
        for dir in build_dir:
            if dir.endswith(".v1"):
                schedule_dir = dir
                break
        assert schedule_dir
        schedule_path = f"{self.benchmark_dir}/build/{schedule_dir}"
        schedule_parser = schedule.gnt_schedule_parser(schedule_path, self.module_map, self.hw.circuit_model.circuit_values["latency"], self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.clk_period])
        schedule_parser.parse()
        schedule_parser.convert(memories=self.hw.circuit_model.memories)
        self.hw.inst_name_map = schedule_parser.inst_name_map
        self.hw.scheduled_dfg = schedule_parser.modified_G

        ## write the scheduled dfg to a file
        with open(f"{self.benchmark_dir}/scheduled-dfg-from-catapult.gml", "wb") as f:
            nx.write_gml(self.hw.scheduled_dfg, f)

    def calculate_wire_parasitics(self):
        """
        Prepares the schedule by setting the end node's start time and getting the longest paths.
        Also updates the hardware netlist with wire parasitics and determines the longest paths.

        Args:
            None
        Returns:
            None
        """

        # netlist_dfg = self.hw.scheduled_dfg.copy() 
        # netlist_dfg.remove_node("end")
        # self.hw.netlist = netlist_dfg

        # update netlist and scheduled dfg with wire parasitics
        run_openroad = (self.checkpoint_controller.check_checkpoint("pd", self.iteration_count) and not self.max_rsc_reached) and not os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp")
        if os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp"):
            shutil.rmtree(f"{self.tmp_dir}/pd", ignore_errors=True)
            shutil.copytree(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp", f"{self.tmp_dir}/pd")
            logger.info(f"copied pd_{self.cur_dsp_usage}_dsp to pd, will use results from previous run")
        if not self.checkpoint_controller.check_checkpoint("pd", self.iteration_count):
            logger.info(f"will not run openroad because of checkpoint, will use results from previous run")
        self.hw.get_wire_parasitics(self.openroad_testfile, self.parasitics, self.benchmark_name, run_openroad, self.cfg["args"]["area"])

        if self.cfg["args"]["hls_tool"] == "catapult":
            # set end node's start time to longest path length
            self.hw.scheduled_dfg.nodes["end"]["start_time"] = nx.dag_longest_path_length(self.hw.scheduled_dfg)

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
                else:
                    d[key.name] = float(self.hw.circuit_model.tech_model.base_params.tech_values[key])
            f.write(yaml.dump(d))

    def symbolic_conversion(self):
        """
        Runs symbolic cacti and saves initial values of symbolic parameters.

        Returns:
            dict: Dictionary of symbolic parameters and their initial values.
        """
        base_cache_cfg = "cfg/base_cache.cfg"
        mem_cache_cfg = "cfg/mem_cache.cfg"
        existing_memories = {}
        for memory in self.hw.circuit_model.memories:
            data_tuple = tuple(self.hw.circuit_model.memories[memory])
            if data_tuple in existing_memories:
                logger.info(f"reusing symbolic cacti values of {existing_memories[data_tuple]} for {memory}")
                mem_type = self.hw.circuit_model.memories[memory]["type"]
                if mem_type == "Mem":
                    self.hw.circuit_model.symbolic_mem[memory] = self.hw.circuit_model.symbolic_mem[existing_memories[data_tuple]]
                else:
                    self.hw.circuit_model.symbolic_buf[memory] = self.hw.circuit_model.symbolic_buf[existing_memories[data_tuple]]
            else:
                opt_vals = {
                    "ndwl": self.hw.circuit_model.memories[memory]["Ndwl"],
                    "ndbl": self.hw.circuit_model.memories[memory]["Ndbl"],
                    "nspd": self.hw.circuit_model.memories[memory]["Nspd"],
                    "ndcm": self.hw.circuit_model.memories[memory]["Ndcm"],
                    "ndsam1": self.hw.circuit_model.memories[memory]["Ndsam_level_1"],
                    "ndsam2": self.hw.circuit_model.memories[memory]["Ndsam_level_2"],
                    "repeater_spacing": self.hw.circuit_model.memories[memory]["Repeater spacing"],
                    "repeater_size": self.hw.circuit_model.memories[memory]["Repeater size"],
                }
                logger.info(f"memory vals: {self.hw.circuit_model.memories[memory]}")
                mem_type = self.hw.circuit_model.memories[memory]["type"]
                logger.info(f"generating symbolic cacti for {memory} of type {mem_type}")
                # generate mem or buf depending on type of memory
                if mem_type == "Mem":
                    self.hw.circuit_model.symbolic_mem[memory] = cacti_util.gen_symbolic("Mem", mem_cache_cfg, opt_vals, use_piecewise=False)
                else:
                    self.hw.circuit_model.symbolic_buf[memory] = cacti_util.gen_symbolic("Buf", base_cache_cfg, opt_vals, use_piecewise=False)
                existing_memories[data_tuple] = memory
        self.hw.save_symbolic_memories()
        self.hw.calculate_objective()


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
        self.symbolic_conversion()
        self.hw.display_objective("after symbolic conversion")

        stdout = sys.stdout
        with open(f"{self.tmp_dir}/ipopt_out.txt", "w") as sys.stdout:
            lag_factor, error = self.opt.optimize("ipopt", improvement=self.inverse_pass_improvement)
            self.inverse_pass_lag_factor *= lag_factor
        sys.stdout = stdout

        self.opt.evaluate_constraints(self.hw.circuit_model.tech_model.constraints, "after optimization")
        
        if self.hls_tool == "vitis":
            # need to update the tech_value for final node arrival time after optimization
            self.hw.calculate_execution_time_vitis(self.hw.top_block_name)

        self.write_back_params()

        self.hw.display_objective("after inverse pass")

        self.obj_over_iterations.append(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
        self.lag_factor_over_iterations.append(self.inverse_pass_lag_factor)

    def log_all_to_file(self, iter_number):
        wire_lengths={}
        for edge in self.hw.circuit_model.edge_to_nets:
            wire_lengths[edge] = self.hw.circuit_model.wire_length(edge)
        self.wire_lengths_over_iterations.append(wire_lengths)
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
        if not os.path.exists(f"{self.tmp_dir}/pd_{self.cur_dsp_usage}_dsp"):
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

    def save_dat(self):
        # Save tech node info to another file prefixed by prev_ so we can restore
        org_dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(org_dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"
        with open(org_dat_file, "r") as src_file, open(prev_dat_file, "w") as dest_file:
            for line in src_file:
                dest_file.write(line)

    def restore_dat(self):
        dat_file = self.hw.cacti_dat_file
        tech_nm = os.path.basename(dat_file)
        tech_nm = os.path.splitext(tech_nm)[0]

        prev_dat_file = f"src/cacti/tech_params/prev_{tech_nm}.dat"

        if os.path.exists(prev_dat_file):
            with open(prev_dat_file, "r") as src_file, open(dat_file, "w") as dest_file:
                for line in src_file:
                    dest_file.write(line)
            os.remove(prev_dat_file)

    def cleanup(self):
        self.restore_dat()

    def end_of_run_plots(self, obj_over_iterations, lag_factor_over_iterations, params_over_iterations, wire_lengths_over_iterations):
        assert len(params_over_iterations) > 1 
        obj = "Energy Delay Product"
        units = "nJ*ns"
        if self.obj_fn == "energy":
            obj = "Energy"
            units = "nJ"
        elif self.obj_fn == "delay":
            obj = "Delay"
            units = "ns"
        trend_plotter = trend_plot.TrendPlot(self, params_over_iterations, obj_over_iterations, lag_factor_over_iterations, wire_lengths_over_iterations, self.save_dir + "/figs", obj, units, self.obj_fn)
        logger.info(f"plotting wire lengths over generations")
        trend_plotter.plot_wire_lengths_over_generations()
        logger.info(f"plotting params over generations")
        trend_plotter.plot_params_over_generations()
        logger.info(f"plotting obj over generations")
        trend_plotter.plot_obj_over_generations()
        logger.info(f"plotting lag factor over generations")
        trend_plotter.plot_lag_factor_over_generations()

    def setup(self):
        if not os.path.exists(self.benchmark_setup_dir):
            shutil.copytree(self.benchmark, self.benchmark_setup_dir)
        assert os.path.exists(self.config_json_path_scalehls)
        shutil.copy(self.config_json_path_scalehls, self.config_json_path)
        self.forward_pass(0, save_dir=self.benchmark_setup_dir, setup=True)

    def execute(self, num_iters):
        self.iteration_count = 0
        self.setup()
        self.checkpoint_controller.check_end_checkpoint("setup")
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
            self.end_of_run_plots(self.obj_over_iterations, self.lag_factor_over_iterations, self.params_over_iterations, self.wire_lengths_over_iterations)
            logger.info(f"current dsp usage: {self.cur_dsp_usage}, max dsp: {self.max_dsp}")
            if self.cur_dsp_usage == self.max_dsp:
                logger.info("Resource constraints have been reached, will skip forward pass steps from now on.")
                self.max_rsc_reached = True        

        # cleanup
        self.cleanup()

        
def main(args):
    codesign_module = Codesign(
        args
    )
    try:
        codesign_module.execute(codesign_module.cfg["args"]["num_iters"])
    finally:
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        
        codesign_module.end_of_run_plots(codesign_module.obj_over_iterations, codesign_module.lag_factor_over_iterations, codesign_module.params_over_iterations, codesign_module.wire_lengths_over_iterations)
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
    parser.add_argument("--stop_at_checkpoint", type=str, help="checkpoint step to stop at (will complete this step and then stop)")
    parser.add_argument("--workload_size", type=int, help="workload size to use, such as the dimension of the matrix for gemm. Only applies to certain benchmarks")
    parser.add_argument("--opt_pipeline", type=str, help="optimization pipeline to use for inverse pass")
    args = parser.parse_args()

    main(args)
