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
from src.forward_pass.vitis_merge_netlists import merge_netlists_vitis
from src.forward_pass.vitis_create_cdfg import create_cdfg_vitis
from src import trend_plot

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


        self.hls_tool = self.cfg["args"]["hls_tool"]
        self.benchmark = f"src/benchmarks/{self.hls_tool}/{self.cfg['args']['benchmark']}"
        self.benchmark_name = self.cfg["args"]["benchmark"]
        self.benchmark_dir = "src/tmp/benchmark"
        self.benchmark_setup_dir = "src/tmp/benchmark_setup"
        self.save_dir = os.path.join(
            self.cfg["args"]["savedir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        if os.path.exists("src/tmp"):
            shutil.rmtree("src/tmp")
        os.mkdir("src/tmp")

        logging.basicConfig(filename=f"{self.save_dir}/codesign.log", level=logging.INFO)
        logger.info(f"args: {self.cfg['args']}")

        self.forward_obj = 0
        self.inverse_obj = 0
        self.openroad_testfile = self.cfg["args"]["openroad_testfile"]
        self.parasitics = self.cfg["args"]["parasitics"]
        self.run_cacti = not self.cfg["args"]["debug_no_cacti"]
        self.no_memory = self.cfg["args"]["no_memory"]
        self.hw = hardwareModel.HardwareModel(self.cfg["args"])
        self.opt = optimize.Optimizer(self.hw)
        self.module_map = {}
        self.inverse_pass_improvement = self.cfg["args"]["inverse_pass_improvement"]
        self.obj_fn = self.cfg["args"]["obj"]
        self.inverse_pass_lag_factor = 1

        self.params_over_iterations = [copy.copy(self.hw.circuit_model.tech_model.base_params.tech_values)]
        self.obj_over_iterations = []
        self.lag_factor_over_iterations = [1.0]
        self.max_unroll = 64

        self.save_dat()

        #with open("src/tmp/tech_params_0.yaml", "w") as f:
        #    f.write(yaml.dump(self.hw.circuit_model.tech_model.base_params.tech_values))

        self.hw.write_technology_parameters(self.save_dir+"/initial_tech_params.yaml")

        self.iteration_count = 0

        self.checkpoint_controller = checkpoint_controller.CheckpointController(self.cfg)
        if not self.check_checkpoint("scalehls"):
            self.checkpoint_controller.load_checkpoint(self.cfg["args"]["checkpoint_step"])

    # only skipping steps in first iteration. This function returns True if we should not skip this step.
    def check_checkpoint(self, checkpoint_step):
        if checkpoint_controller.checkpoint_map[checkpoint_step] > checkpoint_controller.checkpoint_map[self.cfg["args"]["checkpoint_step"]] or self.iteration_count != 0:
            return True
        else:
            return False

    # return True if we should stop the program and save the checkpoint
    def check_save_checkpoint(self, checkpoint_step):
        if checkpoint_controller.checkpoint_map[checkpoint_step] == checkpoint_controller.checkpoint_map[self.cfg["args"]["checkpoint_save_step"]]:
            return True
        else:
            return False

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
        with open(f"ScaleHLS-HIDA/test/Transforms/Directive/config.json", "r") as f:
            config = json.load(f)
        dsp_multiplier = 1e16 # TODO replace with more comprehensive model
        bram_multiplier = 1e16 # TODO replace with more comprehensive model
        if unlimited:
            config["dsp"] = 1000000
            config["bram"] = 1000000
        else:
            config["dsp"] = int(self.cfg["args"]["area"] / (self.hw.circuit_model.tech_model.param_db["A_gate"].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() * dsp_multiplier))
            config["bram"] = int(self.cfg["args"]["area"] / (self.hw.circuit_model.tech_model.param_db["A_gate"].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() * bram_multiplier))
            # setting cur_dsp_usage here instead of with parse_dsp_usage after running scaleHLS
            # because I observed that for a small amount of resources, scaleHLS won't generate the csv file that we need to parse
            self.cur_dsp_usage = config["dsp"] 

        # I don't think "100MHz" has any meaning because scaleHLS should be agnostic to frequency
        config["100MHz"]["fadd"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Add"] / self.clk_period)
        config["100MHz"]["fmul"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["Mult"] / self.clk_period)
        config["100MHz"]["fdiv"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["FloorDiv"] / self.clk_period)
        config["100MHz"]["fcmp"] = math.ceil(self.hw.circuit_model.circuit_values["latency"]["GtE"] / self.clk_period)
        with open(f"ScaleHLS-HIDA/test/Transforms/Directive/config.json", "w") as f:
            json.dump(config, f)

    def run_scalehls(self, save_dir, opt_cmd,setup=False):
        """
        Runs ScaleHLS synthesis tool in a different environment with modified PATH and PYTHONPATH.
        Updates the memory configuration and logs the output.
        Handles directory changes and cleans up temporary files.

        Args:
            None
        Returns:
            None
        """
        self.set_resource_constraint_scalehls(unlimited=setup)
            
        ## get CWD
        cwd = os.getcwd()
        print(f"Running scaleHLS in {cwd}")

        if self.cfg["args"]["pytorch"]:
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
            cmd = [
                'bash', '-c',
                f'''
                cd {cwd}
                cd ScaleHLS-HIDA/
                export PATH=$PATH:$PWD/build/bin:$PWD/polygeist/build/bin
                export PYTHONPATH=$PYTHONPATH:$PWD/build/tools/scalehls/python_packages/scalehls_core
                source mlir_venv/bin/activate
                cd {os.path.join(os.path.dirname(__file__), "..", save_dir)}
                cgeist {self.benchmark_name}.c -function={self.benchmark_name} -S -memref-fullrank -raise-scf-to-affine > {self.benchmark_name}.mlir
                {opt_cmd} | scalehls-translate -scalehls-emit-hlscpp > {os.path.join(os.path.dirname(__file__), "..", save_dir)}/{self.benchmark_name}.cpp
                deactivate
                pwd
                '''
            ]

        with open(f"src/tmp/scalehls_out.log", "w") as outfile:
            p = subprocess.Popen(
                cmd,
                stdout=outfile,
                stderr=subprocess.STDOUT,
                env={}  # clean environment
            )
            p.wait()
        with open(f"src/tmp/scalehls_out.log", "r") as f:
            logger.info(f"scaleHLS output:\n{f.read()}")

        if p.returncode != 0:
            raise Exception(f"scaleHLS failed with error: {p.stderr}")

    def parse_design_space_for_mlir(self, read_dir):
        df = pd.read_csv(f"{read_dir}/{self.benchmark_name}_space.csv")
        for i in range(len(df['dsp'].values)):
            if df['dsp'].values[i] <= self.cur_dsp_usage:
                return f"{read_dir}/{self.benchmark_name}_pareto_{i}.mlir", i
        raise Exception(f"No Pareto solution found for {self.benchmark_name} with dsp usage {self.cur_dsp_usage}")

    def parse_dsp_usage_and_latency(self, mlir_idx):
        df = pd.read_csv(f'''{os.path.join(os.path.dirname(__file__), "..", "src/tmp/benchmark_setup")}/{self.benchmark_name}_space.csv''')
        logger.info(f"dsp usage: {df['dsp'].values[mlir_idx]}, latency: {df['cycle'].values[mlir_idx]}")
        return df['dsp'].values[mlir_idx], df['cycle'].values[mlir_idx]

    def parse_vitis_data(self, save_dir):

        # TODO: Parse the Vitis netlist and write it to netlist-from-vitis.gml and cdfg-from-vitis.gml

        ## print the cwd
        print(f"Current working directory in vitis parse data: {os.getcwd()}")

        if self.no_memory:
            allowed_functions_netlist = {"Add", "Mult"}
            allowed_functions_schedule = {"Add", "Mult", "Call", "II"}
        else:
            allowed_functions_netlist = {"Add", "Mult", "Buf", "MainMem"}
            allowed_functions_schedule = {"Add", "Mult", "Call", "II", "Buf", "MainMem"}

        parse_results_dir = f"{save_dir}/parse_results"

        if self.check_checkpoint("netlist"):
            ## Do preprocessing to the vitis data for the next scripts
            parse_verbose_rpt(f"{save_dir}/{self.benchmark_name}/solution1/.autopilot/db", parse_results_dir)

            ## Create the netlist
            create_vitis_netlist(parse_results_dir)

            ## Create the CDFGs for each FSM
            create_cdfg_vitis(parse_results_dir)

            ## Create the mapping from CDFG nodes to netlist nodes
            #create_cdfg_to_netlist_mapping_vitis(parse_results_dir)

            ## Merge the CDFGs recursivley through the FSM module hierarchy to produce overall CDFG
            #merge_cdfgs_vitis(parse_results_dir, self.vitis_top_function)

            ## Merge the netlists recursivley through the module hierarchy to produce overall netlist
            merge_netlists_vitis(parse_results_dir, self.vitis_top_function, allowed_functions_netlist)
        else:
            logger.info("Skipping Vitis netlist parsing")
        if self.check_save_checkpoint("netlist"):
            raise Exception("Finished Vitis netlist parsing, saving checkpoint")

        if self.check_checkpoint("schedule"):
            schedule_parser = schedule_vitis.vitis_schedule_parser(save_dir, self.benchmark_name, self.vitis_top_function, self.clk_period, allowed_functions_schedule)
            schedule_parser.create_dfgs()
            self.hw.scheduled_dfgs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_standard"] for basic_block_name in schedule_parser.basic_blocks}
            self.hw.loop_1x_graphs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_loop_1x_standard"] for basic_block_name in schedule_parser.basic_blocks if "G_loop_1x" in schedule_parser.basic_blocks[basic_block_name]}
            self.hw.loop_2x_graphs = {basic_block_name: schedule_parser.basic_blocks[basic_block_name]["G_loop_2x_standard"] for basic_block_name in schedule_parser.basic_blocks if "G_loop_2x" in schedule_parser.basic_blocks[basic_block_name]}
            logger.info(f"scheduled dfgs: {self.hw.scheduled_dfgs}")
            logger.info(f"loop 1x graphs: {self.hw.loop_1x_graphs}")
        else:
            for file in os.listdir(parse_results_dir):
                if os.path.isdir(os.path.join(parse_results_dir, file)):
                    self.hw.scheduled_dfgs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_standard.gml")
                    if os.path.exists(f"{parse_results_dir}/{file}/{file}_graph_loop_1x_standard.gml"):
                        self.hw.loop_1x_graphs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_loop_1x_standard.gml")
                        assert os.path.exists(f"{parse_results_dir}/{file}/{file}_graph_loop_2x_standard.gml")
                        self.hw.loop_2x_graphs[file] = nx.read_gml(f"{parse_results_dir}/{file}/{file}_graph_loop_2x_standard.gml")
            logger.info("Skipping Vitis schedule parsing")
        if self.check_save_checkpoint("schedule"):
            raise Exception("Finished Vitis schedule parsing, saving checkpoint")

        print(f"Current working directory at end of vitis parse data: {os.getcwd()}")

        self.hw.netlist = nx.read_gml(f"{parse_results_dir}/{self.vitis_top_function}_full_netlist.gml")

        execution_time = self.hw.calculate_execution_time_vitis(self.vitis_top_function)
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
            opt_cmd = f'''scalehls-opt {self.benchmark_name}.mlir -scalehls-dse-pipeline=\"top-func={self.vitis_top_function} target-spec={os.path.join(os.path.dirname(__file__), "..", "ScaleHLS-HIDA/test/Transforms/Directive/config.json")}\"'''
            mlir_idx = 0
        else:
            mlir_file, mlir_idx = self.parse_design_space_for_mlir(os.path.join(os.path.dirname(__file__), "..", "src/tmp/benchmark_setup"))
            opt_cmd = f"cat {mlir_file}"

        # run scalehls
        if (self.check_checkpoint("scalehls") and not setup) or (self.check_checkpoint("scalehls_unlimited") and setup):
            self.run_scalehls(save_dir, opt_cmd, setup)
        else:
            logger.info("Skipping ScaleHLS")

        # set scale factors if in setup or first iteration
        if self.cfg["args"]["pytorch"]:
            dsp_usage, latency = 1, 1 # TODO replace once pytorch dse working
        else:
            dsp_usage, latency = self.parse_dsp_usage_and_latency(mlir_idx)
        if setup:
            self.max_dsp = dsp_usage
            self.max_latency = latency
        elif iteration_count == 0:
            self.hw.circuit_model.tech_model.max_speedup_factor = latency / self.max_latency
            self.hw.circuit_model.tech_model.max_area_increase_factor = self.max_dsp / dsp_usage
            self.hw.circuit_model.tech_model.init_scale_factors(self.hw.circuit_model.tech_model.max_speedup_factor, self.hw.circuit_model.tech_model.max_area_increase_factor)
        if (self.check_save_checkpoint("scalehls") and not setup) or (self.check_save_checkpoint("scalehls_unlimited") and setup):
            raise Exception("Finished ScaleHLS, saving checkpoint")
        if setup: # setup step ends here, don't need to run rest of forward pass
            return

        os.chdir(os.path.join(os.path.dirname(__file__), "..", save_dir))
        scale_hls_port_fix(f"{self.benchmark_name}.cpp", self.benchmark_name, self.cfg["args"]["pytorch"])
        logger.info(f"Vitis top function: {self.vitis_top_function}")
        command = ["vitis_hls -f tcl_script.tcl"]
        if (self.check_checkpoint("vitis") and not setup) or (self.check_checkpoint("vitis_unlimited") and setup):
            p = subprocess.run(command, shell=True, capture_output=True, text=True)
            logger.info(f"Vitis HLS command output: {p.stdout}")
            if p.returncode != 0:
                logger.error(f"Vitis HLS command failed: {p.stderr}")
                raise Exception(f"Vitis HLS command failed: {p.stderr}")
        else:
            logger.info("Skipping Vitis")
        if (self.check_save_checkpoint("vitis") and not setup) or (self.check_save_checkpoint("vitis_unlimited") and setup):
            raise Exception("Finished Vitis, saving checkpoint")
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
        if iteration_count != 0 or self.cfg["args"]["checkpoint_step"] == "none":
            if os.path.exists(self.benchmark_dir):
                shutil.rmtree(self.benchmark_dir)
            shutil.copytree(self.benchmark, self.benchmark_dir)

        self.clk_period = 1/self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.f] * 1e9 # ns

        if self.hls_tool == "catapult":
            sim_util.change_clk_period_in_script(f"{self.benchmark_dir}/scripts/common.tcl", self.clk_period, self.cfg["args"]["hls_tool"])
            self.catapult_forward_pass()
        else:
            sim_util.change_clk_period_in_script(f"{save_dir}/tcl_script.tcl", self.clk_period, self.cfg["args"]["hls_tool"])
            self.vitis_forward_pass(save_dir=save_dir, iteration_count=iteration_count, setup=setup)
        if setup: return

        # prepare schedule & calculate wire parasitics
        self.prepare_schedule()

        ## create the obj equation 
        self.hw.calculate_objective()

        """if setup:
            self.max_parallel_initial_objective_value = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf()
            print(f"objective value with max parallelism: {self.max_parallel_initial_objective_value}")
        elif iteration_count == 0:
            self.hw.circuit_model.tech_model.max_speedup_factor = self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf() / self.max_parallel_initial_objective_value
            self.hw.circuit_model.tech_model.max_area_increase_factor = self.max_dsp / self.cur_dsp_usage
            print(f"max parallelism factor: {self.hw.circuit_model.tech_model.max_speedup_factor}")
            self.hw.circuit_model.tech_model.init_scale_factors(self.hw.circuit_model.tech_model.max_speedup_factor, self.hw.circuit_model.tech_model.max_area_increase_factor)"""

        self.display_objective("after forward pass")

        self.obj_over_iterations.append(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values).evalf())

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
        schedule_parser = schedule.gnt_schedule_parser(schedule_path, self.module_map, self.hw.circuit_model.circuit_values["latency"], self.hw.circuit_model.tech_model.base_params.tech_values[self.hw.circuit_model.tech_model.base_params.f])
        schedule_parser.parse()
        schedule_parser.convert(memories=self.hw.circuit_model.memories)
        self.hw.inst_name_map = schedule_parser.inst_name_map
        self.hw.scheduled_dfg = schedule_parser.modified_G

        ## write the scheduled dfg to a file
        with open(f"{self.benchmark_dir}/scheduled-dfg-from-catapult.gml", "wb") as f:
            nx.write_gml(self.hw.scheduled_dfg, f)
    
    def prepare_schedule(self):
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
        self.hw.get_wire_parasitics(self.openroad_testfile, self.parasitics, self.benchmark_name)

        if self.cfg["args"]["hls_tool"] == "catapult":
            # set end node's start time to longest path length
            self.hw.scheduled_dfg.nodes["end"]["start_time"] = nx.dag_longest_path_length(self.hw.scheduled_dfg)
    
    def parse_output(self, f):
        """
        Parses the output file from the optimizer in the inverse pass, mapping variable names to
        technology parameters and updating them accordingly.

        Args:
            f (file-like): Opened file object containing the output to parse.

        Returns:
            None
        """
        lines = f.readlines()
        mapping = {}
        max_ind = 0
        i = 0
        while lines[i][0] != "x":
            i += 1
        while lines[i][0] == "x":
            mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
                self.hw.circuit_model.tech_model.base_params.symbol_table[lines[i].split(" ")[-1][:-1]]
            )
            max_ind = int(lines[i][lines[i].find("[") + 1 : lines[i].find("]")])
            i += 1
        while i < len(lines) and lines[i].find("x") != 4:
            i += 1
        i += 2
        #print(f"mapping: {mapping}, max_ind: {max_ind}")
        for _ in range(max_ind):
            key = lines[i].split(":")[0].lstrip().rstrip()
            value = float(lines[i].split(":")[2][1:-1])
            if key in mapping:
                #print(f"key: {key}; mapping: {mapping[key]}; value: {value}")
                self.hw.circuit_model.tech_model.base_params.tech_values[mapping[key]] = (
                    value
                )
            i += 1

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

    
    def display_objective(self, message):
        obj = float(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
        sub_exprs = {}
        for key in self.hw.obj_sub_exprs:
            if not isinstance(self.hw.obj_sub_exprs[key], float):
                sub_exprs[key] = float(self.hw.obj_sub_exprs[key].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
            else:   
                sub_exprs[key] = self.hw.obj_sub_exprs[key]
        print(f"{message}\n {self.obj_fn}: {obj}, sub expressions: {sub_exprs}")


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
        self.display_objective("after symbolic conversion")

        stdout = sys.stdout
        with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
            lag_factor, error = self.opt.optimize("ipopt", improvement=self.inverse_pass_improvement)
            self.inverse_pass_lag_factor *= lag_factor
        sys.stdout = stdout
        f = open("src/tmp/ipopt_out.txt", "r")
        if not error:
            self.parse_output(f)
        
        if self.hls_tool == "vitis":
            # need to update the tech_value for final node arrival time after optimization
            self.hw.calculate_execution_time_vitis(self.hw.top_block_name)

        self.write_back_params()

        self.display_objective("after inverse pass")

        self.obj_over_iterations.append(self.hw.obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
        self.lag_factor_over_iterations.append(self.inverse_pass_lag_factor)

    def log_all_to_file(self, iter_number):
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
        shutil.copy("src/tmp/ipopt_out.txt", f"{self.save_dir}/ipopt_{iter_number}.txt")
        shutil.copy(
            "src/tmp/solver_out.txt", f"{self.save_dir}/solver_{iter_number}.txt"
        )
        """for mem in self.hw.circuit_model.memories:
            shutil.copy(
                f"src/tmp/cacti_exprs_{mem}.txt", f"{self.save_dir}/cacti_exprs_{mem}_{iter_number}.txt"
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

    def end_of_run_plots(self, obj_over_iterations, lag_factor_over_iterations, params_over_iterations):
        assert len(params_over_iterations) > 1 
        obj = "Energy Delay Product"
        units = "nJ*ns"
        if self.obj_fn == "energy":
            obj = "Energy"
            units = "nJ"
        elif self.obj_fn == "delay":
            obj = "Delay"
            units = "ns"
        trend_plotter = trend_plot.TrendPlot(self, params_over_iterations, obj_over_iterations, lag_factor_over_iterations, self.save_dir + "/figs", obj, units, self.obj_fn)
        trend_plotter.plot_params_over_iterations()
        trend_plotter.plot_obj_over_iterations()
        trend_plotter.plot_lag_factor_over_iterations()

    def setup(self):
        if not os.path.exists(self.benchmark_setup_dir):
            shutil.copytree(self.benchmark, self.benchmark_setup_dir)
        self.forward_pass(0, save_dir=self.benchmark_setup_dir, setup=True)

    def execute(self, num_iters):
        self.iteration_count = 0
        self.setup()
        while self.iteration_count < num_iters:
            self.forward_pass(self.iteration_count, self.benchmark_dir)
            self.log_forward_tech_params()
            self.inverse_pass()
            self.hw.circuit_model.update_circuit_values()
            self.log_all_to_file(self.iteration_count)
            self.hw.reset_state()
            self.hw.reset_tech_model()
            self.iteration_count += 1
        self.end_of_run_plots(self.obj_over_iterations, self.lag_factor_over_iterations, self.params_over_iterations)

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
        if codesign_module.cfg["args"]["save_checkpoint"]:
            codesign_module.checkpoint_controller.create_checkpoint()
        codesign_module.end_of_run_plots(codesign_module.obj_over_iterations, codesign_module.lag_factor_over_iterations, codesign_module.params_over_iterations)
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
    parser.add_argument("--checkpoint_step", type=str, help="checkpoint step to resume from")
    parser.add_argument("--save_checkpoint", type=bool, help="save a checkpoint upon exit")
    parser.add_argument("--checkpoint_save_step", type=str, help="checkpoint step to save")
    args = parser.parse_args()

    main(args)
