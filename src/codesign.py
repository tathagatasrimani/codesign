import argparse
import os
import yaml
import sys
import datetime
import logging
import shutil
import subprocess
import copy
import networkx as nx
import matplotlib.pyplot as plt
from src.forward_pass.netlist_parse import parse_yosys_json

logger = logging.getLogger("codesign")

from src import cacti_util
from src import sim_util
from src.hardware_model import hardwareModel
from src.inverse_pass import optimize
from src.forward_pass import schedule
from src import memory
from src.forward_pass import ccore_update
from src import trend_plot

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
        self.benchmark = f"src/benchmarks/{args.benchmark}"
        self.benchmark_name = args.benchmark
        self.save_dir = os.path.join(
            args.savedir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        #shutil.copytree(self.benchmark, f"{self.save_dir}/benchmark")

        logging.basicConfig(filename=f"{self.save_dir}/codesign.log", level=logging.INFO)
        logger.info(f"args: {args}")

        self.forward_edp = 0
        self.inverse_edp = 0
        self.openroad_testfile = args.openroad_testfile
        self.parasitics = args.parasitics
        self.run_cacti = not args.debug_no_cacti
        self.no_memory = args.no_memory
        self.hw = hardwareModel.HardwareModel(args)
        self.opt = optimize.Optimizer(self.hw)
        self.module_map = {}
        self.inverse_pass_improvement = args.inverse_pass_improvement if (hasattr(args, "inverse_pass_improvement") and args.inverse_pass_improvement) else 10
        self.obj_fn = args.obj
        self.inverse_pass_lag_factor = 1

        self.params_over_iterations = []
        self.edp_over_iterations = []
        self.lag_factor_over_iterations = [1.0]
        self.max_unroll = 64

        self.save_dat()

        #with open("src/tmp/tech_params_0.yaml", "w") as f:
        #    f.write(yaml.dump(self.hw.circuit_model.tech_model.base_params.tech_values))

        self.hw.write_technology_parameters(self.save_dir+"/initial_tech_params.yaml")

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

        os.chdir("src/tmp/benchmark")
        clk_period = 300 # ns, TODO: change to actual clk period
        # set correct clk period
        sim_util.change_clk_period_in_script("scripts/common.tcl", clk_period)
        
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
        #     pre_assign_counts = memory.get_pre_assign_counts(f"src/tmp/benchmark/bom.rpt", self.module_map)
        #     logger.info(f"hw.circuit_model.circuit_values: {self.hw.circuit_model.circuit_values}")
        #     self.hw.circuit_model.set_memories(memory.customize_catapult_memories(f"src/tmp/benchmark/memories.rpt", self.benchmark_name, self.hw, pre_assign_counts))
        #     logger.info(f"custom catapult memories: {self.hw.circuit_model.memories}")
        #     os.chdir("src/tmp/benchmark")
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
        cmd = ["yosys", "-p", f"read_verilog src/tmp/benchmark/build/{self.benchmark_name}.v1/rtl.v; hierarchy -top {self.benchmark_name}; proc; write_json src/tmp/benchmark/netlist.json"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if DEBUG_YOSYS:
            logger.info(f"Yosys output: {p.stdout}")
        if p.returncode != 0:
            raise Exception(f"Yosys failed with error: {p.stderr}")

        self.hw.netlist, full_netlist = parse_yosys_json("src/tmp/benchmark/netlist.json", include_memories=(not self.no_memory), top_level_module_type=self.benchmark_name)

        ## write the netlist to a file
        with open("src/tmp/benchmark/netlist-from-catapult.gml", "wb") as f:
            nx.write_gml(self.hw.netlist, f)

        with open("src/tmp/benchmark/full_netlist_debug.gml", "wb") as f:
            nx.write_gml(full_netlist, f)

    def forward_pass(self):
        """
        Executes the forward pass of the codesign process: prepares benchmark, updates ccore
        delays/areas, runs Catapult, calculates wire parasitics, and parses timing reports.

        Args:
            None
        Returns:
            None
        """
        print("\nRunning Forward Pass")
        logger.info("Running Forward Pass")


        ## clear out the existing tmp benchmark directory and copy the benchmark files from the desired benchmark
        if os.path.exists("src/tmp/benchmark"):
            shutil.rmtree("src/tmp/benchmark")
        shutil.copytree(self.benchmark, "src/tmp/benchmark")
        shutil.copytree("src/forward_pass/ccores_base", "src/tmp/benchmark/src/ccores")

        # update delay and area of ccores
        ccore_update.update_ccores(self.hw.circuit_model.circuit_values["area"], self.hw.circuit_model.circuit_values["latency"])

        if self.benchmark_name == "matmult":
            # change the unroll factors in the matmult_basic.cpp file
            with open("src/benchmarks/matmult/src/matmult_basic.cpp", "r") as f:
                lines = f.readlines()
            with open("src/tmp/benchmark/src/matmult_basic.cpp", "w") as f:
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
            with open("src/benchmarks/basic_aes/src/basic_aes.cpp", "r") as f:
                lines = f.readlines()
            with open("src/tmp/benchmark/src/basic_aes.cpp", "w") as f:
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

        

        # prepare schedule & calculate wire parasitics
        self.prepare_schedule()

        ## create the EDP equation 
        self.hw.calculate_objective()

        self.display_objective("after forward pass")

        self.edp_over_iterations.append(self.hw.obj)

    def parse_catapult_timing(self):
        """
        Parses the Catapult timing report, extracts and schedules the data flow graph (DFG).

        Args:
            None
        Returns:
            None
        """
        # make sure to use parasitics here
        build_dir = os.listdir("src/tmp/benchmark/build")
        schedule_dir = None
        for dir in build_dir:
            if dir.endswith(".v1"):
                schedule_dir = dir
                break
        assert schedule_dir
        schedule_path = f"src/tmp/benchmark/build/{schedule_dir}"
        schedule_parser = schedule.gnt_schedule_parser(schedule_path, self.module_map, self.hw.circuit_model.circuit_values["latency"])
        schedule_parser.parse()
        schedule_parser.convert(memories=self.hw.circuit_model.memories)
        self.hw.inst_name_map = schedule_parser.inst_name_map
        self.hw.scheduled_dfg = schedule_parser.modified_G

        ## write the scheduled dfg to a file
        with open("src/tmp/benchmark/scheduled-dfg-from-catapult.gml", "wb") as f:
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

        # set end node's start time to longest path length
        self.hw.scheduled_dfg.nodes["end"]["start_time"] = nx.dag_longest_path_length(self.hw.scheduled_dfg)

        self.hw.longest_paths = schedule.get_longest_paths(self.hw.scheduled_dfg)
    
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
        i = 0
        while lines[i][0] != "x":
            i += 1
        while lines[i][0] == "x":
            mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
                self.hw.circuit_model.tech_model.base_params.symbol_table[lines[i].split(" ")[-1][:-1]]
            )
            i += 1
        while i < len(lines) and lines[i].find("x") != 4:
            i += 1
        i += 2
        for _ in range(len(mapping)):
            key = lines[i].split(":")[0].lstrip().rstrip()
            value = float(lines[i].split(":")[2][1:-1])
            self.hw.circuit_model.tech_model.base_params.tech_values[mapping[key]] = (
                value  # just know that self.hw.circuit_model.tech_model.base_params.tech_values contains all dat
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
        self.hw.calculate_objective(symbolic=True)

    
    def display_objective(self, message,symbolic=False):
        if symbolic:
            obj = float(self.hw.symbolic_obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
            sub_exprs = {}
            for key in self.hw.symbolic_obj_sub_exprs:
                if not isinstance(self.hw.symbolic_obj_sub_exprs[key], float):
                    sub_exprs[key] = float(self.hw.symbolic_obj_sub_exprs[key].xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
                else:   
                    sub_exprs[key] = self.hw.symbolic_obj_sub_exprs[key]
        else:
            obj = self.hw.obj
            sub_exprs = self.hw.obj_sub_exprs
        print(f"{message}\n {self.obj_fn}: {obj}, sub expressions: {sub_exprs}")


    def inverse_pass(self):
        """
        Executes the inverse pass of the codesign process: generates symbolic CACTI values for
        memories, computes EDP (Energy Delay Product), runs optimizer to find better technology
        parameters, and logs results.

        Args:
            None
        Returns:
            None
        """
        print("\nRunning Inverse Pass")
        logger.info("Running Inverse Pass")
        self.symbolic_conversion()
        self.display_objective("after symbolic conversion", symbolic=True)

        stdout = sys.stdout
        with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
            lag_factor, error = self.opt.optimize("ipopt", improvement=self.inverse_pass_improvement)
            self.inverse_pass_lag_factor *= lag_factor
        sys.stdout = stdout
        f = open("src/tmp/ipopt_out.txt", "r")
        if not error:
            self.parse_output(f)

        self.write_back_params()

        self.display_objective("after inverse pass", symbolic=True)

        self.edp_over_iterations.append(self.hw.symbolic_obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values))
        self.lag_factor_over_iterations.append(self.inverse_pass_lag_factor)

    def log_all_to_file(self, iter_number):
        with open(f"{self.save_dir}/results.txt", "a") as f:
            f.write(f"{iter_number}\n")
            f.write(f"Forward EDP: {self.hw.obj}\n")
            f.write(f"Inverse EDP: {self.hw.symbolic_obj.xreplace(self.hw.circuit_model.tech_model.base_params.tech_values)}\n")
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

    def save_checkpoint(self):
        if os.path.exists("src/checkpoint"):
            shutil.rmtree("src/checkpoint")
        os.mkdir("src/checkpoint")
        shutil.copytree("src/tmp/benchmark", "src/checkpoint/benchmark")
        shutil.copy("src/tmp/params_0.yaml", "src/checkpoint/params_0.yaml")
        if self.hw.scheduled_dfg:
            nx.write_gml(self.hw.scheduled_dfg, "src/checkpoint/schedule.gml")

    def end_of_run_plots(self, edp_over_iterations, lag_factor_over_iterations, params_over_iterations):
        assert len(params_over_iterations) > 1 
        obj = "Energy Delay Product"
        units = "nJ*ns"
        if self.obj_fn == "energy":
            obj = "Energy"
            units = "nJ"
        elif self.obj_fn == "delay":
            obj = "Delay"
            units = "ns"
        trend_plotter = trend_plot.TrendPlot(self, params_over_iterations, edp_over_iterations, lag_factor_over_iterations, self.save_dir + "/figs", obj, units, self.obj_fn)
        trend_plotter.plot_params_over_iterations()
        trend_plotter.plot_edp_over_iterations()
        trend_plotter.plot_lag_factor_over_iterations()

    def execute(self, num_iters):
        i = 0
        while i < num_iters:
            self.forward_pass()
            self.log_forward_tech_params()
            self.inverse_pass()
            self.hw.circuit_model.update_circuit_values()
            self.log_all_to_file(i)
            self.hw.reset_state()
            self.hw.reset_tech_model()
            i += 1
        self.end_of_run_plots(self.edp_over_iterations, self.lag_factor_over_iterations, self.params_over_iterations)

        # cleanup
        self.cleanup()

        
def main(args):
    codesign_module = Codesign(
        args
    )
    try:
        codesign_module.execute(args.num_iters)
    finally:
        if args.checkpoint:
            codesign_module.save_checkpoint()
        codesign_module.end_of_run_plots(codesign_module.edp_over_iterations, codesign_module.lag_factor_over_iterations, codesign_module.params_over_iterations)
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
        default=False,
        help="disable memory modeling",
    )
    parser.add_argument('--debug_no_cacti', type=bool, default=False, 
                        help='disable cacti in the first iteration to decrease runtime when debugging')
    parser.add_argument("-c", "--checkpoint", type=bool, default=False, help="save a design checkpoint upon exit")
    parser.add_argument("--logic_node", type=int, default=7, help="logic node size")
    parser.add_argument("--mem_node", type=int, default=32, help="memory node size")
    parser.add_argument("--inverse_pass_improvement", type=float, help="improvement factor for inverse pass")
    parser.add_argument("--tech_node", "-T", type=str, default="default", help="technology node to use as starting point")
    parser.add_argument("--obj", type=str, default="edp", help="objective function")
    parser.add_argument("--model_cfg", type=str, default="default", help="symbolic model configuration")
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, parasitics: {args.parasitics}, num iterations: {args.num_iters}, checkpointing: {args.checkpoint}, area: {args.area}, memory included: {not args.no_memory}, tech node: {args.tech_node}, obj: {args.obj}, model cfg: {args.model_cfg}"
    )

    main(args)
