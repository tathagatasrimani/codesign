import argparse
import os
import yaml
import sys
import datetime
import logging
import shutil
import subprocess

import networkx as nx

logger = logging.getLogger("codesign")

from . import cacti_util
from . import coefficients
from . import sim_util
from . import hardwareModel
from . import hw_symbols
from . import optimize
from . import simulate 
from . import symbolic_simulate
from . import schedule
from . import memory
from . import ccore_update

class Codesign:
    def __init__(self, benchmark_name, save_dir, openroad_testfile, parasitics, no_cacti):
        self.benchmark = f"src/benchmarks/{benchmark_name}"
        self.benchmark_name = benchmark_name
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
            f.write(f"Benchmark: {self.benchmark}\n")
        if os.path.exists("src/tmp"):
            shutil.rmtree("src/tmp")
        os.mkdir("src/tmp")

        #shutil.copytree(self.benchmark, f"{self.save_dir}/benchmark")

        logging.basicConfig(filename=f"{self.save_dir}/codesign.log", level=logging.INFO)

        self.forward_edp = 0
        self.inverse_edp = 0
        self.tech_params = None
        self.initial_tech_params = None
        self.openroad_testfile = openroad_testfile
        self.longest_paths = []
        self.scheduled_dfg = None
        self.parasitics = parasitics
        self.run_cacti = not no_cacti
        self.hw = hardwareModel.HardwareModel()
        self.sim = simulate.ConcreteSimulator()
        self.symbolic_sim = symbolic_simulate.SymbolicSimulator()

        # starting point set by the config we load into the HW model
        coefficients.create_and_save_coefficients([self.hw.transistor_size])

        rcs = self.hw.get_optimization_params_from_tech_params()
        with open("src/tmp/rcs_0.yaml", "w") as f:
            f.write(yaml.dump(rcs))
        initial_tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs)

        self.set_technology_parameters(initial_tech_params)

        self.hw.write_technology_parameters(self.save_dir+"/initial_tech_params.yaml")

    def set_technology_parameters(self, tech_params):
        if self.initial_tech_params == None:
            self.initial_tech_params = tech_params
        self.tech_params = tech_params

    def log_forward_tech_params(self):
        logger.info(f"latency (ns):\n {self.hw.latency}")

        logger.info(f"active power (nW):\n {self.hw.dynamic_power}")

        logger.info(f"active energy (nW):\n {self.hw.dynamic_energy}")

        logger.info(f"passive power (nW):\n {self.hw.leakage_power}")

        #logger.info(f"compute operation totals in fw pass:\n {self.hw.compute_operation_totals}")

    def log_inverse_tech_params(self, cacti_subs):
        logger.info("symbolic active power (W)\n")
        for elem in hw_symbols.symbolic_power_active:
            if elem not in ["MainMem", "Buf", "OffChipIO"]:
                logger.info(f"{elem}: {hw_symbols.symbolic_power_active[elem]().xreplace(self.tech_params)}")

        logger.info("\nsymbolic passive power (W)\n")
        for elem in hw_symbols.symbolic_power_passive:
            if elem not in ["MainMem", "Buf", "OffChipIO"]:
                logger.info(f"{elem}: {hw_symbols.symbolic_power_passive[elem]().xreplace(self.tech_params)}")
        
        logger.info("\nsymbolic latency (ns)\n")
        for elem in hw_symbols.symbolic_latency_wc:
            if elem not in ["MainMem", "Buf", "OffChipIO"]:
                logger.info(f"{elem}: {hw_symbols.symbolic_latency_wc[elem]().xreplace(self.tech_params)}")

        logger.info("\ncacti values\n")
        for elem in cacti_subs:
            logger.info(f"{elem}: {self.tech_params[elem]}")


    def run_catapult(self):

        os.chdir("src/tmp/benchmark")
        clk_period = (1 / self.hw.f) * 1e9 # ns
        # set correct clk period
        sim_util.change_clk_period_in_script("scripts/common.tcl", clk_period)

        p = subprocess.run(["make", "clean"], capture_output=True, text=True)
        cmd = ["make", "build_design"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"first catapult run output: {p.stdout}")
        if p.returncode != 0:
            raise Exception(p.stderr)
        os.chdir("../../..")
        self.hw.memories = memory.customize_catapult_memories(f"src/tmp/benchmark/memories.rpt", self.benchmark_name, self.hw)
        os.chdir("src/tmp/benchmark")
        p = subprocess.run(["make", "clean"], capture_output=True, text=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"custom memory catapult run output: {p.stdout}")
        if p.returncode != 0:
            raise Exception(p.stderr)
        os.chdir("../../..")

        # TODO: extract hw netlist
        #self.hw.netlist = None

    def forward_pass(self):
        print("\nRunning Forward Pass")
        logger.info("Running Forward Pass")

        if os.path.exists("src/tmp/benchmark"):
            shutil.rmtree("src/tmp/benchmark")
        shutil.copytree(self.benchmark, "src/tmp/benchmark")
        shutil.copytree("src/ccores_base", "src/tmp/benchmark/src/ccores")

        # update delay and area of ccores
        ccore_update.update_ccores(self.hw.area, self.hw.latency)

        # run catapult with custom memory configurations
        self.run_catapult()

        # calculate wire parasitics with hardware netlist
        #self.hw.get_wire_parasitics(self.openroad_testfile, self.parasitics)

        # parse catapult timing report, which saves critical paths
        self.parse_catapult_timing()

    def parse_catapult_timing(self):
        # make sure to use parasitics here
        build_dir = os.listdir("src/tmp/benchmark/build")
        schedule_dir = None
        for dir in build_dir:
            if dir.endswith(".v1"):
                schedule_dir = dir
                break
        assert schedule_dir
        schedule_path = f"src/tmp/benchmark/build/{schedule_dir}"
        module_map = {
            "add": "Add",
            "mult": "Mult",
            "ccs_ram_sync_1R1W_rwport": "Buf",
            "ccs_ram_sync_1R1W_rport": "Buf",
            "nop": "nop"
        }
        print(module_map)
        for unit in self.hw.area.keys():
            module_map[unit.lower()] = unit
        print(module_map)
        module_map = {
            "add": "Add",
            "mult": "Mult",
            "ccs_ram_sync_1R1W_rwport": "Buf",
            "ccs_ram_sync_1R1W_rport": "Buf",
            "nop": "nop"
        }
        schedule_parser = schedule.gnt_schedule_parser(schedule_path, module_map)
        schedule_parser.parse()
        schedule_parser.convert()
        self.scheduled_dfg = schedule_parser.modified_G
        self.scheduled_dfg.nodes["end"]["start_time"] = nx.dag_longest_path_length(self.scheduled_dfg)

        self.longest_paths = schedule.get_longest_paths(self.scheduled_dfg)
        netlist_dfg = self.scheduled_dfg.copy()
        netlist_dfg.remove_node("end")
        self.hw.netlist = netlist_dfg

        # schedule operations with parasitic delays
        #schedule.sdc_schedule(self.operations)
    
    def parse_output(self, f):
        lines = f.readlines()
        mapping = {}
        i = 0
        while lines[i][0] != "x":
            i += 1
        while lines[i][0] == "x":
            mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
                hw_symbols.symbol_table[lines[i].split(" ")[-1][:-1]]
            )
            i += 1
        while i < len(lines) and lines[i].find("x") != 4:
            i += 1
        i += 2
        for _ in range(len(mapping)):
            key = lines[i].split(":")[0].lstrip().rstrip()
            value = float(lines[i].split(":")[2][1:-1])
            self.tech_params[mapping[key]] = (
                value  # just know that self.tech_params contains all dat
            )
            i += 1

    def write_back_rcs(self, rcs_path="src/params/rcs_current.yaml"):
        rcs = {"Reff": {}, "Ceff": {}, "Cacti": {}, "Cacti_IO": {}, "other": {}}
        for elem in self.tech_params:
            if (
                elem.name == "f"
                or elem.name == "V_dd"
                or elem.name.startswith("Mem")
                or elem.name.startswith("Buf")
                or elem.name.startswith("OffChipIO")
            ):
                rcs["other"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            elif elem.name in hw_symbols.cacti_tech_params:
                rcs["Cacti"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            elif elem.name in hw_symbols.cacti_io_tech_params:
                rcs["Cacti_IO"][elem.name] = self.tech_params[
                    hw_symbols.symbol_table[elem.name]
                ]
            else:
                rcs[elem.name[: elem.name.find("_")]][
                    elem.name[elem.name.find("_") + 1 :]
                ] = self.tech_params[elem]
        with open(rcs_path, "w") as f:
            f.write(yaml.dump(rcs))

    def inverse_pass(self):
        print("\nRunning Inverse Pass")
        logger.info("Running Inverse Pass")
        base_cache_cfg = "cfg/base_cache.cfg"
        mem_cache_cfg = "cfg/mem_cache.cfg"
        existing_memories = {}
        for memory in self.hw.memories:
            data_tuple = tuple(self.hw.memories[memory])
            if data_tuple in existing_memories:
                logger.info(f"reusing symbolic cacti values of {existing_memories[data_tuple]} for {memory}")
                mem_type = self.hw.memories[memory]["type"]
                if mem_type == "Mem":
                    self.hw.symbolic_mem[memory] = self.hw.symbolic_mem[existing_memories[data_tuple]]
                else:
                    self.hw.symbolic_buf[memory] = self.hw.symbolic_buf[existing_memories[data_tuple]]
            else:
                opt_vals = {
                    "ndwl": self.hw.memories[memory]["Ndwl"],
                    "ndbl": self.hw.memories[memory]["Ndbl"],
                    "nspd": self.hw.memories[memory]["Nspd"],
                    "ndcm": self.hw.memories[memory]["Ndcm"],
                    "ndsam1": self.hw.memories[memory]["Ndsam_level_1"],
                    "ndsam2": self.hw.memories[memory]["Ndsam_level_2"],
                    "repeater_spacing": self.hw.memories[memory]["Repeater spacing"],
                    "repeater_size": self.hw.memories[memory]["Repeater size"],
                }
                logger.info(f"memory vals: {self.hw.memories[memory]}")
                mem_type = self.hw.memories[memory]["type"]
                logger.info(f"generating symbolic cacti for {memory} of type {mem_type}")
                # generate mem or buf depending on type of memory
                if mem_type == "Mem":
                    self.hw.symbolic_mem[memory] = cacti_util.gen_symbolic("Mem", mem_cache_cfg, opt_vals, use_piecewise=False)
                else:
                    self.hw.symbolic_buf[memory] = cacti_util.gen_symbolic("Buf", base_cache_cfg, opt_vals, use_piecewise=False)
                existing_memories[data_tuple] = memory

        cacti_subs = self.symbolic_sim.calculate_edp(self.hw, self.longest_paths, self.scheduled_dfg)

        self.symbolic_sim.save_edp_to_file()

        for cacti_var in cacti_subs:
            if cacti_subs[cacti_var] == 0:
                self.tech_params[cacti_var] = 0
            else:   
                self.tech_params[cacti_var] = cacti_subs[cacti_var].xreplace(self.tech_params).evalf()
        
        for mem in hw_symbols.OffChipIOL:
            self.tech_params[hw_symbols.OffChipIOL[mem]] = self.hw.latency["OffChipIO"]

        self.log_inverse_tech_params(cacti_subs)

        self.inverse_edp = self.symbolic_sim.edp.xreplace(self.tech_params).evalf()
        inverse_exec_time = self.symbolic_sim.execution_time.xreplace(self.tech_params).evalf()
        logger.info(f"execution time: {self.symbolic_sim.execution_time}")
        active_energy = self.symbolic_sim.total_active_energy.xreplace(self.tech_params).evalf()
        logger.info(f"active energy: {self.symbolic_sim.total_active_energy}")
        passive_energy = self.symbolic_sim.total_passive_energy.xreplace(self.tech_params).evalf()
        logger.info(f"passive energy: {self.symbolic_sim.total_passive_energy}")

        # substitute cacti expressions into edp expression
        self.symbolic_sim.edp = self.symbolic_sim.edp.xreplace(cacti_subs)

        assert len(self.inverse_edp.free_symbols) == 0

        print(
            f"Initial EDP: {self.inverse_edp} E-18 Js. Active Energy: {active_energy} nJ. Passive Energy: {passive_energy} nJ. Execution time: {inverse_exec_time} ns"
        )

        stdout = sys.stdout
        with open("src/tmp/ipopt_out.txt", "w") as sys.stdout:
            optimize.optimize(self.tech_params, self.symbolic_sim.edp, "ipopt", cacti_subs)
        sys.stdout = stdout
        f = open("src/tmp/ipopt_out.txt", "r")
        self.parse_output(f)
        # update cacti subs variables
        for cacti_expr in cacti_subs:
            if cacti_subs[cacti_expr] == 0:
                self.tech_params[cacti_expr] = 0
            else:
                self.tech_params[cacti_expr] = float(cacti_subs[cacti_expr].xreplace(self.tech_params).evalf())

        self.write_back_rcs()

        self.inverse_edp = self.symbolic_sim.edp.xreplace(self.tech_params).evalf()
        total_active_energy = (self.symbolic_sim.total_active_energy).xreplace(
            self.tech_params
        ).evalf()
        total_passive_energy = (self.symbolic_sim.total_passive_energy).xreplace(
            self.tech_params
        ).evalf()
        execution_time = self.symbolic_sim.execution_time.xreplace(self.tech_params).evalf()

        print(
            f"Final EDP  : {self.inverse_edp} E-18 Js. Active Energy: {total_active_energy} nJ. Passive Energy: {total_passive_energy} nJ. Execution time: {execution_time} ns"
        )

    def log_all_to_file(self, iter_number):
        with open(f"{self.save_dir}/results.txt", "a") as f:
            f.write(f"{iter_number}\n")
            f.write(f"Forward EDP: {self.forward_edp}\n")
            f.write(f"Inverse EDP: {self.inverse_edp}\n")
        nx.write_gml(
            self.hw.netlist,
            f"{self.save_dir}/netlist_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        nx.write_gml(
            self.scheduled_dfg,
            f"{self.save_dir}/schedule_{iter_number}.gml",
            stringizer=lambda x: str(x),
        )
        self.write_back_rcs(f"{self.save_dir}/rcs_{iter_number}.yaml")
        shutil.copy(
            "src/tmp/symbolic_edp.txt",
            f"{self.save_dir}/symbolic_edp_{iter_number}.txt",
        )
        shutil.copy("src/tmp/ipopt_out.txt", f"{self.save_dir}/ipopt_{iter_number}.txt")
        shutil.copy(
            "src/tmp/solver_out.txt", f"{self.save_dir}/solver_{iter_number}.txt"
        )
        """for mem in self.hw.memories:
            shutil.copy(
                f"src/tmp/cacti_exprs_{mem}.txt", f"{self.save_dir}/cacti_exprs_{mem}_{iter_number}.txt"
            )"""
        #TODO: copy cacti expressions to file, read yaml file from notebook, call sim util fn to get xreplace structure
        #TODO: fw pass save cacti params of interest, with logger unique starting string, then write parsing script in notebook to look at them
        # save latency, power, and tech params
        self.hw.write_technology_parameters(
            f"{self.save_dir}/tech_params_{iter_number}.yaml"
        )

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
        shutil.copy("src/tmp/rcs_0.yaml", "src/checkpoint/rcs_0.yaml")
        if self.scheduled_dfg:
            nx.write_gml(self.scheduled_dfg, "src/checkpoint/schedule.gml")

    def execute(self, num_iters):
        i = 0
        while i < num_iters:
            self.forward_pass()
            self.log_forward_tech_params()
            self.inverse_pass()
            self.hw.update_technology_parameters()
            self.log_all_to_file(i)
            self.hw.reset_state()
            i += 1

        # cleanup
        self.cleanup()

        
def main(args):
    codesign_module = Codesign(
        args.benchmark,
        args.savedir,
        args.openroad_testfile,
        args.parasitics,
        args.debug_no_cacti,
    )
    try:
        codesign_module.execute(args.num_iters)
    except Exception as e:
        if args.checkpoint:
            codesign_module.save_checkpoint()
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
    parser.add_argument("-c", "--checkpoint", type=bool, default=False, help="save a design checkpoint upon exit")
    args = parser.parse_args()
    print(
        f"args: benchmark: {args.benchmark}, parasitics: {args.parasitics}, num iterations: {args.num_iters}, checkpointing: {args.checkpoint}"
    )

    main(args)
