import logging
import random
import yaml
import time
import numpy as np
import math
import copy
import json
import os

logger = logging.getLogger(__name__)

import networkx as nx
import sympy as sp
from src.hardware_model.base_parameters import base_parameters
from src.hardware_model.circuit_models import circuit_model

from src import sim_util

from src.inverse_pass.constraint import Constraint

from src.hardware_model.tech_models import mvs_general_model
from src.hardware_model.tech_models import sweep_model
from src.hardware_model.tech_models import sweep_brute_force_model
from src.hardware_model.tech_models import sweep_basic_model
from src.hardware_model.tech_models import mvs_self_consistent_model
from src.hardware_model.tech_models import mvs_1_spice_model
from openroad_interface import openroad_run
from openroad_interface import openroad_run_hier
from src.hardware_model.objective_evaluator import ObjectiveEvaluator

import cvxpy as cp

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

class HardwareModel:
    """
    Represents a hardware model with configurable technology and hardware parameters. Provides methods
    to set up the hardware, manage netlists, and extract technology-specific timing and power data for
    optimization and simulation purposes.
    """
    def __init__(self, cfg, codesign_root_dir, tmp_dir):

        args = cfg["args"]

        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir
        with open("src/yaml/model_cfg.yaml", "r") as f:
            model_cfgs = yaml.safe_load(f)

        # model cfg is an extension of its base cfg, can create a tree of configs which need to be merged
        self.model_cfg = sim_util.recursive_cfg_merge(model_cfgs, args["model_cfg"])
        print(f"self.model_cfg: {self.model_cfg}")

        symbol_type = "sympy" if args["solver"] != "cvxpy" else "cvxpy"

        if args["checkpoint_load_dir"] != "none" and os.path.exists(f"{self.tmp_dir}/tech_params_latest.yaml"):
            # when loading from checkpoint, use the latest set of tech param values as a starting point. Override "tech_node" argument.
            with open(f"{self.tmp_dir}/tech_params_latest.yaml", "r") as f:
                tech_params = yaml.safe_load(f)
            self.base_params = base_parameters.BaseParameters(args["tech_node"], symbol_type, tech_params)
        else:
            self.base_params = base_parameters.BaseParameters(args["tech_node"], symbol_type)

        self.reset_tech_model()

        self.netlist = nx.DiGraph()
        # for catapult
        self.scheduled_dfg = nx.DiGraph()
        # for vitis
        self.scheduled_dfgs = {}
        self.loop_1x_graphs = {}
        self.loop_2x_graphs = {}
        self.top_block_name = args["benchmark"] if not args["pytorch"] and self.cfg["args"]["arch_opt_pipeline"] != "streamhls" else "forward"
        self.dataflow_blocks = set()

        self.parasitic_graph = nx.DiGraph()
        self.symbolic_mem = {}
        self.symbolic_buf = {}
        self.memories = []
        self.obj_fn = args["obj"]
        self.obj = 0
        self.obj_sub_exprs = {}
        self.area_constraint = args["area"]
        self.hls_tool = args["hls_tool"]
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.constraints = []
        self.sensitivities = {}

        self.block_vectors = {}        

    def reset_state(self):
        self.symbolic_buf = {}
        self.symbolic_mem = {}
        self.netlist = nx.DiGraph()
        self.memories = []
        self.obj = 0
        self.scheduled_dfg = nx.DiGraph()
        self.scheduled_dfgs = {}
        self.loop_1x_graphs = {}
        self.loop_2x_graphs = {}
        self.parasitic_graph = nx.DiGraph()
        #self.obj_sub_exprs = {}
        self.execution_time = 0
        self.total_passive_energy = 0
        self.total_active_energy = 0
        self.inst_name_map = {}
        self.dfg_to_netlist_map = {}
        self.constraints = []

    def write_technology_parameters(self, filename):
        params = {
            "latency": self.circuit_model.circuit_values["latency"],
            "dynamic_energy": self.circuit_model.circuit_values["dynamic_energy"],
            "passive_power": self.circuit_model.circuit_values["passive_power"],
            "area": self.circuit_model.circuit_values["area"], # TODO: make sure we have this
        }
        with open(filename, "w") as f:
            f.write(yaml.dump(params))

    def reset_tech_model(self):
        if self.model_cfg["model_type"] == "sweep":
            self.tech_model = sweep_model.SweepModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "sweep_brute_force":
            self.tech_model = sweep_brute_force_model.SweepBruteForceModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "sweep_basic":
            self.tech_model = sweep_basic_model.SweepBasicModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "mvs_general":
            self.tech_model = mvs_general_model.MVSGeneralModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "mvs_self_consistent":
            self.tech_model = mvs_self_consistent_model.MVSSelfConsistentModel(self.model_cfg, self.base_params)
        elif self.model_cfg["model_type"] == "mvs_1_spice":
            self.tech_model = mvs_1_spice_model.MVS1SpiceModel(self.model_cfg, self.base_params)
        else:
            raise ValueError(f"Invalid model type: {self.model_cfg['model_type']}")
        self.tech_model.create_constraints(self.model_cfg["scaling_mode"])

        # by convention, we should always access bulk model and base params through circuit model
        self.circuit_model = circuit_model.CircuitModel(self.tech_model, cfg=self.cfg)

    def calculate_minimum_clk_period(self):
        self.minimum_clk_period = sim_util.xreplace_safe(self.circuit_model.DFF_DELAY, self.circuit_model.tech_model.base_params.tech_values)
        for edge in self.circuit_model.edge_to_nets:
            self.minimum_clk_period = max(self.minimum_clk_period, sim_util.xreplace_safe(self.circuit_model.wire_delay(edge) + self.circuit_model.DFF_DELAY, self.circuit_model.tech_model.base_params.tech_values))
        return self.minimum_clk_period
    
    def get_wire_parasitics(self, arg_testfile, arg_parasitics, benchmark_name, run_openroad, area_constraint=None):
        if self.hls_tool == "catapult":
            self.catapult_map_netlist_to_scheduled_dfg(benchmark_name)
        
        start_time = time.time()

        netlist_copy = copy.deepcopy(self.netlist)

        logger.info(f"num nodes in netlist before openroad: {len(netlist_copy.nodes)}")

        L_eff = self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.L]
        logger.info(f"current L_eff for get_wire_parascitics: {L_eff}")

        ## hierarchical openroad run
        if (benchmark_name == "resnet18"):
            hier_open_road_run = openroad_run_hier.OpenRoadRunHier(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=run_openroad, circuit_model=self.circuit_model)

            hls_parse_results_dir = f"benchmark/parse_results"

            self.circuit_model.edge_to_nets = hier_open_road_run.run_hierarchical_openroad(
                netlist_copy,
                arg_testfile,
                arg_parasitics,
                area_constraint,
                L_eff,
                hls_parse_results_dir,
                "forward"
            )

        ## flat openroad run
        else:
            open_road_run = openroad_run.OpenRoadRun(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=run_openroad, circuit_model=self.circuit_model)

            self.circuit_model.edge_to_nets, _, _ = open_road_run.run(
                netlist_copy, arg_testfile, arg_parasitics, area_constraint, L_eff
            )

        log_info(f"edge to nets: {self.circuit_model.edge_to_nets}")

        self.minimum_clk_period = self.calculate_minimum_clk_period()
        logger.info(f"minimum clk period: {self.minimum_clk_period}, current clk period: {self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]}")
        if self.minimum_clk_period > self.circuit_model.tech_model.base_params.tech_values[self.circuit_model.tech_model.base_params.clk_period]:
            logger.info(f"minimum clk period is greater than current clk period, setting current clk period to minimum clk period")
            self.circuit_model.tech_model.base_params.set_symbol_value(self.circuit_model.tech_model.base_params.clk_period, self.minimum_clk_period)

        logger.info(f"time to generate wire parasitics: {time.time()-start_time} seconds, {(time.time()-start_time)/60} minutes.")


    def save_display_quantities(self, execution_time):
        if self.model_cfg["model_type"] == "bulk_bsim4":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "subthreshold leakage current": self.circuit_model.tech_model.I_sub,
                "gate tunneling current": self.circuit_model.tech_model.I_tunnel,
                "GIDL current": self.circuit_model.tech_model.I_GIDL,
                "long channel threshold voltage": self.circuit_model.tech_model.base_params.V_th,
                "effective threshold voltage": self.circuit_model.tech_model.V_th_eff,
                "supply voltage": self.circuit_model.tech_model.base_params.V_dd,
                "wire RC": self.circuit_model.tech_model.m1_Rsq * self.circuit_model.tech_model.m1_Csq,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                "f": self.circuit_model.tech_model.base_params.f,
            }
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "bulk":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "subthreshold leakage current": self.circuit_model.tech_model.I_off,
                "gate tunneling current": self.circuit_model.tech_model.I_tunnel,
                "FN term": self.circuit_model.tech_model.FN_term,
                "WKB term": self.circuit_model.tech_model.WKB_term,
                "GIDL current": self.circuit_model.tech_model.I_GIDL,
                "effective threshold voltage": self.circuit_model.tech_model.V_th_eff,
                "supply voltage": self.circuit_model.tech_model.base_params.V_dd,
                "wire RC": self.circuit_model.tech_model.m1_Rsq * self.circuit_model.tech_model.m1_Csq,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                "f": self.circuit_model.tech_model.base_params.f,
            }
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "vs":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "gate length": self.circuit_model.tech_model.param_db["L"],
                "gate width": self.circuit_model.tech_model.param_db["W"],
                "subthreshold leakage current": self.circuit_model.tech_model.param_db["I_sub"],
                "long channel threshold voltage": self.circuit_model.tech_model.param_db["V_th"],
                "effective threshold voltage": self.circuit_model.tech_model.param_db["V_th_eff"],
                "supply voltage": self.circuit_model.tech_model.param_db["V_dd"],
                "wire RC": self.circuit_model.tech_model.param_db["wire RC"],
                "on current per um": self.circuit_model.tech_model.param_db["I_on_per_um"],
                "off current per um": self.circuit_model.tech_model.param_db["I_off_per_um"],
                "gate tunneling current per um": self.circuit_model.tech_model.param_db["I_tunnel_per_um"],
                "subthreshold leakage current per um": self.circuit_model.tech_model.param_db["I_sub_per_um"],
                "DIBL factor": self.circuit_model.tech_model.param_db["DIBL factor"],
                "SS": self.circuit_model.tech_model.param_db["SS"],
                "t_ox": self.circuit_model.tech_model.param_db["t_ox"],
                "eot": self.circuit_model.tech_model.param_db["eot"],
                "scale length": self.circuit_model.tech_model.param_db["scale_length"],
                "C_load": self.circuit_model.tech_model.param_db["C_load"],
                "C_wire": self.circuit_model.tech_model.param_db["C_wire"],
                "R_wire": self.circuit_model.tech_model.param_db["R_wire"],
                "R_device": self.circuit_model.tech_model.param_db["V_dd"]/self.circuit_model.tech_model.param_db["I_on"],
                "F_f": self.circuit_model.tech_model.param_db["F_f"],
                "F_s": self.circuit_model.tech_model.param_db["F_s"],
                "vx0": self.circuit_model.tech_model.param_db["vx0"],
                "v": self.circuit_model.tech_model.param_db["v"],
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                #"f": self.circuit_model.tech_model.base_params.f,
                "parasitic capacitance": self.circuit_model.tech_model.param_db["parasitic capacitance"],
                "k_gate": self.circuit_model.tech_model.param_db["k_gate"],
                "delay": self.circuit_model.tech_model.delay,
                "multiplier delay": self.circuit_model.symbolic_latency_wc["Mult16"](),
                #"scaled power": self.total_passive_power * self.circuit_model.tech_model.capped_power_scale_total + self.total_active_energy/(execution_time * self.circuit_model.tech_model.capped_delay_scale_total),
                "logic_sensitivity": self.circuit_model.tech_model.base_params.logic_sensitivity,
                "logic_resource_sensitivity": self.circuit_model.tech_model.base_params.logic_resource_sensitivity,
                "logic_amdahl_limit": self.circuit_model.tech_model.base_params.logic_amdahl_limit,
                "logic_resource_amdahl_limit": self.circuit_model.tech_model.base_params.logic_resource_amdahl_limit,
                "interconnect sensitivity": self.circuit_model.tech_model.base_params.interconnect_sensitivity,
                "interconnect resource sensitivity": self.circuit_model.tech_model.base_params.interconnect_resource_sensitivity,
                "interconnect amdahl limit": self.circuit_model.tech_model.base_params.interconnect_amdahl_limit,
                "interconnect resource amdahl limit": self.circuit_model.tech_model.base_params.interconnect_resource_amdahl_limit,
                "memory sensitivity": self.circuit_model.tech_model.base_params.memory_sensitivity,
                "memory resource sensitivity": self.circuit_model.tech_model.base_params.memory_resource_sensitivity,
                "memory amdahl limit": self.circuit_model.tech_model.base_params.memory_amdahl_limit,
                "memory resource amdahl limit": self.circuit_model.tech_model.base_params.memory_resource_amdahl_limit,
                "m1_Rsq": self.circuit_model.tech_model.m1_Rsq,
                "m2_Rsq": self.circuit_model.tech_model.m2_Rsq,
                "m3_Rsq": self.circuit_model.tech_model.m3_Rsq,
                "m1_Csq": self.circuit_model.tech_model.m1_Csq,
                "m2_Csq": self.circuit_model.tech_model.m2_Csq,
                "m3_Csq": self.circuit_model.tech_model.m3_Csq,
                "m1_rho": self.circuit_model.tech_model.base_params.m1_rho,
                "m2_rho": self.circuit_model.tech_model.base_params.m2_rho,
                "m3_rho": self.circuit_model.tech_model.base_params.m3_rho,
                "m1_k": self.circuit_model.tech_model.base_params.m1_k,
                "m2_k": self.circuit_model.tech_model.base_params.m2_k,
                "m3_k": self.circuit_model.tech_model.base_params.m3_k,
            }
            if self.circuit_model.tech_model.model_cfg["vs_model_type"] == "base":
                self.obj_sub_exprs["t_1"] = self.circuit_model.tech_model.param_db["t_1"]
            elif self.circuit_model.tech_model.model_cfg["vs_model_type"] == "mvs_si":
                self.obj_sub_exprs["R_s"] = self.circuit_model.tech_model.param_db["R_s"]
                self.obj_sub_exprs["R_d"] = self.circuit_model.tech_model.param_db["R_d"]
                self.obj_sub_exprs["L_ov"] = self.circuit_model.tech_model.param_db["L_ov"]
            elif self.circuit_model.tech_model.model_cfg["vs_model_type"] == "vscnfet":
                self.obj_sub_exprs["Vth_rolloff"] = self.circuit_model.tech_model.param_db["Vth_rolloff"]
                self.obj_sub_exprs["d"] = self.circuit_model.tech_model.param_db["d"]
                self.obj_sub_exprs["L_c"] = self.circuit_model.tech_model.param_db["L_c"]
                self.obj_sub_exprs["H_c"] = self.circuit_model.tech_model.param_db["H_c"]
                self.obj_sub_exprs["H_g"] = self.circuit_model.tech_model.param_db["H_g"]
                self.obj_sub_exprs["k_cnt"] = self.circuit_model.tech_model.param_db["k_cnt"]
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "mvs_general":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "area": self.circuit_model.tech_model.area,
                "delay": self.circuit_model.tech_model.delay,
                "Ieff_n": self.circuit_model.tech_model.Ieff_n,
                "Ieff_p": self.circuit_model.tech_model.Ieff_p,
                "Ioff_n": self.circuit_model.tech_model.Ioff_n,
                "Ioff_p": self.circuit_model.tech_model.Ioff_p,
                "gate tunneling current per um": self.circuit_model.tech_model.param_db["I_tunnel_per_um"],
                "subthreshold leakage current per um": self.circuit_model.tech_model.param_db["I_sub_per_um"],
                "subthreshold leakage current worst case per um": self.circuit_model.tech_model.param_db["I_sub_worst_case_per_um"],
                "on current per um": self.circuit_model.tech_model.param_db["I_on_per_um"],
                "off current per um": self.circuit_model.tech_model.param_db["I_off_per_um"],
                "off current worst case per um": self.circuit_model.tech_model.param_db["I_off_worst_case_per_um"],
                "supply voltage": self.circuit_model.tech_model.base_params.V_dd,
                "long channel threshold voltage": self.circuit_model.tech_model.base_params.V_th,
                "effective threshold voltage": self.circuit_model.tech_model.V_th_eff,
                "effective threshold voltage worst case": self.circuit_model.tech_model.V_th_eff_worst_case,
                "t_ox": self.circuit_model.tech_model.base_params.tox,
                "k_gate": self.circuit_model.tech_model.base_params.k_gate,
                "gate length": self.circuit_model.tech_model.base_params.L,
                "gate width": self.circuit_model.tech_model.base_params.W,
                "mu_eff_n": self.circuit_model.tech_model.base_params.mu_eff_n,
                "mu_eff_p": self.circuit_model.tech_model.base_params.mu_eff_p,
                "eps_semi": self.circuit_model.tech_model.base_params.eps_semi,
                "tsemi": self.circuit_model.tech_model.base_params.tsemi,
                "Lext": self.circuit_model.tech_model.base_params.Lext,
                "L_c": self.circuit_model.tech_model.base_params.Lc,
                "eps_cap": self.circuit_model.tech_model.base_params.eps_cap,
                "rho_c_n": self.circuit_model.tech_model.base_params.rho_c_n,
                "rho_c_p": self.circuit_model.tech_model.base_params.rho_c_p,
                "Rsh_c_n": self.circuit_model.tech_model.base_params.Rsh_c_n,
                "Rsh_c_p": self.circuit_model.tech_model.base_params.Rsh_c_p,
                "Rsh_ext_n": self.circuit_model.tech_model.base_params.Rsh_ext_n,
                "Rsh_ext_p": self.circuit_model.tech_model.base_params.Rsh_ext_p,
                "C_load": self.circuit_model.tech_model.param_db["C_load"],
                "C_wire": self.circuit_model.tech_model.param_db["C_wire"],
                "R_wire": self.circuit_model.tech_model.param_db["R_wire"],
                "f": self.circuit_model.tech_model.base_params.f,
                "n0": self.circuit_model.tech_model.n0,
                "DIBL factor": self.circuit_model.tech_model.delta,
                "dVt": self.circuit_model.tech_model.dVt,
                "scale length": self.circuit_model.tech_model.Lscale,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                "logic_sensitivity": self.circuit_model.tech_model.base_params.logic_sensitivity,
                "logic_resource_sensitivity": self.circuit_model.tech_model.base_params.logic_resource_sensitivity,
                "logic_amdahl_limit": self.circuit_model.tech_model.base_params.logic_amdahl_limit,
                "logic_resource_amdahl_limit": self.circuit_model.tech_model.base_params.logic_resource_amdahl_limit,
                "interconnect sensitivity": self.circuit_model.tech_model.base_params.interconnect_sensitivity,
                "interconnect resource sensitivity": self.circuit_model.tech_model.base_params.interconnect_resource_sensitivity,
                "interconnect amdahl limit": self.circuit_model.tech_model.base_params.interconnect_amdahl_limit,
                "interconnect resource amdahl limit": self.circuit_model.tech_model.base_params.interconnect_resource_amdahl_limit,
                "memory sensitivity": self.circuit_model.tech_model.base_params.memory_sensitivity,
                "memory resource sensitivity": self.circuit_model.tech_model.base_params.memory_resource_sensitivity,
                "memory amdahl limit": self.circuit_model.tech_model.base_params.memory_amdahl_limit,
                "memory resource amdahl limit": self.circuit_model.tech_model.base_params.memory_resource_amdahl_limit,
                "m1_Rsq": self.circuit_model.tech_model.m1_Rsq,
                "m2_Rsq": self.circuit_model.tech_model.m2_Rsq,
                "m3_Rsq": self.circuit_model.tech_model.m3_Rsq,
                "m1_Csq": self.circuit_model.tech_model.m1_Csq,
                "m2_Csq": self.circuit_model.tech_model.m2_Csq,
                "m3_Csq": self.circuit_model.tech_model.m3_Csq,
                "m1_rho": self.circuit_model.tech_model.base_params.m1_rho,
                "m2_rho": self.circuit_model.tech_model.base_params.m2_rho,
                "m3_rho": self.circuit_model.tech_model.base_params.m3_rho,
                "m1_k": self.circuit_model.tech_model.base_params.m1_k,
                "m2_k": self.circuit_model.tech_model.base_params.m2_k,
                "m3_k": self.circuit_model.tech_model.base_params.m3_k,
                "multiplier delay": self.circuit_model.symbolic_latency_wc["Mult16"](),
            }
        elif self.circuit_model.tech_model.model_cfg["model_type"] == "sweep" or self.circuit_model.tech_model.model_cfg["model_type"] == "sweep_brute_force" or self.circuit_model.tech_model.model_cfg["model_type"] == "sweep_basic":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "total power": (self.total_active_energy + self.total_passive_energy)/execution_time,
                "area": self.circuit_model.tech_model.param_db["A_gate"],
                "delay": self.circuit_model.tech_model.delay,
                "gate length": self.circuit_model.tech_model.param_db["L"],
                "gate width": self.circuit_model.tech_model.param_db["W"],
                "C_load": self.circuit_model.tech_model.param_db["C_load"],
                "Inverter VTC gain": self.circuit_model.tech_model.param_db["slope_at_crossing"],
                "R_avg_inv": self.circuit_model.tech_model.param_db["R_avg_inv"],
                "E_act_inv": self.circuit_model.tech_model.E_act_inv,
                "P_pass_inv": self.circuit_model.tech_model.P_pass_inv,
                "Ieff": self.circuit_model.tech_model.param_db["Ieff"],
                "Ioff": self.circuit_model.tech_model.param_db["Ioff"],
                "supply voltage": self.circuit_model.tech_model.param_db["V_dd"],
                "effective threshold voltage": self.circuit_model.tech_model.param_db["V_th_eff"],
                "DIBL factor": self.circuit_model.tech_model.param_db["delta"],
                "n0": self.circuit_model.tech_model.param_db["n0"],
                "scale length": self.circuit_model.tech_model.param_db["Lscale"],
                "GEO": self.circuit_model.tech_model.param_db["GEO"],
                "MUL": self.circuit_model.tech_model.param_db["MUL"],
                "t_ox": self.circuit_model.tech_model.param_db["tox"],
                "tsemi": self.circuit_model.tech_model.param_db["tsemi"],
                "eot": self.circuit_model.tech_model.param_db["eot"],
                "eot_corrected": self.circuit_model.tech_model.param_db["eot_corrected"],
                "k_gate": self.circuit_model.tech_model.param_db["k_gate"],
                "NM_H": self.circuit_model.tech_model.param_db["NM_H"],
                "NM_L": self.circuit_model.tech_model.param_db["NM_L"],
                "noise_margin": self.circuit_model.tech_model.param_db["noise_margin"],
                "multiplier delay": self.circuit_model.symbolic_latency_wc["Mult16"](),
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
                #"scaled power": self.total_passive_power * self.circuit_model.tech_model.capped_power_scale_total + self.total_active_energy/(execution_time * self.circuit_model.tech_model.capped_delay_scale_total),
                "logic_sensitivity": self.circuit_model.tech_model.base_params.logic_sensitivity,
                "logic_resource_sensitivity": self.circuit_model.tech_model.base_params.logic_resource_sensitivity,
                "logic_amdahl_limit": self.circuit_model.tech_model.base_params.logic_amdahl_limit,
                "logic_resource_amdahl_limit": self.circuit_model.tech_model.base_params.logic_resource_amdahl_limit,
                "interconnect sensitivity": self.circuit_model.tech_model.base_params.interconnect_sensitivity,
                "interconnect resource sensitivity": self.circuit_model.tech_model.base_params.interconnect_resource_sensitivity,
                "interconnect amdahl limit": self.circuit_model.tech_model.base_params.interconnect_amdahl_limit,
                "interconnect resource amdahl limit": self.circuit_model.tech_model.base_params.interconnect_resource_amdahl_limit,
                "memory sensitivity": self.circuit_model.tech_model.base_params.memory_sensitivity,
                "memory resource sensitivity": self.circuit_model.tech_model.base_params.memory_resource_sensitivity,
                "memory amdahl limit": self.circuit_model.tech_model.base_params.memory_amdahl_limit,
                "memory resource amdahl limit": self.circuit_model.tech_model.base_params.memory_resource_amdahl_limit,
                "m1_Rsq": self.circuit_model.tech_model.m1_Rsq,
                "m2_Rsq": self.circuit_model.tech_model.m2_Rsq,
                "m3_Rsq": self.circuit_model.tech_model.m3_Rsq,
                "m1_Csq": self.circuit_model.tech_model.m1_Csq,
                "m2_Csq": self.circuit_model.tech_model.m2_Csq,
                "m3_Csq": self.circuit_model.tech_model.m3_Csq,
                "m1_rho": self.circuit_model.tech_model.base_params.m1_rho,
                "m2_rho": self.circuit_model.tech_model.base_params.m2_rho,
                "m3_rho": self.circuit_model.tech_model.base_params.m3_rho,
                "m1_k": self.circuit_model.tech_model.base_params.m1_k,
                "m2_k": self.circuit_model.tech_model.base_params.m2_k,
                "m3_k": self.circuit_model.tech_model.base_params.m3_k,
            }
        else: 
            raise ValueError(f"Model type {self.circuit_model.tech_model.model_cfg['model_type']} not supported")
        self.obj_sub_plot_names = {
            "execution_time": "Execution Time over generations (ns)",
            "passive power": "Passive Power over generations (W)",
            "active power": "Active Power over generations (W)",
            "total power": "Total Power over generations (W)",
            "gate length": "Gate Length over generations (m)",
            "gate width": "Gate Width over generations (m)",
            "NM_H": "Noise Margin High over generations (V)",
            "NM_L": "Noise Margin Low over generations (V)",
            "noise_margin": "Noise Margin over generations (V)",
            "E_act_inv": "Dynamic Energy per Inverter over generations (J)",
            "R_avg_inv": "Inverter average resistance over generations (Ohm)",
            "P_pass_inv": "Passive Power per Inverter over generations (W)",
            "subthreshold leakage current": "Subthreshold Leakage Current over generations (nA)",
            "long channel threshold voltage": "Long Channel Threshold Voltage (V)",
            "effective threshold voltage": "Effective Threshold Voltage over generations (V)",
            "effective threshold voltage worst case": "Effective Threshold Voltage Worst Case over generations (V)",
            "Inverter VTC gain": "Slope of Inverter VTC at Vout=Vin over generations (V/V)",
            "supply voltage": "Supply Voltage over generations (V)",
            "GEO": "GEO flag over generations",
            "MUL": "MUL flag over generations",
            "wire RC": "Wire RC over generations (s)",
            "on current per um": "On Current per um over generations (A/um)",
            "off current per um": "Off Current per um over generations (A/um)",
            "off current worst case per um": "Off Current Worst Case per um over generations (A/um)",
            "gate tunneling current per um": "Gate Tunneling Current per um over generations (A/um)",
            "subthreshold leakage current per um": "Subthreshold Leakage Current per um over generations (A/um)",
            "subthreshold leakage current worst case per um": "Subthreshold Leakage Current Worst Case per um over generations (A/um)",
            "DIBL factor": "DIBL Factor over generations (V/V)",
            "SS": "Subthreshold Slope over generations (V/V)",
            "n0": "n0 over generations",
            "Vth_rolloff": "Vth Rolloff over generations (V)",
            "dVt": "total threshold voltage shift over generations (V)",
            "t_ox": "Gate Oxide Thickness over generations (m)",
            "eot": "Electrical Oxide Thickness over generations (m)",
            "eot_corrected": "Electrical Oxide Thickness Corrected over generations (m)",
            "scale length": "Scale Length over generations (m)",
            "C_load": "Load Capacitance over generations (F)",
            "C_wire": "Wire Capacitance over generations (F)",
            "R_wire": "Wire Resistance over generations (Ohm)",
            "R_device": "Device Resistance over generations (Ohm)",
            "F_f": "F_f over generations",
            "F_s": "F_s over generations",
            "vx0": "virtual source injection velocity over generations (m/s)",
            "v": "effective injection velocity over generations (m/s)",
            "t_1": "T1 over generations (s)",
            "clk_period": "Clock Period over generations (ns)",
            "f": "Frequency over generations (Hz)",
            "parasitic capacitance": "Parasitic Capacitance over generations (F)",
            "L_ov": "L_ov over generations (m)",
            "R_s": "R_s over generations (Ohm)",
            "R_d": "R_d over generations (Ohm)",
            "Vth_rolloff": "Vth Rolloff over generations (V)",
            "d": "CNT diameter over generations (m)",
            "L_c": "Contact length over generations (m)",
            "Lext": "Extension length over generations (m)",
            "H_c": "Contact height over generations (m)",
            "H_g": "CNT gate height over generations (m)",
            "k_cnt": "CNT Dielectric Constant over generations (F/m)",
            "k_gate": "Gate Dielectric Constant over generations (F/m)",
            "eps_cap": "Capacitor Dielectric Constant over generations (F/m)",
            "eps_semi": "Semiconductor Dielectric Constant over generations (F/m)",
            "tsemi": "Semiconductor Thickness over generations (m)",
            "rho_c_n": "n-type Contact Resistance over generations (Ohm-m)",
            "rho_c_p": "p-type Contact Resistance over generations (Ohm-m)",
            "Rsh_c_n": "n-type Shunt Resistance over generations (Ohm)",
            "Rsh_c_p": "p-type Shunt Resistance over generations (Ohm)",
            "Rsh_ext_n": "n-type External Shunt Resistance over generations (Ohm)",
            "Rsh_ext_p": "p-type External Shunt Resistance over generations (Ohm)",
            "Ieff_n": "nMOS Effective Current over generations (A)",
            "Ieff_p": "pMOS Effective Current over generations (A)",
            "Ioff_n": "nMOS Off Current over generations (A)",
            "Ioff_p": "pMOS Off Current over generations (A)",
            "Ioff": "Off Current over generations (A)",
            "Ieff": "Effective Current over generations (A)",
            "Cload": "Load Capacitance over generations (F)",
            "mu_eff_n": "nMOS Effective Mobility over generations (m^2/V-s)",
            "mu_eff_p": "pMOS Effective Mobility over generations (m^2/V-s)",
            "area": "device area over generations (m^2)",
            "delay": "Transistor Delay over generations (s)",
            "multiplier delay": "Multiplier Delay over generations (s)",
            "scaled power": "Scaled Power over generations (W)",
            "logic_sensitivity": "Logic Sensitivity over generations",
            "logic_resource_sensitivity": "Logic Resource Sensitivity over generations",
            "logic_amdahl_limit": "Logic amdahl Limit over generations",
            "logic_resource_amdahl_limit": "Logic Resource amdahl Limit over generations",
            "interconnect sensitivity": "Interconnect Sensitivity over generations",
            "interconnect resource sensitivity": "Interconnect Resource Sensitivity over generations",
            "interconnect amdahl limit": "Interconnect amdahl Limit over generations",
            "interconnect resource amdahl limit": "Interconnect Resource amdahl Limit over generations",
            "memory sensitivity": "Memory Sensitivity over generations",
            "memory resource sensitivity": "Memory Resource Sensitivity over generations",
            "memory amdahl limit": "Memory amdahl Limit over generations",
            "memory resource amdahl limit": "Memory Resource amdahl Limit over generations",
            "m1_Rsq": "Metal 1 Resistance per Square over generations (Ohm/m)",
            "m2_Rsq": "Metal 2 Resistance per Square over generations (Ohm/m)",
            "m3_Rsq": "Metal 3 Resistance per Square over generations (Ohm/m)",
            "m1_Csq": "Metal 1 Capacitance per Square over generations (F/m)",
            "m2_Csq": "Metal 2 Capacitance per Square over generations (F/m)",
            "m3_Csq": "Metal 3 Capacitance per Square over generations (F/m)",
            "m1_rho": "Metal 1 Resistivity over generations (Ohm-m)",
            "m2_rho": "Metal 2 Resistivity over generations (Ohm-m)",
            "m3_rho": "Metal 3 Resistivity over generations (Ohm-m)",
            "m1_k": "Metal 1 Permittivity over generations (F/m)",
            "m2_k": "Metal 2 Permittivity over generations (F/m)",
            "m3_k": "Metal 3 Permittivity over generations (F/m)",
        }
        self.constraints_to_plot = set(
            [
                "total_power <= max_system_power",
                "I_off per (W) <= 100e-9 per (1e-6)",
                "W over L >= 1",
                "V_th_eff >= 0",
                "V_dd >= V_th",
                "delta <= 0.15",
                "latency_FloorDiv16 <= 20*clk_period",
            ]
        )

    def save_obj_vals(self, execution_time, execution_time_override=False, execution_time_override_val=0):
        self.save_display_quantities(execution_time)
        if execution_time_override:
            execution_time = execution_time_override_val
        if self.obj_fn == "edp":
            self.obj = (self.total_passive_energy + self.total_active_energy) * execution_time
            self.obj_scaled = (self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale + self.total_active_energy) * execution_time * self.circuit_model.tech_model.capped_delay_scale
        elif self.obj_fn == "ed2":
            self.obj = (self.total_passive_energy + self.total_active_energy) * (execution_time)**2
            self.obj_scaled = (self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale + self.total_active_energy) * (execution_time * self.circuit_model.tech_model.capped_delay_scale)**2
        elif self.obj_fn == "delay":
            self.obj = execution_time
            self.obj_scaled = execution_time * self.circuit_model.tech_model.capped_delay_scale
        elif self.obj_fn == "energy":
            self.obj = self.total_active_energy + self.total_passive_energy
            self.obj_scaled = (self.total_active_energy + self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale)
        elif self.obj_fn == "eplusd":
            self.obj = ((self.total_active_energy + self.total_passive_energy) * sim_util.xreplace_safe(execution_time, self.circuit_model.tech_model.base_params.tech_values) 
                        + execution_time * sim_util.xreplace_safe(self.total_active_energy + self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values))
            self.obj_scaled = ((self.total_active_energy + self.total_passive_energy * self.circuit_model.tech_model.capped_energy_scale) * sim_util.xreplace_safe(execution_time, self.circuit_model.tech_model.base_params.tech_values) 
                        + execution_time * self.circuit_model.tech_model.capped_delay_scale * sim_util.xreplace_safe(self.total_active_energy + self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values))
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")

    def calculate_sensitivity_analysis(self, blackbox=False, constraints=[]):
        obj = self.obj
        for constraint in constraints:
            eps = 1e-15
            slack_value = -1*sim_util.xreplace_safe(constraint.slack, self.circuit_model.tech_model.base_params.tech_values) + eps
            if (slack_value > 0):
                log_info(f"adding log barrier for constraint {constraint.label}, slack value: {slack_value}, log barrier term: {-math.log(slack_value)}")
                obj += -math.log(slack_value) # adding log barriers to objective to help show effect of constraints on sensitivities
            else:
                logger.warning(f"Constraint {constraint.label} is violated, slack value: {slack_value}")
        for param in self.circuit_model.tech_model.base_params.tech_values:
            #log_info(f"calculating sensitivity for {param}, initial value: {self.circuit_model.tech_model.base_params.tech_values[param]}")
            if blackbox:
                obj_initial_val = sim_util.xreplace_safe(obj, self.circuit_model.tech_model.base_params.tech_values)
                tech_values_param_changed = {k: v for k, v in self.circuit_model.tech_model.base_params.tech_values.items() if k != param}
                tech_values_param_changed[param] = self.circuit_model.tech_model.base_params.tech_values[param]*1.01
                obj_param_changed = sim_util.xreplace_safe(obj, tech_values_param_changed)
                obj_percent_change = (obj_param_changed - obj_initial_val) / obj_initial_val
                if self.circuit_model.tech_model.base_params.tech_values[param] == 0:
                    self.sensitivities[param] = 0
                else:
                    self.sensitivities[param] = (obj_percent_change) / (0.01) # 1% change in param
            else:
                tech_values_without_param = {k: v for k, v in self.circuit_model.tech_model.base_params.tech_values.items() if k != param}
                d_obj_d_param = obj.diff(param, evaluate=True).xreplace(tech_values_without_param)
                self.sensitivities[param] = sim_util.xreplace_safe(d_obj_d_param * (self.circuit_model.tech_model.base_params.tech_values[param] / sim_util.xreplace_safe(obj, self.circuit_model.tech_model.base_params.tech_values)), self.circuit_model.tech_model.base_params.tech_values)
        logger.info(f"sensitivities: {self.sensitivities}")

    def calculate_objective(self, form_dfg=True, do_sensitivity_analysis=False, log_top_vectors=False, clk_period_opt=False):
        start_time = time.time()
        if self.hls_tool == "vitis":
            # Use ObjectiveEvaluator for energy/area calculation (consistent with optimization pass)
            evaluator = ObjectiveEvaluator.from_hardware_model(self)
            evaluator.calculate_objective()
            self.execution_time = evaluator.execution_time
            self.total_passive_energy = evaluator.total_passive_energy
            self.total_active_energy = evaluator.total_active_energy
            self.total_area = evaluator.total_area
        else:
            raise ValueError(f"HLS tool {self.hls_tool} not supported")
        self.save_obj_vals(self.execution_time)
        if do_sensitivity_analysis:
            self.calculate_sensitivity_analysis()
        logger.info(f"time to calculate objective: {time.time()-start_time}")

    def display_objective(self, message):
        self.save_display_quantities(self.execution_time)
        obj = sim_util.xreplace_safe(self.obj, self.circuit_model.tech_model.base_params.tech_values)
        sub_exprs = {}
        for key in self.obj_sub_exprs:
            if not isinstance(self.obj_sub_exprs[key], float):
                sub_exprs[key] = float(sim_util.xreplace_safe(self.obj_sub_exprs[key], self.circuit_model.tech_model.base_params.tech_values))
            else:   
                sub_exprs[key] = self.obj_sub_exprs[key]
        # Also report energies (Joules) alongside the existing power values
        total_energy_val = sim_util.xreplace_safe(self.total_active_energy + self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values)
        passive_energy_val = sim_util.xreplace_safe(self.total_passive_energy, self.circuit_model.tech_model.base_params.tech_values)
        active_energy_val = sim_util.xreplace_safe(self.total_active_energy, self.circuit_model.tech_model.base_params.tech_values)
        sub_exprs["total energy"] = float(total_energy_val)
        sub_exprs["passive energy"] = float(passive_energy_val)
        sub_exprs["active energy"] = float(active_energy_val)
        print(f"{message}\n {self.obj_fn}: {obj}, sub expressions: {sub_exprs}")