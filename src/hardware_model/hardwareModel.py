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
from src.hardware_model.circuit_models import memory_model
from src.hardware_model.circuit_models import logic_unit_model as lum_module

from src import sim_util

from src.inverse_pass.constraint import Constraint

from src.hardware_model.tech_models import sweep_basic_model
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
        self.ram_recurrences = {}
        self.top_block_name = args["benchmark"] if not args["pytorch"] and self.cfg["args"]["arch_opt_pipeline"] != "streamhls" else "forward"
        self.dataflow_blocks = set()

        self.parasitic_graph = nx.DiGraph()
        self.symbolic_mem = {}
        self.symbolic_buf = {}
        self.mem_access_db = {}
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
        self.memory_models = {}
        self.memory_mapping = {}
        self.logic_unit_models = {}

    def reset_state(self):
        self.symbolic_buf = {}
        self.symbolic_mem = {}
        self.netlist = nx.DiGraph()
        self.mem_access_db = {}
        self.obj = 0
        self.scheduled_dfg = nx.DiGraph()
        self.scheduled_dfgs = {}
        self.loop_1x_graphs = {}
        self.loop_2x_graphs = {}
        self.ram_recurrences = {}
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
        self.memory_models = {} # keyed by memory name
        self.logic_unit_models = {} # keyed by name

    def set_memory_models(self, memory_mapping):
        self.memory_mapping = memory_mapping
        for memory_name, memory_info in memory_mapping["flattened"].items():
            self.memory_models[memory_name] = memory_model.MemoryModel(memory_info, name=memory_name)

    def init_default_logic_unit_models(self, index=0):
        """Initialize one default LogicUnitModel per function type at a given pareto index.

        Used before the netlist is available so that circuit_values contain valid
        (non-empty) entries.  Real per-resource models replace these once
        set_logic_unit_models() is called after netlist parsing.
        """
        precomputed = lum_module.precompute_pareto_values(self.circuit_model.tech_model)
        n = len(precomputed["delay"])
        index = max(0, min(index, n - 1))
        logic_fns = set(self.circuit_model.coeffs["gamma"].keys())
        self.logic_unit_models = {}
        for fn in logic_fns:
            lum = lum_module.LogicUnitModel(precomputed, f"{fn}_default", fn)
            lum.set_design_point(index)
            self.logic_unit_models[f"{fn}_default"] = lum
        self.circuit_model.set_logic_unit_models(self.logic_unit_models)

    def set_logic_unit_models(self):
        """Create one LogicUnitModel per unique logic FU resource in the netlist."""
        precomputed = lum_module.precompute_pareto_values(self.circuit_model.tech_model)
        logic_fns = set(self.circuit_model.coeffs["gamma"].keys())
        self.logic_unit_models = {}
        for node, data in self.netlist.nodes(data=True):
            fn = data.get("function", "N/A")
            rsc = data.get("name", None)
            logger.info(f"creating logic unit model for fn: {fn}, rsc: {rsc}")
            if fn in logic_fns and rsc and rsc not in self.logic_unit_models:
                self.logic_unit_models[rsc] = lum_module.LogicUnitModel(precomputed, rsc, fn)
            elif fn == "read" or fn == "write" and rsc and rsc not in self.logic_unit_models: # make one in case this is for a register
                self.logic_unit_models[rsc] = lum_module.LogicUnitModel(precomputed, rsc, "Register16")
            #elif fn in
        self.circuit_model.set_logic_unit_models(self.logic_unit_models)
        logger.info(f"Created {len(self.logic_unit_models)} LogicUnitModel instances from netlist")

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
            open_road_run = openroad_run.OpenRoadRun(cfg=self.cfg, codesign_root_dir=self.codesign_root_dir, tmp_dir=self.tmp_dir, run_openroad=run_openroad, circuit_model=self.circuit_model, memory_models=self.memory_models)

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
        if self.circuit_model.tech_model.model_cfg["model_type"] == "sweep_basic":
            self.obj_sub_exprs = {
                "execution_time": execution_time,
                "passive power": self.total_passive_energy/execution_time,
                "active power": self.total_active_energy/execution_time,
                "total power": (self.total_active_energy + self.total_passive_energy)/execution_time,
                "clk_period": self.circuit_model.tech_model.base_params.clk_period,
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
            for mem_name, mem_model in self.memory_models.items():
                if mem_model.num_design_points == 0:
                    continue
                row = mem_model.get_design_point_row()
                self.obj_sub_exprs[f"mem_{mem_name}_index"] = mem_model.design_point_index
                area_key = next((k for k in row if "area" in k.lower()), None)
                latency_key = next((k for k in row if "latency" in k.lower()), None)
                leakage_key = next((k for k in row if "leak" in k.lower()), None)
                for label, col in [("area", area_key), ("latency", latency_key), ("leakage", leakage_key)]:
                    if col is not None:
                        self.obj_sub_exprs[f"mem_{mem_name}_{label}"] = row[col]
                self.obj_sub_exprs[f"mem_{mem_name}_capacity"] = mem_model.total_size_bits
            for fu_name, lum in self.logic_unit_models.items():
                row = lum.get_design_point_row()
                self.obj_sub_exprs[f"fu_{fu_name}_index"]     = row["index"]
                self.obj_sub_exprs[f"fu_{fu_name}_delay"]     = row["delay"]
                self.obj_sub_exprs[f"fu_{fu_name}_Eactinv"] = row["E_act_inv"]
                self.obj_sub_exprs[f"fu_{fu_name}_Ppassinv"]= row["P_pass_inv"]
                self.obj_sub_exprs[f"fu_{fu_name}_area"]      = row["area"]
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
            "effective threshold voltage": "Effective Threshold Voltage over generations (V)",
            "supply voltage": "Supply Voltage over generations (V)",
            "GEO": "GEO flag over generations",
            "MUL": "MUL flag over generations",
            "DIBL factor": "DIBL Factor over generations (V/V)",
            "n0": "n0 over generations",
            "t_ox": "Gate Oxide Thickness over generations (m)",
            "eot": "Electrical Oxide Thickness over generations (m)",
            "eot_corrected": "Electrical Oxide Thickness Corrected over generations (m)",
            "scale length": "Scale Length over generations (m)",
            "clk_period": "Clock Period over generations (ns)",
            "k_gate": "Gate Dielectric Constant over generations (F/m)",
            "tsemi": "Semiconductor Thickness over generations (m)",
            "Ioff": "Off Current over generations (A)",
            "Ieff": "Effective Current over generations (A)",
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
        for mem_name, mem_model in self.memory_models.items():
            if mem_model.num_design_points == 0:
                continue
            self.obj_sub_plot_names[f"mem_{mem_name}_index"] = f"Memory {mem_name} design point index over generations"
            self.obj_sub_plot_names[f"mem_{mem_name}_area"] = f"Memory {mem_name} area over generations (mm^2)"
            self.obj_sub_plot_names[f"mem_{mem_name}_latency"] = f"Memory {mem_name} latency over generations (ns)"
            self.obj_sub_plot_names[f"mem_{mem_name}_leakage"] = f"Memory {mem_name} leakage over generations (mW)"
            self.obj_sub_plot_names[f"mem_{mem_name}_capacity"] = f"Memory {mem_name} capacity over generations (KB)"
        for fu_name, lum in self.logic_unit_models.items():
            fn = lum.function
            self.obj_sub_plot_names[f"fu_{fu_name}_index"]     = f"FU {fu_name} ({fn}) design point index over generations"
            self.obj_sub_plot_names[f"fu_{fu_name}_delay"]     = f"FU {fu_name} ({fn}) delay over generations (s)"
            self.obj_sub_plot_names[f"fu_{fu_name}_Eactinv"] = f"FU {fu_name} ({fn}) dynamic energy over generations (J)"
            self.obj_sub_plot_names[f"fu_{fu_name}_Ppassinv"]= f"FU {fu_name} ({fn}) passive power over generations (W)"
            self.obj_sub_plot_names[f"fu_{fu_name}_area"]      = f"FU {fu_name} ({fn}) area over generations (m^2)"

    def calculate_objective(self, form_dfg=True, log_top_vectors=False, clk_period_opt=False):
        start_time = time.time()
        if self.hls_tool == "vitis":
            # Use ObjectiveEvaluator for energy/area calculation (consistent with optimization pass)
            evaluator = ObjectiveEvaluator.from_hardware_model(self)
            evaluator.calculate_objective()
            self.execution_time = evaluator.execution_time
            self.total_passive_energy = evaluator.total_passive_energy
            self.total_active_energy = evaluator.total_active_energy
            self.total_area = evaluator.total_area
            self.obj = evaluator.obj
            #self.save_display_quantities(self.execution_time)
        else:
            raise ValueError(f"HLS tool {self.hls_tool} not supported")
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
        # Group sub_exprs by unit for readable output
        system_entries = {}
        mem_groups = {}
        fu_groups = {}
        energy_entries = {}
        for key, val in sub_exprs.items():
            if key.startswith("mem_"):
                unit_name = key.split("_")[1]
                mem_groups.setdefault(unit_name, {})[key] = val
            elif key.startswith("fu_"):
                unit_name = '_'.join(key.split("_")[1:-1])
                fu_groups.setdefault(unit_name, {})[key] = val
            elif key in ("total energy", "passive energy", "active energy"):
                energy_entries[key] = val
            else:
                system_entries[key] = val
        lines = [f"{message}", f" {self.obj_fn}: {obj}"]
        if system_entries:
            lines.append(f"  system: {system_entries}")
        for unit_name, entries in mem_groups.items():
            lines.append(f"  mem/{unit_name}: {entries}")
        for unit_name, entries in fu_groups.items():
            lines.append(f"  fu/{unit_name}: {entries}")
        if energy_entries:
            lines.append(f"  energy: {energy_entries}")
        print("\n".join(lines))