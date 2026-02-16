import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
import math
import json
import numpy as np
import pandas as pd
import os
import cvxpy as cp
from src.inverse_pass.constraint import Constraint
from src import sim_util
logger = logging.getLogger(__name__)

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

class SweepBasicModel(TechModel):
    def __init__(self, model_cfg, base_params):
        self.sweep_file_prefix = model_cfg["sweep_file_prefix"]
        self.sweep_file_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + ".csv")
        self.sweep_model_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + "_model.json")
        self.cur_design_point_index = None
        super().__init__(model_cfg, base_params)

    def process_optimization_results(self):
        metric_vals = {}
        for metric in self.sweep_model["output_metrics"]:
            metric_vals[metric] = getattr(self, metric).value
            log_info(f"after optimization, {metric} is {metric_vals[metric]}")
        self.set_params_from_design_point(self.cur_design_point)
        self.config_param_db()

    def find_nearest_design_by_values(self, values):
        """
        Find the nearest design point in pareto_df based on input parameter values.

        Args:
            values: dict mapping input column names to their values
                          e.g., {'L': 15e-9, 'W': 10e-9, 'V_dd': 0.8, 'V_th': 0.3}
            input_columns: list of column names to use for matching. If None,
                          uses all keys from input_values that exist in pareto_df.

        Returns:
            tuple: (design_dict, distance)
                - design_dict: the nearest design point as a dict
                - distance: normalized Euclidean distance to the nearest point
        """
        from sklearn.neighbors import NearestNeighbors

        # Determine which columns to use
        columns = [c for c in values.keys() if c in self.pareto_df.columns]

        if len(columns) == 0:
            raise ValueError(f"No matching columns found. Available: {list(self.pareto_df.columns)}. Input values: {values}")

        # Extract the input data from pareto_df
        X = self.pareto_df[columns].values

        # Build query point
        query = np.array([[values[c] for c in columns]])

        # Find nearest neighbor
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn.fit(X)
        distance, index = nn.kneighbors(query)

        print(f"distance: {distance[0][0]}, index: {index[0][0]}")

        return self.pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0]), index[0][0]

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

    def init_transistor_equations(self):
        super().init_transistor_equations()

        # Read Pareto CSV
        log_info(f"Reading Pareto front from: {self.sweep_file_path}")
        self.pareto_df = pd.read_csv(self.sweep_file_path)
        #self.pareto_df_log = np.log(self.pareto_df)
        
        log_info(f"Loading Pareto surface model from: {self.sweep_model_path}")
        with open(self.sweep_model_path, 'r') as f:
            self.sweep_model = json.load(f)
        
        self.config_pareto_metric_db()
        if not self.base_params.output_parameters_initialized:
            self.base_params.init_output_parameters_basic(self.sweep_model["output_metrics"])
            tech_values_str_ind = {}
            for k, v in self.base_params.tech_values.items():
                if k in self.base_params.names:
                    tech_values_str_ind[self.base_params.names[k]] = v
            self.cur_design_point, dist, self.cur_design_point_index = self.find_nearest_design_by_values(tech_values_str_ind)
            log_info(f"distance to nearest design: {dist}")
            log_info(f"index of nearest design: {self.cur_design_point_index}")
        else:
            self.cur_design_point = self.base_params.cur_design_point
        self.set_params_from_design_point(self.cur_design_point)
        log_info(f"Current design point: {self.cur_design_point}")

        if "delay" not in self.sweep_model["output_metrics"]:
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
            log_info(f"set delay to {sim_util.xreplace_safe(self.delay, self.base_params.tech_values)}")
        if "E_act_inv" not in self.sweep_model["output_metrics"]:
            self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
            log_info(f"set E_act_inv to {sim_util.xreplace_safe(self.E_act_inv, self.base_params.tech_values)}")
        if "P_pass_inv" not in self.sweep_model["output_metrics"]:
            self.P_pass_inv = self.Ioff * self.base_params.V_dd
            log_info(f"set P_pass_inv to {sim_util.xreplace_safe(self.P_pass_inv, self.base_params.tech_values)}")

        self.config_param_db()

    def set_params_from_design_point(self, design_point):
        # Extract logic params; fall back to the whole dict for backward compat
        logic_params = design_point.get("logic", design_point)
        for param in logic_params.keys():
            if not hasattr(self.base_params, param):
                setattr(self.base_params, param, self.base_params.symbol_init(param))
            self.base_params.set_symbol_value(getattr(self.base_params, param), logic_params[param])
            setattr(self, param, getattr(self.base_params, param))
            log_info(f"set {param} to {sim_util.xreplace_safe(getattr(self, param), self.base_params.tech_values)}")
        if hasattr(self, "C_gate"):
            self.C_load = self.C_gate
            self.C_diff = self.C_gate
        if hasattr(self, "C_par"):
            self.C_diff = self.C_par
        self.base_params.cur_design_point = logic_params
        

    def config_param_db(self):
        super().config_param_db()
        for param in self.cur_design_point.keys():
            log_info(f"setting {param} in param_db to {sim_util.xreplace_safe(getattr(self.base_params, param), self.base_params.tech_values)}")
            self.param_db[param] = getattr(self.base_params, param)
        
        if "Ioff" in self.sweep_model["output_metrics"]:
            self.I_off = getattr(self.base_params, "Ioff")
            self.param_db["I_off"] = self.I_off
        
        if "A_gate" not in self.sweep_model["output_metrics"]:
            log_info(f"setting A_gate to {sim_util.xreplace_safe(self.area, self.base_params.tech_values)}")
            self.param_db["A_gate"] = self.area
        
        self.param_db["eot_corrected"] = self.param_db["eot"] * 3.9/self.e_sio2

    def config_pareto_metric_db(self):
        self.pareto_metric_db = self.sweep_model["output_metrics"]

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints_cvxpy = []