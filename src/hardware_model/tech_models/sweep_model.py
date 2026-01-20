import logging
from src.hardware_model.tech_models.tech_model_base import TechModel
from src.sim_util import symbolic_convex_max, symbolic_min, custom_cosh, custom_exp
import math
import json
from sympy import symbols, ceiling, expand, exp, Abs, cosh, log
import sympy as sp
import numpy as np
import pandas as pd
import os
import cvxpy as cp
from src.inverse_pass.constraint import Constraint
from src.hardware_model.tech_models.tech_library.sweep_tech_codesign import just_fit_pareto_surface
from src import sim_util
logger = logging.getLogger(__name__)

class SweepModel(TechModel):
    def __init__(self, model_cfg, base_params):
        self.sweep_file_prefix = model_cfg["sweep_file_prefix"]
        self.sweep_file_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + ".csv")
        self.sweep_model_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + "_model.json")
        self.cur_design_point_index = None
        super().__init__(model_cfg, base_params)

    def process_design_point(self):

        if "delay" not in self.sweep_model["output_metrics"]:
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
            logger.info(f"set delay to {self.delay.value}")
        if "E_act_inv" not in self.sweep_model["output_metrics"]:
            self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
            logger.info(f"set E_act_inv to {self.E_act_inv.value}")
        if "P_pass_inv" not in self.sweep_model["output_metrics"]:
            eps = 1e-20
            self.P_pass_inv = self.Ioff * self.base_params.V_dd + eps
            logger.info(f"set P_pass_inv to {self.P_pass_inv.value}")

        self.param_constant_constraints = []
        for i in range(self.base_params.w.size):
            self.param_constant_constraints.append(Constraint(self.base_params.w[i] == self.base_params.w.value[i], f"w[{i}] == {self.base_params.w.value[i]}"))

    def eval_metric(self, metric):
        input_vec = self.pareto_df[metric].values
        return input_vec @ self.A[metric]

    def process_optimization_results(self):
        metric_vals = {}
        for metric in self.sweep_model["output_metrics"]:
            metric_vals[metric] = getattr(self, metric).value
            logger.info(f"after optimization, {metric} is {metric_vals[metric]}")
        self.cur_design_point, dist, self.cur_design_point_index = self.find_nearest_design_by_values(metric_vals)
        logger.info(f"nearest design: {self.cur_design_point}, index: {self.cur_design_point_index}, distance: {dist}")

        self.process_design_point()

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
        logger.info(f"Reading Pareto front from: {self.sweep_file_path}")
        self.pareto_df = pd.read_csv(self.sweep_file_path)
        #self.pareto_df_log = np.log(self.pareto_df)
        
        logger.info(f"Loading Pareto surface model from: {self.sweep_model_path}")
        with open(self.sweep_model_path, 'r') as f:
            self.sweep_model = json.load(f)
        
        self.config_pareto_metric_db()

        self.X = self.pareto_df[self.sweep_model["input_metrics"]].values

        self.A = np.array([[self.sweep_model["constraints"][output_metric]["exponents"][metric] for metric in self.sweep_model["input_metrics"]] for output_metric in self.sweep_model["output_metrics"]])

        self.Y = self.pareto_df[self.sweep_model["output_metrics"]].values
        
        self.base_params.init_weights(len(self.sweep_model["input_metrics"]))
        tech_values_str_ind = {}
        for k, v in self.base_params.tech_values.items():
            if k in self.base_params.names:
                tech_values_str_ind[self.base_params.names[k]] = v
        self.cur_design_point, dist, self.cur_design_point_index = self.find_nearest_design_by_values(tech_values_str_ind)
        weight_values = np.zeros(len(self.sweep_model["input_metrics"]))
        for i, input_metric in enumerate(self.sweep_model["input_metrics"]):
            weight_values[i] = self.cur_design_point[input_metric]
        self.base_params.set_symbol_value(self.base_params.w, weight_values)

        logger.info(f"Current design point: {self.cur_design_point}")
        logger.info(f"distance to nearest design: {dist}")
        logger.info(f"index of nearest design: {self.cur_design_point_index}")

        for metric in self.sweep_model["output_metrics"]:
            metric_val = self.sweep_model["constraints"][metric]["coefficient"]
            for i in range(len(self.sweep_model["input_metrics"])):
                metric_val = metric_val * cp.power(self.base_params.w[i], self.sweep_model["constraints"][metric]["exponents"][self.sweep_model["input_metrics"][i]])
            setattr(self, metric, metric_val)

        if hasattr(self, "C_gate"):
            self.C_load = self.C_gate
            self.C_diff = self.C_gate
        self.config_param_db()
        
        self.process_design_point()

    def config_param_db(self):
        super().config_param_db()
        for param in self.cur_design_point.keys():
            if param not in self.sweep_model["output_metrics"]:
                self.param_db[param] = self.cur_design_point[param]
        
        for metric in self.sweep_model["output_metrics"]:
            self.param_db[metric] = getattr(self, metric)
        
        if "Ioff" in self.sweep_model["output_metrics"]:
            self.I_off = getattr(self, "Ioff")
            self.param_db["I_off"] = self.I_off
        
        if "A_gate" not in self.param_db:
            self.param_db["A_gate"] = self.area

    def config_pareto_metric_db(self):
        self.pareto_metric_db = self.sweep_model["output_metrics"]

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        self.constraints_cvxpy = []

        for i in range(len(self.sweep_model["input_metrics"])):
            min_constraint = self.base_params.w[i] >= self.sweep_model["param_bounds"][self.sweep_model["input_metrics"][i]]["min"]
            max_constraint = self.base_params.w[i] <= self.sweep_model["param_bounds"][self.sweep_model["input_metrics"][i]]["max"]
            self.constraints_cvxpy.append(Constraint(min_constraint, f"w[{i}] >= {self.sweep_model['param_bounds'][self.sweep_model['input_metrics'][i]]['min']}"))
            self.constraints_cvxpy.append(Constraint(max_constraint, f"w[{i}] <= {self.sweep_model['param_bounds'][self.sweep_model['input_metrics'][i]]['max']}"))
            assert min_constraint.is_dgp(), f"min_constraint is not DGP: {min_constraint}"
            assert max_constraint.is_dgp(), f"max_constraint is not DGP: {max_constraint}"
            logger.info(f"constraint cvx in tech model: {min_constraint}")