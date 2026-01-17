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
from src.inverse_pass.constraint import Constraint
from src.hardware_model.tech_models.tech_library.sweep_tech_codesign import just_fit_pareto_surface, ParamNormalizer

logger = logging.getLogger(__name__)

class SweepModel(TechModel):
    def __init__(self, model_cfg, base_params):
        self.sweep_file_prefix = model_cfg["sweep_file_prefix"]
        self.sweep_file_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + ".csv")
        self.sweep_model_path = os.path.join(os.path.dirname(__file__), "tech_library", "design_spaces", "pareto_fronts", self.sweep_file_prefix + "_model.json")
        super().__init__(model_cfg, base_params)

    # evaluate a metric given the values of the fit parameters (monomial form)
    def eval_metric(self, metric, param_values):
        assert metric in self.sweep_model["output_metrics"]
        coef = self.sweep_model["constraints"][metric]["coefficient"]
        exps = self.sweep_model["constraints"][metric]["exponents"]
        params = self.sweep_model["param_names"]
        param_dict = {param: param_values[param] for param in params}
        result = coef
        for param, exp in exps.items():
            result *= np.power(param_dict[param], exp)
        return result
    
    # Helper: find nearest design given abstract parameter values
    def find_nearest_design(self, param_values):
        query_params = np.array([[param_values[p] for p in self.sweep_model["param_names"]]])
        query_normalized = np.zeros_like(query_params)
        for i, param in enumerate(self.sweep_model["param_names"]):
            param_range = self.sweep_model["param_bounds"][param]["max"] - self.sweep_model["param_bounds"][param]["min"]
            query_normalized[0, i] = (query_params[0, i] - self.sweep_model["param_bounds"][param]["min"]) / param_range
        distance, index = self.sweep_model["param_nn_index"].kneighbors(query_normalized)
        return self.pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0])

    def find_nearest_design_by_inputs(self, input_values, input_columns=None):
        """
        Find the nearest design point in pareto_df based on input parameter values.

        Args:
            input_values: dict mapping input column names to their values
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
        if input_columns is None:
            input_columns = [c for c in input_values.keys() if c in self.pareto_df.columns]

        if len(input_columns) == 0:
            raise ValueError(f"No matching columns found. Available: {list(self.pareto_df.columns)}")

        # Extract the input data from pareto_df
        X = self.pareto_df[input_columns].values

        # Normalize by range for fair distance calculation
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min + 1e-10  # Avoid division by zero

        X_normalized = (X - X_min) / X_range

        # Build query point
        query = np.array([[input_values[c] for c in input_columns]])
        query_normalized = (query - X_min) / X_range

        # Find nearest neighbor
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_normalized)
        distance, index = nn.kneighbors(query_normalized)

        return self.pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0])

    def init_tech_specific_constants(self):
        super().init_tech_specific_constants()

    def init_transistor_equations(self):
        super().init_transistor_equations()

        # Read Pareto CSV
        logger.info(f"Reading Pareto front from: {self.sweep_file_path}")
        self.pareto_df = pd.read_csv(self.sweep_file_path)
        
        logger.info(f"Loading Pareto surface model from: {self.sweep_model_path}")
        with open(self.sweep_model_path, 'r') as f:
            self.sweep_model = json.load(f)

        Y = self.pareto_df[self.sweep_model["output_metrics"]].values
        self.normalizer = ParamNormalizer(Y, len(self.sweep_model["param_names"]), param_min=self.sweep_model["param_min"], param_max=self.sweep_model["param_max"])

        tech_values_str_ind = self.base_params.tech_values
        tech_values_str_ind = {k.name: v for k, v in tech_values_str_ind.items()}
        self.cur_design_point, _ = self.find_nearest_design_by_inputs(tech_values_str_ind)

        logger.info(f"Current design point: {self.cur_design_point}")

        self.config_param_db()

    def config_param_db(self):
        super().config_param_db()
        for param in self.cur_design_point.keys():
            self.param_db[param] = self.cur_design_point[param]
        self.I_off = self.param_db["Ioff"]
        self.P_pass_inv = self.param_db["Pstatic"]
        self.A_gate = self.param_db["area"]
        self.delay = self.param_db["delay"]
        self.E_act_inv = self.param_db["Edynamic"]
        self.C_gate = self.param_db["C_gate"]
        self.C_diff = self.C_gate
        self.C_load = self.C_gate
        self.R_avg_inv = self.param_db["R_avg_inv"]
        self.param_db["A_gate"] = self.A_gate

    def apply_base_parameter_effects(self):
        pass

    def apply_additional_effects(self):
        super().apply_additional_effects()

    def create_constraints(self, dennard_scaling_type="constant_field"):
        super().create_constraints(dennard_scaling_type)
        