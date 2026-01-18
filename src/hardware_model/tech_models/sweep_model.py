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
from src.hardware_model.tech_models.tech_library.sweep_tech_codesign import just_fit_pareto_surface, ParamNormalizer
from src import sim_util
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
            result *= param_dict[param]**exp
        return result
    
    # Helper: find nearest design given abstract parameter values
    def find_nearest_design(self, param_values):
        from sklearn.neighbors import NearestNeighbors

        param_names = self.sweep_model["param_names"]

        # Build query point from param_values
        query_params = np.array([[param_values[p] for p in param_names]])

        # Normalize query using param bounds
        query_normalized = np.zeros_like(query_params)
        for i, param in enumerate(param_names):
            param_min = self.sweep_model["param_bounds"][param]["min"]
            param_max = self.sweep_model["param_bounds"][param]["max"]
            param_range = param_max - param_min
            if param_range > 0:
                query_normalized[0, i] = (query_params[0, i] - param_min) / param_range
            else:
                query_normalized[0, i] = 0.0

        # Build normalized param space from pareto data using the normalizer
        # The params are derived from output metrics, so use the normalizer to get them
        Y = self.pareto_df[self.sweep_model["output_metrics"]].values
        params_all = self.normalizer.metrics_to_params(Y, as_array=True)

        # Normalize the param space
        params_min = np.array([self.sweep_model["param_bounds"][p]["min"] for p in param_names])
        params_max = np.array([self.sweep_model["param_bounds"][p]["max"] for p in param_names])
        params_range = params_max - params_min
        params_range[params_range == 0] = 1.0  # Avoid division by zero
        params_normalized = (params_all - params_min) / params_range

        # Build KNN index and find nearest neighbor
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(params_normalized)
        distance, index = nn.kneighbors(query_normalized)

        return self.pareto_df.iloc[index[0][0]].to_dict(), float(distance[0][0])

    def calculate_and_process_design_point(self):
        logger.info(f"Current param values: {self.cur_param_values}")
        for param in self.cur_param_values.keys():
            if not hasattr(self, param):
                setattr(self, param, cp.Variable(pos=True))
            param_var = getattr(self, param)
            param_var.value = self.cur_param_values[param]
            self.cur_param_values[param] = param_var
            setattr(self.base_params, param, param_var)
            self.base_params.set_symbol_value(getattr(self.base_params, param), param_var.value)
            self.base_params.symbol_table[param] = param_var
            self.base_params.names[param_var] = param

        self.output_metric_vals = {
            metric: self.eval_metric(metric, self.cur_param_values) for metric in self.sweep_model["output_metrics"]
        }


        for metric in self.sweep_model["output_metrics"]:
            logger.info(f"evaluated {metric}: {self.output_metric_vals[metric].value}")
            setattr(self, metric, self.output_metric_vals[metric])
            #self.base_params.tech_values[self.base_params.symbol_table[metric]] = self.output_metric_vals[metric]

        for metric in self.cur_design_point.keys():
            if metric not in self.sweep_model["output_metrics"] and hasattr(self.base_params, metric):
                self.base_params.set_symbol_value(self.base_params.symbol_table[metric], self.cur_design_point[metric])
        
        if hasattr(self, "C_gate"):
            self.C_diff = self.C_gate
            self.C_load = self.C_gate

        self.config_param_db()

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

    def process_optimization_results(self, variables):
        param_values = {}
        for var in variables:
            if var in self.base_params.names:
                var_name = self.base_params.names[var]
                if var_name in self.sweep_model["param_names"]:
                    param_values[var_name] = var.value
        assert len(param_values) == len(self.sweep_model["param_names"]), f"number of parameters in optimization results does not match number of parameters in sweep model: {len(param_values)} != {len(self.sweep_model['param_names'])}"
        self.cur_param_values = param_values
        self.cur_design_point, dist = self.find_nearest_design(self.cur_param_values)
        logger.info(f"optimization results: {self.cur_param_values}")
        logger.info(f"nearest design: {self.cur_design_point}, distance: {dist}")
        self.calculate_and_process_design_point()


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
            raise ValueError(f"No matching columns found. Available: {list(self.pareto_df.columns)}. Input values: {input_values}")

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
        
        self.config_pareto_metric_db()

        Y = self.pareto_df[self.sweep_model["output_metrics"]].values
        self.normalizer = ParamNormalizer(Y, len(self.sweep_model["param_names"]), param_min=self.sweep_model["param_min"], param_max=self.sweep_model["param_max"])

        model_was_already_initialized = False
        for param_name in self.sweep_model["param_names"]:
            if hasattr(self.base_params, param_name):
                model_was_already_initialized = True
                self.cur_param_values = {p_name: getattr(self.base_params, p_name).value for p_name in self.sweep_model["param_names"]}
                self.cur_design_point, dist = self.find_nearest_design(self.cur_param_values)
                logger.info(f"loaded parameter values: {self.cur_param_values}")
                logger.info(f"nearest design: {self.cur_design_point}, distance: {dist}")
                break
        if not model_was_already_initialized:
            tech_values_str_ind = {}
            for k, v in self.base_params.tech_values.items():
                if k in self.base_params.names:
                    tech_values_str_ind[self.base_params.names[k]] = v
            self.cur_design_point, _ = self.find_nearest_design_by_inputs(tech_values_str_ind)

            logger.info(f"Current design point: {self.cur_design_point}")

            self.cur_output_metrics = {metric: self.cur_design_point[metric] for metric in self.sweep_model["output_metrics"]}

            self.cur_param_values = self.normalizer.metrics_to_params(self.cur_output_metrics)
        
        self.calculate_and_process_design_point()

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

        for param in self.cur_param_values.keys():
            min_constraint = getattr(self, param) >= self.sweep_model["param_bounds"][param]["min"]
            max_constraint = getattr(self, param) <= self.sweep_model["param_bounds"][param]["max"]
            self.constraints_cvxpy.append(Constraint(min_constraint, f"{param} >= {self.sweep_model['param_bounds'][param]['min']}"))
            self.constraints_cvxpy.append(Constraint(max_constraint, f"{param} <= {self.sweep_model['param_bounds'][param]['max']}"))
            assert min_constraint.is_dgp(), f"min_constraint is not DGP: {min_constraint}"
            assert max_constraint.is_dgp(), f"max_constraint is not DGP: {max_constraint}"
            logger.info(f"constraint cvx in tech model: {min_constraint}")
            logger.info(f"constraint cvx in tech model: {max_constraint}")

        self.param_constant_constraints = []
        for param in self.cur_param_values.keys():
            self.param_constant_constraints.append(Constraint(getattr(self, param) == sim_util.xreplace_safe(self.cur_param_values[param], self.base_params.tech_values), f"{param} == {self.cur_param_values[param]}"))
            assert self.param_constant_constraints[-1].constraint.is_dgp(), f"param_constant_constraint is not DGP: {self.param_constant_constraints[-1]}"
            logger.info(f"param_constant_constraint in tech model: {self.param_constant_constraints[-1]}")
        