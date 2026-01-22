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
from src import coefficients
logger = logging.getLogger(__name__)

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

class objective_evaluator:
    def __init__(self, tech_model, total_active_energy, total_passive_energy, scheduled_dfgs, dataflow_blocks, obj_fn, top_block_name, loop_1x_graphs, edge_to_nets):
        # hardcoded tech node to reference for logical effort coefficients
        self.coeffs = coefficients.create_and_save_coefficients([7])
        self.set_coefficients()
        self.tech_model = tech_model
        self.total_active_energy = total_active_energy
        self.total_passive_energy = total_passive_energy
        self.scheduled_dfgs = scheduled_dfgs
        self.loop_1x_graphs = loop_1x_graphs
        self.dataflow_blocks = dataflow_blocks
        self.obj = 0
        self.obj_fn = obj_fn
        self.top_block_name = top_block_name
        self.edge_to_nets = edge_to_nets

    def set_coefficients(self):
        self.alpha = self.coeffs["alpha"]
        self.beta = self.coeffs["beta"]
        self.gamma = self.coeffs["gamma"]
        self.area_coeffs = self.coeffs["area"]

        # TODO: add actual data for Exp16
        self.alpha["Exp16"] = 3*(self.alpha["Mult16"] + self.alpha["Add16"])
        self.beta["Exp16"] = self.beta["Mult16"] + self.beta["Add16"]
        self.gamma["Exp16"] = 3*(self.gamma["Mult16"] + self.gamma["Add16"])
        self.area_coeffs["Exp16"] = self.area_coeffs["Mult16"] + self.area_coeffs["Add16"]
    
    def calculate_objective(self):
        self.execution_time = self.calculate_execution_time()
        if self.obj_fn == "edp":
            self.obj = (self.total_passive_energy + self.total_active_energy) * self.execution_time
        elif self.obj_fn == "ed2":
            self.obj = (self.total_passive_energy + self.total_active_energy) * (self.execution_time)**2
        elif self.obj_fn == "delay":
            self.obj = self.execution_time
        elif self.obj_fn == "energy":
            self.obj = self.total_active_energy + self.total_passive_energy
        else:
            raise ValueError(f"Objective function {self.obj_fn} not supported")
        self.obj = sim_util.xreplace_safe(self.obj, self.tech_model.base_params.tech_values)
        self.total_power = sim_util.xreplace_safe((self.total_passive_energy + self.total_active_energy) / self.execution_time, self.tech_model.base_params.tech_values)

    def calculate_execution_time(self):
        self.node_arrivals = {}
        self.graph_delays = {}
        for basic_block_name in self.scheduled_dfgs:
            #self.node_arrivals[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}}
            self.node_arrivals[basic_block_name] = {"full": {}, "loop_1x": {}, "loop_2x": {}}
        log_info(f"scheduled dfgs: {self.scheduled_dfgs.keys()}")
        graph_end_node = f"graph_end_{self.top_block_name}" if self.top_block_name not in self.dataflow_blocks else f"{self.top_block_name}_graph_end_{self.top_block_name}"
        return self.calculate_execution_time_vitis_recursive(self.top_block_name, self.scheduled_dfgs[self.top_block_name], graph_end_node=graph_end_node)

    def get_rsc_edge(self, edge, dfg):
        if "rsc" in dfg.nodes[edge[0]] and "rsc" in dfg.nodes[edge[1]]:
            return (dfg.nodes[edge[0]]["rsc"], dfg.nodes[edge[1]]["rsc"])
        else:
            return edge

    def calculate_execution_time_vitis_recursive(self, basic_block_name, dfg, graph_end_node="graph_end", graph_type="full", resource_delays_only=False):
        log_info(f"calculating execution time for {basic_block_name} with graph end node {graph_end_node}")
        for node in dfg.nodes:
            #self.node_arrivals[basic_block_name][graph_type][node] = sp.symbols(f"node_arrivals_{basic_block_name}_{graph_type}_{node}")
            self.node_arrivals[basic_block_name][graph_type][node] = 0
        for node in dfg.nodes:   
            preds = list(dfg.predecessors(node))
            for pred in preds:
                pred_delay = 0.0
                if dfg.edges[pred, node]["resource_edge"]:
                    if dfg.nodes[pred]["function"] == "II":
                        loop_name = dfg.nodes[pred]["loop_name"]
                        delay_1x = self.calculate_execution_time_vitis_recursive(basic_block_name, self.loop_1x_graphs[loop_name][True], graph_end_node="loop_end_1x", graph_type="loop_1x", resource_delays_only=True)
                        # TODO add dependence of II on loop-carried dependency
                        pred_delay = delay_1x * (int(dfg.nodes[pred]["count"])-1)
                    else:
                        pred_delay = sim_util.xreplace_safe(self.tech_model.base_params.clk_period, self.tech_model.base_params.tech_values)
                elif dfg.nodes[pred]["function"] == "Call": # if function call, recursively calculate its delay 
                    if dfg.nodes[pred]["call_function"] not in self.graph_delays:
                        self.graph_delays[dfg.nodes[pred]["call_function"]] = self.calculate_execution_time_vitis_recursive(dfg.nodes[pred]["call_function"], self.scheduled_dfgs[dfg.nodes[pred]["call_function"]], graph_end_node=f"graph_end_{dfg.nodes[pred]['call_function']}")
                    pred_delay = self.graph_delays[dfg.nodes[pred]["call_function"]]
                elif not resource_delays_only:
                    if dfg.nodes[pred]["function"] == "Wire":
                        src = dfg.nodes[pred]["src_node"]
                        dst = dfg.nodes[pred]["dst_node"]
                        rsc_edge = self.get_rsc_edge((src, dst), dfg)
                        if rsc_edge in self.edge_to_nets:
                            pred_delay = self.wire_delay(rsc_edge)
                            log_info(f"added wire delay {self.wire_delay(rsc_edge)} for edge {rsc_edge}")
                        else:
                            log_info(f"no wire delay for edge {rsc_edge}")
                    else:
                        pred_delay = self.latency(dfg.nodes[pred]["function"]) 
                log_info(f"pred_delay: {pred_delay}")
                self.node_arrivals[basic_block_name][graph_type][node] = max(self.node_arrivals[basic_block_name][graph_type][pred] + pred_delay, self.node_arrivals[basic_block_name][graph_type][node])
        return self.node_arrivals[basic_block_name][graph_type][graph_end_node]

    #TODO come back and replace C_diff and C_load with the capacitance correctly sized for src and dst of each net
    def wire_delay(self, edge):
        wire_delay = 0
        for net in self.edge_to_nets[edge]:
            #logger.info(f"calculating wire delay for net {net.net_id}")
            R_on_line = self.tech_model.R_avg_inv
            C_current = self.tech_model.C_diff
            wire_delay += R_on_line * C_current
            for segment in net.segments:
                #logger.info(f"calculating wire delay for segment in layer {segment.layer} with length {segment.length}")
                C_current = segment.length * self.tech_model.wire_parasitics["C"][segment.layer]
                R_on_line += segment.length * self.tech_model.wire_parasitics["R"][segment.layer]
                wire_delay += R_on_line * C_current
            C_current = self.tech_model.C_load
            wire_delay += R_on_line * C_current
        return sim_util.xreplace_safe(wire_delay * 1e9, self.tech_model.base_params.tech_values)

    def latency(self, op_type):
        return math.ceil(self.gamma[op_type] * sim_util.xreplace_safe(self.tech_model.delay, self.tech_model.base_params.tech_values))


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
            logger.info(f"after optimization, {metric} is {metric_vals[metric]}")
        self.cur_design_point, dist, self.cur_design_point_index = self.find_nearest_design_by_values(metric_vals)
        assert dist < 1e-6, f"distance to nearest design should be 0: {dist}"
        self.set_params_from_design_point(self.cur_design_point)

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
        if not self.base_params.output_parameters_initialized:
            self.base_params.init_output_parameters_basic(self.sweep_model["output_metrics"])
        tech_values_str_ind = {}
        for k, v in self.base_params.tech_values.items():
            if k in self.base_params.names:
                tech_values_str_ind[self.base_params.names[k]] = v
        self.cur_design_point, dist, self.cur_design_point_index = self.find_nearest_design_by_values(tech_values_str_ind)
        self.set_params_from_design_point(self.cur_design_point)

        logger.info(f"Current design point: {self.cur_design_point}")
        logger.info(f"distance to nearest design: {dist}")
        logger.info(f"index of nearest design: {self.cur_design_point_index}")

        if "delay" not in self.sweep_model["output_metrics"]:
            self.delay = (self.R_avg_inv * (self.C_diff + self.C_wire/2) + (self.R_avg_inv + self.R_wire) * (self.C_wire/2 + self.C_load)) * 1e9  # ns
            logger.info(f"set delay to {sim_util.xreplace_safe(self.delay, self.base_params.tech_values)}")
        if "E_act_inv" not in self.sweep_model["output_metrics"]:
            self.E_act_inv = (0.5*(self.C_load + self.C_diff + self.C_wire)*self.base_params.V_dd*self.base_params.V_dd) * 1e9  # nJ
            logger.info(f"set E_act_inv to {sim_util.xreplace_safe(self.E_act_inv, self.base_params.tech_values)}")
        if "P_pass_inv" not in self.sweep_model["output_metrics"]:
            self.P_pass_inv = self.Ioff * self.base_params.V_dd
            logger.info(f"set P_pass_inv to {sim_util.xreplace_safe(self.P_pass_inv, self.base_params.tech_values)}")

        self.config_param_db()

    def set_params_from_design_point(self, design_point):
        for metric in self.sweep_model["output_metrics"]:
            self.base_params.set_symbol_value(getattr(self.base_params, metric), design_point[metric])
            setattr(self, metric, getattr(self.base_params, metric))
        if hasattr(self, "C_gate"):
            self.C_load = self.C_gate
            self.C_diff = self.C_gate
        

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
        
        if "A_gate" not in self.sweep_model["output_metrics"]:
            logger.info(f"setting A_gate to {sim_util.xreplace_safe(self.area, self.base_params.tech_values)}")
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