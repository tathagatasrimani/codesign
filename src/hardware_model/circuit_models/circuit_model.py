import logging

from src import coefficients
from src import sim_util
import cvxpy as cp
import sympy as sp
from src.inverse_pass.constraint import Constraint
logger = logging.getLogger(__name__)

DATA_WIDTH = 16

DEBUG = False
def log_info(msg):
    if DEBUG:
        logger.info(msg)
def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

class CircuitModel:
    def __init__(self, tech_model, cfg=None):
        self.tech_model = tech_model
        # Check if wirelength costs should be set to zero
        self.zero_wirelength_costs = False
        self.constant_wire_length_cost = None
        if cfg is not None and "args" in cfg and "constant_wire_length_cost" in cfg["args"]:
            # print(f"Setting constant_wire_length_cost to {cfg['args']['constant_wire_length_cost']} from config!!!")
            self.constant_wire_length_cost = cfg["args"]["constant_wire_length_cost"]
        if cfg is not None and "args" in cfg and "zero_wirelength_costs" in cfg["args"]:
            # print(f"Setting zero_wirelength_costs to {cfg['args']['zero_wirelength_costs']} from config!!!")
            self.zero_wirelength_costs = cfg["args"]["zero_wirelength_costs"]
        self.constraints = []

        # hardcoded tech node to reference for logical effort coefficients
        self.coeffs = coefficients.create_and_save_coefficients([7])
        self.set_coefficients()

        # UNITS: ns
        self.symbolic_latency_wc = {
            "And16": lambda: self._make_fu_lat_dict("And16"),
            "Or16": lambda: self._make_fu_lat_dict("Or16"),
            "Add16": lambda: self._make_fu_lat_dict("Add16"),
            "Sub16": lambda: self._make_fu_lat_dict("Sub16"),
            "Mult16": lambda: self._make_fu_lat_dict("Mult16"),
            "FloorDiv16": lambda: self._make_fu_lat_dict("FloorDiv16"),
            "Modulus16": lambda: self._make_fu_lat_dict("Modulus16"),
            "LShift16": lambda: self._make_fu_lat_dict("LShift16"),
            "RShift16": lambda: self._make_fu_lat_dict("RShift16"),
            "BitOr16": lambda: self._make_fu_lat_dict("BitOr16"),
            "BitXor16": lambda: self._make_fu_lat_dict("BitXor16"),
            "BitAnd16": lambda: self._make_fu_lat_dict("BitAnd16"),
            "Eq16": lambda: self._make_fu_lat_dict("Eq16"),
            "NotEq16": lambda: self._make_fu_lat_dict("NotEq16"),
            "Lt16": lambda: self._make_fu_lat_dict("Lt16"),
            "LtE16": lambda: self._make_fu_lat_dict("LtE16"),
            "Gt16": lambda: self._make_fu_lat_dict("Gt16"),
            "GtE16": lambda: self._make_fu_lat_dict("GtE16"),
            "Not16": lambda: self._make_fu_lat_dict("Not16"),
            "Exp16": lambda: self._make_fu_lat_dict("Exp16"),
            "Register16": lambda: self._make_fu_lat_dict("Register16"),
            "Mux16": lambda: self._make_fu_lat_dict("Mux16"),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 1,
            "write": lambda: 1,
            "load": lambda: 1,
            "store": lambda: 1,
            "fifo": lambda: 1,
            "memory": lambda: 1,
        }
        self.DFF_DELAY = 10*self.tech_model.delay # ~10 FO4 delays

        # UNITS: nJ
        self.symbolic_energy_active = {
            "And16": lambda: self._make_fu_energy_dict("And16"),
            "Or16": lambda: self._make_fu_energy_dict("Or16"),
            "Add16": lambda: self._make_fu_energy_dict("Add16"),
            "Sub16": lambda: self._make_fu_energy_dict("Sub16"),
            "Mult16": lambda: self._make_fu_energy_dict("Mult16"),
            "FloorDiv16": lambda: self._make_fu_energy_dict("FloorDiv16"),
            "Modulus16": lambda: self._make_fu_energy_dict("Modulus16"),
            "LShift16": lambda: self._make_fu_energy_dict("LShift16"),
            "RShift16": lambda: self._make_fu_energy_dict("RShift16"),
            "BitOr16": lambda: self._make_fu_energy_dict("BitOr16"),
            "BitXor16": lambda: self._make_fu_energy_dict("BitXor16"),
            "BitAnd16": lambda: self._make_fu_energy_dict("BitAnd16"),
            "Eq16": lambda: self._make_fu_energy_dict("Eq16"),
            "NotEq16": lambda: self._make_fu_energy_dict("NotEq16"),
            "Lt16": lambda: self._make_fu_energy_dict("Lt16"),
            "LtE16": lambda: self._make_fu_energy_dict("LtE16"),
            "Gt16": lambda: self._make_fu_energy_dict("Gt16"),
            "GtE16": lambda: self._make_fu_energy_dict("GtE16"),
            "Not16": lambda: self._make_fu_energy_dict("Not16"),
            "Exp16": lambda: self._make_fu_energy_dict("Exp16"),
            "Register16": lambda: self._make_fu_energy_dict("Register16"),
            "Mux16": lambda: self._make_fu_energy_dict("Mux16"),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
            "load": lambda: 0,
            "store": lambda: 0,
            "fifo": lambda: 0,
            "memory": lambda: 0,
        }
        self.DFF_ENERGY = 20*self.tech_model.E_act_inv # TODO: get actual value

        # UNITS: W
        self.symbolic_power_passive = {
            "And16": lambda: self._make_fu_power_dict("And16"),
            "Or16": lambda: self._make_fu_power_dict("Or16"),
            "Add16": lambda: self._make_fu_power_dict("Add16"),
            "Sub16": lambda: self._make_fu_power_dict("Sub16"),
            "Mult16": lambda: self._make_fu_power_dict("Mult16"),
            "FloorDiv16": lambda: self._make_fu_power_dict("FloorDiv16"),
            "Modulus16": lambda: self._make_fu_power_dict("Modulus16"),
            "LShift16": lambda: self._make_fu_power_dict("LShift16"),
            "RShift16": lambda: self._make_fu_power_dict("RShift16"),
            "BitOr16": lambda: self._make_fu_power_dict("BitOr16"),
            "BitXor16": lambda: self._make_fu_power_dict("BitXor16"),
            "BitAnd16": lambda: self._make_fu_power_dict("BitAnd16"),
            "Eq16": lambda: self._make_fu_power_dict("Eq16"),
            "NotEq16": lambda: self._make_fu_power_dict("NotEq16"),
            "Lt16": lambda: self._make_fu_power_dict("Lt16"),
            "LtE16": lambda: self._make_fu_power_dict("LtE16"),
            "Gt16": lambda: self._make_fu_power_dict("Gt16"),
            "GtE16": lambda: self._make_fu_power_dict("GtE16"),
            "Not16": lambda: self._make_fu_power_dict("Not16"),
            "Exp16": lambda: self._make_fu_power_dict("Exp16"),
            "Register16": lambda: self._make_fu_power_dict("Register16"),
            "Mux16": lambda: self._make_fu_power_dict("Mux16"),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
            "load": lambda: 0,
            "store": lambda: 0,
            "fifo": lambda: 0,
            "memory": lambda: 0,
        }
        self.DFF_PASSIVE_POWER = 20*self.tech_model.P_pass_inv # TODO: get actual value

        # UNITS: um^2
        self.symbolic_area = {
            "And16": lambda: self._make_fu_area_dict("And16"),
            "Or16": lambda: self._make_fu_area_dict("Or16"),
            "Add16": lambda: self._make_fu_area_dict("Add16"),
            "Sub16": lambda: self._make_fu_area_dict("Sub16"),
            "Mult16": lambda: self._make_fu_area_dict("Mult16"),
            "FloorDiv16": lambda: self._make_fu_area_dict("FloorDiv16"),
            "Modulus16": lambda: self._make_fu_area_dict("Modulus16"),
            "LShift16": lambda: self._make_fu_area_dict("LShift16"),
            "RShift16": lambda: self._make_fu_area_dict("RShift16"),
            "BitOr16": lambda: self._make_fu_area_dict("BitOr16"),
            "BitXor16": lambda: self._make_fu_area_dict("BitXor16"),
            "BitAnd16": lambda: self._make_fu_area_dict("BitAnd16"),
            "Eq16": lambda: self._make_fu_area_dict("Eq16"),
            "NotEq16": lambda: self._make_fu_area_dict("NotEq16"),
            "Lt16": lambda: self._make_fu_area_dict("Lt16"),
            "LtE16": lambda: self._make_fu_area_dict("LtE16"),
            "Gt16": lambda: self._make_fu_area_dict("Gt16"),
            "GtE16": lambda: self._make_fu_area_dict("GtE16"),
            "Not16": lambda: self._make_fu_area_dict("Not16"),
            "Exp16": lambda: self._make_fu_area_dict("Exp16"),
            "Register16": lambda: self._make_fu_area_dict("Register16"),
            "Mux16": lambda: self._make_fu_area_dict("Mux16"),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
            "load": lambda: 0,
            "store": lambda: 0,
            "fifo": lambda: 0,
            "memory": lambda: 0,
        }
        self.DFF_AREA = 20*self.tech_model.area # TODO: get actual value

        # memories output from forward pass
        self.memories = {}

        # per-FU logic unit models (keyed by rsc_name_unique); populated after netlist is loaded
        self.logic_unit_models = {}

        # main mem from inverse pass
        self.symbolic_mem = {}

        # buffers from inverse pass
        self.symbolic_buf = {}

        # symbolic expressions for resource attributes (i.e. Buf latency) from inverse pass
        self.symbolic_rsc_exprs = {}

        # circuit level parameter values
        self.circuit_values = {}

        # wire length by edge
        self.edge_to_nets = {}

        self.metal_layers = ["metal1", "metal2", "metal3", "metal4", "metal5", "metal6", "metal7", "metal8", "metal9", "metal10"]

        self.update_circuit_values()

        self.create_constraints()
    
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

    def set_memories(self, memories):
        self.memories = memories
        self.update_circuit_values()

    def set_logic_unit_models(self, logic_unit_models: dict):
        self.logic_unit_models = logic_unit_models
        self.update_circuit_values()

    def compare_symbolic_mem(self):
        for key in self.symbolic_mem:
            assert key in self.memories, f"symbolic memory {key} not found in memories"      


    def update_circuit_values(self):
        # derive curcuit level values from technology values
        tv = self.tech_model.base_params.tech_values

        def resolve(val):
            """Evaluate val to float(s): handles both scalar sympy exprs and
            {rsc_name: sympy_expr} dicts produced by the per-FU _make_fu_* helpers."""
            if isinstance(val, dict):
                return {k: float(sim_util.xreplace_safe(v, tv)) for k, v in val.items()}
            return float(sim_util.xreplace_safe(val, tv))

        self.circuit_values["latency"] = {
            key: resolve(self.symbolic_latency_wc[key]()) for key in self.symbolic_latency_wc if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["dynamic_energy"] = {
            key: resolve(self.symbolic_energy_active[key]()) for key in self.symbolic_energy_active if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["passive_power"] = {
            key: resolve(self.symbolic_power_passive[key]()) for key in self.symbolic_power_passive if key not in ["Buf", "MainMem"]
        }
        self.circuit_values["area"] = {
            key: resolve(self.symbolic_area[key]()) for key in self.symbolic_area
        }

        # memory values
        self.circuit_values["latency"]["rsc"] = {
            key: self.memories[key]["Access time (ns)"] for key in self.memories
        }
        self.circuit_values["dynamic_energy"]["rsc"] = {
            "Read": {
                key: self.memories[key]["Dynamic read energy (nJ)"] for key in self.memories
            },
            "Write": {
                key: self.memories[key]["Dynamic write energy (nJ)"] for key in self.memories
            }
        }
        self.circuit_values["passive_power"]["rsc"] = {
            key: self.memories[key]["Standby leakage per bank(mW)"] * 1e-3 for key in self.memories
        }
        self.circuit_values["area"]["rsc"] = {
            key: self.memories[key]["Area (mm2)"] * 1e6 for key in self.memories
        } 

    #TODO come back and replace C_diff and C_load with the capacitance correctly sized for src and dst of each net
    def wire_delay(self, edge):
        wire_delay = 0

        if self.zero_wirelength_costs:
            return 0.0

        if self.constant_wire_length_cost is not None:
            # print(f"Using constant wire length cost of {self.constant_wire_length_cost} for edge {edge}!!")
            wire_delay = self.constant_wire_length_cost  # ns
            return wire_delay
        
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
        return wire_delay * 1e9

    # for 1 bit
    def wire_length(self, edge):
        # print(f"calculating wire length for edge {edge} and zero_wirelength_costs = {self.zero_wirelength_costs}!!")
        if self.zero_wirelength_costs:
            return 0
        if self.constant_wire_length_cost is not None:
            # print(f"Using constant wire length cost of {self.constant_wire_length_cost} for edge {edge}!!")
            return self.constant_wire_length_cost
        # wire length = sum of lengths of all segments in all nets on this edge
        wire_length = 0
        for net in self.edge_to_nets[edge]:
            for segment in net.segments:
                wire_length += segment.length
        log_info(f"wire_length for edge {edge} is {wire_length}")
        return wire_length
        
    # multiplying wire length by DATA_WIDTH because there are multiple bits on the wire.
    def wire_energy(self, edge, symbolic=False):
        # print(f"calculating wire energy for edge {edge} and zero_wirelength_costs = {self.zero_wirelength_costs}!!")
        if self.zero_wirelength_costs:
            return 0
        if self.constant_wire_length_cost is not None:
            wire_energy = 5 * 1e-3 * self.constant_wire_length_cost
            # print(f"Using constant wire length cost of {self.constant_wire_length_cost} for edge {edge} : Wire Energy {wire_energy}!!")
            return wire_energy
        # wire energy = 0.5 * C * V_dd^2 * length
        wire_energy = 0
        for net in self.edge_to_nets[edge]:
            for segment in net.segments:
                wire_energy += 0.5 * segment.length*DATA_WIDTH * self.tech_model.wire_parasitics["C"][segment.layer] * self.tech_model.base_params.V_dd**2
        return wire_energy * 1e9

    # --- Per-FU dict helpers (used when logic_unit_models is populated) ---
    # Each returns {rsc_name_unique: value} for all FUs of the given function type.
    # lum.delay/E_act_inv/P_pass_inv/area are numeric (float); clk_period/DFF_* are symbolic.

    def _fu_pipeline_cost(self, fu_delay):
        """Pipeline stage cost factor for a FU with the given combinational delay (float)."""
        clk_period = self.tech_model.base_params.clk_period
        return clk_period / (clk_period - self.DFF_DELAY)

    def _make_fu_lat_dict(self, fn_type):
        clk_period = self.tech_model.base_params.clk_period
        result = {}
        for rsc, lum in self.logic_unit_models.items():
            if lum.function == fn_type:
                result[rsc] = self.gamma[fn_type] * lum.delay * clk_period / (clk_period - self.DFF_DELAY)
        return result

    def _make_fu_energy_dict(self, fn_type):
        clk_period = self.tech_model.base_params.clk_period
        result = {}
        for rsc, lum in self.logic_unit_models.items():
            if lum.function == fn_type:
                lat = self.gamma[fn_type] * lum.delay * clk_period / (clk_period - self.DFF_DELAY)
                pipeline_cost = DATA_WIDTH * self.DFF_ENERGY * (lat / clk_period)
                result[rsc] = self.alpha[fn_type] * lum.E_act_inv + pipeline_cost
        return result

    def _make_fu_power_dict(self, fn_type):
        clk_period = self.tech_model.base_params.clk_period
        result = {}
        for rsc, lum in self.logic_unit_models.items():
            if lum.function == fn_type:
                lat = self.gamma[fn_type] * lum.delay * clk_period / (clk_period - self.DFF_DELAY)
                pipeline_cost = DATA_WIDTH * self.DFF_PASSIVE_POWER * (lat / clk_period)
                result[rsc] = self.beta[fn_type] * lum.P_pass_inv + pipeline_cost
        return result

    def _make_fu_area_dict(self, fn_type):
        clk_period = self.tech_model.base_params.clk_period
        result = {}
        for rsc, lum in self.logic_unit_models.items():
            if lum.function == fn_type:
                lat = self.gamma[fn_type] * lum.delay * clk_period / (clk_period - self.DFF_DELAY)
                pipeline_cost = DATA_WIDTH * self.DFF_AREA * (lat / clk_period)
                result[rsc] = self.area_coeffs[fn_type] * lum.area + pipeline_cost
        return result

    def create_constraints(self):
        self.constraints = []
        if self.tech_model.model_cfg["effects"]["frequency"]:
            for key in self.symbolic_latency_wc:
                if key not in ["Buf", "MainMem", "OffChipIO", "Call", "N/A"]:
                    # cycle limit to constrain the amount of pipelining
                    #self.constraints.append((self.symbolic_latency_wc[key]()* 1e-9) * self.tech_model.base_params.f <= 20) # num cycles <= 20 (cycles = time(s) * frequency(Hz))
                    latency_expr = self.symbolic_latency_wc[key]()
                    if not latency_expr: continue
                    clk_period_expr = 20 * self.tech_model.base_params.clk_period
                    self.constraints.append(Constraint(self.symbolic_latency_wc[key]() <= 20 * self.tech_model.base_params.clk_period, f"latency_{key} <= 20*clk_period")) # num cycles <= 20 (cycles = time(s) * frequency(Hz))
        for edge in self.edge_to_nets:
            self.constraints.append(Constraint(self.wire_delay(edge) + self.DFF_DELAY >= self.tech_model.base_params.clk_period, f"wire_delay_{edge} + DFF_DELAY >= clk_period"))