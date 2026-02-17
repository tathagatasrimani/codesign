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
            "And16": lambda: self.make_sym_lat_wc(self.gamma["And16"]),
            "Or16": lambda: self.make_sym_lat_wc(self.gamma["Or16"]),
            "Add16": lambda: self.make_sym_lat_wc(self.gamma["Add16"]),
            "Sub16": lambda: self.make_sym_lat_wc(self.gamma["Sub16"]),
            "Mult16": lambda: self.make_sym_lat_wc(self.gamma["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_lat_wc(self.gamma["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_lat_wc(self.gamma["Modulus16"]),
            "LShift16": lambda: self.make_sym_lat_wc(self.gamma["LShift16"]),
            "RShift16": lambda: self.make_sym_lat_wc(self.gamma["RShift16"]),
            "BitOr16": lambda: self.make_sym_lat_wc(self.gamma["BitOr16"]),
            "BitXor16": lambda: self.make_sym_lat_wc(self.gamma["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_lat_wc(self.gamma["BitAnd16"]),
            "Eq16": lambda: self.make_sym_lat_wc(self.gamma["Eq16"]),
            "NotEq16": lambda: self.make_sym_lat_wc(self.gamma["NotEq16"]),
            "Lt16": lambda: self.make_sym_lat_wc(self.gamma["Lt16"]),
            "LtE16": lambda: self.make_sym_lat_wc(self.gamma["LtE16"]),
            "Gt16": lambda: self.make_sym_lat_wc(self.gamma["Gt16"]),
            "GtE16": lambda: self.make_sym_lat_wc(self.gamma["GtE16"]),
            "Not16": lambda: self.make_sym_lat_wc(self.gamma["Not16"]),
            "Exp16": lambda: self.make_sym_lat_wc(self.gamma["Exp16"]),
            "Register16": lambda: self.make_sym_lat_wc(self.gamma["Register16"]),   
            "Mux16": lambda: self.make_sym_lat_wc(self.gamma["Mux16"]),
            "Fpadd16": lambda: self.make_sym_lat_wc(self.gamma["Fpadd16"]),
            "Fpsub16": lambda: self.make_sym_lat_wc(self.gamma["Fpsub16"]),
            "Fpmul16": lambda: self.make_sym_lat_wc(self.gamma["Fpmul16"]),
            "Fpdiv16": lambda: self.make_sym_lat_wc(self.gamma["Fpdiv16"]),
            "Buf": lambda: self.make_buf_lat_dict(),    
            "MainMem": lambda: self.make_mem_lat_dict(),
            "OffChipIO": lambda: self.make_io_lat_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
        }
        self.DFF_DELAY = 10*self.tech_model.delay # ~10 FO4 delays

        # UNITS: nJ
        self.symbolic_energy_active = {
            "And16": lambda: self.make_sym_energy_act("And16", self.alpha["And16"]),
            "Or16": lambda: self.make_sym_energy_act("Or16", self.alpha["Or16"]),
            "Add16": lambda: self.make_sym_energy_act("Add16", self.alpha["Add16"]),
            "Sub16": lambda: self.make_sym_energy_act("Sub16", self.alpha["Sub16"]),
            "Mult16": lambda: self.make_sym_energy_act("Mult16", self.alpha["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_energy_act("FloorDiv16", self.alpha["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_energy_act("Modulus16", self.alpha["Modulus16"]),
            "LShift16": lambda: self.make_sym_energy_act("LShift16", self.alpha["LShift16"]),
            "RShift16": lambda: self.make_sym_energy_act("RShift16", self.alpha["RShift16"]),
            "BitOr16": lambda: self.make_sym_energy_act("BitOr16", self.alpha["BitOr16"]),
            "BitXor16": lambda: self.make_sym_energy_act("BitXor16", self.alpha["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_energy_act("BitAnd16", self.alpha["BitAnd16"]),
            "Eq16": lambda: self.make_sym_energy_act("Eq16", self.alpha["Eq16"]),
            "NotEq16": lambda: self.make_sym_energy_act("NotEq16", self.alpha["NotEq16"]),
            "Lt16": lambda: self.make_sym_energy_act("Lt16", self.alpha["Lt16"]),
            "LtE16": lambda: self.make_sym_energy_act("LtE16", self.alpha["LtE16"]),
            "Gt16": lambda: self.make_sym_energy_act("Gt16", self.alpha["Gt16"]),
            "GtE16": lambda: self.make_sym_energy_act("GtE16", self.alpha["GtE16"]),
            "Not16": lambda: self.make_sym_energy_act("Not16", self.alpha["Not16"]),
            "Exp16": lambda: self.make_sym_energy_act("Exp16", self.alpha["Exp16"]),
            "Register16": lambda: self.make_sym_energy_act("Register16", self.alpha["Register16"]),
            "Mux16": lambda: self.make_sym_energy_act("Mux16", self.alpha["Mux16"]),
            "Fpadd16": lambda: self.make_sym_energy_act("Fpadd16", self.alpha["Fpadd16"]),
            "Fpsub16": lambda: self.make_sym_energy_act("Fpsub16", self.alpha["Fpsub16"]),
            "Fpmul16": lambda: self.make_sym_energy_act("Fpmul16", self.alpha["Fpmul16"]),
            "Fpdiv16": lambda: self.make_sym_energy_act("Fpdiv16", self.alpha["Fpdiv16"]),
            "Buf": lambda: self.make_buf_energy_active_dict(),
            "MainMem": lambda: self.make_mainmem_energy_active_dict(),
            "OffChipIO": lambda: self.make_io_energy_active_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
        }
        self.DFF_ENERGY = 20*self.tech_model.E_act_inv # TODO: get actual value

        # UNITS: W
        self.symbolic_power_passive = {
            "And16": lambda: self.make_sym_power_pass("And16", self.beta["And16"]),
            "Or16": lambda: self.make_sym_power_pass("Or16", self.beta["Or16"]),
            "Add16": lambda: self.make_sym_power_pass("Add16", self.beta["Add16"]),
            "Sub16": lambda: self.make_sym_power_pass("Sub16", self.beta["Sub16"]),
            "Mult16": lambda: self.make_sym_power_pass("Mult16", self.beta["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_power_pass("FloorDiv16", self.beta["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_power_pass("Modulus16", self.beta["Modulus16"]),
            "LShift16": lambda: self.make_sym_power_pass("LShift16", self.beta["LShift16"]),
            "RShift16": lambda: self.make_sym_power_pass("RShift16", self.beta["RShift16"]),
            "BitOr16": lambda: self.make_sym_power_pass("BitOr16", self.beta["BitOr16"]),
            "BitXor16": lambda: self.make_sym_power_pass("BitXor16", self.beta["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_power_pass("BitAnd16", self.beta["BitAnd16"]),
            "Eq16": lambda: self.make_sym_power_pass("Eq16", self.beta["Eq16"]),
            "NotEq16": lambda: self.make_sym_power_pass("NotEq16", self.beta["NotEq16"]),
            "Lt16": lambda: self.make_sym_power_pass("Lt16", self.beta["Lt16"]),
            "LtE16": lambda: self.make_sym_power_pass("LtE16", self.beta["LtE16"]),
            "Gt16": lambda: self.make_sym_power_pass("Gt16", self.beta["Gt16"]),
            "GtE16": lambda: self.make_sym_power_pass("GtE16", self.beta["GtE16"]),
            "Not16": lambda: self.make_sym_power_pass("Not16", self.beta["Not16"]),
            "Exp16": lambda: self.make_sym_power_pass("Exp16", self.beta["Exp16"]),
            "Register16": lambda: self.make_sym_power_pass("Register16", self.beta["Register16"]),
            "Mux16": lambda: self.make_sym_power_pass("Mux16", self.beta["Mux16"]),
            "Fpadd16": lambda: self.make_sym_power_pass("Fpadd16", self.beta["Fpadd16"]),
            "Fpsub16": lambda: self.make_sym_power_pass("Fpsub16", self.beta["Fpsub16"]),
            "Fpmul16": lambda: self.make_sym_power_pass("Fpmul16", self.beta["Fpmul16"]),
            "Fpdiv16": lambda: self.make_sym_power_pass("Fpdiv16", self.beta["Fpdiv16"]),
            "MainMem": lambda: self.make_mainmem_power_passive_dict(),
            "Buf": lambda: self.make_buf_power_passive_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
        }
        self.DFF_PASSIVE_POWER = 20*self.tech_model.P_pass_inv # TODO: get actual value

        # UNITS: um^2
        self.symbolic_area = {
            "And16": lambda: self.make_sym_area(self.area_coeffs["And16"]),
            "Or16": lambda: self.make_sym_area(self.area_coeffs["Or16"]),
            "Add16": lambda: self.make_sym_area(self.area_coeffs["Add16"]),
            "Sub16": lambda: self.make_sym_area(self.area_coeffs["Sub16"]),
            "Mult16": lambda: self.make_sym_area(self.area_coeffs["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_area(self.area_coeffs["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_area(self.area_coeffs["Modulus16"]),
            "LShift16": lambda: self.make_sym_area(self.area_coeffs["LShift16"]),
            "RShift16": lambda: self.make_sym_area(self.area_coeffs["RShift16"]),
            "BitOr16": lambda: self.make_sym_area(self.area_coeffs["BitOr16"]),
            "BitXor16": lambda: self.make_sym_area(self.area_coeffs["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_area(self.area_coeffs["BitAnd16"]),
            "Eq16": lambda: self.make_sym_area(self.area_coeffs["Eq16"]),
            "NotEq16": lambda: self.make_sym_area(self.area_coeffs["NotEq16"]),
            "Lt16": lambda: self.make_sym_area(self.area_coeffs["Lt16"]),
            "LtE16": lambda: self.make_sym_area(self.area_coeffs["LtE16"]),
            "Gt16": lambda: self.make_sym_area(self.area_coeffs["Gt16"]),
            "GtE16": lambda: self.make_sym_area(self.area_coeffs["GtE16"]),
            "Not16": lambda: self.make_sym_area(self.area_coeffs["Not16"]),
            "Exp16": lambda: self.make_sym_area(self.area_coeffs["Exp16"]),
            "Register16": lambda: self.make_sym_area(self.area_coeffs["Register16"]),
            "Mux16": lambda: self.make_sym_area(self.area_coeffs["Mux16"]),
            "Fpadd16": lambda: self.make_sym_area(self.area_coeffs["Fpadd16"]),
            "Fpsub16": lambda: self.make_sym_area(self.area_coeffs["Fpsub16"]),
            "Fpmul16": lambda: self.make_sym_area(self.area_coeffs["Fpmul16"]),
            "Fpdiv16": lambda: self.make_sym_area(self.area_coeffs["Fpdiv16"]),
            "N/A": lambda: 0,
            "Call": lambda: 0,
            "read": lambda: 0,
            "write": lambda: 0,
        }

        # memories output from forward pass
        self.memories = {}

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

    def compare_symbolic_mem(self):
        for key in self.symbolic_mem:
            assert key in self.memories, f"symbolic memory {key} not found in memories"      


    def update_circuit_values(self):
        # derive curcuit level values from technology values
        self.circuit_values["latency"] = {
            key: float(sim_util.xreplace_safe(self.symbolic_latency_wc[key](), self.tech_model.base_params.tech_values)) for key in self.symbolic_latency_wc if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["dynamic_energy"] = {
            key: float(sim_util.xreplace_safe(self.symbolic_energy_active[key](), self.tech_model.base_params.tech_values)) for key in self.symbolic_energy_active if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["passive_power"] = {
            key: float(sim_util.xreplace_safe(self.symbolic_power_passive[key](), self.tech_model.base_params.tech_values)) for key in self.symbolic_power_passive if key not in ["Buf", "MainMem"]
        }
        self.circuit_values["area"] = {
            key: float(sim_util.xreplace_safe(self.symbolic_area[key](), self.tech_model.base_params.tech_values)) for key in self.symbolic_area
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
        
    def make_sym_lat_wc(self, gamma):
        unpipelined_delay = gamma * self.tech_model.delay
        # more pipelining means that DFFs become a larger portion of the total delay
        pipeline_cost = self.tech_model.base_params.clk_period/(self.tech_model.base_params.clk_period - self.DFF_DELAY)
        return unpipelined_delay * pipeline_cost
    
    def make_buf_lat_dict(self):
        return self.tech_model.base_params.BufL

    def make_mem_lat_dict(self):    
        d = {}
        for mem in self.tech_model.base_params.MemReadL:
            d[mem] = (self.tech_model.base_params.MemReadL[mem] + self.tech_model.base_params.MemWriteL[mem]) / 2
        return d

    def make_io_lat_dict(self):
        return self.tech_model.base_params.OffChipIOL
    
    def make_buf_energy_active_dict(self):
        d = {}
        for mem in self.tech_model.base_params.BufReadEact:
            d[mem] = ((self.tech_model.base_params.BufReadEact[mem] + self.tech_model.base_params.BufWriteEact[mem]) / 2)
        return d

    def make_mainmem_energy_active_dict(self):
        d = {}
        for mem in self.tech_model.base_params.MemWriteEact:
            d[mem] = ((self.tech_model.base_params.MemWriteEact[mem] + self.tech_model.base_params.MemReadEact[mem]) / 2)
        return d    

    def make_io_energy_active_dict(self):
        return self.tech_model.base_params.OffChipIOPact * self.tech_model.base_params.OffChipIOL
    
    def make_sym_energy_act(self, key, alpha):
        unpipelined_energy = alpha * self.tech_model.E_act_inv
        pipeline_cost = DATA_WIDTH * self.DFF_ENERGY * (self.symbolic_latency_wc[key]()/self.tech_model.base_params.clk_period) # DATA_WIDTH DFFs needed for each extra cycle
        return unpipelined_energy + pipeline_cost

    def make_mainmem_power_passive_dict(self):
        return self.tech_model.base_params.MemPpass
    
    def make_buf_power_passive_dict(self):
        return self.tech_model.base_params.BufPpass
    
    def make_sym_power_pass(self, key, beta):
        unpipelined_power = beta * self.tech_model.P_pass_inv
        pipeline_cost = DATA_WIDTH * self.DFF_PASSIVE_POWER * (self.symbolic_latency_wc[key]()/self.tech_model.base_params.clk_period) # DATA_WIDTH DFFs needed for each extra cycle
        return unpipelined_power + pipeline_cost

    def make_sym_area(self, area_coeff):
        return area_coeff * self.tech_model.base_params.area

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