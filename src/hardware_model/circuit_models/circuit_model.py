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
        self.constraints_cvx = []

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
            "Buf": lambda: self.make_buf_lat_dict(),    
            "MainMem": lambda: self.make_mem_lat_dict(),
            "OffChipIO": lambda: self.make_io_lat_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
        }

        # UNITS: nJ
        self.symbolic_energy_active = {
            "And16": lambda: self.make_sym_energy_act(self.alpha["And16"]),
            "Or16": lambda: self.make_sym_energy_act(self.alpha["Or16"]),
            "Add16": lambda: self.make_sym_energy_act(self.alpha["Add16"]),
            "Sub16": lambda: self.make_sym_energy_act(self.alpha["Sub16"]),
            "Mult16": lambda: self.make_sym_energy_act(self.alpha["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_energy_act(self.alpha["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_energy_act(self.alpha["Modulus16"]),
            "LShift16": lambda: self.make_sym_energy_act(self.alpha["LShift16"]),
            "RShift16": lambda: self.make_sym_energy_act(self.alpha["RShift16"]),
            "BitOr16": lambda: self.make_sym_energy_act(self.alpha["BitOr16"]),
            "BitXor16": lambda: self.make_sym_energy_act(self.alpha["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_energy_act(self.alpha["BitAnd16"]),
            "Eq16": lambda: self.make_sym_energy_act(self.alpha["Eq16"]),
            "NotEq16": lambda: self.make_sym_energy_act(self.alpha["NotEq16"]),
            "Lt16": lambda: self.make_sym_energy_act(self.alpha["Lt16"]),
            "LtE16": lambda: self.make_sym_energy_act(self.alpha["LtE16"]),
            "Gt16": lambda: self.make_sym_energy_act(self.alpha["Gt16"]),
            "GtE16": lambda: self.make_sym_energy_act(self.alpha["GtE16"]),
            "Not16": lambda: self.make_sym_energy_act(self.alpha["Not16"]),
            "Exp16": lambda: self.make_sym_energy_act(self.alpha["Exp16"]),
            "Register16": lambda: self.make_sym_energy_act(self.alpha["Register16"]),
            "Buf": lambda: self.make_buf_energy_active_dict(),
            "MainMem": lambda: self.make_mainmem_energy_active_dict(),
            "OffChipIO": lambda: self.make_io_energy_active_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
        }

        # UNITS: W
        self.symbolic_power_passive = {
            "And16": lambda: self.make_sym_power_pass(self.beta["And16"]),
            "Or16": lambda: self.make_sym_power_pass(self.beta["Or16"]),
            "Add16": lambda: self.make_sym_power_pass(self.beta["Add16"]),
            "Sub16": lambda: self.make_sym_power_pass(self.beta["Sub16"]),
            "Mult16": lambda: self.make_sym_power_pass(self.beta["Mult16"]),
            "FloorDiv16": lambda: self.make_sym_power_pass(self.beta["FloorDiv16"]),
            "Modulus16": lambda: self.make_sym_power_pass(self.beta["Modulus16"]),
            "LShift16": lambda: self.make_sym_power_pass(self.beta["LShift16"]),
            "RShift16": lambda: self.make_sym_power_pass(self.beta["RShift16"]),
            "BitOr16": lambda: self.make_sym_power_pass(self.beta["BitOr16"]),
            "BitXor16": lambda: self.make_sym_power_pass(self.beta["BitXor16"]),
            "BitAnd16": lambda: self.make_sym_power_pass(self.beta["BitAnd16"]),
            "Eq16": lambda: self.make_sym_power_pass(self.beta["Eq16"]),
            "NotEq16": lambda: self.make_sym_power_pass(self.beta["NotEq16"]),
            "Lt16": lambda: self.make_sym_power_pass(self.beta["Lt16"]),
            "LtE16": lambda: self.make_sym_power_pass(self.beta["LtE16"]),
            "Gt16": lambda: self.make_sym_power_pass(self.beta["Gt16"]),
            "GtE16": lambda: self.make_sym_power_pass(self.beta["GtE16"]),
            "Not16": lambda: self.make_sym_power_pass(self.beta["Not16"]),
            "Exp16": lambda: self.make_sym_power_pass(self.beta["Exp16"]),
            "Register16": lambda: self.make_sym_power_pass(self.beta["Register16"]),
            "MainMem": lambda: self.make_mainmem_power_passive_dict(),
            "Buf": lambda: self.make_buf_power_passive_dict(),
            "N/A": lambda: 0,
            "Call": lambda: 0,
        }

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
            "N/A": lambda: 0,
            "Call": lambda: 0,
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

    def set_uarch_constraints(self):
        self.tech_model.constraints.append(self.logic_delay >= self.tech_model.delay)
        self.tech_model.constraints.append(self.logic_energy_active >= self.tech_model.E_act_inv)
        self.tech_model.constraints.append(self.logic_power_passive >= self.tech_model.P_pass_inv)
        for layer in self.metal_layers:
            self.tech_model.constraints.append(self.wire_unit_delay[layer] >= self.tech_model.wire_parasitics["R"][layer]*self.tech_model.wire_parasitics["C"][layer])
            self.tech_model.constraints.append(self.wire_unit_energy[layer] >= 0.5*self.tech_model.wire_parasitics["C"][layer]*self.tech_model.base_params.V_dd**2)

    def set_uarch_parameters(self):
        self.clk_period_cvx = cp.Variable(pos=True)
        self.clk_period_cvx.value = float(sim_util.xreplace_safe(self.tech_model.base_params.clk_period, self.tech_model.base_params.tech_values))
        self.logic_delay_cvx = cp.Parameter(pos=True)
        self.logic_delay_cvx.value = float(sim_util.xreplace_safe(self.tech_model.delay, self.tech_model.base_params.tech_values))
        self.logic_energy_active_cvx = cp.Parameter(pos=True)
        self.logic_energy_active_cvx.value = float(sim_util.xreplace_safe(self.tech_model.E_act_inv, self.tech_model.base_params.tech_values))
        self.logic_power_passive_cvx = cp.Parameter(pos=True)
        self.logic_power_passive_cvx.value = float(sim_util.xreplace_safe(self.tech_model.P_pass_inv, self.tech_model.base_params.tech_values))

        
        self.uarch_lat_cvx = {
            key: self.gamma[key]*self.logic_delay_cvx for key in self.gamma
        }
        self.uarch_lat_cvx["N/A"] = 0
        self.uarch_lat_cvx["Call"] = 0
        self.uarch_energy_active_cvx = {
            key: self.alpha[key]*self.logic_energy_active_cvx for key in self.alpha
        }
        self.uarch_energy_active_cvx["N/A"] = 0
        self.uarch_energy_active_cvx["Call"] = 0
        self.uarch_power_passive_cvx = {
            key: self.beta[key]*self.logic_power_passive_cvx for key in self.beta
        }
        self.uarch_power_passive_cvx["N/A"] = 0
        self.uarch_power_passive_cvx["Call"] = 0
        self.wire_unit_delay_cvx = {
            layer: cp.Variable(pos=True) for layer in self.metal_layers
        }
        self.wire_unit_energy_cvx = {
            layer: cp.Variable(pos=True) for layer in self.metal_layers
        }
        for layer in self.metal_layers:
            self.wire_unit_delay_cvx[layer].value = float(sim_util.xreplace_safe(self.tech_model.wire_parasitics["R"][layer]*self.tech_model.wire_parasitics["C"][layer], self.tech_model.base_params.tech_values))
            self.wire_unit_energy_cvx[layer].value = float(sim_util.xreplace_safe(0.5*self.tech_model.wire_parasitics["C"][layer]*self.tech_model.base_params.V_dd**2, self.tech_model.base_params.tech_values))

        self.wire_C_values = {}
        self.wire_R_values = {}
        for layer in self.metal_layers:
            self.wire_C_values[layer] = sim_util.xreplace_safe(self.tech_model.wire_parasitics["C"][layer], self.tech_model.base_params.tech_values)
            self.wire_R_values[layer] = sim_util.xreplace_safe(self.tech_model.wire_parasitics["R"][layer], self.tech_model.base_params.tech_values)
        
        self.device_C_diff = sim_util.xreplace_safe(self.tech_model.C_diff, self.tech_model.base_params.tech_values)
        self.device_C_load = sim_util.xreplace_safe(self.tech_model.C_load, self.tech_model.base_params.tech_values)
        self.device_R_avg_inv = sim_util.xreplace_safe(self.tech_model.R_avg_inv, self.tech_model.base_params.tech_values)

    def update_uarch_parameters(self):
        self.logic_delay_cvx.value = float(sim_util.xreplace_safe(self.tech_model.delay, self.tech_model.base_params.tech_values))
        self.logic_energy_active_cvx.value = float(sim_util.xreplace_safe(self.tech_model.E_act_inv, self.tech_model.base_params.tech_values))
        self.logic_power_passive_cvx.value = float(sim_util.xreplace_safe(self.tech_model.P_pass_inv, self.tech_model.base_params.tech_values))
        for layer in self.metal_layers:
            log_info(f"wire_unit_delay_cvx[{layer}] = {self.wire_unit_delay_cvx[layer].value}, with R[{layer}] = {sim_util.xreplace_safe(self.tech_model.wire_parasitics['R'][layer], self.tech_model.base_params.tech_values)}, C[{layer}] = {sim_util.xreplace_safe(self.tech_model.wire_parasitics['C'][layer], self.tech_model.base_params.tech_values)}")
            self.wire_unit_delay_cvx[layer].value = float(sim_util.xreplace_safe(self.tech_model.wire_parasitics["R"][layer]*self.tech_model.wire_parasitics["C"][layer], self.tech_model.base_params.tech_values))
            self.wire_unit_energy_cvx[layer].value = float(sim_util.xreplace_safe(0.5*self.tech_model.wire_parasitics["C"][layer]*self.tech_model.base_params.V_dd**2, self.tech_model.base_params.tech_values))

    def set_memories(self, memories):
        self.memories = memories
        self.update_circuit_values()

    def compare_symbolic_mem(self):
        for key in self.symbolic_mem:
            assert key in self.memories, f"symbolic memory {key} not found in memories"      


    def update_circuit_values(self):
        self.set_uarch_parameters()
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
    def wire_delay(self, edge, symbolic=False):
        # If wires are disabled, delay contribution is zero
        if self.zero_wirelength_costs:
            return 0.0

        if self.constant_wire_length_cost is not None:
            # print(f"Using constant wire length cost of {self.constant_wire_length_cost} for edge {edge}!!")
            wire_delay = self.constant_wire_length_cost  # ns
            return wire_delay

        if symbolic:
            wire_delay = 0.0
            for net in self.edge_to_nets.get(edge, []):
                R_on_line = self.tech_model.R_avg_inv
                C_current = self.tech_model.C_diff
                wire_delay += R_on_line * C_current
                for segment in net.segments:
                    C_current = segment.length * self.tech_model.wire_parasitics["C"][segment.layer]
                    R_on_line += segment.length * self.tech_model.wire_parasitics["R"][segment.layer]
                    wire_delay += R_on_line * C_current
                C_current = self.tech_model.C_load
                wire_delay += R_on_line * C_current

        else:
            # wire delay = R * C * length^2 (ns)
            wire_delay = 0
            #logger.info(f"calculating wire delay for edge {edge}")
            for net in self.edge_to_nets[edge]:
                #logger.info(f"calculating wire delay for net {net.net_id}")
                R_on_line = self.device_R_avg_inv
                C_current = self.device_C_diff
                wire_delay += R_on_line * C_current
                for segment in net.segments:
                    #logger.info(f"calculating wire delay for segment in layer {segment.layer} with length {segment.length}")
                    C_current = segment.length * self.wire_C_values[segment.layer]
                    R_on_line += segment.length * self.wire_R_values[segment.layer]
                    wire_delay += R_on_line * C_current
                C_current = self.device_C_load
                wire_delay += R_on_line * C_current
        print(f"wire_delay for edge {edge} is {wire_delay} ns!!")
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
        # print(f"wire_length for edge {edge} is {wire_length}")
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
        if not symbolic and wire_energy != 0:
            wire_energy = sim_util.xreplace_safe(wire_energy, self.tech_model.base_params.tech_values)
        # print(f"wire_energy for edge {edge} is {wire_energy} nJ")
        return wire_energy * 1e9
        
    def make_sym_lat_wc(self, gamma):
        return gamma * self.tech_model.delay
    
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
    
    def make_sym_energy_act(self, alpha):
        return alpha * self.tech_model.E_act_inv

    def make_mainmem_power_passive_dict(self):
        return self.tech_model.base_params.MemPpass
    
    def make_buf_power_passive_dict(self):
        return self.tech_model.base_params.BufPpass
    
    def make_sym_power_pass(self, beta):
        return beta * self.tech_model.P_pass_inv

    def make_sym_area(self, area_coeff):
        return area_coeff * self.tech_model.base_params.area

    def create_constraints(self):
        if self.tech_model.model_cfg["effects"]["frequency"]:
            for key in self.symbolic_latency_wc:
                #if key not in ["Buf", "MainMem", "OffChipIO", "Call", "N/A"]:
                if key == "FloorDiv16": # most stringent constraint
                    # cycle limit to constrain the amount of pipelining
                    #self.constraints.append((self.symbolic_latency_wc[key]()* 1e-9) * self.tech_model.base_params.f <= 20) # num cycles <= 20 (cycles = time(s) * frequency(Hz))
                    self.constraints.append(Constraint((self.symbolic_latency_wc[key]())<= 20*self.tech_model.base_params.clk_period, f"latency_{key} <= 20*clk_period")) # num cycles <= 20 (cycles = time(s) * frequency(Hz))
    
    def create_constraints_cvx(self, scale_cvx):
        self.constraints_cvx = []
        for key in self.symbolic_latency_wc:
            if key not in ["Buf", "MainMem", "OffChipIO", "Call", "N/A"]:
                self.constraints_cvx.append(self.uarch_lat_cvx[key]<= 20*self.clk_period_cvx) # num cycles <= 20 (cycles = time(s) * frequency(Hz))