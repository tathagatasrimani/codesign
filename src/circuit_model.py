import logging

from src import coefficients

logger = logging.getLogger(__name__)

class CircuitModel:
    def __init__(self, tech_model):
        self.tech_model = tech_model

        # hardcoded tech node to reference for logical effort coefficients
        self.coeffs = coefficients.create_and_save_coefficients([7])
        self.set_coefficients()

        # UNITS: ns
        self.symbolic_latency_wc = {
            "And": lambda: self.make_sym_lat_wc(self.gamma["And"]),
            "Or": lambda: self.make_sym_lat_wc(self.gamma["Or"]),
            "Add": lambda: self.make_sym_lat_wc(self.gamma["Add"]),
            "Sub": lambda: self.make_sym_lat_wc(self.gamma["Sub"]),
            "Mult": lambda: self.make_sym_lat_wc(self.gamma["Mult"]),
            "FloorDiv": lambda: self.make_sym_lat_wc(self.gamma["FloorDiv"]),
            "Mod": lambda: self.make_sym_lat_wc(self.gamma["Mod"]),
            "LShift": lambda: self.make_sym_lat_wc(self.gamma["LShift"]),
            "RShift": lambda: self.make_sym_lat_wc(self.gamma["RShift"]),
            "BitOr": lambda: self.make_sym_lat_wc(self.gamma["BitOr"]),
            "BitXor": lambda: self.make_sym_lat_wc(self.gamma["BitXor"]),
            "BitAnd": lambda: self.make_sym_lat_wc(self.gamma["BitAnd"]),
            "Eq": lambda: self.make_sym_lat_wc(self.gamma["Eq"]),
            "NotEq": lambda: self.make_sym_lat_wc(self.gamma["NotEq"]),
            "Lt": lambda: self.make_sym_lat_wc(self.gamma["Lt"]),
            "LtE": lambda: self.make_sym_lat_wc(self.gamma["LtE"]),
            "Gt": lambda: self.make_sym_lat_wc(self.gamma["Gt"]),
            "GtE": lambda: self.make_sym_lat_wc(self.gamma["GtE"]), 
            "USub": lambda: self.make_sym_lat_wc(self.gamma["USub"]),   
            "UAdd": lambda: self.make_sym_lat_wc(self.gamma["UAdd"]),
            "IsNot": lambda: self.make_sym_lat_wc(self.gamma["IsNot"]),
            "Not": lambda: self.make_sym_lat_wc(self.gamma["Not"]),
            "Invert": lambda: self.make_sym_lat_wc(self.gamma["Invert"]),
            "Regs": lambda: self.make_sym_lat_wc(self.gamma["Regs"]),   
            "Buf": lambda: self.make_buf_lat_dict(),    
            "MainMem": lambda: self.make_mem_lat_dict(),
            "OffChipIO": lambda: self.make_io_lat_dict(),
        }

        # UNITS: nJ
        self.symbolic_energy_active = {
            "And": lambda: self.make_sym_energy_act(self.alpha["And"]),
            "Or": lambda: self.make_sym_energy_act(self.alpha["Or"]),
            "Add": lambda: self.make_sym_energy_act(self.alpha["Add"]),
            "Sub": lambda: self.make_sym_energy_act(self.alpha["Sub"]),
            "Mult": lambda: self.make_sym_energy_act(self.alpha["Mult"]),
            "FloorDiv": lambda: self.make_sym_energy_act(self.alpha["FloorDiv"]),
            "Mod": lambda: self.make_sym_energy_act(self.alpha["Mod"]),
            "LShift": lambda: self.make_sym_energy_act(self.alpha["LShift"]),
            "RShift": lambda: self.make_sym_energy_act(self.alpha["RShift"]),
            "BitOr": lambda: self.make_sym_energy_act(self.alpha["BitOr"]),
            "BitXor": lambda: self.make_sym_energy_act(self.alpha["BitXor"]),
            "BitAnd": lambda: self.make_sym_energy_act(self.alpha["BitAnd"]),
            "Eq": lambda: self.make_sym_energy_act(self.alpha["Eq"]),
            "NotEq": lambda: self.make_sym_energy_act(self.alpha["NotEq"]),
            "Lt": lambda: self.make_sym_energy_act(self.alpha["Lt"]),
            "LtE": lambda: self.make_sym_energy_act(self.alpha["LtE"]),
            "Gt": lambda: self.make_sym_energy_act(self.alpha["Gt"]),
            "GtE": lambda: self.make_sym_energy_act(self.alpha["GtE"]),
            "USub": lambda: self.make_sym_energy_act(self.alpha["USub"]),
            "UAdd": lambda: self.make_sym_energy_act(self.alpha["UAdd"]),
            "IsNot": lambda: self.make_sym_energy_act(self.alpha["IsNot"]),
            "Not": lambda: self.make_sym_energy_act(self.alpha["Not"]),
            "Invert": lambda: self.make_sym_energy_act(self.alpha["Invert"]),   
            "Regs": lambda: self.make_sym_energy_act(self.alpha["Regs"]),
            "Buf": lambda: self.make_buf_energy_active_dict(),
            "MainMem": lambda: self.make_mainmem_energy_active_dict(),
            "OffChipIO": lambda: self.make_io_energy_active_dict(),
        }

        # UNITS: W
        self.symbolic_power_passive = {
            "And": lambda: self.make_sym_power_pass(self.beta["And"]),
            "Or": lambda: self.make_sym_power_pass(self.beta["Or"]),
            "Add": lambda: self.make_sym_power_pass(self.beta["Add"]),
            "Sub": lambda: self.make_sym_power_pass(self.beta["Sub"]),
            "Mult": lambda: self.make_sym_power_pass(self.beta["Mult"]),
            "FloorDiv": lambda: self.make_sym_power_pass(self.beta["FloorDiv"]),
            "Mod": lambda: self.make_sym_power_pass(self.beta["Mod"]),
            "LShift": lambda: self.make_sym_power_pass(self.beta["LShift"]),
            "RShift": lambda: self.make_sym_power_pass(self.beta["RShift"]),
            "BitOr": lambda: self.make_sym_power_pass(self.beta["BitOr"]),
            "BitXor": lambda: self.make_sym_power_pass(self.beta["BitXor"]),
            "BitAnd": lambda: self.make_sym_power_pass(self.beta["BitAnd"]),
            "Eq": lambda: self.make_sym_power_pass(self.beta["Eq"]),
            "NotEq": lambda: self.make_sym_power_pass(self.beta["NotEq"]),
            "Lt": lambda: self.make_sym_power_pass(self.beta["Lt"]),
            "LtE": lambda: self.make_sym_power_pass(self.beta["LtE"]),
            "Gt": lambda: self.make_sym_power_pass(self.beta["Gt"]),
            "GtE": lambda: self.make_sym_power_pass(self.beta["GtE"]),
            "USub": lambda: self.make_sym_power_pass(self.beta["USub"]),
            "UAdd": lambda: self.make_sym_power_pass(self.beta["UAdd"]),
            "IsNot": lambda: self.make_sym_power_pass(self.beta["IsNot"]),
            "Not": lambda: self.make_sym_power_pass(self.beta["Not"]),
            "Invert": lambda: self.make_sym_power_pass(self.beta["Invert"]),
            "Regs": lambda: self.make_sym_power_pass(self.beta["Regs"]),
            "MainMem": lambda: self.make_mainmem_power_passive_dict(),
            "Buf": lambda: self.make_buf_power_passive_dict(),
        }

        # UNITS: um^2
        self.symbolic_area = {
            "And": lambda: self.make_sym_area(self.area_coeffs["And"]),
            "Or": lambda: self.make_sym_area(self.area_coeffs["Or"]),
            "Add": lambda: self.make_sym_area(self.area_coeffs["Add"]),
            "Sub": lambda: self.make_sym_area(self.area_coeffs["Sub"]),
            "Mult": lambda: self.make_sym_area(self.area_coeffs["Mult"]),
            "FloorDiv": lambda: self.make_sym_area(self.area_coeffs["FloorDiv"]),
            "Mod": lambda: self.make_sym_area(self.area_coeffs["Mod"]), 
            "LShift": lambda: self.make_sym_area(self.area_coeffs["LShift"]),
            "RShift": lambda: self.make_sym_area(self.area_coeffs["RShift"]),
            "BitOr": lambda: self.make_sym_area(self.area_coeffs["BitOr"]),
            "BitXor": lambda: self.make_sym_area(self.area_coeffs["BitXor"]),
            "BitAnd": lambda: self.make_sym_area(self.area_coeffs["BitAnd"]),
            "Eq": lambda: self.make_sym_area(self.area_coeffs["Eq"]),
            "NotEq": lambda: self.make_sym_area(self.area_coeffs["NotEq"]),
            "Lt": lambda: self.make_sym_area(self.area_coeffs["Lt"]),
            "LtE": lambda: self.make_sym_area(self.area_coeffs["LtE"]),
            "Gt": lambda: self.make_sym_area(self.area_coeffs["Gt"]),
            "GtE": lambda: self.make_sym_area(self.area_coeffs["GtE"]),
            "USub": lambda: self.make_sym_area(self.area_coeffs["USub"]),
            "UAdd": lambda: self.make_sym_area(self.area_coeffs["UAdd"]),
            "IsNot": lambda: self.make_sym_area(self.area_coeffs["IsNot"]),
            "Not": lambda: self.make_sym_area(self.area_coeffs["Not"]),
            "Invert": lambda: self.make_sym_area(self.area_coeffs["Invert"]),
            "Regs": lambda: self.make_sym_area(self.area_coeffs["Regs"]),
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
        self.wire_length_by_edge = {}

        self.metal_layers = ["metal1", "metal2", "metal3"]

        self.update_circuit_values()
    
    def set_coefficients(self):
        self.alpha = self.coeffs["alpha"]
        self.beta = self.coeffs["beta"]
        self.gamma = self.coeffs["gamma"]
        self.area_coeffs = self.coeffs["area"]

    def set_memories(self, memories):
        self.memories = memories
        self.update_circuit_values()

    def compare_symbolic_mem(self):
        for key in self.symbolic_mem:
            assert key in self.memories, f"symbolic memory {key} not found in memories"      

    def update_circuit_values(self):
        # derive curcuit level values from technology values
        self.circuit_values["latency"] = {
            key: float(self.symbolic_latency_wc[key]().xreplace(self.tech_model.base_params.tech_values).evalf()) for key in self.symbolic_latency_wc if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["dynamic_energy"] = {
            key: float(self.symbolic_energy_active[key]().xreplace(self.tech_model.base_params.tech_values).evalf()) for key in self.symbolic_energy_active if key not in ["Buf", "MainMem", "OffChipIO"]
        }
        self.circuit_values["passive_power"] = {
            key: float(self.symbolic_power_passive[key]().xreplace(self.tech_model.base_params.tech_values).evalf()) for key in self.symbolic_power_passive if key not in ["Buf", "MainMem"]
        }
        self.circuit_values["area"] = {
            key: float(self.symbolic_area[key]().xreplace(self.tech_model.base_params.tech_values).evalf()) for key in self.symbolic_area
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

    def wire_delay(self, edge, symbolic=False):
        # wire delay = R * C * length^2 (ns)
        if symbolic:
            return sum([self.wire_length_by_edge[edge][layer]**2 * 
                        self.tech_model.wire_parasitics["R"][layer] * 
                        self.tech_model.wire_parasitics["C"][layer] 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        else:
            return sum([self.wire_length_by_edge[edge][layer]**2 * 
                        self.tech_model.base_params.tech_values[self.tech_model.wire_parasitics["R"][layer]] * 
                        self.tech_model.base_params.tech_values[self.tech_model.wire_parasitics["C"][layer]] 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        
    def wire_energy(self, edge, symbolic=False):
        # wire energy = 0.5 * C * V_dd^2 * length
        if symbolic:
            return 0.5*sum([self.wire_length_by_edge[edge][layer] * 
                        self.tech_model.wire_parasitics["C"][layer] * self.tech_model.base_params.V_dd**2 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        else:
            return 0.5*sum([self.wire_length_by_edge[edge][layer] * 
                        self.tech_model.base_params.tech_values[self.tech_model.wire_parasitics["C"][layer]] * self.tech_model.base_params.tech_values[self.tech_model.base_params.V_dd]**2 
                        if layer in self.wire_length_by_edge[edge]
                        else 0
                        for layer in self.metal_layers]) * 1e9
        
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