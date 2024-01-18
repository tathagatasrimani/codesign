from sympy import symbols, ceiling, expand, exp


C_int_inv = symbols("C_int_inv", positive=True)
V_dd = symbols("V_dd", positive=True)
f = symbols("f", positive=True)
C_input_inv = symbols("C_input_inv", positive=True)

u_p = 1
C_ox = 1
W = 1
L = 1
V_T = 1
I_s = 1
q = 1
V_offset = 1
n = 1
K = 1
T = 1
R_wire = 1
C_wire = 1e-10

I_on = (u_p*C_ox*W*(V_dd-V_T)**2)/(2*L)
I_off = I_s*exp((q*(V_dd-V_T-V_offset)) / (n*K*T)) * (1 - exp((-q*V_dd) / (K*T)))
I_avg = (I_on + I_off) / 2
R_avg_inv = V_dd / I_avg
C_inv = C_input_inv + C_int_inv
P_act_inv = 0.5*C_inv*V_dd*V_dd*f
P_pass_inv = I_off*V_dd

# coefficients used in logical effort formulation
# active power
alpha = {
    "And": 1,
    "Or": 1,
    "Add": 1,
    "Sub": 1,
    "Mult": 1,
    "FloorDiv": 1,
    "Mod": 1,
    "LShift": 1,
    "RShift": 1,
    "BitOr": 1,
    "BitXor": 1,
    "BitAnd": 1,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 1,
    "UAdd": 1,
    "IsNot": 1,
    "Not": 1, 
    "Invert": 1,
    "Regs": 1
}

# passive power
beta = {
    "And": 2,
    "Or": 2,
    "Add": 2,
    "Sub": 2,
    "Mult": 2,
    "FloorDiv": 2,
    "Mod": 2,
    "LShift": 2,
    "RShift": 2,
    "BitOr": 2,
    "BitXor": 2,
    "BitAnd": 2,
    "Eq": 2,
    "NotEq": 2,
    "Lt": 2,
    "LtE": 2,
    "Gt": 2,
    "GtE": 2,
    "USub": 2,
    "UAdd": 2,
    "IsNot": 2,
    "Not": 2, 
    "Invert": 2,
    "Regs": 2
}

# delay
gamma = {
    "And": 1,
    "Or": 1,
    "Add": 1,
    "Sub": 1,
    "Mult": 1,
    "FloorDiv": 1,
    "Mod": 1,
    "LShift": 1,
    "RShift": 1,
    "BitOr": 1,
    "BitXor": 1,
    "BitAnd": 1,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 1,
    "UAdd": 1,
    "IsNot": 1,
    "Not": 1, 
    "Invert": 1,
    "Regs": 1
}

def make_sym_lat_wc(gamma):
    return gamma * R_avg_inv * C_input_inv + (gamma * R_avg_inv + R_wire) * C_wire

symbolic_latency_wc = {
    "And": make_sym_lat_wc(gamma["And"]),
    "Or": make_sym_lat_wc(gamma["Or"]),
    "Add": make_sym_lat_wc(gamma["Add"]),
    "Sub": make_sym_lat_wc(gamma["Sub"]),
    "Mult": make_sym_lat_wc(gamma["Mult"]),
    "FloorDiv": make_sym_lat_wc(gamma["FloorDiv"]),
    "Mod": make_sym_lat_wc(gamma["Mod"]),
    "LShift": make_sym_lat_wc(gamma["LShift"]),
    "RShift": make_sym_lat_wc(gamma["RShift"]),
    "BitOr": make_sym_lat_wc(gamma["BitOr"]),
    "BitXor": make_sym_lat_wc(gamma["BitXor"]),
    "BitAnd": make_sym_lat_wc(gamma["BitAnd"]),
    "Eq": make_sym_lat_wc(gamma["Eq"]),
    "NotEq": make_sym_lat_wc(gamma["NotEq"]),
    "Lt": make_sym_lat_wc(gamma["Lt"]),
    "LtE": make_sym_lat_wc(gamma["LtE"]),
    "Gt": make_sym_lat_wc(gamma["Gt"]),
    "GtE": make_sym_lat_wc(gamma["GtE"]),
    "USub": make_sym_lat_wc(gamma["USub"]),
    "UAdd": make_sym_lat_wc(gamma["UAdd"]),
    "IsNot": make_sym_lat_wc(gamma["IsNot"]),
    "Not": make_sym_lat_wc(gamma["Not"]),
    "Invert": make_sym_lat_wc(gamma["Invert"]),
    "Regs": make_sym_lat_wc(gamma["Regs"]),
}

def make_sym_lat_cyc(f, lat_wc):
    return ceiling(f*lat_wc)/f

symbolic_latency_cyc = {
    "And": make_sym_lat_cyc(f, symbolic_latency_wc["And"]),
    "Or": make_sym_lat_cyc(f, symbolic_latency_wc["Or"]),
    "Add": make_sym_lat_cyc(f, symbolic_latency_wc["Add"]),
    "Sub": make_sym_lat_cyc(f, symbolic_latency_wc["Sub"]),
    "Mult": make_sym_lat_cyc(f, symbolic_latency_wc["Mult"]),
    "FloorDiv": make_sym_lat_cyc(f, symbolic_latency_wc["FloorDiv"]),
    "Mod": make_sym_lat_cyc(f, symbolic_latency_wc["Mod"]),
    "LShift": make_sym_lat_cyc(f, symbolic_latency_wc["LShift"]),
    "RShift": make_sym_lat_cyc(f, symbolic_latency_wc["RShift"]),
    "BitOr": make_sym_lat_cyc(f, symbolic_latency_wc["BitOr"]),
    "BitXor": make_sym_lat_cyc(f, symbolic_latency_wc["BitXor"]),
    "BitAnd": make_sym_lat_cyc(f, symbolic_latency_wc["BitAnd"]),
    "Eq": make_sym_lat_cyc(f, symbolic_latency_wc["Eq"]),
    "NotEq": make_sym_lat_cyc(f, symbolic_latency_wc["NotEq"]),
    "Lt": make_sym_lat_cyc(f, symbolic_latency_wc["Lt"]),
    "LtE": make_sym_lat_cyc(f, symbolic_latency_wc["LtE"]),
    "Gt": make_sym_lat_cyc(f, symbolic_latency_wc["Gt"]),
    "GtE": make_sym_lat_cyc(f, symbolic_latency_wc["GtE"]),
    "USub": make_sym_lat_cyc(f, symbolic_latency_wc["USub"]),
    "UAdd": make_sym_lat_cyc(f, symbolic_latency_wc["UAdd"]),
    "IsNot": make_sym_lat_cyc(f, symbolic_latency_wc["IsNot"]),
    "Not": make_sym_lat_cyc(f, symbolic_latency_wc["Not"]),
    "Invert": make_sym_lat_cyc(f, symbolic_latency_wc["Invert"]),
    "Regs": make_sym_lat_cyc(f, symbolic_latency_wc["Regs"]),
}

def make_sym_power_act(alpha):
    return alpha * P_act_inv

symbolic_power_active = {
    "And": make_sym_power_act(alpha["And"]),
    "Or": make_sym_power_act(alpha["Or"]),
    "Add": make_sym_power_act(alpha["Add"]),
    "Sub": make_sym_power_act(alpha["Sub"]),
    "Mult": make_sym_power_act(alpha["Mult"]),
    "FloorDiv": make_sym_power_act(alpha["FloorDiv"]),
    "Mod": make_sym_power_act(alpha["Mod"]),
    "LShift": make_sym_power_act(alpha["LShift"]),
    "RShift": make_sym_power_act(alpha["RShift"]),
    "BitOr": make_sym_power_act(alpha["BitOr"]),
    "BitXor": make_sym_power_act(alpha["BitXor"]),
    "BitAnd": make_sym_power_act(alpha["BitAnd"]),
    "Eq": make_sym_power_act(alpha["Eq"]),
    "NotEq": make_sym_power_act(alpha["NotEq"]),
    "Lt": make_sym_power_act(alpha["Lt"]),
    "LtE": make_sym_power_act(alpha["LtE"]),
    "Gt": make_sym_power_act(alpha["Gt"]),
    "GtE": make_sym_power_act(alpha["GtE"]),
    "USub": make_sym_power_act(alpha["USub"]),
    "UAdd": make_sym_power_act(alpha["UAdd"]),
    "IsNot": make_sym_power_act(alpha["IsNot"]),
    "Not": make_sym_power_act(alpha["Not"]),
    "Invert": make_sym_power_act(alpha["Invert"]),
    "Regs": make_sym_power_act(alpha["Regs"]),
}

def make_sym_power_pass(beta):
    return beta * P_pass_inv

symbolic_power_passive = {
    "And": make_sym_power_pass(beta["And"]),
    "Or": make_sym_power_pass(beta["Or"]),
    "Add": make_sym_power_pass(beta["Add"]),
    "Sub": make_sym_power_pass(beta["Sub"]),
    "Mult": make_sym_power_pass(beta["Mult"]),
    "FloorDiv": make_sym_power_pass(beta["FloorDiv"]),
    "Mod": make_sym_power_pass(beta["Mod"]),
    "LShift": make_sym_power_pass(beta["LShift"]),
    "RShift": make_sym_power_pass(beta["RShift"]),
    "BitOr": make_sym_power_pass(beta["BitOr"]),
    "BitXor": make_sym_power_pass(beta["BitXor"]),
    "BitAnd": make_sym_power_pass(beta["BitAnd"]),
    "Eq": make_sym_power_pass(beta["Eq"]),
    "NotEq": make_sym_power_pass(beta["NotEq"]),
    "Lt": make_sym_power_pass(beta["Lt"]),
    "LtE": make_sym_power_pass(beta["LtE"]),
    "Gt": make_sym_power_pass(beta["Gt"]),
    "GtE": make_sym_power_pass(beta["GtE"]),
    "USub": make_sym_power_pass(beta["USub"]),
    "UAdd": make_sym_power_pass(beta["UAdd"]),
    "IsNot": make_sym_power_pass(beta["IsNot"]),
    "Not": make_sym_power_pass(beta["Not"]),
    "Invert": make_sym_power_pass(beta["Invert"]),
    "Regs": make_sym_power_pass(beta["Regs"]),
}