from sympy import symbols, ceiling


C_tr = symbols("C_tr", positive=True)
R_tr = symbols("R_tr", positive=True)
V_dd = symbols("V_dd", positive=True)
f = symbols("f", positive=True)
I_leak = symbols("I_leak", positive=True)
C_eff = symbols("C_eff", positive=True)

latency_tr_wc = R_tr * C_tr
power_tr_act = C_eff * (V_dd ** 2) * f
power_tr_pass = I_leak * V_dd

nsw = {
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

nt = {
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

"""nsw = {
    "And": symbols("nsw_And", integer=True),
    "Or": symbols("nsw_Or", integer=True),
    "Add": symbols("nsw_Add", integer=True),
    "Sub": symbols("nsw_Sub", integer=True),
    "Mult": symbols("nsw_Mul", integer=True),
    "FloorDiv": symbols("nsw_FloorDiv", integer=True),
    "Mod": symbols("nsw_Mod", integer=True),
    "LShift": symbols("nsw_LShift", integer=True),
    "RShift": symbols("nsw_RShift", integer=True),
    "BitOr": symbols("nsw_BitOr", integer=True),
    "BitXor": symbols("nsw_BitXor", integer=True),
    "BitAnd": symbols("nsw_BitAnd", integer=True),
    "Eq": symbols("nsw_Eq", integer=True),
    "NotEq": symbols("nsw_NotEq", integer=True),
    "Lt": symbols("nsw_Lt", integer=True),
    "LtE": symbols("nsw_LtE", integer=True),
    "Gt": symbols("nsw_Gt", integer=True),
    "GtE": symbols("nsw_GtE", integer=True),
    "USub": symbols("nsw_USub", integer=True),
    "UAdd": symbols("nsw_UAdd", integer=True),
    "IsNot": symbols("nsw_IsNot", integer=True),
    "Not": symbols("nsw_Not", integer=True), 
    "Invert": symbols("nsw_Invert", integer=True),
    "Regs": symbols("nsw_Regs", integer=True)
}

nt = {
    "And": symbols("nt_And", integer=True),
    "Or": symbols("nt_Or", integer=True),
    "Add": symbols("nt_Add", integer=True),
    "Sub": symbols("nt_Sub", integer=True),
    "Mult": symbols("nt_Mul", integer=True),
    "FloorDiv": symbols("nt_FloorDiv", integer=True),
    "Mod": symbols("nt_Mod", integer=True),
    "LShift": symbols("nt_LShift", integer=True),
    "RShift": symbols("nt_RShift", integer=True),
    "BitOr": symbols("nt_BitOr", integer=True),
    "BitXor": symbols("nt_BitXor", integer=True),
    "BitAnd": symbols("nt_BitAnd", integer=True),
    "Eq": symbols("nt_Eq", integer=True),
    "NotEq": symbols("nt_NotEq", integer=True),
    "Lt": symbols("nt_Lt", integer=True),
    "LtE": symbols("nt_LtE", integer=True),
    "Gt": symbols("nt_Gt", integer=True),
    "GtE": symbols("nt_GtE", integer=True),
    "USub": symbols("nt_USub", integer=True),
    "UAdd": symbols("nt_UAdd", integer=True),
    "IsNot": symbols("nt_IsNot", integer=True),
    "Not": symbols("nt_Not", integer=True), 
    "Invert": symbols("nt_Invert", integer=True),
    "Regs": symbols("nt_Regs", integer=True)
}

gamma = {
    "And": symbols("gamma_And"),
    "Or": symbols("gamma_Or"),
    "Add": symbols("gamma_Add"),
    "Sub": symbols("gamma_Sub"),
    "Mult": symbols("gamma_Mul"),
    "FloorDiv": symbols("gamma_FloorDiv"),
    "Mod": symbols("gamma_Mod"),
    "LShift": symbols("gamma_LShift"),
    "RShift": symbols("gamma_RShift"),
    "BitOr": symbols("gamma_BitOr"),
    "BitXor": symbols("gamma_BitXor"),
    "BitAnd": symbols("gamma_BitAnd"),
    "Eq": symbols("gamma_Eq"),
    "NotEq": symbols("gamma_NotEq"),
    "Lt": symbols("gamma_Lt"),
    "LtE": symbols("gamma_LtE"),
    "Gt": symbols("gamma_Gt"),
    "GtE": symbols("gamma_GtE"),
    "USub": symbols("gamma_USub"),
    "UAdd": symbols("gamma_UAdd"),
    "IsNot": symbols("gamma_IsNot"),
    "Not": symbols("gamma_Not"), 
    "Invert": symbols("gamma_Invert"),
    "Regs": symbols("gamma_Regs")
}"""

def make_sym_lat_wc(gamma):
    return gamma * latency_tr_wc

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
    #return ceiling(f*lat_wc)/f
    return lat_wc

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

def make_sym_power_act(nsw):
    return nsw * power_tr_act

symbolic_power_active = {
    "And": make_sym_power_act(nsw["And"]),
    "Or": make_sym_power_act(nsw["Or"]),
    "Add": make_sym_power_act(nsw["Add"]),
    "Sub": make_sym_power_act(nsw["Sub"]),
    "Mult": make_sym_power_act(nsw["Mult"]),
    "FloorDiv": make_sym_power_act(nsw["FloorDiv"]),
    "Mod": make_sym_power_act(nsw["Mod"]),
    "LShift": make_sym_power_act(nsw["LShift"]),
    "RShift": make_sym_power_act(nsw["RShift"]),
    "BitOr": make_sym_power_act(nsw["BitOr"]),
    "BitXor": make_sym_power_act(nsw["BitXor"]),
    "BitAnd": make_sym_power_act(nsw["BitAnd"]),
    "Eq": make_sym_power_act(nsw["Eq"]),
    "NotEq": make_sym_power_act(nsw["NotEq"]),
    "Lt": make_sym_power_act(nsw["Lt"]),
    "LtE": make_sym_power_act(nsw["LtE"]),
    "Gt": make_sym_power_act(nsw["Gt"]),
    "GtE": make_sym_power_act(nsw["GtE"]),
    "USub": make_sym_power_act(nsw["USub"]),
    "UAdd": make_sym_power_act(nsw["UAdd"]),
    "IsNot": make_sym_power_act(nsw["IsNot"]),
    "Not": make_sym_power_act(nsw["Not"]),
    "Invert": make_sym_power_act(nsw["Invert"]),
    "Regs": make_sym_power_act(nsw["Regs"]),
}

def make_sym_power_pass(nt):
    return nt * power_tr_pass

symbolic_power_passive = {
    "And": make_sym_power_pass(nt["And"]),
    "Or": make_sym_power_pass(nt["Or"]),
    "Add": make_sym_power_pass(nt["Add"]),
    "Sub": make_sym_power_pass(nt["Sub"]),
    "Mult": make_sym_power_pass(nt["Mult"]),
    "FloorDiv": make_sym_power_pass(nt["FloorDiv"]),
    "Mod": make_sym_power_pass(nt["Mod"]),
    "LShift": make_sym_power_pass(nt["LShift"]),
    "RShift": make_sym_power_pass(nt["RShift"]),
    "BitOr": make_sym_power_pass(nt["BitOr"]),
    "BitXor": make_sym_power_pass(nt["BitXor"]),
    "BitAnd": make_sym_power_pass(nt["BitAnd"]),
    "Eq": make_sym_power_pass(nt["Eq"]),
    "NotEq": make_sym_power_pass(nt["NotEq"]),
    "Lt": make_sym_power_pass(nt["Lt"]),
    "LtE": make_sym_power_pass(nt["LtE"]),
    "Gt": make_sym_power_pass(nt["Gt"]),
    "GtE": make_sym_power_pass(nt["GtE"]),
    "USub": make_sym_power_pass(nt["USub"]),
    "UAdd": make_sym_power_pass(nt["UAdd"]),
    "IsNot": make_sym_power_pass(nt["IsNot"]),
    "Not": make_sym_power_pass(nt["Not"]),
    "Invert": make_sym_power_pass(nt["Invert"]),
    "Regs": make_sym_power_pass(nt["Regs"]),
}