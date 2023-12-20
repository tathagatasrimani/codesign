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

symbolic_latency_wc = {
    "And": gamma["And"] * latency_tr_wc,
    "Or": gamma["Or"] * latency_tr_wc,
    "Add": gamma["Add"] * latency_tr_wc,
    "Sub": gamma["Sub"] * latency_tr_wc,
    "Mult": gamma["Mult"] * latency_tr_wc,
    "FloorDiv": gamma["FloorDiv"] * latency_tr_wc,
    "Mod": gamma["Mod"] * latency_tr_wc,
    "LShift": gamma["LShift"] * latency_tr_wc,
    "RShift": gamma["RShift"] * latency_tr_wc,
    "BitOr": gamma["BitOr"] * latency_tr_wc,
    "BitXor": gamma["BitXor"] * latency_tr_wc,
    "BitAnd": gamma["BitAnd"] * latency_tr_wc,
    "Eq": gamma["Eq"] * latency_tr_wc,
    "NotEq": gamma["NotEq"] * latency_tr_wc,
    "Lt": gamma["Lt"] * latency_tr_wc,
    "LtE": gamma["LtE"] * latency_tr_wc,
    "Gt": gamma["Gt"] * latency_tr_wc,
    "GtE": gamma["GtE"] * latency_tr_wc,
    "USub": gamma["USub"] * latency_tr_wc,
    "UAdd": gamma["UAdd"] * latency_tr_wc,
    "IsNot": gamma["IsNot"] * latency_tr_wc,
    "Not": gamma["Not"] * latency_tr_wc,
    "Invert": gamma["Invert"] * latency_tr_wc,
    "Regs": gamma["Regs"] * latency_tr_wc,
}



symbolic_latency_cyc = {
    "And": ceiling(f*symbolic_latency_wc["And"]) / f,
    "Or": ceiling(f*symbolic_latency_wc["Or"]) / f,
    "Add": ceiling(f*symbolic_latency_wc["Add"]) / f,
    "Sub": ceiling(f*symbolic_latency_wc["Sub"]) / f,
    "Mult": ceiling(f*symbolic_latency_wc["Mult"]) / f,
    "FloorDiv": ceiling(f*symbolic_latency_wc["FloorDiv"]) / f,
    "Mod": ceiling(f*symbolic_latency_wc["Mod"]) / f,
    "LShift": ceiling(f*symbolic_latency_wc["LShift"]) / f,
    "RShift": ceiling(f*symbolic_latency_wc["RShift"]) / f,
    "BitOr": ceiling(f*symbolic_latency_wc["BitOr"]) / f,
    "BitXor": ceiling(f*symbolic_latency_wc["BitXor"]) / f,
    "BitAnd": ceiling(symbolic_latency_wc["BitAnd"]) / f,
    "Eq": ceiling(f*symbolic_latency_wc["Eq"]) / f,
    "NotEq": ceiling(f*symbolic_latency_wc["NotEq"]) / f,
    "Lt": ceiling(f*symbolic_latency_wc["Lt"]) / f,
    "LtE": ceiling(f*symbolic_latency_wc["LtE"]) / f,
    "Gt": ceiling(f*symbolic_latency_wc["Gt"]) / f,
    "GtE": ceiling(f*symbolic_latency_wc["GtE"]) / f,
    "USub": ceiling(f*symbolic_latency_wc["USub"]) / f,
    "UAdd": ceiling(f*symbolic_latency_wc["UAdd"]) / f,
    "IsNot": ceiling(f*symbolic_latency_wc["IsNot"]) / f,
    "Not": ceiling(f*symbolic_latency_wc["Not"]) / f,
    "Invert": ceiling(f*symbolic_latency_wc["Invert"]) / f,
    "Regs": ceiling(f*symbolic_latency_wc["Regs"]) / f,
}

def make_sym_lat_cyc(f, lat_wc):
    #return ceiling(f*lat_wc)/f
    return lat_wc

for key in symbolic_latency_cyc:
    symbolic_latency_cyc[key] = make_sym_lat_cyc(f, symbolic_latency_wc[key])

symbolic_power_active = {
    "And": nsw["And"] * power_tr_act,
    "Or": nsw["Or"] * power_tr_act,
    "Add": nsw["Add"] * power_tr_act,
    "Sub": nsw["Sub"] * power_tr_act,
    "Mult": nsw["Mult"] * power_tr_act,
    "FloorDiv": nsw["FloorDiv"] * power_tr_act,
    "Mod": nsw["Mod"] * power_tr_act,
    "LShift": nsw["LShift"] * power_tr_act,
    "RShift": nsw["RShift"] * power_tr_act,
    "BitOr": nsw["BitOr"] * power_tr_act,
    "BitXor": nsw["BitXor"] * power_tr_act,
    "BitAnd": nsw["BitAnd"] * power_tr_act,
    "Eq": nsw["Eq"] * power_tr_act,
    "NotEq": nsw["NotEq"] * power_tr_act,
    "Lt": nsw["Lt"] * power_tr_act,
    "LtE": nsw["LtE"] * power_tr_act,
    "Gt": nsw["Gt"] * power_tr_act,
    "GtE": nsw["GtE"] * power_tr_act,
    "USub": nsw["USub"] * power_tr_act,
    "UAdd": nsw["UAdd"] * power_tr_act,
    "IsNot": nsw["IsNot"] * power_tr_act,
    "Not": nsw["Not"] * power_tr_act,
    "Invert": nsw["Invert"] * power_tr_act,
    "Regs": nsw["Regs"] * power_tr_act,
}

symbolic_power_passive = {
    "And": nt["And"] * power_tr_pass,
    "Or": nt["Or"] * power_tr_pass,
    "Add": nt["Add"] * power_tr_pass,
    "Sub": nt["Sub"] * power_tr_pass,
    "Mult": nt["Mult"] * power_tr_pass,
    "FloorDiv": nt["FloorDiv"] * power_tr_pass,
    "Mod": nt["Mod"] * power_tr_pass,
    "LShift": nt["LShift"] * power_tr_pass,
    "RShift": nt["RShift"] * power_tr_pass,
    "BitOr": nt["BitOr"] * power_tr_pass,
    "BitXor": nt["BitXor"] * power_tr_pass,
    "BitAnd": nt["BitAnd"] * power_tr_pass,
    "Eq": nt["Eq"] * power_tr_pass,
    "NotEq": nt["NotEq"] * power_tr_pass,
    "Lt": nt["Lt"] * power_tr_pass,
    "LtE": nt["LtE"] * power_tr_pass,
    "Gt": nt["Gt"] * power_tr_pass,
    "GtE": nt["GtE"] * power_tr_pass,
    "USub": nt["USub"] * power_tr_pass,
    "UAdd": nt["UAdd"] * power_tr_pass,
    "IsNot": nt["IsNot"] * power_tr_pass,
    "Not": nt["Not"] * power_tr_pass,
    "Invert": nt["Invert"] * power_tr_pass,
    "Regs": nt["Regs"] * power_tr_pass,
}