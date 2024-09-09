import os
import yaml

from sympy import symbols, ceiling, expand, exp

V_dd = symbols("V_dd", positive=True)
f = symbols("f", positive=True)
MemReadL = symbols("MemReadL", positive=True)
MemWriteL = symbols("MemWriteL", positive=True)
MemReadEact = symbols("MemReadEact", positive=True)
MemWriteEact = symbols("MemWriteEact", positive=True)
MemPpass = symbols("MemPpass", positive=True)
BufL = symbols("BufL", positive=True)
BufReadEact = symbols("BufReadEact", positive=True)
BufWriteEact = symbols("BufWriteEact", positive=True)
BufPpass = symbols("BufPpass", positive=True)

# Where do these show up in the optimization objective
BufPeriphAreaEff = symbols("buf_peripheral_area_proportion", positive=True)
MemPeriphAreaEff = symbols("mem_peripheral_area_propportion", positive=True)

OffChipIOL = symbols("OffChipIOL", positive=True)
OffChipIOPact = symbols("OffChipIOPact", positive=True)

Reff = {
    "And": symbols("Reff_And", positive=True),
    "Or": symbols("Reff_Or", positive=True),
    "Add": symbols("Reff_Add", positive=True),
    "Sub": symbols("Reff_Sub", positive=True),
    "Mult": symbols("Reff_Mult", positive=True),
    "FloorDiv": symbols("Reff_FloorDiv", positive=True),
    "Mod": symbols("Reff_Mod", positive=True),
    "LShift": symbols("Reff_LShift", positive=True),
    "RShift": symbols("Reff_RShift", positive=True),
    "BitOr": symbols("Reff_BitOr", positive=True),
    "BitXor": symbols("Reff_BitXor", positive=True),
    "BitAnd": symbols("Reff_BitAnd", positive=True),
    "Eq": symbols("Reff_Eq", positive=True),
    "NotEq": symbols("Reff_NotEq", positive=True),
    "Lt": symbols("Reff_Lt", positive=True),
    "LtE": symbols("Reff_LtE", positive=True),
    "Gt": symbols("Reff_Gt", positive=True),
    "GtE": symbols("Reff_GtE", positive=True),
    "USub": symbols("Reff_USub", positive=True),
    "UAdd": symbols("Reff_UAdd", positive=True),
    "IsNot": symbols("Reff_IsNot", positive=True),
    "Not": symbols("Reff_Not", positive=True),
    "Invert": symbols("Reff_Invert", positive=True),
    "Regs": symbols("Reff_Regs", positive=True)
}

Ceff = {
    "And": symbols("Ceff_And", positive=True),
    "Or": symbols("Ceff_Or", positive=True),
    "Add": symbols("Ceff_Add", positive=True),
    "Sub": symbols("Ceff_Sub", positive=True),
    "Mult": symbols("Ceff_Mult", positive=True),
    "FloorDiv": symbols("Ceff_FloorDiv", positive=True),
    "Mod": symbols("Ceff_Mod", positive=True),
    "LShift": symbols("Ceff_LShift", positive=True),
    "RShift": symbols("Ceff_RShift", positive=True),
    "BitOr": symbols("Ceff_BitOr", positive=True),
    "BitXor": symbols("Ceff_BitXor", positive=True),
    "BitAnd": symbols("Ceff_BitAnd", positive=True),
    "Eq": symbols("Ceff_Eq", positive=True),
    "NotEq": symbols("Ceff_NotEq", positive=True),
    "Lt": symbols("Ceff_Lt", positive=True),
    "LtE": symbols("Ceff_LtE", positive=True),
    "Gt": symbols("Ceff_Gt", positive=True),
    "GtE": symbols("Ceff_GtE", positive=True),
    "USub": symbols("Ceff_USub", positive=True),
    "UAdd": symbols("Ceff_UAdd", positive=True),
    "IsNot": symbols("Ceff_IsNot", positive=True),
    "Not": symbols("Ceff_Not", positive=True),
    "Invert": symbols("Ceff_Invert", positive=True),
    "Regs": symbols("Ceff_Regs", positive=True),
}

symbol_table = {
    "V_dd": V_dd,
    "MemReadL": MemReadL,
    "MemWriteL": MemWriteL,
    "MemReadEact": MemReadEact,
    "MemWriteEact": MemWriteEact,
    "MemPpass": MemPpass,
    "BufL": BufL,
    "BufReadEact": BufReadEact,
    "BufWriteEact": BufWriteEact,
    "BufPpass": BufPpass,
    "OffChipIOL": OffChipIOL,
    "OffChipIOPact": OffChipIOPact,
    "Reff_And": Reff["And"],
    "Reff_Or": Reff["Or"],
    "Reff_Add": Reff["Add"],
    "Reff_Sub": Reff["Sub"],
    "Reff_Mult": Reff["Mult"],
    "Reff_FloorDiv": Reff["FloorDiv"],
    "Reff_Mod": Reff["Mod"],
    "Reff_LShift": Reff["LShift"],
    "Reff_RShift": Reff["RShift"],
    "Reff_BitOr": Reff["BitOr"],
    "Reff_BitXor": Reff["BitXor"],
    "Reff_BitAnd": Reff["BitAnd"],
    "Reff_Eq": Reff["Eq"],
    "Reff_NotEq": Reff["NotEq"],
    "Reff_Lt": Reff["Lt"],
    "Reff_LtE": Reff["LtE"],
    "Reff_Gt": Reff["Gt"],
    "Reff_GtE": Reff["GtE"],
    "Reff_USub": Reff["USub"],
    "Reff_UAdd": Reff["UAdd"],
    "Reff_IsNot": Reff["IsNot"],
    "Reff_Not": Reff["Not"],
    "Reff_Invert": Reff["Invert"],
    "Reff_Regs": Reff["Regs"],
    "Ceff_And": Ceff["And"],
    "Ceff_Or": Ceff["Or"],
    "Ceff_Add": Ceff["Add"],
    "Ceff_Sub": Ceff["Sub"],
    "Ceff_Mult": Ceff["Mult"],
    "Ceff_FloorDiv": Ceff["FloorDiv"],
    "Ceff_Mod": Ceff["Mod"],
    "Ceff_LShift": Ceff["LShift"],
    "Ceff_RShift": Ceff["RShift"],
    "Ceff_BitOr": Ceff["BitOr"],
    "Ceff_BitXor": Ceff["BitXor"],
    "Ceff_BitAnd": Ceff["BitAnd"],
    "Ceff_Eq": Ceff["Eq"],
    "Ceff_NotEq": Ceff["NotEq"],
    "Ceff_Lt": Ceff["Lt"],
    "Ceff_LtE": Ceff["LtE"],
    "Ceff_Gt": Ceff["Gt"],
    "Ceff_GtE": Ceff["GtE"],
    "Ceff_USub": Ceff["USub"],
    "Ceff_UAdd": Ceff["UAdd"],
    "Ceff_IsNot": Ceff["IsNot"],
    "Ceff_Not": Ceff["Not"],
    "Ceff_Invert": Ceff["Invert"],
    "Ceff_Regs": Ceff["Regs"],
}


# passive power
beta = yaml.load(
    open(os.path.join(os.path.dirname(__file__), "params/coefficients.yaml"), "r"),
    Loader=yaml.Loader,
)["beta"]

def make_sym_lat_wc(elem):
    return Reff[elem] * Ceff[elem]

symbolic_latency_wc = {
    "And": make_sym_lat_wc("And"),
    "Or": make_sym_lat_wc("Or"),
    "Add": make_sym_lat_wc("Add"),
    "Sub": make_sym_lat_wc("Sub"),
    "Mult": make_sym_lat_wc("Mult"),
    "FloorDiv": make_sym_lat_wc("FloorDiv"),
    "Mod": make_sym_lat_wc("Mod"),
    "LShift": make_sym_lat_wc("LShift"),
    "RShift": make_sym_lat_wc("RShift"),
    "BitOr": make_sym_lat_wc("BitOr"),
    "BitXor": make_sym_lat_wc("BitXor"),
    "BitAnd": make_sym_lat_wc("BitAnd"),
    "Eq": make_sym_lat_wc("Eq"),
    "NotEq": make_sym_lat_wc("NotEq"),
    "Lt": make_sym_lat_wc("Lt"),
    "LtE": make_sym_lat_wc("LtE"),
    "Gt": make_sym_lat_wc("Gt"),
    "GtE": make_sym_lat_wc("GtE"),
    "USub": make_sym_lat_wc("USub"),
    "UAdd": make_sym_lat_wc("UAdd"),
    "IsNot": make_sym_lat_wc("IsNot"),
    "Not": make_sym_lat_wc("Not"),
    "Invert": make_sym_lat_wc("Invert"),
    "Regs": make_sym_lat_wc("Regs"),
    "Buf": BufL,
    "MainMem": (MemReadL + MemWriteL)/2, # this needs to change later to sep the two.
    "OffChipIO": OffChipIOL,
}

# def make_sym_lat_cyc(f, lat_wc): # bad name, output is not in units of cycles, its in units of time.
#     return ceiling(f*lat_wc)/f

# symbolic_latency_cyc = {
#     "And": make_sym_lat_cyc(f, symbolic_latency_wc["And"]),
#     "Or": make_sym_lat_cyc(f, symbolic_latency_wc["Or"]),
#     "Add": make_sym_lat_cyc(f, symbolic_latency_wc["Add"]),
#     "Sub": make_sym_lat_cyc(f, symbolic_latency_wc["Sub"]),
#     "Mult": make_sym_lat_cyc(f, symbolic_latency_wc["Mult"]),
#     "FloorDiv": make_sym_lat_cyc(f, symbolic_latency_wc["FloorDiv"]),
#     "Mod": make_sym_lat_cyc(f, symbolic_latency_wc["Mod"]),
#     "LShift": make_sym_lat_cyc(f, symbolic_latency_wc["LShift"]),
#     "RShift": make_sym_lat_cyc(f, symbolic_latency_wc["RShift"]),
#     "BitOr": make_sym_lat_cyc(f, symbolic_latency_wc["BitOr"]),
#     "BitXor": make_sym_lat_cyc(f, symbolic_latency_wc["BitXor"]),
#     "BitAnd": make_sym_lat_cyc(f, symbolic_latency_wc["BitAnd"]),
#     "Eq": make_sym_lat_cyc(f, symbolic_latency_wc["Eq"]),
#     "NotEq": make_sym_lat_cyc(f, symbolic_latency_wc["NotEq"]),
#     "Lt": make_sym_lat_cyc(f, symbolic_latency_wc["Lt"]),
#     "LtE": make_sym_lat_cyc(f, symbolic_latency_wc["LtE"]),
#     "Gt": make_sym_lat_cyc(f, symbolic_latency_wc["Gt"]),
#     "GtE": make_sym_lat_cyc(f, symbolic_latency_wc["GtE"]),
#     "USub": make_sym_lat_cyc(f, symbolic_latency_wc["USub"]),
#     "UAdd": make_sym_lat_cyc(f, symbolic_latency_wc["UAdd"]),
#     "IsNot": make_sym_lat_cyc(f, symbolic_latency_wc["IsNot"]),
#     "Not": make_sym_lat_cyc(f, symbolic_latency_wc["Not"]),
#     "Invert": make_sym_lat_cyc(f, symbolic_latency_wc["Invert"]),
#     "Regs": make_sym_lat_cyc(f, symbolic_latency_wc["Regs"]),
#     "Buf": BufL,
#     "MainMem": (MemReadL + MemWriteL) / 2,
#     "OffChipIO": OffChipIOL,
# }

def make_sym_power_act(elem):
    return 0.5 * V_dd * V_dd  / Reff[elem] # C dependence will reappear explicitly when we go back to Energy from power.

symbolic_power_active = {
    "And": make_sym_power_act("And"),
    "Or": make_sym_power_act("Or"),
    "Add": make_sym_power_act("Add"),
    "Sub": make_sym_power_act("Sub"),
    "Mult": make_sym_power_act("Mult"),
    "FloorDiv": make_sym_power_act("FloorDiv"),
    "Mod": make_sym_power_act("Mod"),
    "LShift": make_sym_power_act("LShift"),
    "RShift": make_sym_power_act("RShift"),
    "BitOr": make_sym_power_act("BitOr"),
    "BitXor": make_sym_power_act("BitXor"),
    "BitAnd": make_sym_power_act("BitAnd"),
    "Eq": make_sym_power_act("Eq"),
    "NotEq": make_sym_power_act("NotEq"),
    "Lt": make_sym_power_act("Lt"),
    "LtE": make_sym_power_act("LtE"),
    "Gt": make_sym_power_act("Gt"),
    "GtE": make_sym_power_act("GtE"),
    "USub": make_sym_power_act("USub"),
    "UAdd": make_sym_power_act("UAdd"),
    "IsNot": make_sym_power_act("IsNot"),
    "Not": make_sym_power_act("Not"),
    "Invert": make_sym_power_act("Invert"),
    "Regs": make_sym_power_act("Regs"),
    "OffChipIO": OffChipIOPact,
}

symbolic_energy_active = {
    "Buf": (BufReadEact + BufWriteEact) / 2,
    "MainMem": (MemReadEact + MemWriteEact)
    / 2,
}

def make_sym_power_pass(beta, P_pass_inv=V_dd**2 / (Reff["Not"] * 100)):
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
    "MainMem": MemPpass,
    "Buf": BufPpass,
}

def update_symbolic_passive_power(R_off_on_ratio):
    for elem in symbolic_power_passive:
        if elem != "MainMem" and elem != "Buf":
            symbolic_power_passive[elem] = make_sym_power_pass(beta[elem], V_dd**2 / (R_off_on_ratio * Reff["Not"]))
