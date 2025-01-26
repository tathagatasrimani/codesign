import os
import sys
import sympy as sp
import yaml
import time
os.chdir("..")
sys.path.append(os.getcwd())
from src import sim_util
from src import hw_symbols
from src import optimize
start_time = time.time()

def parse_output(f, tech_params):
    lines = f.readlines()
    mapping = {}
    i = 0
    while lines[i][0] != "x":
        i += 1
    while lines[i][0] == "x":
        mapping[lines[i][lines[i].find("[") + 1 : lines[i].find("]")]] = (
            hw_symbols.symbol_table[lines[i].split(" ")[-1][:-1]]
        )
        i += 1
    while i < len(lines) and lines[i].find("x") != 4:
        i += 1
    i += 2
    for _ in range(len(mapping)):
        key = lines[i].split(":")[0].lstrip().rstrip()
        value = float(lines[i].split(":")[2][1:-1])
        print(mapping[key], value)
        tech_params[mapping[key]] = (
            value  # just know that tech_params contains all dat
        )
        i += 1

transistor_size = 7

log_dir = sim_util.get_latest_log_dir()

text_files = {}

with open('src/cacti/symbolic_expressions/Mem_access_time.txt', 'r') as file:
    text_files["MemL"] = file.read()

with open("src/cacti/symbolic_expressions/Mem_read_dynamic.txt", "r") as file:
    mem_read_dynamic_text = file.read()

with open("src/cacti/symbolic_expressions/Mem_write_dynamic.txt", "r") as file:
    mem_write_dynamic_text = file.read()

with open("src/cacti/symbolic_expressions/Mem_read_leakage.txt", "r") as file:
    mem_read_leakage_text = file.read()

with open("src/cacti/symbolic_expressions/Buf_access_time.txt", "r") as file:
    buf_access_time_text = file.read()

with open("src/cacti/symbolic_expressions/Buf_read_dynamic.txt", "r") as file:
    buf_read_dynamic_text = file.read()

with open("src/cacti/symbolic_expressions/Buf_write_dynamic.txt", "r") as file:
    buf_write_dynamic_text = file.read()

with open("src/cacti/symbolic_expressions/Buf_read_leakage.txt", "r") as file:
    buf_read_leakage_text = file.read()

with open(log_dir+"/symbolic_edp_0.txt", "r") as file:
    edp_txt = file.read()

print("file reads completed")
print(time.time()-start_time)


MemL_expr = sp.sympify(text_files["MemL"], locals=hw_symbols.symbol_table) * 1e9 # convert from s to ns
MemReadEact_expr = sp.sympify(mem_read_dynamic_text, locals=hw_symbols.symbol_table)
MemWriteEact_expr = sp.sympify(mem_write_dynamic_text, locals=hw_symbols.symbol_table)
MemPpass_expr = sp.sympify(mem_read_leakage_text, locals=hw_symbols.symbol_table)

BufL_expr = sp.sympify(buf_access_time_text, locals=hw_symbols.symbol_table) * 1e9 # convert from s to ns
BufReadEact_expr = sp.sympify(buf_read_dynamic_text, locals=hw_symbols.symbol_table)
BufWriteEact_expr = sp.sympify(buf_write_dynamic_text, locals=hw_symbols.symbol_table)
BufPpass_expr = sp.sympify(buf_read_leakage_text, locals=hw_symbols.symbol_table)

# TODO: print these vs fw pass values
cacti_subs = {
    hw_symbols.MemReadL: (MemL_expr / 2),
    hw_symbols.MemWriteL: (MemL_expr / 2),
    hw_symbols.MemReadEact: MemReadEact_expr,
    hw_symbols.MemWriteEact: MemWriteEact_expr,
    hw_symbols.MemPpass: MemPpass_expr,

    hw_symbols.BufL: BufL_expr,
    hw_symbols.BufReadEact: BufReadEact_expr,
    hw_symbols.BufWriteEact: BufWriteEact_expr,
    hw_symbols.BufPpass: BufPpass_expr,
}

edp = sp.sympify(edp_txt, locals=hw_symbols.symbol_table)

print("exprs converted to sympy")
print(time.time()-start_time)

rcs_dict = yaml.load(open(log_dir+"/rcs_0.yaml", "r"), Loader=yaml.Loader)
tech_params = sim_util.generate_init_params_from_rcs_as_symbols(rcs_dict)

print("generated tech params")

edp = edp.xreplace(cacti_subs)
initial_value = edp.xreplace(tech_params).simplify()
print(f"initial value: {initial_value}")

stdout = sys.stdout
with open("notebooks/test_files/ipopt_out.txt", "w") as sys.stdout:
    optimize.optimize(tech_params, edp, "ipopt", cacti_subs)
sys.stdout = stdout
f = open("notebooks/test_files/ipopt_out.txt", "r")
parse_output(f, tech_params)
final_value = edp.xreplace(tech_params).simplify()
print(f"final value: {final_value}")