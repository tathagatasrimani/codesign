import os
import sys
import sympy as sp
import yaml
import concurrent.futures
import time
os.chdir("..")
sys.path.append(os.getcwd())
from src import sim_util
from src import hw_symbols
from src import optimize


NUM_CORES = os.cpu_count()

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

def run_sympify(file):
    return sp.sympify(file, locals=hw_symbols.symbol_table)

def main():
    start_time = time.time()

    log_dir = sim_util.get_latest_log_dir()

    text_files = {}

    with open('src/cacti/symbolic_expressions/Mem_access_time.txt', 'r') as file:
        text_files["MemL"] = file.read()

    with open("src/cacti/symbolic_expressions/Mem_read_dynamic.txt", "r") as file:
        text_files["MemR"] = file.read()

    with open("src/cacti/symbolic_expressions/Mem_write_dynamic.txt", "r") as file:
        text_files["MemW"] = file.read()

    with open("src/cacti/symbolic_expressions/Mem_read_leakage.txt", "r") as file:
        text_files["MemRL"] = file.read()

    with open("src/cacti/symbolic_expressions/Buf_access_time.txt", "r") as file:
        text_files["BufL"] = file.read()

    with open("src/cacti/symbolic_expressions/Buf_read_dynamic.txt", "r") as file:
        text_files["BufR"] = file.read()

    with open("src/cacti/symbolic_expressions/Buf_write_dynamic.txt", "r") as file:
        text_files["BufW"] = file.read()

    with open("src/cacti/symbolic_expressions/Buf_read_leakage.txt", "r") as file:
        text_files["BufRL"] = file.read()

    with open(log_dir+"/symbolic_edp_0.txt", "r") as file:
        text_files["edp"] = file.read()

    print("file reads completed")
    print(time.time()-start_time)

    exprs = {}

    processes = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        for file in text_files:
            process = executor.submit(run_sympify, text_files[file])
            processes[process] = file

        for future in concurrent.futures.as_completed(processes):
            name = processes[future]
            try:
                exprs[name] = future.result()
                print(f"saving result for {name}")
            except Exception as exc:
                print('%r generated an exception: %s' % (name, exc))

    exprs["MemL"] *= 1e9
    exprs["BufL"] *= 1e9

    # TODO: print these vs fw pass values
    cacti_subs = {
        hw_symbols.MemReadL: (exprs["MemL"] / 2),
        hw_symbols.MemWriteL: (exprs["MemL"] / 2),
        hw_symbols.MemReadEact: exprs["MemR"],
        hw_symbols.MemWriteEact: exprs["MemW"],
        hw_symbols.MemPpass: exprs["MemRL"],
        hw_symbols.BufL: exprs["BufL"],
        hw_symbols.BufReadEact: exprs["BufR"],
        hw_symbols.BufWriteEact: exprs["BufW"],
        hw_symbols.BufPpass: exprs["BufRL"],
    }

    edp = exprs["edp"]

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


if __name__ == "__main__":
    main()