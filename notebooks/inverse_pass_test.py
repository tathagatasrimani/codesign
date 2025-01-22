import os
import sys
import sympy as sp
os.chdir("..")
sys.path.append(os.getcwd())
from src import sim_util
from src import hw_symbols

with open("src/cacti/symbolic_expressions/Buf_access_time.txt", "r") as file:
    buf_access_time_text = file.read()

print("file read completed")

BufL_expr = sp.sympify(buf_access_time_text, locals=hw_symbols.symbol_table) * 1e9

cacti_subs = {hw_symbols.BufL: BufL_expr}

print("expr converted to sympy")