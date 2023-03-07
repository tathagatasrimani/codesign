import json
import graphviz as gv
from collections import deque
import ast
import astor
import math

from cfg.staticfg.builder import CFGBuilder
from cfg.staticfg.model import Block
from cfg.staticfg.model import Link
from hls_instrument import instrument_and_run
from cfg.ast_utils import ASTUtils
from hls import HardwareModel
import hls

cycles = 0
main_cfg = None

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)
    
def cycle_sim(hw_inuse, max_cycles):
    global cycles
    for i in range(max_cycles):
        for elem in hw_inuse:
            for j in range(len(hw_inuse[elem])):
                if hw_inuse[elem][j] > 0:
                    hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1)
        print("This is after cycle " + str(cycles))
        print(hw_inuse)
        print("")
        cycles += 1

def sim(cfg, hw, first):
    global cycles, main_cfg
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw:
        hw_inuse[elem] = [0] * hw[elem]
    print(hw_inuse)
    while True:
        for statement in cur_node.statements:
            hw_need = HardwareModel(0, 0)
            hw_need.eval_expr(statement)
            calls = []
            # find any function calls in the statement and step into them before we continue in the current function
            func_calls(statement, calls)
            if len(calls) != 0:
                for call in calls[::-1]:
                    sim(main_cfg.functioncfgs[call], hw, False)
            print(hw_need.hw_allocated)
            max_cycles = 0
            for elem in hw_need.hw_allocated:
                cur_elem_count = hw_need.hw_allocated[elem]
                if cur_elem_count == 0: continue
                if hw[elem] < cur_elem_count: 
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = cur_elem_count * hls.latency[elem]
                print("cycles needed for " + elem + ": " + str(cur_cycles_needed))
                max_cycles = max(cur_cycles_needed, max_cycles)
                i = 0
                while cur_cycles_needed > 0:
                    hw_inuse[elem][i] += hls.latency[elem]
                    i = (i + 1) % hw[elem]
                    cur_cycles_needed -= hls.latency[elem]
            cycle_sim(hw_inuse, max_cycles)
        if len(cur_node.exits) != 0:
            cur_node = cur_node.exits[0].target
        else:
            break
    return cycles
