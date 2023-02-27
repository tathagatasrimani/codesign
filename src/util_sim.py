import json
import graphviz as gv
from collections import deque
import ast
import astor

from cfg.staticfg.builder import CFGBuilder
from cfg.staticfg.model import Block
from cfg.staticfg.model import Link
from hls_instrument import instrument_and_run
from cfg.ast_utils import ASTUtils
from hls import HardwareModel

cycles = 0
main_cfg = None

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)
    

def sim(cfg, hw, first):
    global cycles, main_cfg
    offset = 0
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    while True:
        for statement in cur_node.statements:
            hw_need = HardwareModel(0, 0)
            hw_need.eval_expr(statement)
            calls = []
            func_calls(statement, calls)
            if len(calls) != 0:
                for call in calls[::-1]:
                    sim(main_cfg.functioncfgs[call], hw, False)
            print(hw_need.hw_allocated)
            offset += 1
        if len(cur_node.exits) != 0:
            cur_node = cur_node.exits[0].target
        else:
            break
    return cycles

if __name__ == "__main__":
    sim()

