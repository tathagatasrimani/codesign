import json
import graphviz as gv
from collections import deque
import ast

from cfg.staticfg.builder import CFGBuilder
from cfg.staticfg.model import Block
from cfg.staticfg.model import Link
from hls_instrument import instrument_and_run
from cfg.ast_utils import ASTUtils
from hls import HardwareModel


def sim(cfg, hw):
    cycle = 0
    offset = 0
    stack = deque()
    cur_node = cfg.entryblock.exits[0].target
    while True:
        hw_need = HardwareModel(0, 0)
        for statement in cur_node.statements:
            hw_need.eval_expr(statement)
            print(hw_need.hw_allocated)
        break
    return cycle

if __name__ == "__main__":
    sim()

