import json
import graphviz
import re
from collections import deque
import ast

from src.cfg_builder import CFGBuilder
from src.hls_instrument import instrument_and_run
from src.ast_utils import ASTUtils

op2sym_map = {
    "And": "and",
    "Or": "or",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "IsNot": "!=",
    "USub": "-",
    "UAdd": "+",
    "Not": "!",
    "Invert": "~",
}
delimiters = (
    "+",
    "-",
    "*",
    "//",
    "%",
    "=",
    ">>",
    "<<",
    "<",
    "<=",
    ">",
    ">=",
    "!=",
    "~",
    "!",
    "^",
    "&",
)

latency = {
    "And": 1,
    "Or": 1,
    "Add": 4,
    "Sub": 4,
    "Mult": 5,
    "FloorDiv": 16,
    "Mod": 3,
    "LShift": 0.70,
    "RShift": 0.70,
    "BitOr": 0.06,
    "BitXor": 0.06,
    "BitAnd": 0.06,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 0.42,
    "UAdd": 0.42,
    "IsNot": 1,
    "Not": 0.06,
    "Invert": 0.06,
    "Regs": 1,
}
power = {
    "And": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Or": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Add": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Sub": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Mult": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "FloorDiv": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "Mod": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "LShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "RShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "BitOr": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitXor": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitAnd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Eq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "NotEq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Lt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "LtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Gt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "GtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "USub": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "UAdd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "IsNot": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Not": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Invert": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Regs": [7.936518e-03, 1.062977e-03, 8.999495e-03, 8.999495e-03, 7.395312e-05],
}

class Node:
    def __init__(self) -> None:
        self.start = -1
        self.end = -1
        self.hw = {}

def assign(expr):

    return expr

def main():
    # note: must specify path to run the program, this is just an example path
    path = '/Users/PatrickMcEwen/high_level_synthesis/venv/hls_and_schedule/benchmarks/nonai_models/'
    benchmark = 'simple'
    cfg = CFGBuilder().build_from_file('main.c', path + benchmark + '.py')
    cfg.build_visual(path + benchmark, 'pdf', show = False)
    for node in cfg:
        #print(node.predecessors, node.exits)
        for statement in node.statements:
            #print(statement, ASTUtils.get_identifier(statement))
            if type(statement) == ast.Assign:
                print(statement.targets, statement.value)
                for target in statement.targets:
                    if type(target) == ast.Name:
                        print(target.id)
                if type(statement.value.left) == ast.Name and type(statement.value.right) == ast.Name:
                    print(statement.value.left.id, statement.value.op, statement.value.right.id)
            #num_cycles, hw_need, energy_need = assign(statement)
    return 0

if __name__ == "__main__":
    main()