import json
import graphviz as gv
import re
from collections import deque
import ast

from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils

benchmark = 'simple'
expr_to_node = {}
func_ref = {}

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
    "Regs": "Regs"
}

latency = {
    "And": 1,
    "Or": 1,
    "Add": 2,
    "Sub": 2,
    "Mult": 6,
    "FloorDiv": 36,
    "Mod": 3,
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
    "Regs": 2
}

power = {
    "And": 64 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Or": 64 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Add": 2 * [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Sub": 2 * [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Mult": 2 * [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "FloorDiv": 2 * [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "Mod": 2 * [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "LShift": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "RShift": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "BitOr": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitXor": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitAnd": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Eq": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "NotEq": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Lt": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "LtE": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Gt": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "GtE": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "USub": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "UAdd": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "IsNot": 2 * [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Not": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Invert": 2 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    #"Regs": 2 * [7.936518e-03, 1.062977e-03, 8.999495e-03, 8.999495e-03, 7.395312e-05],
    "Regs": 2 * [2.5, 2.5, 2.5, 2.5, 0.25]
}

area = {
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

latency_scale = {
    512: 1,
    1024: 2,
    2048: 3,
    4096: 4,
    8192: 5,
    16384: 6,
    32768: 7,
    65536: 8,
    131072: 9,
    262144: 10,
    524288: 11,
    1048576: 12,
    2097152: 13,
    4194304: 14,
    8388608: 15,
    16777216: 16,
    33554432: 17,
    67108864: 18,
    134217728: 19,
    268435456: 20,
    536870912: 21
}

power_scale = {
    512: 1,
    1024: 2,
    2048: 3,
    4096: 4,
    8192: 5,
    16384: 6,
    32768: 7,
    65536: 8,
    131072: 9,
    262144: 10,
    524288: 11,
    1048576: 12,
    2097152: 13,
    4194304: 14,
    8388608: 15,
    16777216: 16,
    33554432: 17,
    67108864: 18,
    134217728: 19,
    268435456: 20,
    536870912: 21
}

class HardwareModel:

    def __init__(self,id,bandwidth,loop_counts={},var_sizes={}):
        self.max_bw = bandwidth
        self.bw_avail = bandwidth

        self.loop_counts = loop_counts

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable]=False

        # number of non-memory elements allocated
        self.hw_allocated = {}
        self.hw_allocated["Regs"] = 0
        self.loop_variables = loop_counts
        self.var_sizes = var_sizes
        self.id = id


        for key in op2sym_map.keys():
                self.hw_allocated[key] = 0

        self.cycles = 0


    def print_stats(self):
        s = '''
        cycles={cycles}
        allocated={allocated}
        utilized={utilized}
        '''.format(cycles=self.cycles, \
                   allocated=str(self.hw_allocated))
        return s

    
    