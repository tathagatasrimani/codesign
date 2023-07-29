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
    7: {
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
    },
    5: {
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
    },
    3: {
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
}

dynamic_power = {
    7: { 
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
    },
    5: { 
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
    },
    3: { 
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
}

leakage_power = {
     7: { 
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
    },
    5: { 
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
    },
    3: { 
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
}

area = {
    7: { 
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
    },
    5: { 
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
    },
    3: { 
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

    def __init__(self,id,bandwidth,mem_layers,pitch,transistor_size,loop_counts={},var_sizes={}):
        self.max_bw = bandwidth
        self.bw_avail = bandwidth

        self.loop_counts = loop_counts

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable]=False

        # number of non-memory elements allocated
        self.transistor_size = transistor_size
        self.mem_layers = mem_layers
        self.pitch = pitch
        self.hw_allocated = {}
        self.hw_allocated["Regs"] = 0
        self.loop_variables = loop_counts
        self.var_sizes = var_sizes
        self.id = id
        self.area = area[transistor_size]
        self.latency = latency[transistor_size]
        self.latency_scale = latency_scale
        self.dynamic_power = dynamic_power[transistor_size]
        self.leakage_power = leakage_power[transistor_size]
        self.power_scale = power_scale
        if mem_layers == 2:
            if pitch == 100:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 10:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 1:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 0.1:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
        elif mem_layers == 4:
            if pitch == 100:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 10:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 1:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
            elif pitch == 0.1:
                 self.area["Regs"] = 1
                 self.dynamic_power["Regs"] = 1
                 self.leakage_power["Regs"] = 1
                 self.latency["Regs"] = 1
        else: # mem_layers == 1
            self.area["Regs"] = 1
            self.dynamic_power["Regs"] = 1
            self.leakage_power["Regs"] = 1
            self.latency["Regs"] = 1
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

    
    