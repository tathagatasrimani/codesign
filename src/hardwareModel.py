import json
import graphviz as gv
import re
from collections import deque
import ast
from sympy import *
from cfg.staticfg.builder import CFGBuilder
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
}

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

symbolic_latency = {
    "And": symbols("latency_And"),
    "Or": symbols("latency_Or"),
    "Add": symbols("latency_Add"),
    "Sub": symbols("latency_Sub"),
    "Mult": symbols("latency_Mult"),
    "FloorDiv": symbols("latency_FloorDiv"),
    "Mod": symbols("latency_Mod"),
    "LShift": symbols("latency_LShift"),
    "RShift": symbols("latency_RShift"),
    "BitOr": symbols("latency_BitOr"),
    "BitXor": symbols("latency_BitXor"),
    "BitAnd": symbols("latency_BitAnd"),
    "Eq": symbols("latency_Eq"),
    "NotEq": symbols("latency_NotEq"),
    "Lt": symbols("latency_Lt"),
    "LtE": symbols("latency_LtE"),
    "Gt": symbols("latency_Gt"),
    "GtE": symbols("latency_GtE"),
    "USub": symbols("latency_USub"),
    "UAdd": symbols("latency_UAdd"),
    "IsNot": symbols("latency_IsNot"),
    "Not": symbols("latency_Not"),
    "Invert": symbols("latency_Invert"),
    "Regs": symbols("latency_Regs"),
}

symbolic_power = {
    "And": [symbols("power_And_0"), symbols("power_And_1"), symbols("power_And_2"), symbols("power_And_3"), symbols("power_And_4")],
    "Or": [symbols("power_Or_0"), symbols("power_Or_1"), symbols("power_Or_2"), symbols("power_Or_3"), symbols("power_Or_4")],
    "Add": [symbols("power_Add_0"), symbols("power_Add_1"), symbols("power_Add_2"), symbols("power_Add_3"), symbols("power_Add_4")],
    "Sub": [symbols("power_Sub_0"), symbols("power_Sub_1"), symbols("power_Sub_2"), symbols("power_Sub_3"), symbols("power_Sub_4")],
    "Mult": [symbols("power_Mult_0"), symbols("power_Mult_1"), symbols("power_Mult_2"), symbols("power_Mult_3"), symbols("power_Mult_4")],
    "FloorDiv": [symbols("power_FloorDiv_0"), symbols("power_FloorDiv_1"), symbols("power_FloorDiv_2"), symbols("power_FloorDiv_3"), symbols("power_FloorDiv_4")],
    "Mod": [symbols("power_Mod_0"), symbols("power_Mod_1"), symbols("power_Mod_2"), symbols("power_Mod_3"), symbols("power_Mod_4")],
    "LShift": [symbols("power_LShift_0"), symbols("power_LShift_1"), symbols("power_LShift_2"), symbols("power_LShift_3"), symbols("power_LShift_4")],
    "RShift": [symbols("power_RShift_0"), symbols("power_RShift_1"), symbols("power_RShift_2"), symbols("power_RShift_3"), symbols("power_RShift_4")],
    "BitOr": [symbols("power_BitOr_0"), symbols("power_BitOr_1"), symbols("power_BitOr_2"), symbols("power_BitOr_3"), symbols("power_BitOr_4")],
    "BitXor": [symbols("power_BitXor_0"), symbols("power_BitXor_1"), symbols("power_BitXor_2"), symbols("power_BitXor_3"), symbols("power_BitXor_4")],
    "BitAnd": [symbols("power_BitAnd_0"), symbols("power_BitAnd_1"), symbols("power_BitAnd_2"), symbols("power_BitAnd_3"), symbols("power_BitAnd_4")],
    "Eq": [symbols("power_Eq_0"), symbols("power_Eq_1"), symbols("power_Eq_2"), symbols("power_Eq_3"), symbols("power_Eq_4")],
    "NotEq": [symbols("power_NotEq_0"), symbols("power_NotEq_1"), symbols("power_NotEq_2"), symbols("power_NotEq_3"), symbols("power_NotEq_4")],
    "Lt": [symbols("power_Lt_0"), symbols("power_Lt_1"), symbols("power_Lt_2"), symbols("power_Lt_3"), symbols("power_Lt_4")],
    "LtE": [symbols("power_LtE_0"), symbols("power_LtE_1"), symbols("power_LtE_2"), symbols("power_LtE_3"), symbols("power_LtE_4")],
    "Gt": [symbols("power_Gt_0"), symbols("power_Gt_1"), symbols("power_Gt_2"), symbols("power_Gt_3"), symbols("power_Gt_4")],
    "GtE": [symbols("power_GtE_0"), symbols("power_GtE_1"), symbols("power_GtE_2"), symbols("power_GtE_3"), symbols("power_GtE_4")],
    "USub": [symbols("power_USub_0"), symbols("power_USub_1"), symbols("power_USub_2"), symbols("power_USub_3"), symbols("power_USub_4")],
    "UAdd": [symbols("power_UAdd_0"), symbols("power_UAdd_1"), symbols("power_UAdd_2"), symbols("power_UAdd_3"), symbols("power_UAdd_4")],
    "IsNot": [symbols("power_IsNot_0"), symbols("power_IsNot_1"), symbols("power_IsNot_2"), symbols("power_IsNot_3"), symbols("power_IsNot_4")],
    "Not": [symbols("power_Not_0"), symbols("power_Not_1"), symbols("power_Not_2"), symbols("power_Not_3"), symbols("power_Not_4")],
    "Invert": [symbols("power_Invert_0"), symbols("power_Invert_1"), symbols("power_Invert_2"), symbols("power_Invert_3"), symbols("power_Invert_4")],
    "Regs": [symbols("power_Regs_0"), symbols("power_Regs_1"), symbols("power_Regs_2"), symbols("power_Regs_3"), symbols("power_Regs_4")],
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
        
        # a dict of symbols, only assigned and compute the value when needed


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


class SymbolicHardwareModel:

    def __init__(self, id, bandwidth, loop_counts={}, var_sizes={}):
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
        
        # a dict of symbols, only assigned and compute the value when needed


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