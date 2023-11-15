from hardwareModel import HardwareModel, SymbolicHardwareModel
from ast_utils import ASTUtils
import schedule
import dfg_algo
import matplotlib.pyplot as plt
import ast
import hardwareModel
import math
import json
import os
import numpy as np
import sys
from sympy import *
import itertools
import sympy
import pyomo.environ as pyo
from pyomo.core.expr import Expr_if
import pyomo.core.expr.sympy_tools as sympy_tools
from pyomo.opt import SolverFactory
from MyPyomoSympyBimap import MyPyomoSympyBimap
opt = pyo.SolverFactory('glpk')


class SymbolicHardwareSimulator():

    def __init__(self):
        self.cycles = 0
        self.id_to_node = {}
        self.path = os.getcwd()
        self.data_path = []
        self.node_intervals = []
        self.node_sum_power = {}
        self.node_sum_cycles = {}
        self.unroll_at = {}
        self.vars_allocated = {}
        self.where_to_free = {}
        self.compute_element_to_node_id = {}
        self.compute_element_neighbors = {}
        self.memory_needed = 0
        self.cur_memory_size = 0
        self.new_graph = None
        self.mem_layers = 0
        self.transistor_size = 0
        self.pitch = 0
        self.cache_size = 0
        self.reads = 0
        self.writes = 0
        self.total_read_size = 0
        self.total_write_size = 0
        self.max_regs_inuse = 0
        self.max_mem_inuse = 0

    def get_hw_need(self, state):
        # not the original simulate model now, so we can use a non-symbolic hardware model
        hw_need = HardwareModel(0,0,self.mem_layers, self.pitch, self.transistor_size, self.cache_size)
        for op in state:
            if not op.operation: continue
            else: hw_need.hw_allocated[op.operation] += 1
        return hw_need.hw_allocated

    def get_batch(self, need, spec):
        batch = 0
        for i in range(need):
            # add 1 to batch if need / spec > i
            batch += (functions.elementary.hyperbolic.tanh((need/spec) - i + 0.5) + 1) / 2
        return batch

    def symbolic_cycle_sim_parallel(self, hw_spec, hw_need):
        max_cycles = 0
        power_sum = 0
        # print(hw_need)
        for elem in hw_need:
            if hw_need[elem] == 0: continue
            hw_spec[elem] = hw_need[elem] # assuming that hw_need exactly matches the spec for now (dfg is the hardware)
            batch = math.ceil(hw_need[elem] / hw_spec[elem])
            # batch = self.get_batch(hw_need[elem], hw_spec[elem])
            # print("batch for", elem, "with need of", hw_need[elem], "and spec of", hw_spec[elem], "is", batch)
            active_power = hw_need[elem] * hardwareModel.symbolic_power[elem][2]
            power_sum += active_power
            power_sum += batch * hw_spec[elem] * hardwareModel.symbolic_power[elem][2] / 10 # idle dividor still need passive power
            cycles_per_node = batch * hardwareModel.symbolic_latency[elem] # real latency in self.cycles
            max_cycles = 0.5 * (max_cycles + cycles_per_node + abs(max_cycles - cycles_per_node))
            #max_cycles = Max(max_cycles, cycles_per_node)
            print("max_cycles:", max_cycles)
        
        self.cycles += max_cycles
        return max_cycles, power_sum

    def symbolic_simulate(self, cfg, symbolic_node_operations, hw_spec, symbolic_first):
        cur_node = cfg.entryblock
        if symbolic_first: 
            cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        i = 0
        # focus on symbolizing the node_operations
        while i < len(self.data_path):
            node_id = self.data_path[i][0]
            cur_node = self.id_to_node[node_id]
            self.node_intervals.append([node_id, [self.cycles, 0]])
            if not node_id in self.node_sum_power:
                self.node_sum_power[node_id] = 0 # just reset because we will end up overwriting it
            if not node_id in self.node_sum_cycles:
                self.node_sum_cycles[node_id] = 0
            iters = 0
            if self.unroll_at[cur_node.id]:
                j = i
                while True:
                    j += 1
                    if len(self.data_path) <= j: break
                    next_node_id = self.data_path[j][0]
                    if next_node_id != node_id: break
                    iters += 1
                i = j - 1 # skip over loop iterations because we execute them all at once
            for state in symbolic_node_operations[cur_node]:
                # if unroll, take each operation in a state and create more of them
                if self.unroll_at[cur_node.id]:
                    new_state = state.copy()
                    for op in state:
                        for j in range(iters):
                            new_state.append(op)
                    state = new_state
                hw_need = self.get_hw_need(state)
                max_cycles, power_sum = self.symbolic_cycle_sim_parallel(hw_spec, hw_need)
                self.node_sum_power[node_id] += power_sum
                self.node_sum_cycles[node_id] += max_cycles
            self.node_intervals[-1][1][1] = self.cycles
            i += 1

    def set_data_path(self):
        with open('/Users/PatrickMcEwen/git_container/codesign/src/instrumented_files/output.txt', 'r') as f:
            src = f.read()
            l = src.split('\n')
            for i in range(len(l)):
                l[i] = l[i].split()
            #print(l)
            last_line = '-1'
            last_node = '-1'
            for item in l:
                if len(item) == 2 and (item[0] != last_node or item[1] == last_line):
                    last_node = item[0]
                    last_line = item[1]
                    self.data_path.append(item)
            print(f"data_path: {self.data_path}")
            print("memory needed: ", self.memory_needed)

    def simulator_prep(self, benchmark):
        cfg, graphs, self.unroll_at = dfg_algo.main_fn(self.path, benchmark)
        node_operations = schedule.schedule(cfg, graphs, benchmark)
        self.set_data_path()
        for node in cfg: self.id_to_node[str(node.id)] = node
        #print(self.id_to_node)
        return cfg, graphs, node_operations

# creates nested if expression to represent sympy Max function
def nested_if(x0, xrest):
    if len(xrest) == 1:
        return Expr_if(IF=(x0>xrest[0]), THEN=x0, ELSE=xrest[0])
    else:
        return Expr_if(IF=(x0>xrest[0]), THEN=nested_if(x0, xrest[1:]), ELSE=nested_if(xrest[0], xrest[1:]))


def main():
    benchmark = sys.argv[1]
    print(benchmark)
    simulator = SymbolicHardwareSimulator()

    cfg, graphs, node_operations = simulator.simulator_prep(benchmark)

    simulator.transistor_size = 3 # in nm
    simulator.pitch = 100
    simulator.mem_layers = 2
    if simulator.memory_needed < 1000000:
        simulator.cache_size = 1
    elif simulator.memory_needed < 2000000:
        simulator.cache_size = 2
    elif simulator.memory_needed < 4000000:
        simulator.cache_size = 4
    elif simulator.memory_needed < 8000000:
        simulator.cache_size = 8
    else: 
        simulator.cache_size = 16
    hw = HardwareModel(0, 0, simulator.mem_layers, simulator.pitch, simulator.transistor_size, simulator.cache_size)
    symbolic_hw = SymbolicHardwareModel(0, 0)
    
    hw.hw_allocated['Add'] = 1
    hw.hw_allocated['Regs'] = 30
    hw.hw_allocated['Mult'] = 15
    hw.hw_allocated['Sub'] = 15
    hw.hw_allocated['FloorDiv'] = 15
    hw.hw_allocated['Gt'] = 1
    hw.hw_allocated['And'] = 1
    hw.hw_allocated['Or'] = 1
    hw.hw_allocated['Mod'] = 1
    hw.hw_allocated['LShift'] = 1
    hw.hw_allocated['RShift'] = 1
    hw.hw_allocated['BitOr'] = 1
    hw.hw_allocated['BitXor'] = 1
    hw.hw_allocated['BitAnd'] = 1
    hw.hw_allocated['Eq'] = 1
    hw.hw_allocated['NotEq'] = 1
    hw.hw_allocated['Lt'] = 1
    hw.hw_allocated['LtE'] = 1
    hw.hw_allocated['GtE'] = 1
    hw.hw_allocated['IsNot'] = 1
    hw.hw_allocated['USub'] = 1
    hw.hw_allocated['UAdd'] = 1
    hw.hw_allocated['Not'] = 1
    hw.hw_allocated['Invert'] = 1
    from sympy import symbols
    
    symbolic_hw.hw_allocated['Add'] = symbols('Add')
    symbolic_hw.hw_allocated['Regs'] = symbols('Regs')
    symbolic_hw.hw_allocated['Mult'] = symbols('Mult')
    symbolic_hw.hw_allocated['Sub'] = symbols('Sub')
    symbolic_hw.hw_allocated['FloorDiv'] = symbols('FloorDiv')
    symbolic_hw.hw_allocated['Gt'] = symbols('Gt')
    symbolic_hw.hw_allocated['And'] = symbols('And')
    symbolic_hw.hw_allocated['Or'] = symbols('Or')
    symbolic_hw.hw_allocated['Mod'] = symbols('Mod')
    symbolic_hw.hw_allocated['LShift'] = symbols('LShift')
    symbolic_hw.hw_allocated['RShift'] = symbols('RShift')
    symbolic_hw.hw_allocated['BitOr'] = symbols('BitOr')
    symbolic_hw.hw_allocated['BitXor'] = symbols('BitXor')
    symbolic_hw.hw_allocated['BitAnd'] = symbols('BitAnd')
    symbolic_hw.hw_allocated['Eq'] = symbols('Eq')
    symbolic_hw.hw_allocated['NotEq'] = symbols('NotEq')
    symbolic_hw.hw_allocated['Lt'] = symbols('Lt')
    symbolic_hw.hw_allocated['LtE'] = symbols('LtE')
    symbolic_hw.hw_allocated['GtE'] = symbols('GtE')
    symbolic_hw.hw_allocated['IsNot'] = symbols('IsNot')
    symbolic_hw.hw_allocated['USub'] = symbols('USub')
    symbolic_hw.hw_allocated['UAdd'] = symbols('UAdd')
    symbolic_hw.hw_allocated['Not'] = symbols('Not')
    symbolic_hw.hw_allocated['Invert'] = symbols('Invert')
    
    first = True
    
    simulator.symbolic_simulate(cfg, node_operations, symbolic_hw.hw_allocated, first)
    
    node_avg_power = {}
    #for node_id in simulator.node_sum_power:
        # node_sum_cycles_is_zero = 0.5 * tanh(node_sum_cycles[node_id]) + 0.5
        # probably node_sum_cycles[node_id] is not zero, because it's the max of all the self.cycles, just divide by it
        # node_avg_power[node_id] = (simulator.node_sum_power[node_id] / simulator.node_sum_cycles[node_id]).simplify()

    total_cycles = 0
    for node_id in simulator.node_sum_cycles:
        total_cycles += simulator.node_sum_cycles[node_id]

    total_power = 0
    for node_id in simulator.node_sum_power:
        total_power += simulator.node_sum_power[node_id]

    edp = total_cycles * total_power
    edp = edp.simplify()
    
    
    expr_symbols = {}
    free_symbols = []
    mapping = {}
    for s in edp.free_symbols:
        free_symbols.append(s)
        if s.name == 'latency_Add':
            continue
        if s.name == 'latency_Mult':
            continue
        if "latency" in s.name:
            expr_symbols[s] = hardwareModel.latency[simulator.transistor_size][s.name.split('_')[1]]
        elif "power" in s.name:
            expr_symbols[s] = hardwareModel.dynamic_power[simulator.transistor_size][s.name.split('_')[1]]
        else:
            expr_symbols[s] = hw.hw_allocated[s.name]

    model = pyo.ConcreteModel()
    model.nVars = pyo.Param(initialize=len(edp.free_symbols))
    model.N = pyo.RangeSet(model.nVars)
    model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    i = 0
    for j in model.x:
        mapping[free_symbols[i]] = j
        print(j, free_symbols[i])
        i += 1

    m = MyPyomoSympyBimap()
    for symbol in edp.free_symbols:
        m.sympy2pyomo[symbol] = model.x[mapping[symbol]]
    sympy_tools._operatorMap.update({sympy.Max: lambda x: nested_if(x[0], x[1:])})
    #print(mapping.sympyVars())
    py_exp = sympy_tools.sympy2pyomo_expression(edp, m)
    # py_exp = sympy_tools.sympy2pyomo_expression(hardwareModel.symbolic_latency["Add"] ** (1/2), m)
    print(py_exp)
    model.obj = pyo.Objective(expr=py_exp)
    model.cuts = pyo.ConstraintList()
    model.Constraint = pyo.Constraint( expr = py_exp >= 10)
    
    opt = SolverFactory('ipopt')
    #opt.options['max_iter'] = 1000
    results = opt.solve(model)  
    model.display()
    
if __name__ == '__main__':
    main()
    
    
    

