from hardwareModel import HardwareModel, SymbolicHardwareModel
from ast_utils import ASTUtils
import schedule
import dfg_algo
import matplotlib.pyplot as plt
import ast
import hardwareModel
import math
import json
import sys
from sympy import *

cycles = 0
main_cfg = None
id_to_node = {}
path = '/home/ubuntu/codesign/src/cfg/benchmarks/' # change path variable for local computer
data_path = []
node_intervals = []
node_sum_power = {}
node_sum_cycles = {}
unroll_at = {}

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)

def get_hw_need(state):
    # not the original simulate model now, so we can use a non-symbolic hardware model
    hw_need = HardwareModel(0,0)
    for op in state:
        if not op.operation: continue
        else: hw_need.hw_allocated[op.operation] += 1
    return hw_need.hw_allocated

def symbolic_cycle_sim_parallel(hw_spec, hw_need):
    global cycles
    max_cycles = 0
    power_sum = 0
    for elem in hw_need:
        batch = math.ceil(hw_need[elem] / hw_spec[elem])
        active_power = hw_need[elem] * hardwareModel.symbolic_power[elem][2]
        power_sum += active_power
        power_sum += batch * hw_spec[elem] * hardwareModel.symbolic_power[elem][2] / 10 # idle dividor still need passive power
        cycles_per_node = batch * hardwareModel.symbolic_latency[elem] # real latency in cycles
        max_cycles = Max(max_cycles, cycles_per_node)
    cycles += max_cycles
    return max_cycles, power_sum

def symbolic_simulate(cfg, data_path, symbolic_node_operations, hw_spec, symbolic_first):
    global main_cfg, id_to_node, unroll_at
    cur_node = cfg.entryblock
    if symbolic_first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    i = 0
    # focus on symbolizing the node_operations
    while i < len(data_path):
        node_id = data_path[i][0]
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_sum_power[node_id] = 0 # just reset because we will end up overwriting it
        node_sum_cycles[node_id] = 0
        iters = 0
        if unroll_at[cur_node.id]:
            j = i
            while True:
                j += 1
                if len(data_path) <= j: break
                next_node_id = data_path[j][0]
                if next_node_id != node_id: break
                iters += 1
            i = j - 1 # skip over loop iterations because we execute them all at once
        for state in symbolic_node_operations[cur_node]:
            # if unroll, take each operation in a state and create more of them
            if unroll_at[cur_node.id]:
                new_state = state.copy()
                for op in state:
                    for j in range(iters):
                        new_state.append(op)
                state = new_state
            hw_need = get_hw_need(state)
            max_cycles, power_sum = symbolic_cycle_sim_parallel(hw_spec, hw_need)
            node_sum_power[node_id] += power_sum
            node_sum_cycles[node_id] += max_cycles
        node_intervals[-1][1][1] = cycles
        i += 1


cur_node_id = 0

def main():
    global unroll_at
    benchmark = sys.argv[1]
    print(benchmark)
    # for next step we would start from makeing unroll_at symbolic, is that right?
    cfg, graphs, unroll_at = dfg_algo.main_fn(path, benchmark)
    # I think we need to make graphs symbolic, so that we could optimize the schedule procedure?
    cfg, node_operations = schedule.schedule(cfg, graphs, sys.argv[1])
    # symbolic_hw = SymbolicHardwareModel(0, 0)
    
    hw = HardwareModel(0, 0)
    
    hw.hw_allocated['Add'] = 15
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
    
    
    # symbolic_hw.hw_allocated['Add'] = symbols('Add')
    # symbolic_hw.hw_allocated['Regs'] = symbols('Regs')
    # symbolic_hw.hw_allocated['Mult'] = symbols('Mult')
    # symbolic_hw.hw_allocated['Sub'] = symbols('Sub')
    # symbolic_hw.hw_allocated['FloorDiv'] = symbols('FloorDiv')
    # symbolic_hw.hw_allocated['Gt'] = symbols('Gt')
    # symbolic_hw.hw_allocated['And'] = symbols('And')
    # symbolic_hw.hw_allocated['Or'] = symbols('Or')
    # symbolic_hw.hw_allocated['Mod'] = symbols('Mod')
    # symbolic_hw.hw_allocated['LShift'] = symbols('LShift')
    # symbolic_hw.hw_allocated['RShift'] = symbols('RShift')
    # symbolic_hw.hw_allocated['BitOr'] = symbols('BitOr')
    # symbolic_hw.hw_allocated['BitXor'] = symbols('BitXor')
    # symbolic_hw.hw_allocated['BitAnd'] = symbols('BitAnd')
    # symbolic_hw.hw_allocated['Eq'] = symbols('Eq')
    # symbolic_hw.hw_allocated['NotEq'] = symbols('NotEq')
    # symbolic_hw.hw_allocated['Lt'] = symbols('Lt')
    # symbolic_hw.hw_allocated['LtE'] = symbols('LtE')
    # symbolic_hw.hw_allocated['GtE'] = symbols('GtE')
    # symbolic_hw.hw_allocated['IsNot'] = symbols('IsNot')
    # symbolic_hw.hw_allocated['USub'] = symbols('USub')
    # symbolic_hw.hw_allocated['UAdd'] = symbols('UAdd')
    # symbolic_hw.hw_allocated['Not'] = symbols('Not')
    # symbolic_hw.hw_allocated['Invert'] = symbols('Invert')
    
    for node in cfg:
        id_to_node[str(node.id)] = node
    
    # set up sequence of cfg nodes to visit
    with open('/home/ubuntu/codesign/src/instrumented_files/output.txt', 'r') as f:
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
                data_path.append(item)
    # but for now we just begin with symbolic simulation
    # data = simulate(cfg, data_path, node_operations, hw.hw_allocated, True)
    first = True
    
    symbolic_simulate(cfg, data_path, node_operations, hw.hw_allocated, first)
    
    node_avg_power = {}
    for node_id in node_sum_power:
        # node_sum_cycles_is_zero = 0.5 * tanh(node_sum_cycles[node_id]) + 0.5
        # probably node_sum_cycles[node_id] is not zero, because it's the max of all the cycles, just divide by it
        node_avg_power[node_id] = (node_sum_power[node_id] / node_sum_cycles[node_id]).simplify()
    # print("node_sum_power", node_sum_power)
    # print("node_sum_cycles", node_sum_cycles)
    # print("node_intervals", node_intervals)
    
    node_avg_power_value = {}
    for node_id in node_avg_power:
        expr_symbols = {}
        expr = node_avg_power[node_id]
        for s in expr.free_symbols:
            if not s in expr_symbols:
                if "latency" in s.name:
                    expr_symbols[s] = hardwareModel.latency[s.name.split('_')[1]]
                else:
                    expr_symbols[s] = hardwareModel.power[s.name.split('_')[1]][int(s.name.split('_')[2])]
        expr_value = expr.subs(expr_symbols)
        node_avg_power_value[node_id] = float(expr_value)

    print(node_avg_power_value)

if __name__ == '__main__':
    main()
    
    
    

