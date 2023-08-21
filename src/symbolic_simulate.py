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
    # print(hw_need)
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
    global main_cfg, id_to_node, unroll_at, node_sum_cycles, node_sum_power
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
        if not node_id in node_sum_power:
            node_sum_power[node_id] = 0 # just reset because we will end up overwriting it
        if not node_id in node_sum_cycles:
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
    
    # node_avg_power_value = {}
    # for node_id in node_avg_power:
    #     expr_symbols = {}
    #     expr = node_avg_power[node_id]
    #     for s in expr.free_symbols:
    #         if not s in expr_symbols:
    #             if "latency" in s.name:
    #                 expr_symbols[s] = hardwareModel.latency[s.name.split('_')[1]]
    #             else:
    #                 expr_symbols[s] = hardwareModel.power[s.name.split('_')[1]][int(s.name.split('_')[2])]
    #     expr_value = expr.subs(expr_symbols)
    #     node_avg_power_value[node_id] = float(expr_value)
        
    # node_avg_power_value = {}
    total_cycles = 0
    for node_id in node_sum_cycles:
        total_cycles += node_sum_cycles[node_id]
    
    print("total_cycles", total_cycles)
    
    delta = 0.1
    
    expr_symbols = {}
    for s in total_cycles.free_symbols:
        if "latency" in s.name:
            expr_symbols[s] = hardwareModel.latency[s.name.split('_')[1]]
        else:
            expr_symbols[s] = hardwareModel.power[s.name.split('_')[1]][int(s.name.split('_')[2])]
    print("expr_symbols", expr_symbols)
    
    print("before modification ", total_cycles.subs(expr_symbols))
    
    # prime_expr = {}
    # prime_expr_value = {}
    
    # for s in total_cycles.free_symbols:
    #     print("total_cycles", total_cycles)
    #     print("s", s)
    #     prime_s = total_cycles.diff(s)
    #     print("prime_s", prime_s)
    #     prime_expr[s] = prime_s
    #     prime_expr_value[s] = prime_s.subs(expr_symbols)
    
    # print("prime_expr", prime_expr)
    # print("prime_expr_value", prime_expr_value)
    
    # max_key = max(prime_expr_value, key=prime_expr_value.get) # positive, apply - 0.1
    # print("max_key", max_key)
    # min_key = min(prime_expr_value, key=prime_expr_value.get) # negative, apply + 0.1
    # print("min_key", min_key)
    
    # expr_symbols[max_key] -= delta
    # expr_symbols[min_key] += delta
    
    # print("min latency ", total_cycles.subs(expr_symbols))
    
    # expr_symbols[max_key] += delta
    # expr_symbols[min_key] -= delta
    
    # print("original latency ", total_cycles.subs(expr_symbols))
    
    # expr_symbols[max_key] += delta
    # expr_symbols[min_key] -= delta
    
    # print("maximum latency ", total_cycles.subs(expr_symbols))
    cost_sum = 0
    for s in total_cycles.free_symbols:
        cost_sum += 1/(s+1)
    
    print('cost_sum', cost_sum)
    expr_symbols_with_cost = total_cycles * cost_sum
    
    
    
    
    expr_symbols = {}
    cnt=0
    for s in expr_symbols_with_cost.free_symbols:
        cnt+=1
        if cnt<=2:
            continue
        if "latency" in s.name:
            expr_symbols[s] = hardwareModel.latency[s.name.split('_')[1]]
        else:
            expr_symbols[s] = hardwareModel.power[s.name.split('_')[1]][int(s.name.split('_')[2])]

    # print("only keep 2 variables ", expr_symbols_with_cost.subs(expr_symbols))
    expr_symbols_with_cost_with_2_symbols = expr_symbols_with_cost.subs(expr_symbols)
    
    diffs=[]
    symbols=[]
    for s in expr_symbols_with_cost_with_2_symbols.free_symbols:
        diffs.append(expr_symbols_with_cost_with_2_symbols.diff(s))
        symbols.append(s)
    from design_space import DesignSpace
    ds=DesignSpace(expr_symbols_with_cost_with_2_symbols,symbols,diffs)
    ds.solve()
    
    # stationary_points = solve(diffs, symbols, dict=True)   

    # # Append boundary points
    # stationary_points.append({x:0, y:0})
    # stationary_points.append({x:1, y:0})
    # stationary_points.append({x:1, y:1})
    # stationary_points.append({x:0, y:1})
    
    # # store results after evaluation
    # results = []
    
    # # iteration counter
    # j = -1
    
    # for i in range(len(stationary_points)):
    #     j = j+1
    #     x1 = stationary_points[j].get(x)
    #     y1 = stationary_points[j].get(y)
        
    #     # If point is in the domain evalute and append it
    #     if (0 <= x1 <=  1) and ( 0 <= y1 <=  1):
    #         tmp = f.subs({x:x1, y:y1})
    #         results.append(tmp)
    #     else:
    #         # else remove the point
    #         stationary_points.pop(j)
    #         j = j-1
            
    # # Variables to store info
    # returnMax = []
    # returnMin = []
    
    # # Get the maximum value
    # maximum = max(results)
    
    # # Get the position of all the maximum values
    # maxpos = [i for i,j in enumerate(results) if j==maximum]
    
    # # Append only unique points
    # append = False
    # for item in maxpos:
    #     for i in returnMax:
    #         if (stationary_points[item] in i.values()):
    #             append = True
               
    #     if (not(append)):
    #         returnMax.append({maximum: stationary_points[item]})
    
    # # Get the minimum value
    # minimum  = min(results)
    
    # # Get the position of all the minimum  values
    # minpos = [i for i,j in enumerate(results) if j==minimum ]
    
    # # Append only unique points
    # append = False
    # for item in minpos:
    #     for i in returnMin:
    #         if (stationary_points[item] in i.values()):
    #             append = True
               
    #     if (not(append)):
    #         returnMin.append({minimum: stationary_points[item]})
    

    
    # print([returnMax, returnMin])

    # import scipy.optimize as optimize

    # def f(params):
    #     # print(params)  # <-- you'll see that params is a NumPy array
    #     a, b, c = params # <-- for readability you may wish to assign names to the component variables
    #     return a**2 + b**2 + c**2

    # initial_guess = [1, 1, 1]
    # result = optimize.minimize(f, initial_guess)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # energy delay product
    # realistic model what you can change and how much you can change it
    
    # total_cycles_value = total_cycles.subs(expr_symbols)
    # print("total_cycles_value", total_cycles_value)
        
    # node_sum_cycles_value = {}
    # for node_id in node_sum_cycles:
    #     expr_symbols = {}
    #     expr = node_sum_cycles[node_id]
    #     for s in expr.free_symbols:
    #         if not s in expr_symbols:
    #             if "latency" in s.name:
    #                 expr_symbols[s] = hardwareModel.latency[s.name.split('_')[1]]
    #             else:
    #                 expr_symbols[s] = hardwareModel.power[s.name.split('_')[1]][int(s.name.split('_')[2])]
    #     expr_value = expr.subs(expr_symbols)
    #     node_sum_cycles_value[node_id] = float(expr_value)
    
    # fprime_And = total_cycles_value.diff(hardwareModel.symbolic_latency["And"])
    # fprime_Or = total_cycles_value.diff(hardwareModel.symbolic_latency["Or"])
    # stationary_points = solve([fprime_And, fprime_Or], [hardwareModel.symbolic_latency["And"], hardwareModel.symbolic_latency["Or"]], dict=True)   
    # print(stationary_points)
    
    
    # x_interval = [0.5, 5]
    # y_interval = [0.5, 5]
    # constraints = [
    #     total_cycles_value.And(hardwareModel.symbolic_latency["Add"] >= x_interval[0], hardwareModel.symbolic_latency["Add"] <= x_interval[1]),
    #     total_cycles_value.And(hardwareModel.symbolic_latency["Regs"] >= y_interval[0], hardwareModel.symbolic_latency["Regs"] <= y_interval[1])
    # ]
    
    # prime_Add = total_cycles_value.diff(hardwareModel.symbolic_latency["Add"])
    # prime_Regs = total_cycles_value.diff(hardwareModel.symbolic_latency["Regs"])
    # stationary_points = solve([prime_Add, prime_Regs], [hardwareModel.symbolic_latency["Add"], hardwareModel.symbolic_latency["Regs"]], dict=True)
    # print("stationary_points", stationary_points)
    # valid_critical_points = [point for point in stationary_points if all(con.subs(point) for con in constraints)]

    
    


    # print(node_sum_cycles_value)
    # {'1': 2.0, '102': 6.0, '29': 2.0, '7': 1.0, '9': 11.76, '31': 5.0, '33': 4.12, '39': 4.06, '36': 4.06, '43': 3.0, '60': 4.06, '73': 3.0, '84': 4.0, '90': 5.359999999999999, '17': 1.0, '21': 20.759999999999998}
    # /home/ubuntu/codesign/src/cfg/benchmarks/models/testme.py
if __name__ == '__main__':
    main()
    
    
    

