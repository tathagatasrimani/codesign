from hardwareModel import HardwareModel
from ast_utils import ASTUtils
import schedule
import dfg_algo
import symbolic_dfg_algo
import symbolic_schedule
import matplotlib.pyplot as plt
import ast
import hardwareModel
import math
import json
import sys

data = {}
cycles = 0
main_cfg = None
id_to_node = {}
path = '/home/ubuntu/codesign/src/cfg/benchmarks/' # change path variable for local computer
data_path = []
power_use = []
node_intervals = []
node_avg_power = {}
unroll_at = {}

def func_calls(expr, calls):
    if type(expr) == ast.Call:
        calls.append(expr.func.id)
    for sub_expr in ASTUtils.get_sub_expr(expr):
        func_calls(sub_expr, calls)

def get_hw_need(state):
    hw_need = HardwareModel(0,0)
    for op in state:
        if not op.operation: continue
        else: hw_need.hw_allocated[op.operation] += 1
    return hw_need.hw_allocated

def cycle_sim(hw_inuse, max_cycles):
    global cycles, data, power_use
    node_power_sum = 0
    for i in range(max_cycles):
        power_use.append(0)
        #print("This is during cycle " + str(cycles))
        #print(hw_inuse)
        #print("")
        # save current state of hardware to data array
        cur_data = ""
        for elem in hw_inuse:
            if len(hw_inuse[elem]) > 0:
                cur_data += elem + ": "
                count = 0
                for i in hw_inuse[elem]:
                    if i > 0:
                        count += 1
                        power_use[cycles] += hardwareModel.power[elem][2]
                power_use[cycles] += (hardwareModel.power[elem][2] / 10) * len(hw_inuse[elem]) # passive power
                cur_data += str(count) + "/" + str(len(hw_inuse[elem])) + " in use. || "
        data[cycles] = cur_data
        # simulate one cycle
        for elem in hw_inuse:
            for j in range(len(hw_inuse[elem])):
                if hw_inuse[elem][j] > 0:
                    hw_inuse[elem][j] = max(0, hw_inuse[elem][j] - 1)
        node_power_sum += power_use[cycles]
        cycles += 1
    return node_power_sum

def simulate(cfg, data_path, node_operations, hw_spec, first):
    global main_cfg, id_to_node, unroll_at
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    #print(hw_inuse)
    i = 0
    while i < len(data_path):
        node_id = data_path[i][0]
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
        start_cycles = cycles # for calculating average power
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
        for state in node_operations[cur_node]:
            # if unroll, take each operation in a state and create more of them
            if unroll_at[cur_node.id]:
                new_state = state.copy()
                for op in state:
                    for j in range(iters):
                        new_state.append(op)
                state = new_state
            hw_need = get_hw_need(state)
            #print(hw_need)
            max_cycles = 0
            for elem in hw_need:
                cur_elem_count = hw_need[elem]
                if cur_elem_count == 0: continue
                if hw_spec[elem] == 0 and cur_elem_count > 0:
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = int(math.ceil(cur_elem_count / hw_spec[elem]) * hardwareModel.latency[elem])
                # symbolic max
                # https://www.sympy.org/en/index.html
                # plug in the number and see if it's correct
                # print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                max_cycles = max(cur_cycles_needed, max_cycles)
                j = 0
                while cur_elem_count > 0:
                    hw_inuse[elem][j] += hardwareModel.latency[elem]
                    j = (j + 1) % hw_spec[elem]
                    cur_elem_count -= 1
            node_avg_power[node_id] += cycle_sim(hw_inuse, max_cycles)
        if cycles - start_cycles > 0: node_avg_power[node_id] /= cycles - start_cycles
        node_intervals[-1][1][1] = cycles
        i += 1
    print("done with simulation")
    return data


def symbolic_simulate(cfg, data_path, node_operations, hw_spec, first):
    global main_cfg, id_to_node, unroll_at
    cur_node = cfg.entryblock
    if first: 
        cur_node = cur_node.exits[0].target # skip over the first node in the main cfg
        main_cfg = cfg
    hw_inuse = {}
    for elem in hw_spec:
        hw_inuse[elem] = [0] * hw_spec[elem]
    #print(hw_inuse)
    i = 0
    while i < len(data_path):
        node_id = data_path[i][0]
        cur_node = id_to_node[node_id]
        node_intervals.append([node_id, [cycles, 0]])
        node_avg_power[node_id] = 0 # just reset because we will end up overwriting it
        start_cycles = cycles # for calculating average power
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
        for state in node_operations[cur_node]:
            # if unroll, take each operation in a state and create more of them
            if unroll_at[cur_node.id]:
                new_state = state.copy()
                for op in state:
                    for j in range(iters):
                        new_state.append(op)
                state = new_state
            hw_need = get_hw_need(state)
            #print(hw_need)
            max_cycles = 0
            for elem in hw_need:
                cur_elem_count = hw_need[elem]
                if cur_elem_count == 0: continue
                if hw_spec[elem] == 0 and cur_elem_count > 0:
                    raise Exception("hardware specification insufficient to run program")
                cur_cycles_needed = int(math.ceil(cur_elem_count / hw_spec[elem]) * hardwareModel.latency[elem])
                # symbolic max
                # https://www.sympy.org/en/index.html
                # plug in the number and see if it's correct
                # print("cycles needed for " + elem + ": " + str(cur_cycles_needed) + ' (element count = ' + str(cur_elem_count) + ')')
                max_cycles = max(cur_cycles_needed, max_cycles)
                j = 0
                while cur_elem_count > 0:
                    hw_inuse[elem][j] += hardwareModel.latency[elem]
                    j = (j + 1) % hw_spec[elem]
                    cur_elem_count -= 1
            node_avg_power[node_id] += cycle_sim(hw_inuse, max_cycles)
        if cycles - start_cycles > 0: node_avg_power[node_id] /= cycles - start_cycles
        node_intervals[-1][1][1] = cycles
        i += 1
    print("done with simulation")
    return data

def main():
    global power_use, unroll_at
    benchmark = sys.argv[1]
    print(benchmark)
    cfg, graphs, unroll_at = symbolic_dfg_algo.main_fn(path, benchmark)
    cfg, node_operations = symbolic_schedule.schedule(cfg, graphs, sys.argv[1])
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
    data = simulate(cfg, data_path, node_operations, hw.hw_allocated, True)
    text = json.dumps(data, indent=4)
    names = sys.argv[1].split('/')
    with open(path + 'json_data/' + names[-1], 'w') as fh:
        fh.write(text)
    t = []
    for i in range(len(power_use)):
        t.append(i)
    plt.plot(t,power_use)
    plt.title("power use for " + names[-1])
    plt.xlabel("Cycle")
    plt.ylabel("Power")
    plt.savefig("cfg/benchmarks/power_plots/power_use_" + names[-1] + ".pdf")
    plt.clf() 
    print("done!")

if __name__ == '__main__':
    main()
    
    
    
